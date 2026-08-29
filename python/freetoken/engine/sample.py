from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from freetoken.utils import is_sm90_supported, nvtx_annotate

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from freetoken.core import Batch, Req


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    # True when at least one request carries a presence/frequency penalty; the sampler
    # then lowers each request's logits over its generated tokens before sampling.
    apply_penalties: bool = False


def apply_penalties(
    logits: torch.Tensor,
    reqs: List["Req"],
) -> None:
    """Apply OpenAI presence/frequency penalties to ``logits`` in place (row per req).

    For a token ``t`` the request already generated, its score is lowered by
    ``presence_penalty`` plus ``frequency_penalty * count(t)``. The prompt is excluded
    (only ``input_ids[req.prompt_len:]`` counts), so the penalty grows with the
    generation itself -- positive values push the model away from repeating itself,
    which breaks reasoning loops; negative values nudge it toward repetition.
    """
    for i, req in enumerate(reqs):
        sp = req.sampling_params
        pp, fp = sp.presence_penalty, sp.frequency_penalty
        if not pp and not fp:
            continue
        gen = req.input_ids[req.prompt_len :]
        if gen.numel() == 0:
            continue
        uniq, counts = torch.unique(gen, return_counts=True)
        vals = torch.full_like(counts, pp, dtype=torch.float32) + fp * counts.to(
            torch.float32
        )
        logits[i, uniq.to(logits.device)] -= vals.to(logits.device)


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    # pin_memory needs CUDA; the tinygrad path is CPU-only.
    return torch.tensor(data, dtype=dtype, pin_memory=device.type == "cuda").to(
        device, non_blocking=True
    )


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    from freetoken.kernel.backend import is_flashinfer_installed

    if is_flashinfer_installed():
        import flashinfer.sampling as sampling
    else:
        import freetoken.kernel.triton.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        apply_penalties = any(
            p.presence_penalty != 0.0 or p.frequency_penalty != 0.0 for p in params
        )
        if all(p.is_greedy for p in params) and not apply_penalties:
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(
            temperatures,
            top_k=top_k,
            top_p=top_p,
            apply_penalties=apply_penalties,
        )

    @nvtx_annotate("Sampler")
    def sample_cpu(
        self, logits: torch.Tensor, args: BatchSamplingArgs, batch: Batch
    ) -> torch.Tensor:
        """Tinygrad path: logits are host-side torch CPU tensors; plain-torch sampler."""
        if args.apply_penalties:
            apply_penalties(logits, batch.reqs)
        # A malformed forward here previously killed the scheduler process with
        # ``torch.multinomial: probability tensor contains either inf, nan ...``
        # (backend exit with no restart). Clamp instead so one bad forward costs
        # a flagged token, not the server. NOTE: the greedy path in
        # engine.forward_batch calls runner.forward_greedy (GPU argmax) and never
        # reaches this guard — content regressions there are caught by the
        # decode-nan probe / serve smoke, not here.
        if not torch.isfinite(logits).all():
            batch_uid = batch.reqs[0].uid if batch.reqs else -1
            logger.error(
                "sampler: non-finite logits (uid=%s, nan=%d, inf=%d); clamping to "
                "-1e9 (root cause is upstream in the model forward, not sampling)",
                batch_uid, int(torch.isnan(logits).sum()), int(torch.isinf(logits).sum()),
            )
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=-1e9, neginf=-1e9)
        if args.temperatures is None:  # greedy
            return torch.argmax(logits, dim=-1)
        temps = args.temperatures.to(torch.float32)
        probs = torch.softmax(logits / temps.unsqueeze(-1), dim=-1)
        top_k = args.top_k if args.top_k is not None else logits.shape[-1]
        if args.top_p is not None:
            sorted_p, sorted_idx = probs.sort(descending=True, dim=-1)
            cum = sorted_p.cumsum(dim=-1)
            keep = cum - sorted_p < args.top_p.unsqueeze(-1)
            keep = keep | (keep.cumsum(dim=-1) < top_k.unsqueeze(-1))
            mask = torch.zeros_like(probs, dtype=torch.bool).scatter_(-1, sorted_idx, keep)
            probs = probs * mask
            probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-12)
        else:
            probs, idx = probs.topk(int(top_k), dim=-1)
            probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-12)
            return idx.gather(-1, torch.multinomial(probs, 1))
        return torch.multinomial(probs, 1).squeeze(-1)

    def sample(
        self, logits: torch.Tensor, args: BatchSamplingArgs, batch: Batch
    ) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.apply_penalties:
                apply_penalties(logits, batch.reqs)
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
