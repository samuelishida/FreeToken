"""TinygradModelRunner: FreeToken engine seam over tinygrad's Transformer.

``--device tinygrad`` runs the model through the tinygrad fork's AMD backend
(direct kfd/hsa ioctls -- no ROCm userspace, no Vulkan). The runner owns ONE
``Transformer`` instance (single-request stateful: per-block KV cache +
GatedDeltaNet recurrent state) and maps FreeToken batches to
``model.logits()`` calls.

Scope: ``max_running_req=1`` (the scheduler serializes requests; the tinygrad
Transformer is batch=1). Multi-request is a future extension.

JIT notes (mirrors ``Transformer.generate``):
- ``start_pos`` and the prefill token count are UOp variables bound at the
  call site, so the AMD flash kernels see symbolic shapes. The prefill flash
  kernel pads the query tile to a multiple of BLOCK_M=32 internally and
  slices garbage rows off; the decode kernel requires ``max_context % 128 == 0``
  (block_n=128), so the runner rounds ``max_len`` up to a multiple of 128.
- The bind must happen OUTSIDE the JIT'd function (like ``generate``), or the
  JIT's variable bookkeeping breaks on the second decode call.
"""

from __future__ import annotations

import logging
import random

import numpy as np
import torch

from freetoken.core import Batch


class TinygradModelRunner:
    def __init__(
        self,
        model_path: str,
        model_config,
        max_len: int,
        max_batch: int = 256,
        max_slots: int = 1,
    ) -> None:
        if max_slots > 1:
            raise NotImplementedError(
                "--device tinygrad supports max_running_req=1 only "
                "(the tinygrad Transformer is single-request stateful)"
            )
        from tinygrad import Tensor, TinyJit, UOp
        from tinygrad.llm.model import Transformer

        # AMD flash decode kernel: max_kv_len % block_n(128) == 0.
        max_context = max_len
        if max_context % 128:
            max_context = (max_context // 128 + 1) * 128
        self.max_len = max_context
        self.max_batch = max_batch
        self.vocab_size = model_config.vocab_size

        self.model, self._kv = Transformer.from_gguf(
            model_path, max_context=max_context
        )
        self._Tensor = Tensor

        # JIT specialization: symbolic start_pos + prefill token count, bound at
        # the call site (see module docstring).
        self._v_sp = UOp.variable("start_pos", 0, max_context - 1)
        self._v_nt = UOp.variable("n_toks", 1, max_batch)
        self._buf = Tensor.zeros(1, max_context, dtype="int32")
        # Host mirror of the prompt buffer: the scheduler sends each prefill
        # chunk as batch.input_ids (the extend only), so the runner accumulates
        # the full prompt here and copies the chunk at its global position.
        self._buf_np = np.zeros((1, max_context), dtype=np.int32)

        def _prefill(tokens_buf, sp, nt):
            return self.model.logits(tokens_buf[:, sp : sp + nt], sp)

        def _decode(tokens, sp):
            return self.model.logits(tokens, sp)

        self._prefill_jit = TinyJit(_prefill)
        self._decode_jit = TinyJit(_decode)
        self._warmup()
        # GPU-resident greedy decode state (see forward_greedy): a persistent
        # realized token buffer fed back from the in-graph argmax — no host
        # round-trip per step (an H2D copy costs ~10-13 ms of fixed queue
        # latency on this kfd path).
        self._tok_buf = self._Tensor(np.zeros((1, 1), dtype=np.int32)).realize()
        self._greedy_last_id: int | None = None

    def _warmup(self) -> None:
        """Compile both JIT graphs and realize the weights once at init.

        TinyJit's first call is eager, the second captures, the third executes;
        the warmup runs each JIT twice with the real request shapes (256-token
        prefill chunk + decode at a non-trivial position) so the first request
        doesn't pay a recompile.
        """
        warm = np.array([random.randint(0, 1000) for _ in range(256)], dtype=np.int32)
        self._buf_np[0, :256] = warm
        self._buf.assign(self._Tensor(self._buf_np))
        sp, nt = self._v_sp.bind(0), self._v_nt.bind(256)
        lg = self._prefill_jit(self._buf, sp, nt).realize()  # eager
        lg = self._prefill_jit(self._buf, sp, nt).realize()  # capture
        sp = self._v_sp.bind(256)
        tok = np.array([[int(lg.argmax().item())]], dtype=np.int32)
        self._decode_jit(self._Tensor(tok), sp).realize()  # eager
        self._decode_jit(self._Tensor(tok), sp).realize()  # capture
        # Return the eager warmup's cached scratch to the driver (~0.6 GB of
        # allocator LRU): the VRAM headroom gate needs it at 128K context.
        try:
            from tinygrad import Device
            from tinygrad.nn.state import get_state_dict

            dev = next(iter(get_state_dict(self.model).values())).device
            Device[dev].allocator.free_cache()
        except Exception as exc:
            logging.debug("tinygrad_runner: free_cache() failed: %s", exc)

    def forward(self, batch: Batch) -> torch.Tensor:
        """Logits [nreq, V] (last token of each req's extend) as a CPU tensor.

        Mirrors ``VulkanModelRunner.forward``'s return contract: one row per
        request, the logits of the last token of the request's extend, which
        ``sample_cpu`` + the scheduler's ``_process_last_data`` consume.
        """
        assert len(batch.reqs) == 1, "tinygrad runner: max_running_req=1 only"
        req = batch.reqs[0]
        cached_len = req.cached_len
        # The scheduler's batch.input_ids is the EXTEND (the prefill chunk or the
        # single decode token), not the full sequence.
        ids = batch.input_ids.cpu().numpy().astype(np.int32)
        n_toks = len(ids)
        if n_toks == 0:
            # Empty extend: the scheduler never sends it; return zeros defensively.
            return torch.zeros((1, self.vocab_size), dtype=torch.float32)

        sp = self._v_sp.bind(cached_len)
        if n_toks == 1:
            lg = self._decode_jit(self._Tensor(ids.reshape(1, 1)), sp)
        else:
            # Accumulate the chunk at its global position and run the symbolic
            # prefill (the AMD flash kernel pads the query tile internally).
            self._buf_np[0, cached_len : cached_len + n_toks] = ids
            self._buf.assign(self._Tensor(self._buf_np))
            nt = self._v_nt.bind(n_toks)
            lg = self._prefill_jit(self._buf, sp, nt)
        logits = lg.realize().numpy().astype(np.float32)
        return torch.from_numpy(logits)

    def forward_greedy(self, batch: Batch) -> int:
        """Greedy decode step with zero per-step host transfers.

        The input token stays on the device: forward_greedy reads the device
        token buffer, runs the decode graph, takes argmax ON THE GPU, feeds the
        sampled token back into the device buffer (device->device copy), and
        returns the id to the host (4-byte D2H). Only the FIRST greedy step
        after a prefill/new request reads batch.input_ids (one H2D, since the
        device buffer is stale then).

        Enforces consistency with the scheduler: batch.input_ids must carry the
        id this runner returned last time; on divergence (new request) the
        buffer is re-injected from host.
        """
        assert len(batch.reqs) == 1, "tinygrad runner: max_running_req=1 only"
        req = batch.reqs[0]
        cached_len = req.cached_len
        ids_host = batch.input_ids.cpu().numpy().astype(np.int32)
        if self._greedy_last_id is None or int(ids_host[0]) != self._greedy_last_id:
            # first greedy step after a prefill / request switch: re-inject
            self._tok_buf.assign(self._Tensor(ids_host.reshape(1, 1)))
        sp = self._v_sp.bind(cached_len)
        lg = self._decode_jit(self._tok_buf, sp)
        nxt = lg.argmax(-1, keepdim=True).realize()  # (1,1,1) int32, on device
        tok_id = int(nxt.numpy().flatten()[0])  # 4-byte D2H
        # feed the sampled token back for the next step (device->device)
        self._tok_buf.assign(nxt.cast("int32"))
        self._greedy_last_id = tok_id
        return tok_id

    def close(self) -> None:
        self.model = None
        self._kv = None
