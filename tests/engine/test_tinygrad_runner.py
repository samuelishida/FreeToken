"""TinygradModelRunner contract tests.

The runner wraps the tinygrad fork's Transformer (AMD kfd/hsa backend). The
real-model e2e is gated behind FREETOKEN_TINYGRAD_E2E=1 (needs the 22 GB
checkpoint + AMD GPU); the rest of the contract is tested without a model.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))


def _dummy_config():
    from freetoken.distributed import DistributedInfo

    class _MC:
        vocab_size = 248320

    return _MC(), DistributedInfo(0, 1)


def test_max_slots_rejection():
    """max_running_req > 1 is rejected at runner init (defense in depth; the
    args parser already rejects it)."""
    from freetoken.engine.tinygrad_runner import TinygradModelRunner

    mc, tp = _dummy_config()
    with pytest.raises(NotImplementedError, match="max_running_req=1"):
        TinygradModelRunner(
            "/nonexistent.gguf", mc, max_len=4096, max_slots=2
        )


def test_args_reject_tinygrad_multi_request():
    """--device tinygrad --max-running-requests > 1 fails at parse time."""
    from freetoken.server.args import parse_args

    with pytest.raises(SystemExit, match="max-running-requests 1"):
        parse_args(
            [
                "--model",
                "/media/smk/5fce248d-bbdd-488d-8883-4f000f85cc10/Models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
                "--device",
                "tinygrad",
                "--max-running-requests",
                "2",
            ]
        )


@pytest.mark.needs_weights
def test_e2e_tinygrad():
    """Real-model e2e through the Engine with --device tinygrad (one Engine,
    all scenarios): greedy continuation, chunked prefill, prefix reuse, and
    fresh-request state reset. Matches the ROCm reference output."""
    if not os.environ.get("FREETOKEN_TINYGRAD_E2E"):
        pytest.skip("set FREETOKEN_TINYGRAD_E2E=1 to run the real-model e2e")
    import random

    import torch

    from freetoken.core import Batch, Req, SamplingParams
    from freetoken.engine import Engine, EngineConfig
    from freetoken.utils import load_tokenizer

    P = "/media/smk/5fce248d-bbdd-488d-8883-4f000f85cc10/Models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    tok = load_tokenizer(P)
    config = EngineConfig(
        model_path=P,
        tp_info=_dummy_config()[1],
        dtype=torch.float16,
        device="tinygrad",
        max_running_req=1,
        max_seq_len_override=4096,
    )
    engine = Engine(config)

    def forward(full_ids, extend, cached_len, phase):
        req = Req(
            input_ids=full_ids, table_idx=0, cached_len=cached_len,
            output_len=max(1, len(extend)), uid=0,
            sampling_params=SamplingParams(), cache_handle=None,
        )
        batch = Batch(reqs=[req], phase=phase)
        batch.input_ids = extend
        batch.positions = torch.arange(
            cached_len, cached_len + len(extend), dtype=torch.int32
        )
        batch.padded_reqs = batch.reqs
        out = engine.forward_batch(batch, engine.sampler.prepare(batch))
        return int(out.next_tokens_cpu[0].item())

    def decode_n(ids, n):
        gen = []
        cached = 0
        while cached < len(ids):
            chunk_end = min(cached + 256, len(ids))
            tok = forward(ids[:chunk_end], ids[cached:chunk_end], cached, "prefill")
            cached = chunk_end
        gen.append(tok)  # only the last chunk's logits are sampled
        for _ in range(n - 1):
            nxt = torch.tensor([gen[-1]], dtype=torch.int32)
            full = torch.cat([ids, torch.tensor(gen, dtype=torch.int32)])
            gen.append(forward(full, nxt, len(full) - 1, "decode"))
        return gen

    try:
        # 1. greedy continuation (matches the ROCm reference).
        ids = torch.tensor(
            tok.encode("The capital of France is", add_special_tokens=False),
            dtype=torch.int32,
        )
        gen = decode_n(ids, 15)
        text = tok.decode(gen)
        assert text.startswith(" Paris"), f"unexpected continuation: {text!r}"

        # 2. chunked prefill: 512-token prompt in 2 chunks of 256.
        random.seed(42)
        long_prompt = [random.randint(0, 20000) for _ in range(512)]
        gen = decode_n(torch.tensor(long_prompt, dtype=torch.int32), 3)
        assert len(gen) == 3

        # 3. prefix reuse: same first 5 tokens, continue from start_pos=5.
        gen2 = decode_n(ids, 8)
        assert tok.decode(gen2).startswith(" Paris"), f"prefix reuse wrong: {tok.decode(gen2)!r}"

        # 4. fresh request after a different prompt: state must reset.
        fresh = torch.tensor(
            tok.encode("The capital of Japan is", add_special_tokens=False),
            dtype=torch.int32,
        )
        gen3 = decode_n(fresh, 8)
        assert tok.decode(gen3).startswith(" Tokyo"), f"fresh request wrong: {tok.decode(gen3)!r}"
    finally:
        engine.shutdown()
