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
def test_e2e_greedy_continuation():
    """Greedy continuation of 'The capital of France is' -> ' Paris...' through
    the Engine with --device tinygrad (matches the ROCm reference)."""
    if not os.environ.get("FREETOKEN_TINYGRAD_E2E"):
        pytest.skip("set FREETOKEN_TINYGRAD_E2E=1 to run the real-model e2e")
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
    try:
        ids = torch.tensor(
            tok.encode("The capital of France is", add_special_tokens=False),
            dtype=torch.int32,
        )
        req = Req(
            input_ids=ids, table_idx=0, cached_len=0, output_len=16, uid=0,
            sampling_params=SamplingParams(), cache_handle=None,
        )
        batch = Batch(reqs=[req], phase="prefill")
        batch.input_ids = ids
        batch.positions = torch.arange(len(ids), dtype=torch.int32)
        batch.padded_reqs = batch.reqs
        out = engine.forward_batch(batch, engine.sampler.prepare(batch))
        gen = [int(out.next_tokens_cpu[0].item())]
        for _ in range(14):
            full = torch.cat([ids, torch.tensor(gen, dtype=torch.int32)])
            req2 = Req(
                input_ids=full, table_idx=0, cached_len=len(full) - 1,
                output_len=1, uid=0, sampling_params=SamplingParams(),
                cache_handle=None,
            )
            batch2 = Batch(reqs=[req2], phase="decode")
            batch2.input_ids = full
            batch2.positions = torch.arange(len(full), dtype=torch.int32)
            batch2.padded_reqs = batch2.reqs
            out2 = engine.forward_batch(batch2, engine.sampler.prepare(batch2))
            gen.append(int(out2.next_tokens_cpu[0].item()))
        text = tok.decode(gen)
        assert text.startswith(" Paris"), f"unexpected continuation: {text!r}"
    finally:
        engine.shutdown()
