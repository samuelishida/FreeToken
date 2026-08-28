"""Benchmark prefill/decode tok/s for the --device tinygrad path.

Warm JIT (the runner warms up both JIT graphs at init, so the one-time compile
is excluded). Prefill is measured as chunked prefill (256-token chunks, the
scheduler's cap); decode is measured as single-token steps at the given
context length.

Usage:
  .venv-rocm/bin/python scripts/bench-tinygrad.py --model /path/to/model.gguf
                              [--ctx 4096,16384,65536,131072]
                              [--decode 20] [--kernels]

--kernels also reports the number of captured kernels per decode step
(counted from the JIT's captured graph; falls back to wall-clock only if the
internal API is unavailable).
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from typing import Optional

import torch

sys.path.insert(0, "python")
from freetoken.core import Batch, Req, SamplingParams
from freetoken.distributed import DistributedInfo
from freetoken.engine import Engine, EngineConfig
from freetoken.utils import load_tokenizer


def make_engine(model_path: str, max_seq_len_override: int) -> Engine:
    tok = load_tokenizer(model_path)
    config = EngineConfig(
        model_path=model_path,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.float16,
        device="tinygrad",
        max_running_req=1,
        max_seq_len_override=max_seq_len_override,
    )
    engine = Engine(config)
    return engine


def forward(
    engine: Engine,
    full_ids: torch.Tensor,
    extend: torch.Tensor,
    cached_len: int,
    phase: str,
) -> int:
    req = Req(
        input_ids=full_ids,
        table_idx=0,
        cached_len=cached_len,
        output_len=max(1, len(extend)),
        uid=0,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )
    batch = Batch(reqs=[req], phase=phase)
    batch.input_ids = extend
    batch.positions = torch.arange(
        cached_len, cached_len + len(extend), dtype=torch.int32
    )
    batch.padded_reqs = batch.reqs
    out = engine.forward_batch(batch, engine.sampler.prepare(batch))
    return int(out.next_tokens_cpu[0].item())


def prefill_toks(engine: Engine, ids: torch.Tensor) -> tuple[float, int]:
    """Chunked prefill of ids; returns (elapsed, n_tokens)."""
    cached = 0
    t0 = time.monotonic()
    while cached < len(ids):
        chunk_end = min(cached + 256, len(ids))
        forward(engine, ids[:chunk_end], ids[cached:chunk_end], cached, "prefill")
        cached = chunk_end
    return time.monotonic() - t0, len(ids)


def decode_toks(
    engine: Engine, ids: torch.Tensor, n: int = 20
) -> tuple[float, int]:
    """n single-token decode steps after prefill; returns (elapsed, n)."""
    gen: list[int] = []
    cached = 0
    while cached < len(ids):
        chunk_end = min(cached + 256, len(ids))
        forward(engine, ids[:chunk_end], ids[cached:chunk_end], cached, "prefill")
        cached = chunk_end
    t0 = time.monotonic()
    for _ in range(n):
        nxt = torch.tensor([gen[-1] if gen else 0], dtype=torch.int32)
        full = torch.cat([ids, torch.tensor(gen, dtype=torch.int32)])
        gen.append(forward(engine, full, nxt, len(full) - 1, "decode"))
    return time.monotonic() - t0, n


def decode_kernel_count(engine: Engine) -> Optional[int]:
    """Number of kernels in the captured decode JIT graph, or None."""
    try:
        from tinygrad.uop.ops import Ops

        jit = engine.tinygrad_runner._decode_jit
        if getattr(jit, "captured", None) is None:
            return None
        lin = jit.captured.linear
        return sum(1 for u in lin.toposort() if u.op is Ops.LINEAR)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to the GGUF model")
    ap.add_argument("--ctx", default="4096,16384,65536,131072")
    ap.add_argument("--decode", type=int, default=20)
    ap.add_argument("--kernels", action="store_true")
    args = ap.parse_args()
    ctxs = [int(c) for c in args.ctx.split(",")]

    # Match the serve default (FT_KV_TOKENS=131072) so the tinygrad kernel
    # cache keys align: the runner's max_context is a multiple of 128, and
    # kernels are cached per-shape. Using 131584 here would recompile the
    # whole graph on a cold start.
    max_seq_len_override = ((max(ctxs) + 127) // 128) * 128
    engine = make_engine(args.model, max_seq_len_override)
    print("engine ready (JIT warm); runner max_len =", engine.tinygrad_runner.max_len)

    if args.kernels:
        kc = decode_kernel_count(engine)
        print(f"decode JIT kernel count: {kc if kc is not None else 'N/A (API unavailable)'}")

    random.seed(7)
    print(f"{'ctx':>7} {'prefill tok/s':>14} {'decode tok/s':>14}")
    for ctx in ctxs:
        prompt_len = ctx - args.decode - 32  # leave room for decode steps
        ids = torch.tensor(
            [random.randint(0, 20000) for _ in range(prompt_len)], dtype=torch.int32
        )
        pt, pn = prefill_toks(engine, ids)
        dt, dn = decode_toks(engine, ids, n=args.decode)
        print(f"{ctx:>7} {pn / pt:>14.1f} {dn / dt:>14.1f}")

    engine.shutdown()
    print("BENCH DONE")


if __name__ == "__main__":
    main()
