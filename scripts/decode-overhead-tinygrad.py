"""Decode step overhead breakdown for the --device tinygrad path.

Splits one decode step's wall time into its components:

  - full forward_batch wall   (the bench's per-step number, ~61 ms)
  - JIT dispatch              (Python schedule of the captured graph, ~39 ms)
  - GPU wait                  (realize/sync, ~0.05 ms)
  - kernel launches per step  (counted from a Context(DEBUG=2) trace)

Also: --cprofile runs the step loop under cProfile (top-20 cumulative) and
--hcq2 re-runs a short measurement with the fork's alternative dispatch path
(the HCQ2 knob in realize.py) to check whether it changes the dispatch cost.

Usage:
  .venv-rocm/bin/python scripts/decode-overhead-tinygrad.py --model /path/to/model.gguf
                              [--ctx 4096] [--steps 30] [--cprofile] [--hcq2]
"""
from __future__ import annotations

import argparse
import contextlib
import cProfile
import io
import os
import random
import re
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "python")
from freetoken.core import Batch, Req, SamplingParams
from freetoken.distributed import DistributedInfo
from freetoken.engine import Engine, EngineConfig
from freetoken.utils import load_tokenizer

PREFILL_CHUNK = 256


def build_engine(model_path: str, max_len: int) -> Engine:
    load_tokenizer(model_path)
    config = EngineConfig(
        model_path=model_path,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.float16,
        device="tinygrad",
        max_running_req=1,
        max_seq_len_override=max_len,
    )
    return Engine(config)


def forward(engine: Engine, full_ids: torch.Tensor, extend: torch.Tensor, cached_len: int, phase: str) -> int:
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
    batch.positions = torch.arange(cached_len, cached_len + len(extend), dtype=torch.int32)
    batch.padded_reqs = batch.reqs
    out = engine.forward_batch(batch, engine.sampler.prepare(batch))
    return int(out.next_tokens_cpu[0].item())


def prefill(engine: Engine, ids: torch.Tensor) -> None:
    """Chunked prefill of ids (advances the model state)."""
    cached = 0
    while cached < len(ids):
        chunk_end = min(cached + PREFILL_CHUNK, len(ids))
        forward(engine, ids[:chunk_end], ids[cached:chunk_end], cached, "prefill")
        cached = chunk_end


def run_steps(engine: Engine, full_ids: torch.Tensor, n: int) -> tuple[list[float], list[float], list[float]]:
    """n decode steps; returns (full_wall, dispatch, gpu_wait) in ms.

    Per step: (a) one full forward_batch wall sample, (b) one direct JIT
    dispatch sample (schedule without wait), (c) the realize/GPU-wait sample.
    The direct probe advances the model state by one more token (KV at the
    next position) — the same call the runner makes internally.
    """
    runner = engine.tinygrad_runner
    gen: list[int] = []
    full_wall: list[float] = []
    dispatch: list[float] = []
    gpu_wait: list[float] = []
    d2h: list[float] = []
    full = full_ids
    for _ in range(n):
        nxt = torch.tensor([gen[-1] if gen else 0], dtype=torch.int32)
        full = torch.cat([full, nxt])  # grow the sequence by the sampled token
        pos = len(full) - 1  # the single decode token sits at the last slot
        t0 = time.perf_counter()
        tok_id = forward(engine, full, nxt, pos, "decode")
        full_wall.append((time.perf_counter() - t0) * 1000)
        gen.append(tok_id)
        sp = runner._v_sp.bind(pos + 1)
        tok_np = np.array([[gen[-1]]], dtype=np.int32)
        t0 = time.perf_counter()
        out = runner._decode_jit(runner._Tensor(tok_np), sp)
        t1 = time.perf_counter()
        out.realize()
        t2 = time.perf_counter()
        _ = out.numpy()
        t3 = time.perf_counter()
        dispatch.append((t1 - t0) * 1000)
        gpu_wait.append((t2 - t1) * 1000)
        d2h.append((t3 - t2) * 1000)
    return full_wall, dispatch, gpu_wait, d2h


def count_launches(engine: Engine, pos: int, tok_id: int) -> int:
    """Kernel launches for one decode step, from a Context(DEBUG=2) trace."""
    from tinygrad.helpers import Context

    runner = engine.tinygrad_runner
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        with Context(DEBUG=2):
            sp = runner._v_sp.bind(pos)
            out = runner._decode_jit(runner._Tensor(np.array([[tok_id]], dtype=np.int32)), sp)
            out.realize()
    return len(re.findall(r"\*\*\* AMD", buf.getvalue()))


def median_ms(vals: list[float]) -> float:
    return statistics.median(vals) if vals else float("nan")


def cprofile_run(engine: Engine, full_ids: torch.Tensor, n: int) -> str:
    """Run the step loop under cProfile; returns top-20 cumulative text."""
    pr = cProfile.Profile()
    pr.enable()
    run_steps(engine, full_ids, n)
    pr.disable()
    s = io.StringIO()
    pstats = __import__("pstats").Stats(pr, stream=s)
    pstats.sort_stats("cumulative").print_stats(20)
    with open("/tmp/hawk-decode-overhead.log", "w") as f:
        f.write(s.getvalue())
    return s.getvalue()


def hcq2_probe(model_path: str, ctx: int, steps: int) -> tuple[int, str]:
    """Re-run a short measurement with HCQ2=1 (fork's alt dispatch path)."""
    env = {**os.environ, "HCQ2": "1"}
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--model", model_path,
        "--ctx", str(ctx),
        "--steps", str(steps),
    ]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return 1, "HCQ2 probe timed out"
    tail = (r.stdout or "")[-2500:] + ("\n" + (r.stderr or "")[-1500:])
    return r.returncode, tail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to the GGUF model")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cprofile", action="store_true")
    ap.add_argument("--hcq2", action="store_true", help="probe the fork's HCQ2 dispatch path")
    args = ap.parse_args()

    # The HCQ2 knob is read at tinygrad import time; probe it in a subprocess.
    if args.hcq2 and os.environ.get("HCQ2") != "1":
        rc, tail = hcq2_probe(args.model, args.ctx, 10)
        print(f"=== HCQ2 probe: {'OK' if rc == 0 else 'FAILED (rc=%d)' % rc} ===")
        print(tail)
        return

    engine = build_engine(args.model, args.ctx)
    random.seed(7)
    prompt_len = min(512, max(256, args.ctx // 4))
    ids = torch.tensor([random.randint(0, 20000) for _ in range(prompt_len)], dtype=torch.int32)
    prefill(engine, ids)
    print("engine ready (JIT warm); measuring", args.steps, "decode steps")
    if os.environ.get("HCQ2") == "1":
        print("HCQ2: ENABLED")

    full_wall, dispatch, gpu_wait, d2h = run_steps(engine, ids, args.steps)
    print(f"{'component':28} {'mediana ms':>10} {'min ms':>8} {'max ms':>8}")
    print(f"{'full forward_batch':28} {median_ms(full_wall):10.2f} {min(full_wall):8.2f} {max(full_wall):8.2f}")
    print(f"{'JIT dispatch (CPU)':28} {median_ms(dispatch):10.2f} {min(dispatch):8.2f} {max(dispatch):8.2f}")
    print(f"{'GPU wait (realize)':28} {median_ms(gpu_wait):10.2f} {min(gpu_wait):8.2f} {max(gpu_wait):8.2f}")
    print(f"{'D2H logits (numpy)':28} {median_ms(d2h):10.2f} {min(d2h):8.2f} {max(d2h):8.2f}")

    samples = [count_launches(engine, 512 + args.steps * 2 + i, 7) for i in range(3)]
    print(f"launches per decode step: median {statistics.median(samples)} (samples {samples})")

    if args.cprofile:
        print("\n=== cProfile top-20 (cumulative) — also /tmp/hawk-decode-overhead.log ===")
        print(cprofile_run(engine, ids, args.steps))

    engine.shutdown()
    print("OVERHEAD DONE")


if __name__ == "__main__":
    main()
