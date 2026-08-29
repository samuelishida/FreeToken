"""Startup split timing for the tinygrad runner (kernel-disk-cache question).

Times the REAL server init phases with a warm kernel DB:
  model build (Transformer.from_gguf) vs runner warmup (4 TinyJit calls:
  prefill eager/capture + decode eager/capture) vs the first real prefill chunk
  and first decode.

Usage:
  .venv-rocm/bin/python scripts/time-startup.py --model <gguf> [--max-seq-len 131072]
"""
import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "python")
torch.set_grad_enabled(False)

from freetoken.distributed import DistributedInfo  # noqa: E402
from freetoken.engine import Engine, EngineConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-seq-len", type=int, default=131072)
    a = ap.parse_args()

    import tinygrad.engine.jit as tj
    import tinygrad.llm.gguf as gg
    tj.TinyJit = tj._TinyJit  # the class, not the @overload wrapper function

    orig_parse = gg._gguf_parse
    t_parse = {}

    def timed_parse(tensor):
        t = time.monotonic()
        r = orig_parse(tensor)
        t_parse["s"] = time.monotonic() - t
        return r

    gg._gguf_parse = timed_parse
    import tinygrad.llm.model as gmod

    orig_gguf = gmod.Transformer.from_gguf
    t_gguf = {}

    def timed_from_gguf(*args, **kwargs):
        t = time.monotonic()
        r = orig_gguf(*args, **kwargs)
        t_gguf["s"] = time.monotonic() - t
        return r

    gmod.Transformer.from_gguf = staticmethod(timed_from_gguf)

    orig_call = tj.TinyJit.__call__
    t_calls = []

    def timed_call(self, *args, **kwargs):
        t = time.monotonic()
        r = orig_call(self, *args, **kwargs)
        t_calls.append((getattr(self.fxn, "__name__", "?"), time.monotonic() - t))
        return r

    tj.TinyJit.__call__ = timed_call

    t0 = time.monotonic()
    eng = Engine(EngineConfig(
        model_path=a.model, tp_info=DistributedInfo(0, 1), dtype=torch.float16,
        device="tinygrad", max_running_req=1, max_seq_len_override=a.max_seq_len))
    t_init = time.monotonic() - t0

    print(f"=== startup split (max_context={a.max_seq_len}) ===", flush=True)
    print(f"gguf file read/parse:        {t_parse.get('s', float('nan')):8.1f}s", flush=True)
    print(f"model build (from_gguf):   {t_gguf.get('s', float('nan')):8.1f}s", flush=True)
    for i, (name, dt) in enumerate(t_calls):
        print(f"TinyJit call {i}: {name:12s} {dt:8.1f}s", flush=True)
    print(f"runner init total:          {t_init:8.1f}s", flush=True)


if __name__ == "__main__":
    main()
