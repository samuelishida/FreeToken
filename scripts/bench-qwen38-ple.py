#!/usr/bin/env python3
"""Record one-user Qwen4-Exp PLE cold/warm/cache-thrash metrics."""
from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path


def run_case(config: dict) -> dict:
    """Run supplied callback and attach process RSS; callback owns serving setup."""
    callback = config.get("run")
    if callback is None: raise ValueError("benchmark config requires run callback")
    started = time.perf_counter()
    result = dict(callback(config))
    result["elapsed_s"] = time.perf_counter() - started
    result["rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if result.get("finite_logits") is not True: raise RuntimeError("benchmark output is not finite")
    return result


def assert_memory_budget(result: dict) -> None:
    cap = int(result.get("ram_cap_bytes", 0) or 0)
    if cap and result.get("rss_bytes", 0) > cap: raise MemoryError("process RSS exceeded configured RAM cap")
    if result.get("swap_in_bytes", 0): raise MemoryError("swap-in observed during benchmark")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ple-store", required=True)
    parser.add_argument("--decode", type=int, default=128)
    args = parser.parse_args()
    # Launch orchestration is intentionally explicit: this script does not invent a model path
    # or hide server instrumentation. Callers can import run_case with a serving callback.
    result = {"model": args.model, "ple_store": args.ple_store, "decode_tokens": args.decode,
              "finite_logits": False, "error": "provide an instrumented run callback"}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(2)


if __name__ == "__main__": main()
