"""VRAM breakdown for the --device tinygrad path + headroom gate.

Reports total device bytes (from the amdgpu sysfs entries — the source of
truth), the model weights, the fp16/Q8 KV cache, the GDN recurrent state, and
the remainder (activations + overhead). Enforces the project's hard constraint:
free VRAM must stay >= 1 GB (headroom), exiting non-zero otherwise.

Usage:
  .venv-rocm/bin/python scripts/vram-tinygrad.py --model /path/to/model.gguf
                              [--max-len 131072] [--no-gate]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Optional

import torch

sys.path.insert(0, "python")
from freetoken.distributed import DistributedInfo
from freetoken.engine import Engine, EngineConfig

MIN_HEADROOM_GB = 1.0


def amdgpu_sysfs() -> tuple[int, int] | None:
    """Auto-detect the amdgpu render device and read VRAM used/total (bytes).

    Returns (used, total) or None if no sysfs entry is present (non-amdgpu /
    VM). The render device with mem_info_vram_total is the amdgpu GPU.
    """
    for p in glob.glob("/sys/class/drm/renderD*/device/mem_info_vram_total"):
        try:
            total = int(open(p).read().strip())
            used = int(open(p.replace("vram_total", "vram_used")).read().strip())
            return used, total
        except (OSError, ValueError):
            continue
    return None


def state_dict_nbytes(model) -> dict[str, int]:
    """Walk the model's tensors (name -> nbytes), skipping non-device bufs."""
    from tinygrad.nn.state import get_state_dict

    out: dict[str, int] = {}
    for name, t in get_state_dict(model).items():
        try:
            out[name] = t.nbytes()
        except Exception:
            continue
    return out


def breakdown(engine: Engine, model_path: str) -> dict[str, int]:
    """dict of category -> bytes. Weights use the GGUF file size (the packed
    Q4_K_M footprint); the state-dict nbytes() over-counts quantized weights
    by reporting their logical fp16 size. KV / GDN state are counted from
    their (unquantized) tensors."""
    model = engine.tinygrad_runner.model
    tensors = state_dict_nbytes(model)
    kv = sum(v for k, v in tensors.items() if "cache_kv" in k)
    state = sum(v for k, v in tensors.items() if "recurrent_state" in k)
    weights = os.path.getsize(model_path)
    return {"weights": weights, "kv": kv, "gdn_state": state, "misc": 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to the GGUF model")
    ap.add_argument("--max-len", type=int, default=131072)
    ap.add_argument("--no-gate", action="store_true", help="skip the headroom exit")
    args = ap.parse_args()

    engine = Engine(
        EngineConfig(
            model_path=args.model,
            tp_info=DistributedInfo(0, 1),
            dtype=torch.float16,
            device="tinygrad",
            max_running_req=1,
            max_seq_len_override=args.max_len,
        )
    )

    b = breakdown(engine, args.model)
    sysfs = amdgpu_sysfs()
    if sysfs is not None:
        used, total = sysfs
        total_b, used_b = total, used
        label = "sysfs"
    else:
        used_b = sum(b.values())
        total_b = used_b
        label = "sum-of-tensors (no sysfs)"

    free = total_b - used_b
    print(f"device bytes source : {label}")
    print(f"total               : {total_b / 1e9:8.2f} GB ({total_b / 1073741824:.2f} GiB)")
    print(f"used                : {used_b / 1e9:8.2f} GB ({used_b / 1073741824:.2f} GiB)")
    print(f"free (headroom)     : {free / 1e9:8.2f} GB")
    for k, v in b.items():
        print(f"  {k:12}: {v / 1e9:8.2f} GB")
    print(f"  {'remainder':12}: {(used_b - sum(b.values())) / 1e9:8.2f} GB (activations + overhead + JIT graphs)")

    engine.shutdown()

    if not args.no_gate and sysfs is not None and free < MIN_HEADROOM_GB * 1e9:
        print(f"HEADROOM GATE FAILED: free {free / 1e9:.2f} GB < {MIN_HEADROOM_GB} GB")
        sys.exit(1)
    if sysfs is None and not args.no_gate:
        print("WARNING: no amdgpu sysfs entry; headroom gate skipped")
    print("VRAM OK" if (args.no_gate or sysfs is None or free >= MIN_HEADROOM_GB * 1e9) else "VRAM GATE FAILED")


if __name__ == "__main__":
    main()
