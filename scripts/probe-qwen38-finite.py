#!/usr/bin/env python3
"""Finite-logit trace for Qwen4-Exp tinygrad baseline or native runner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _digest(values) -> str:
    import hashlib
    import numpy as np
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def probe(model, prompt, steps: int, checkpoints=()) -> dict:
    import numpy as np
    import time
    report = {"format": "qwen38-finite-trace-v1", "steps": steps, "finite": True,
              "checkpoints": {name: [] for name in checkpoints}, "records": []}
    tokens = prompt
    for step in range(steps):
        started = time.perf_counter()
        logits = model.logits(tokens, step)
        values = np.asarray(logits.numpy() if hasattr(logits, "numpy") else logits)
        finite = bool(np.isfinite(values).all())
        record = {"position": step, "finite": finite, "logits_crc": _digest(values),
                  "elapsed_ms": (time.perf_counter() - started) * 1000.0}
        if finite: record["token_id"] = int(values.reshape(-1).argmax())
        report["records"].append(record)
        if not finite:
            report["finite"] = False; report["first_nonfinite"] = {"step": step}; break
        token = record["token_id"]
        try:
            from tinygrad import Tensor
            tokens = Tensor([[token]], dtype="int32")
        except ImportError:
            tokens = np.asarray([[token]], dtype=np.int32)
    return report


def record_trace(model_path: str, prompt_tokens: list[int], steps: int, max_context: int) -> dict:
    import os
    import sys
    root = os.environ.get("FREETOKEN_TINYGRAD_ROOT", "/media/smk/5fce248d-bbdd-488d-8883-4f000f85cc10/Code/ollama-tg/tg-fork")
    if root not in sys.path: sys.path.insert(0, root)
    from tinygrad import Tensor
    from tinygrad.llm.model import Transformer
    model, _ = Transformer.from_gguf(model_path, max_context=max_context)
    model.reset_state()
    return probe(model, Tensor([prompt_tokens], dtype="int32"), steps)


def record_native_trace(model_path: str, prompt_tokens: list[int], steps: int, max_context: int) -> dict:
    import time
    import torch
    from freetoken.models.qwen4_exp.model import Qwen4ExpNativeModel
    from freetoken.models.qwen4_exp.packed import Qwen4ExpPackedSource
    source = Qwen4ExpPackedSource(model_path)
    cfg = type("TraceConfig", (), {"model_path": model_path, "max_seq_len": max_context})()
    model = Qwen4ExpNativeModel(source, cfg, torch.device("cuda"))
    ids = torch.tensor([prompt_tokens], dtype=torch.long, device="cuda")
    model.reset()
    try:
        logits = model.forward(ids, 0)
        report = {"format": "qwen38-native-finite-trace-v1", "steps": steps,
                  "finite": True, "records": []}
        for step in range(steps):
            torch.cuda.synchronize(); started = time.perf_counter()
            if step:
                logits = model.forward(next_id, len(prompt_tokens) + step - 1)
            torch.cuda.synchronize()
            finite = bool(torch.isfinite(logits).all())
            values = logits.float().to("cpu")
            record = {"position": len(prompt_tokens) + step - 1, "finite": finite,
                      "logits_crc": _digest(values), "elapsed_ms": (time.perf_counter() - started) * 1000.0}
            if finite:
                next_id = torch.argmax(logits, dim=-1).reshape(1, 1).to("cuda")
                record["token_id"] = int(next_id.item())
            report["records"].append(record)
            if not finite:
                report["finite"] = False; report["first_nonfinite"] = {"step": step}; break
        return report
    finally:
        source.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt-tokens", default="248044,100")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--all-positions", type=int, metavar="N", help="run exactly N native decode positions")
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--device", choices=("tinygrad", "native-rocm"), default="tinygrad")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--checkpoints", default="ple,qsa,gdn,moe,head")
    args = parser.parse_args()
    if args.all_positions is not None: args.steps = args.all_positions
    prompt = [int(x) for x in args.prompt_tokens.split(",") if x]
    report = (record_trace(args.model, prompt, args.steps, args.max_context)
              if args.device == "tinygrad" else record_native_trace(args.model, prompt, args.steps, args.max_context))
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if args.strict and not report["finite"]: raise SystemExit(2)


if __name__ == "__main__": main()
