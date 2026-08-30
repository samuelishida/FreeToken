#!/usr/bin/env python3
"""Emit and validate a header-only manifest for split Qwen4-Exp GGUF."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


EXPECTED_TYPES = {0, 8, 12, 13, 14, 17, 18, 20, 30}


def inspect_gguf(path: str | Path) -> dict:
    from freetoken.models.gguf.reader import load_gguf_headers

    metadata, shards, headers = load_gguf_headers(str(path))
    tensors = [
        {
            "name": h.name,
            "ggml_type": h.ggml_type,
            "shape": list(h.shape),
            "ggml_shape": list(h.ggml_shape),
            "row_bytes": h.row_bytes,
            "nbytes": h.nbytes,
            "offset": h.data_offset,
            "shard": h.shard_index + 1,
        }
        for h in headers
    ]
    ple = next((t for t in tensors if t["name"] == "per_layer_token_embd.weight"), None)
    manifest = {
        "format": "qwen4exp-gguf-manifest-v1",
        "model": str(path),
        "architecture": metadata.get("general.architecture"),
        "shards": [
            {"index": s.index + 1, "count": s.count, "tensor_count": s.tensor_count,
             "bytes": s.file_size}
            for s in shards
        ],
        "metadata": {
            k: metadata[k]
            for k in sorted(metadata)
            if k.startswith("qwen4exp.")
        },
        "type_counts": {str(k): v for k, v in sorted(Counter(h.ggml_type for h in headers).items())},
        "types": sorted({h.ggml_type for h in headers}),
        "ple": ple,
        "tensors": tensors,
    }
    return manifest


def verify_manifest(manifest: dict) -> None:
    if manifest.get("architecture") != "qwen4exp":
        raise ValueError(f"expected qwen4exp, got {manifest.get('architecture')!r}")
    shards = manifest.get("shards", [])
    if len(shards) != 3 or [s["index"] for s in shards] != [1, 2, 3]:
        raise ValueError(f"expected ordered three-shard GGUF, got {shards}")
    types = set(manifest.get("types", []))
    missing = EXPECTED_TYPES - types
    if missing:
        raise ValueError(f"target GGUF missing expected types: {sorted(missing)}")
    ple = manifest.get("ple")
    if ple is None or ple["shape"] != [320001536, 160] or ple["ggml_type"] != 20 or ple["shard"] != 2:
        raise ValueError(f"unexpected Qwen4-Exp PLE tensor: {ple}")
    if manifest["metadata"].get("qwen4exp.ple.layers") != [1]:
        raise ValueError("expected qwen4exp.ple.layers=[1] for GGUF layer blk.1")
    names = [t["name"] for t in manifest.get("tensors", [])]
    if len(names) != len(set(names)):
        raise ValueError("duplicate tensor name")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    manifest = inspect_gguf(args.model)
    if args.verify:
        verify_manifest(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} ({len(manifest['tensors'])} tensors)")


if __name__ == "__main__":
    main()
