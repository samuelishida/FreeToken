#!/usr/bin/env python3
"""Pack pre-extracted Qwen4-Exp PLE FP8 rows into the tinygrad page store.

The checkpoint inspector must produce/validate source rows first. This command
never downloads a checkpoint and never mutates its source directory.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ollama-tg" / "tg-fork"))
from tinygrad.llm.ple_store import PlePack, PleStore  # noqa: E402


def pack_ple(source: Path, destination: Path, *, replace: bool = False):
    if replace and destination.exists():
        for child in destination.iterdir():
            if child.is_file(): child.unlink()
    manifest = PlePack.create(source, destination)
    PleStore.open(destination / "manifest.json", ram_bytes=0).close()
    return manifest


def verify_packed_ple(manifest) -> None:
    store = PleStore.open(Path(manifest.root) / "manifest.json", ram_bytes=0)
    store.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="local directory containing extracted PLE *.bin rows")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    manifest = pack_ple(args.model, args.out, replace=args.replace)
    if args.verify: verify_packed_ple(manifest)
    print(f"packed {manifest.rows} rows into {manifest.root}")


if __name__ == "__main__": main()
