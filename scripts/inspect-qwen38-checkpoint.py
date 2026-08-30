#!/usr/bin/env python3
"""Inspect a local Qwen4-Exp safetensors checkpoint without loading weights."""
from dataclasses import asdict, dataclass
import argparse, json, struct
from pathlib import Path

@dataclass(frozen=True)
class TensorRecord:
  name: str; dtype: str; shape: tuple[int, ...]; shard: str; bytes: int; ple_rows: tuple[int, int] | None = None

@dataclass(frozen=True)
class CheckpointManifest:
  format_version: int
  config: str
  tensors: tuple[TensorRecord, ...]

  def to_dict(self):
    return {"format_version": self.format_version, "config": self.config,
            "tensors": [asdict(t) for t in self.tensors]}

def _header(path: Path):
  with path.open("rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    return json.loads(f.read(n))

def inspect_checkpoint(model_dir: Path, output: Path) -> CheckpointManifest:
  config = model_dir / "config.json"
  if not config.is_file(): raise ValueError(f"missing config.json in {model_dir}")
  index_path = model_dir / "model.safetensors.index.json"
  index = json.loads(index_path.read_text()) if index_path.exists() else {"weight_map": {}}
  shards = sorted(model_dir.glob("*.safetensors"))
  if not shards: raise ValueError(f"no safetensors shards in {model_dir}")
  records = []; ple_cursor = 0
  for shard in shards:
    header = _header(shard)
    for name, meta in sorted((k, v) for k, v in header.items() if k != "__metadata__"):
      shape = tuple(int(x) for x in meta["shape"]); dtype = str(meta["dtype"])
      byte_count = int(meta["data_offsets"][1] - meta["data_offsets"][0])
      ple = None
      if ("ple" in name.lower() or "ngram" in name.lower()) and len(shape) >= 2 and shape[-1] == 160:
        ple = (ple_cursor, ple_cursor + shape[-2]); ple_cursor += shape[-2]
      records.append(TensorRecord(name, dtype, shape, shard.name, byte_count, ple))
  manifest = CheckpointManifest(1, config.name, tuple(records))
  output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
  return manifest

def verify_manifest(manifest: CheckpointManifest) -> None:
  names = [t.name for t in manifest.tensors]
  if len(names) != len(set(names)): raise ValueError("duplicate tensor name")
  rows = [t.ple_rows for t in manifest.tensors if t.ple_rows is not None]
  if rows and rows != sorted(rows): raise ValueError("non-contiguous PLE row intervals")
  if rows and any(a != b for (_, b), (a, _) in zip(rows, rows[1:])): raise ValueError("non-contiguous PLE row intervals")

def main():
  p = argparse.ArgumentParser(); p.add_argument("--model", type=Path, required=True); p.add_argument("--out", type=Path, required=True); p.add_argument("--verify", action="store_true")
  a = p.parse_args(); m = inspect_checkpoint(a.model, a.out)
  if a.verify: verify_manifest(m)
  print(f"wrote {a.out} ({len(m.tensors)} tensors)")

if __name__ == "__main__": main()
