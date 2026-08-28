#!/usr/bin/env bash
# Profile the --device tinygrad path: VRAM breakdown (+ headroom gate) and
# prefill/decode tok/s. Read-only; does not start a server.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${FT_PY:-.venv-rocm/bin/python}"
MODEL="${FT_MODEL:-/media/smk/5fce248d-bbdd-488d-8883-4f000f85cc10/Models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
KV_TOKENS="${FT_KV_TOKENS:-131072}"

if [ ! -f "$MODEL" ]; then
  echo "model not found: $MODEL (set FT_MODEL)" >&2
  exit 1
fi

echo "=== VRAM breakdown (max_len=$KV_TOKENS) ==="
"$PY" scripts/vram-tinygrad.py --max-len "$KV_TOKENS"

echo
echo "=== Benchmark (prefill/decode tok/s) ==="
"$PY" scripts/bench-tinygrad.py --kernels

echo
echo "PROFILE DONE"