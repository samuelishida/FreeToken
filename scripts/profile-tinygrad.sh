#!/usr/bin/env bash
# Profile the --device tinygrad path: VRAM breakdown (+ headroom gate) and
# prefill/decode tok/s. Read-only; does not start a server.
# All configuration is via launch args (no env vars).
#
# Usage:
#   ./scripts/profile-tinygrad.sh --model /path/to/model.gguf \
#       [--max-len 131072] [--ctx 4096,16384,65536,131072] [--decode 20] \
#       [--kernels] [--no-gate]
set -euo pipefail

cd "$(dirname "$0")/.."

PY=".venv-rocm/bin/python"
MODEL=""
MAX_LEN="131072"
CTX="4096,16384,65536,131072"
DECODE="20"
KERNELS=0
NO_GATE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --model)      MODEL="$2"; shift 2 ;;
    --max-len)    MAX_LEN="$2"; shift 2 ;;
    --ctx)        CTX="$2"; shift 2 ;;
    --decode)     DECODE="$2"; shift 2 ;;
    --kernels)    KERNELS=1; shift ;;
    --no-gate)    NO_GATE=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$MODEL" ]; then
  echo "--model is required" >&2
  exit 1
fi
if [ ! -f "$MODEL" ]; then
  echo "model not found: $MODEL" >&2
  exit 1
fi

VRAM_ARGS=(--model "$MODEL" --max-len "$MAX_LEN")
[ "$NO_GATE" = 1 ] && VRAM_ARGS+=(--no-gate)

BENCH_ARGS=(--model "$MODEL" --ctx "$CTX" --decode "$DECODE")
[ "$KERNELS" = 1 ] && BENCH_ARGS+=(--kernels)

echo "=== VRAM breakdown (max_len=$MAX_LEN) ==="
"$PY" scripts/vram-tinygrad.py "${VRAM_ARGS[@]}"

echo
echo "=== Benchmark (prefill/decode tok/s) ==="
"$PY" scripts/bench-tinygrad.py "${BENCH_ARGS[@]}"

echo
echo "PROFILE DONE"
