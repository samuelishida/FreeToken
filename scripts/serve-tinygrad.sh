#!/usr/bin/env bash
# serve-tinygrad.sh
#
# Serve the Qwen3.5/3.6-35B-A3B hybrid MoE GGUF (GatedDeltaNet + full-attention)
# on AMD via the tinygrad fork's direct kfd/hsa backend (no ROCm userspace, no
# Vulkan). FreeToken's engine (scheduler, sampler, OpenAI API) drives tinygrad's
# Transformer; the runner owns the single-request model state.
#
# Constraints:
#   - max_running_req=1 (the tinygrad Transformer is single-request stateful).
#   - --num-tokens sizes max_context; the AMD flash decode kernel requires a
#     multiple of 128 (the runner rounds up). VRAM: 22 GB weights + fp16 KV
#     (~2.7 GB at 128K for this model's 10 full-attn layers) + recurrent state;
#     default 32K is conservative, raise with FT_KV_TOKENS if VRAM allows.
#   - The first request pays the tinygrad JIT compile (kernels are compiled at
#     engine init; the runner warms up both graphs at startup).
#
# Usage:
#   ./serve-tinygrad.sh                 # launch on 127.0.0.1:1920
#   FT_PORT=1930 ./serve-tinygrad.sh    # pick another port
#   ./serve-tinygrad.sh stop            # kill the running server
#   ./serve-tinygrad.sh status          # running? + tail of the log

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$REPO/.venv-rocm/bin/python}"

MODEL="${FT_MODEL:-}"
HOST="${FT_HOST:-127.0.0.1}"
PORT="${FT_PORT:-1920}"
# max_context in tokens (rounded up to a multiple of 128 by the runner).
KV_TOKENS="${FT_KV_TOKENS:-32768}"
MAX_OUTPUT="${FT_MAX_OUTPUT:-65536}"
LOG="${FT_LOG:-/tmp/serve_tinygrad.log}"

die() { echo "ERROR: $*" >&2; exit 1; }

if grep -q $'\r' "${BASH_SOURCE[0]}"; then
    die "CRLF line endings detected in $(basename "${BASH_SOURCE[0]}"). Run: sed -i 's/\r$//' ${BASH_SOURCE[0]}"
fi

[ -x "$PY" ] || die "python not found: $PY (set PY=/path/to/venv-python)"

if [ -z "$MODEL" ]; then
    die "set FT_MODEL=/path/to/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
fi

case "${1:-}" in
    stop)
        pkill -f "ft serve.*--device tinygrad" 2>/dev/null || true
        echo "stopped (if it was running)"
        exit 0
        ;;
    status)
        if pgrep -f "ft serve.*--device tinygrad" >/dev/null; then
            echo "running; log tail:"
            tail -20 "$LOG" 2>/dev/null || true
        else
            echo "not running"
        fi
        exit 0
        ;;
esac

exec "$PY" -m freetoken.cli serve \
    --model "$MODEL" \
    --device tinygrad \
    --max-running-requests 1 \
    --num-tokens "$KV_TOKENS" \
    --max-output-tokens "$MAX_OUTPUT" \
    --host "$HOST" \
    --port "$PORT" \
    >>"$LOG" 2>&1
