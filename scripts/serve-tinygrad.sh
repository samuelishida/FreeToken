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
#   - Startup takes ~2.5 min (model load + JIT warmup); the first request then
#     pays no recompile.
#
# Usage:
#   ./serve-tinygrad.sh                 # launch on 127.0.0.1:1920
#   FT_MODEL=/path/to/model.gguf ./serve-tinygrad.sh
#   FT_PORT=1930 ./serve-tinygrad.sh    # pick another port
#   FT_KV_TOKENS=131072 ./serve-tinygrad.sh   # bigger context (VRAM permitting)
#   ./serve-tinygrad.sh status          # running? + tail of the log
#   ./serve-tinygrad.sh test            # one chat request against the server
#   ./serve-tinygrad.sh stop            # kill the server AND free its VRAM

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$REPO/.venv-rocm/bin/python}"

# Default: the Qwen3.6-35B-A3B GGUF on this machine. Override with FT_MODEL.
MODEL="${FT_MODEL:-/media/smk/5fce248d-bbdd-488d-8883-4f000f85cc10/Models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
HOST="${FT_HOST:-127.0.0.1}"
PORT="${FT_PORT:-1920}"
# max_context in tokens (rounded up to a multiple of 128 by the runner).
KV_TOKENS="${FT_KV_TOKENS:-32768}"
MAX_OUTPUT="${FT_MAX_OUTPUT:-65536}"
LOG="${FT_LOG:-/tmp/serve_tinygrad.log}"
PIDFILE="${FT_PIDFILE:-/tmp/serve_tinygrad.pid}"

die() { echo "ERROR: $*" >&2; exit 1; }

if grep -q $'\r' "${BASH_SOURCE[0]}"; then
    die "CRLF line endings detected in $(basename "${BASH_SOURCE[0]}"). Run: sed -i 's/\r$//' ${BASH_SOURCE[0]}"
fi

[ -x "$PY" ] || die "python not found: $PY (set PY=/path/to/venv-python)"
[ -f "$MODEL" ] || die "model not found: $MODEL (set FT_MODEL=/path/to/model.gguf)"

# Kill the server process tree AND any leftover kfd scheduler subprocess that
# still holds the model's VRAM (a plain pkill of the launcher leaves the
# scheduler alive with ~22 GB resident).
_stop() {
    if [ -f "$PIDFILE" ]; then
        local pid
        pid="$(cat "$PIDFILE")"
        kill -9 "$pid" 2>/dev/null || true
        rm -f "$PIDFILE"
    fi
    # The scheduler subprocess (freetoken-TP0-scheduler) survives the launcher
    # kill; it is the one holding the kfd VRAM. Kill it too.
    pkill -9 -f "freetoken.cli serve" 2>/dev/null || true
    sleep 1
    for p in /sys/class/kfd/kfd/proc/*/; do
        [ -d "$p" ] || continue
        kill -9 "$(basename "$p")" 2>/dev/null || true
    done
    echo "stopped (VRAM freed)"
}

case "${1:-}" in
    stop)
        _stop
        exit 0
        ;;
    status)
        if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            echo "running (pid $(cat "$PIDFILE")); log tail:"
            tail -20 "$LOG" 2>/dev/null || true
        else
            echo "not running"
        fi
        exit 0
        ;;
    test)
        [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null || die "server not running (start it first)"
        echo "POST /v1/chat/completions ->"
        curl -s "http://$HOST:$PORT/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d '{"model":"qwen3.6","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":256,"stream":false}' \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print('  content:', repr(d['choices'][0]['message'].get('content',''))); print('  finish:', d['choices'][0]['finish_reason'])"
        exit 0
        ;;
esac

# Refuse to double-launch.
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    die "already running (pid $(cat "$PIDFILE")); use '$0 stop' first"
fi

echo "serving $MODEL on $HOST:$PORT (max_context=$KV_TOKENS)"
echo "  log: $LOG   (startup ~2.5 min: model load + JIT warmup)"
echo "  status: $0 status | test: $0 test | stop: $0 stop"

# Launch detached; record the launcher PID. The scheduler subprocess is killed
# by `stop` (see _stop).
nohup "$PY" -m freetoken.cli serve \
    --model "$MODEL" \
    --device tinygrad \
    --max-running-requests 1 \
    --num-tokens "$KV_TOKENS" \
    --max-output-tokens "$MAX_OUTPUT" \
    --host "$HOST" \
    --port "$PORT" \
    >>"$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "launched (pid $(cat "$PIDFILE"))"
