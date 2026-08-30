#!/usr/bin/env bash
# Standard ROCm Engine route for split Qwen3.8 Flash-Next GGUF.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "$REPO/.env" ]; then
  while IFS='=' read -r k v; do
    case "$k" in ''|\#*) continue ;; esac
    case "$k" in *[!a-zA-Z0-9_]*) continue ;; esac
    if [ -z "${!k+x}" ]; then export "$k=${v%$'\r'}"; fi
  done < "$REPO/.env"
fi

# Qwen3.8 GGUF projections use validated ROCm GEMV/dequant kernels. Keep Torch
# dequant as opt-in fallback for older deployments and gfx1100 kernel gaps.
export FREETOKEN_ROCM_GGUF_TORCH_FALLBACK="${FREETOKEN_ROCM_GGUF_TORCH_FALLBACK:-1}"
# Expert banks remain file-backed, but cold routed admissions must not serialize
# one pread per bank. Keep bounded parallel I/O independent of PLE workers.
export FREETOKEN_QWEN38_EXPERT_IO_WORKERS="${FT_QWEN38_EXPERT_IO_WORKERS:-16}"
# RDNA3 warm-prefill measurements favor 64 output lanes for IQ2/IQ3/IQ4_NL;
# keep operator override for older Triton/ROCm builds.
export FREETOKEN_QWEN4_TRITON_BLOCK_OUT="${FT_QWEN38_TRITON_BLOCK_OUT:-64}"

MODEL="${FT_QWEN38_MODEL:-/home/smk/models/Qwen3.8-Flash-Next-GGUF/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf}"
HOST="${FT_QWEN38_HOST:-127.0.0.1}"
PORT="${FT_QWEN38_ROCM_PORT:-1922}"
MAX_SEQ="${FT_QWEN38_MAX_SEQ_LEN:-32768}"
MAX_OUT="${FT_QWEN38_MAX_OUTPUT:-8192}"
KV_TOKENS="${FT_QWEN38_KV_TOKENS:-$MAX_SEQ}"
KV_FALLBACK="${FT_QWEN38_KV_FALLBACK_TOKENS:-}"
KV_DTYPE="${FT_QWEN38_KV_DTYPE:-auto}"
GPU="${FT_QWEN38_GPU:-0}"
PLE_PREFETCH="${FT_QWEN38_PLE_PREFETCH:-1}"
PLE_MODE="${FT_QWEN38_PLE_MODE:-auto}"
PLE_STORE="${FT_QWEN38_PLE_STORE:-}"
PLE_STORE_BUILD="${FT_QWEN38_PLE_STORE_BUILD:-auto}"
# Keep PLE cache bounded while giving routed experts most available hot-tier RAM;
# resolver still enforces runtime/safety reserves before accepting explicit caps.
PLE_RAM_CACHE_MIB="${FT_QWEN38_PLE_RAM_CACHE_MIB:-1536}"
PLE_ROW_CACHE_MIB="${FT_QWEN38_PLE_ROW_CACHE_MIB:-512}"
# Qwen3.8 routed IQ banks are ~43 GiB packed, while this path keeps them
# file-backed. Reserve a 28 GiB shared hot tier by default on 48+ GiB hosts:
# 26 GiB for routed experts, 1.5 GiB for PLE pages, 0.5 GiB for PLE rows.
# Resolver rejects this cap when MemAvailable cannot cover runtime reserves;
# explicit env overrides remain authoritative. The cache is lazy, so startup
# does not fault 30 GiB or pin it; hot routed banks stay in RAM after first use.
HOST_CACHE_MIB="${FT_QWEN38_HOST_CACHE_MIB:-28672}"
EXPERT_HOST_CACHE_MIB="${FT_QWEN38_EXPERT_HOST_CACHE_MIB:-26624}"
PLE_GPU_CACHE_MIB="${FT_QWEN38_PLE_GPU_CACHE_MIB:-128}"
PLE_STAGING_MIB="${FT_QWEN38_PLE_STAGING_MIB:-32}"
PLE_IO="${FT_QWEN38_PLE_IO:-auto}"
PLE_IO_DEPTH="${FT_QWEN38_PLE_IO_DEPTH:-64}"
PLE_CACHE_POLICY="${FT_QWEN38_PLE_CACHE_POLICY:-2q}"
PLE_PREFETCH_DEPTH="${FT_QWEN38_PLE_PREFETCH_DEPTH:-auto}"
PLE_BATCHED_CACHE="${FT_QWEN38_PLE_BATCHED_CACHE:-1}"
PLE_FUSED_DEQUANT="${FT_QWEN38_PLE_FUSED_DEQUANT:-0}"
EXPERT_RESIDENCY="${FT_QWEN38_EXPERT_RESIDENCY:-auto}"
MOE_GROUPED="${FT_QWEN38_MOE_GROUPED:-1}"
MOE_SCRATCH_MIB="${FT_QWEN38_MOE_SCRATCH_MIB:-128}"
PREFILL_ADAPTIVE="${FT_QWEN38_PREFILL_ADAPTIVE:-1}"
QSA_PREFILL_LIVE_WIDTH="${FT_QWEN38_QSA_PREFILL_LIVE_WIDTH:-0}"
# Large Qwen GGUF agent/tool prompts can take >10 minutes at configured
# 1024-token prefill cap. Keep watchdog finite, with one hour default.
REQUEST_TIMEOUT="${FT_QWEN38_REQUEST_TIMEOUT_S:-3600}"
SSE_HEARTBEAT="${FT_QWEN38_SSE_HEARTBEAT_S:-15}"
PLE_PROBE_TIMEOUT="${FT_QWEN38_PLE_PROBE_TIMEOUT_S:-300}"
# Keep one GPU generation at a time, but queue Copilot retries/parallel editor
# requests instead of returning 429 while long prefill/decode is active.
MAX_PENDING="${FT_QWEN38_MAX_PENDING_REQUESTS:-8}"
KV_FALLBACK_ARGS=()
PLE_STORE_ARGS=()
if [[ -n "$PLE_STORE" ]]; then PLE_STORE_ARGS=(--ple-store "$PLE_STORE"); fi
if [[ -n "$KV_FALLBACK" && "$KV_FALLBACK" =~ ^[0-9]+$ && "$KV_TOKENS" =~ ^[0-9]+$ && "$KV_FALLBACK" -lt "$KV_TOKENS" ]]; then
  KV_FALLBACK_ARGS=(--kv-reserve-fallback-tokens "$KV_FALLBACK")
fi

if [[ "$PLE_PREFETCH" == "0" || "$PLE_PREFETCH" == "false" || "$PLE_PREFETCH" == "no" ]]; then
  PLE_FLAG=(--no-ple-prefetch)
else
  PLE_FLAG=(--ple-prefetch)
fi
if [[ "$PLE_BATCHED_CACHE" == "0" || "$PLE_BATCHED_CACHE" == "false" || "$PLE_BATCHED_CACHE" == "no" ]]; then
  PLE_BATCHED_FLAG=(--no-ple-batched-cache)
else
  PLE_BATCHED_FLAG=(--ple-batched-cache)
fi
if [[ "$PLE_FUSED_DEQUANT" == "0" || "$PLE_FUSED_DEQUANT" == "false" || "$PLE_FUSED_DEQUANT" == "no" ]]; then
  PLE_FUSED_FLAG=(--no-ple-fused-dequant)
else
  PLE_FUSED_FLAG=(--ple-fused-dequant)
fi
if [[ "$QSA_PREFILL_LIVE_WIDTH" == "0" || "$QSA_PREFILL_LIVE_WIDTH" == "false" || "$QSA_PREFILL_LIVE_WIDTH" == "no" ]]; then
  QSA_PREFILL_FLAG=(--no-qwen38-qsa-prefill-live-width)
else
  QSA_PREFILL_FLAG=(--qwen38-qsa-prefill-live-width)
fi
if [[ "$MOE_GROUPED" == "0" || "$MOE_GROUPED" == "false" || "$MOE_GROUPED" == "no" ]]; then
  MOE_GROUPED_FLAG=(--no-qwen38-moe-grouped)
else
  MOE_GROUPED_FLAG=(--qwen38-moe-grouped)
fi
if [[ "$PREFILL_ADAPTIVE" == "0" || "$PREFILL_ADAPTIVE" == "false" || "$PREFILL_ADAPTIVE" == "no" ]]; then
  PREFILL_ADAPTIVE_FLAG=(--no-qwen38-prefill-adaptive)
else
  PREFILL_ADAPTIVE_FLAG=(--qwen38-prefill-adaptive)
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  exec "$REPO/.venv-rocm/bin/python" -m freetoken.cli serve --help
fi

# Production ROCm route never honors tinygrad's diagnostic host fallback switch.
unset QWEN4EXP_MOE_HOST

exec "$REPO/.venv-rocm/bin/python" -m freetoken.cli serve \
  --model-path "$MODEL" --device cuda --gpu "$GPU" --tensor-parallel-size 1 \
  --host "$HOST" --port "$PORT" --max-running-requests 1 --max-pending-requests "$MAX_PENDING" \
  --max-seq-len-override "$MAX_SEQ" --max-output-tokens "$MAX_OUT" \
  --kv-cache-dtype "$KV_DTYPE" --kv-reserve-tokens "$KV_TOKENS" \
  --max-prefill-length "${FT_QWEN38_PREFILL_TOKENS:-1024}" \
  --memory-ratio "${FT_QWEN38_MEMORY_RATIO:-0.90}" \
  --served-model-name "${FT_QWEN38_SERVED_MODEL:-qwen3.8-flash-next}" \
  --moe-backend "${FT_QWEN38_MOE_BACKEND:-auto}" \
  --qwen38-expert-residency "$EXPERT_RESIDENCY" --qwen38-moe-scratch-mib "$MOE_SCRATCH_MIB" \
  --moe-cache-auto --ple-mode "$PLE_MODE" --ple-store-build "$PLE_STORE_BUILD" \
  --qwen38-host-cache-mib "$HOST_CACHE_MIB" --qwen38-expert-host-cache-mib "$EXPERT_HOST_CACHE_MIB" \
  --ple-ram-cache-mib "$PLE_RAM_CACHE_MIB" --ple-row-cache-mib "$PLE_ROW_CACHE_MIB" \
  --ple-prefetch-depth "$PLE_PREFETCH_DEPTH" --ple-gpu-cache-mib "$PLE_GPU_CACHE_MIB" \
  --ple-staging-mib "$PLE_STAGING_MIB" --ple-io "$PLE_IO" --ple-io-depth "$PLE_IO_DEPTH" "${PLE_FLAG[@]}" \
  --ple-cache-policy "$PLE_CACHE_POLICY" "${PLE_BATCHED_FLAG[@]}" "${PLE_FUSED_FLAG[@]}" "${MOE_GROUPED_FLAG[@]}" "${QSA_PREFILL_FLAG[@]}" "${PREFILL_ADAPTIVE_FLAG[@]}" \
  --request-timeout-s "$REQUEST_TIMEOUT" --sse-heartbeat-s "$SSE_HEARTBEAT" \
  --ple-probe-timeout-s "$PLE_PROBE_TIMEOUT" \
  "${PLE_STORE_ARGS[@]}" \
  "${KV_FALLBACK_ARGS[@]}" "$@"
