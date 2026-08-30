#!/usr/bin/env bash
# Qwen3.8 Flash-Next wrapper around the existing tinygrad serve lifecycle.
# Override FT_QWEN38_MODEL when local filename differs; no weights downloaded.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export FT_MODEL="${FT_QWEN38_MODEL:-/home/smk/models/Qwen3.8-Flash-Next-GGUF/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf}"
export FT_HOST="${FT_QWEN38_HOST:-127.0.0.1}"
export FT_PORT="${FT_QWEN38_PORT:-1921}"
export FT_KV_TOKENS="${FT_QWEN38_KV_TOKENS:-32768}"
export FT_MAX_OUTPUT="${FT_QWEN38_MAX_OUTPUT:-8192}"
export FT_SERVED_MODEL="${FT_QWEN38_SERVED_MODEL:-qwen3.8-flash-next}"
export FT_TG_LOG="${FT_QWEN38_LOG:-/tmp/serve_qwen38_next.log}"
export FT_TG_PIDFILE="${FT_QWEN38_PIDFILE:-/tmp/serve_qwen38_next.pid}"
# Current tinygrad QSA is host-routed, and per-token packed AMD expert GEMMs
# compile too slowly for 48-layer warmup. Use bounded packed-row host BLAS until
# fused AMD expert kernels land; override to 0 only for kernel experiments.
export QWEN4EXP_MOE_HOST="${QWEN4EXP_MOE_HOST:-1}"

exec "$REPO/scripts/serve-tinygrad.sh" "$@"
