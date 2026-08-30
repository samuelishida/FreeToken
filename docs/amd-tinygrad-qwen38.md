# Qwen3.8 Flash-Next on tinygrad/AMD

## Standard ROCm production path

Use `scripts/serve-qwen38-rocm.sh` for GGUF on ROCm. Its default `FT_QWEN38_PLE_MODE=auto`
resolves to sidecar-backed `paged`; set `FT_QWEN38_PLE_STORE` and
`FT_QWEN38_PLE_STORE_BUILD=auto` to control `.ftple` placement. The script keeps the
configured `FT_QWEN38_PREFILL_TOKENS` cap (1024 by default) and exposes RAM/GPU/staging,
I/O, request timeout, SSE heartbeat, and `FT_QWEN38_PLE_PROBE_TIMEOUT_S` knobs. It
reserves a lazy 28 GiB host tier by default (26 GiB routed-expert cache, 1.5 GiB
PLE pages, 0.5 GiB PLE rows); the resolver refuses that cap when headroom is low.
Cold expert admissions use 16 bounded file-read workers; RDNA3 IQ GEMV uses 64
output lanes. Override with `FT_QWEN38_EXPERT_IO_WORKERS` and
`FT_QWEN38_TRITON_BLOCK_OUT` when validating another ROCm stack.

### ROCm prefill rollout

Baseline uses legacy full-width QSA geometry for sub-512-token prefills, then
automatically switches to live-width scoring for longer prompts. It uses 2Q PLE
pages, bounded direct lookup, and primitive IQ4_NL decode. Enable Qwen3.8 GGUF
ROCm experiments only after a
restart:

```bash
FT_QWEN38_QSA_PREFILL_LIVE_WIDTH=1 \
FT_QWEN38_PLE_CACHE_POLICY=2q \
FT_QWEN38_PLE_BATCHED_CACHE=1 \
FT_QWEN38_PLE_FUSED_DEQUANT=1 \
scripts/serve-qwen38-rocm.sh
```

Collect cold, warm, and cache-thrash evidence with
`scripts/bench-qwen38-rocm.py --prefill-ctx 1024,8192,24576,32768
--prefill-samples 5`. Promote only if warm 1024 reaches 20 tok/s and longer
context p95 stays within 5% of baseline. PLE page RAM is 1.5 GiB by default;
choose larger values only after >=10% warm-p95 improvement without swap or lost
MoE/KV headroom. Roll back with every boolean set to `0` and policy `lru`.

Readiness requires successful one-row IQ4_NL probe. Check `/health` and `/v1/cache/status`
for `ple_probe`, resolved storage topology, budgets, and live PLE counters. HTTP 200 stream
headers alone do not prove generation; require first data, terminal `[DONE]`, and
`/v1/stats.requests.active == 0`. If probe fails or times out, fix sidecar fingerprint,
GPU/kernel support, or budgets before retrying. No worker swap is an acceptance pass.

### Copilot latency and retries

`--max-running-requests` sizes scheduler slots. Serial engines admit up to four queued
requests per slot, so a Copilot retry waits behind an active prefill instead of failing with
HTTP 429. Additional requests still get HTTP 429 (`Retry-After: 1`) once that bounded queue is
full; this prevents unbounded SSD and MoE work.

Qwen3.8 GGUF prefill remains capped at the configured `FT_QWEN38_PREFILL_TOKENS` value
(1024 by default). Large Copilot agent/tool contexts can therefore take several minutes even
when GPU and PLE counters are advancing. The ROCm script defaults
`FT_QWEN38_REQUEST_TIMEOUT_S=3600` (one hour); override it when a shorter or longer bounded
deadline is appropriate. Use Ask mode or reduce the custom model's `maxInputTokens` for short
turns. A post-header timeout is
sent as an OpenAI chunk with `choices`, error metadata, and `[DONE]`, so VS Code shows a finite
error instead of reporting `Response contained no choices`.

Experimental milestone: local text-only Qwen4-Exp checkpoint, TP=1, one
request at a time, Linux AMD tinygrad backend. Vision, MTP, batching, and swap
as cache are unsupported.

Prepare a local checkpoint, inspect it, then pack extracted FP8 PLE rows to a
local NVMe directory:

```bash
python3 scripts/inspect-qwen38-checkpoint.py --model <checkpoint> \
  --out tests/fixtures/qwen38/manifest.json --verify
python3 scripts/pack-qwen38-ple.py --model <ple-rows> \
  --out <nvme>/qwen38-ple --verify
```

Launch with explicit storage and bounded host RAM:

```bash
ft serve --device tinygrad --model <checkpoint> --max-running-requests 1 \
  --ple-mode ssd --ple-store <nvme>/qwen38-ple --ple-ram-gib 8
```

PLE pages are owned by a bounded CPU LRU. SSD waits, hits, misses,
coalescing, evictions, and bytes are reported through `/v1/cache/status`.
Swap is emergency OS backing only; stop benchmark runs when swap-in occurs.
Measure cold, warm, and deliberately undersized-cache cases. Do not publish a
throughput target until finite logits, RSS/VRAM, SSD latency, and cache metrics
are recorded on the target RX 7900 XTX.
