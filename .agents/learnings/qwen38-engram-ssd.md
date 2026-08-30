# Qwen38 Engram SSD Runtime

## Context

Resumed Qwen4-Exp/tinygrad plan across FreeToken and sibling fork. Added
reference-first GR/QSA/PLE primitives, single-request runner wiring, reports,
and gated benchmark entrypoints.

## Hardest decision

Keep raw QSA keys and pending modulo-four state separate from compressed keys;
make PLE storage a supplier boundary so resident and SSD paths share lookup
semantics and observable cache metrics.

## Alternatives rejected

- Dense full-context production attention — violates selected-width and memory contract.
- Swap-backed or mmap-only PLE residency — hides synchronous SSD/cache misses.
- Implicit PLE hash authority — no pinned checkpoint hash vectors exist locally; keep hash function injectable until official vectors are supplied.

## Least confident

Exact released-checkpoint tensor names, Engram hash/padding/projection details,
and real Qwen4-Exp GDN/QSA/MoE weight execution remain unverified without the
official local checkpoint and RX 7900 XTX run. Inc 7 stays pending.

## Reuse

Read before continuing `.plans/qwen38-engram-ssd/plan.md`,
`tinygrad/llm/qwen4exp_*.py`, `ple_store.py`, and the tinygrad runner/API integration.

## 2026-08-29 implementation gate

### Hardest decision

Use `qwen3_moe` as Transformers' GGUF tokenizer converter for `qwen4exp`, because
the model architecture is new but its embedded Qwen BPE contract is supported by
the existing converter. Keep Qwen4Exp direct runner mode out of legacy TinyJit
buffers; model owns resettable GDN/QSA/PLE state.

### Alternatives rejected

- Launching the default per-expert AMD graph path — 48 layers caused unbounded
  compile time; wrapper now enables bounded host-BLAS evaluation of selected rows.
- Passing whole multi-block rows to `gguf.quants.dequantize_blocks` — decoder
  expects individual GGUF blocks, so rows must reshape to `(blocks, block_bytes)`.
- Claiming fused-device performance — live gate proves functional inference only;
  QSA routing and expert kernels remain explicit optimization work.

### Least confident

Host fallback produces finite full-model logits and one-token HTTP output, but
numerical parity against an official reference vector and production-speed AMD
packed kernels still need validation.

### Reuse

Read when touching `qwen4exp_gguf.py`, `qwen4exp.py`,
`python/freetoken/models/gguf/tokenizer.py`, or
`python/freetoken/engine/tinygrad_runner.py`; repeat real HTTP and reset gates
after changing any state or quantization path.

## 2026-08-29 ROCm Q8 KV gate

### Context

Standard ROCm Engine now serves the three-shard Qwen3.8 GGUF through Qwen4Exp
QSA/PLE/GDN state, with optional Q8 QSA K/V storage and 131072-token reserve.

### Hardest decision

Keep Q8 limited to ordinary QSA K/V rows; retain BF16 index slabs, pending-ring
state, scales, and all GDN/PLE state. This preserves QSA routing geometry while
cutting cache bytes without changing model semantics.

### Alternatives rejected

- Tinygrad/native Qwen4Exp route — separate diagnostic path; standard ROCm Engine
  already owns working HIP dispatch and scheduler lifecycle.
- `torch.topk` as default on ROCm — fixed tie behavior but lost wide-QSA scaling;
  retain Triton top-k and validate score-contract semantics for ties.
- Forcing 98304 KV when 131072 fits — violates requested capacity policy; fallback
  stays startup-fit-only.

### Least confident

RX 7900 XTX HIP graph capture is unavailable, and current Q8 fresh-process speed
is 17.18 tok/s at 1K / 13.70 tok/s at 32K, below 20/15 targets. Cold first-request
kernel compilation can exceed three minutes; warm HTTP benchmark evidence is the
valid throughput signal.

### Reuse

Read before changing `kvcache/qsa_pool.py`, `kernel/triton/q8_kv.py`,
`kernel/triton/qsa/{attend,topk}.py`, or `scripts/serve-qwen38-rocm.sh`.

## 2026-08-29 chat-prefill stall

### Context

Live VSCode `/v1/chat/completions` held an SSE connection with 24,141 prompt
tokens and no completion while ROCm reported zero activity. Standard GGUF
Qwen4 storage was confirmed as GPU-resident dense weights plus mmap expert/PLE
sources and GPU slot staging; process swap stayed at zero.

### Hardest decision

Treat HTTP 200 as stream-header acceptance, not successful completion. Add
OpenAI SSE keepalives during long prefill and preserve request cancellation while
reducing GGUF PLE gather allocation overhead.

### Alternatives rejected

- Claiming packed SSD PLE is active on standard ROCm — `--ple-store` is wired to
  tinygrad; standard GGUF currently uses OS mmap/page cache.
- Killing the user's live server to test patched code — restart is required, but
  external test state must remain untouched.

### Least confident

The old live request's exact backend wait point cannot be obtained under current
ptrace restrictions. It may be cold HIP compilation, long host-page prefill, or
a scheduler-level stall; fresh post-restart request telemetry is required.

### Reuse

Read before touching `server/openai_api.py`, `models/qwen4_exp/packed.py`, or
standard GGUF storage telemetry. Restart before validating keepalive behavior.

## 2026-08-29 PLE queue and full-order test hardening

### Context

Hardened paged `.ftple` reads and Copilot request lifecycle after a gfx1100
request stalled in unsafe IQ4_NL/expert dispatch. Also ran the complete test
order, which exposed scheduler fixture and optional-backend import interactions.

### Hardest decision

Keep synchronous PLE lookup bounded without holding `_cache_lock` while waiting
for `_read_slots`; reserve an in-flight placeholder and consumer reference first,
then launch the blocking read outside the lock so completion callbacks can admit
pages and release capacity.

### Alternatives rejected

- Increase I/O queue size — hides deadlock and violates fixed-budget paging.
- Remove telemetry from abort paths — loses production stage visibility; make
  telemetry optional for lightweight scheduler doubles instead.
- Change tests to import submodules explicitly — package lazy recovery belongs in
  runtime and fixes stale optional-probe module state for real callers too.

### Least confident

ROCm non-PLE Triton/FP8/DSV4 suites still fail on this gfx1100 + ROCm 7.2
environment; Qwen PLE/engine/server suites and production smoke pass, but those
backend failures need a supported compiler/device matrix before claiming global
GPU-suite health.

### Reuse

Read before changing `models/qwen4_exp/ple_gguf.py`, `scheduler/scheduler.py`,
`freetoken/__init__.py`, or `kernel/__init__.py`; preserve lock-free bounded I/O
submission and optional telemetry semantics.
