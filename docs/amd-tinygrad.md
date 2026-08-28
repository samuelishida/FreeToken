# FreeToken on AMD via tinygrad

`--device tinygrad` runs the model through the **tinygrad fork's** inference
stack on AMD (RX 7900 XTX / gfx1100). The tinygrad AMD backend talks directly
to the kernel driver (kfd/hsa/amdgpu_drm ioctls) — **no ROCm userspace, no
Vulkan**. FreeToken's engine (scheduler, sampler, OpenAI API) stays; model
execution goes through tinygrad's `Transformer`.

This replaces the earlier custom Vulkan path (`--device vulkan`), which was
removed. The Vulkan work lives on the `feat/amd-vulkan-support` branch for
history.

## How it works

- `Engine._init_tinygrad` builds a `TinygradModelRunner` that owns ONE tinygrad
  `Transformer` instance (single-request stateful: per-block KV cache +
  GatedDeltaNet recurrent state).
- The runner maps each FreeToken batch to `model.logits(tokens, start_pos)`
  (a public method added to the fork) and returns logits `[nreq, V]` — the
  same contract the scheduler's `sample_cpu` path consumes.
- Two `TinyJit` specializations mirror `Transformer.generate`: `start_pos` and
  the prefill token count are UOp variables bound at the call site, so the AMD
  flash kernels see symbolic shapes (the prefill kernel pads the query tile to
  a multiple of 32 internally; the decode kernel needs `max_context % 128 == 0`,
  which the runner rounds up to).
- The runner warms up both JIT graphs at init (TinyJit's first call is eager,
  the second captures, the third executes — the warmup runs each twice with
  real request shapes), so the first request doesn't pay a recompile.

## Constraints

- **`max_running_req=1`** — the tinygrad Transformer is single-request
  stateful. `--device tinygrad --max-running-requests > 1` is rejected at
  argument parse time.
- **Context ceiling** — `--num-tokens` sizes `max_context` (rounded up to a
  multiple of 128). `/v1/models` reports it; the scheduler rejects over-long
  prompts with a clean `context_length_exceeded` 400.
- **VRAM** — 22 GB weights + fp16 KV (~2.7 GB at 128K for this model's 10
  full-attn layers) + recurrent state. The serve script defaults to 32K
  (`FT_KV_TOKENS`); raise it if VRAM allows.
- **Startup** — model load + JIT warmup takes ~2.5 min before the first
  request (one-time).

## Dependency

The tinygrad fork (`../ollama-tg/tg-fork`) must be importable. On this machine
a `.pth` file in the venv's `site-packages` points at it (equivalent to
`pip install -e ../ollama-tg/tg-fork`). The fork carries the qwen35moe support,
the AMD kernels, and the public `Transformer.logits()` method.

## Usage

```bash
FT_MODEL=/path/to/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf ./scripts/serve-tinygrad.sh
# then point Copilot / any OpenAI client at http://127.0.0.1:1920/v1
```

## Benchmarks (RX 7900 XTX, gfx1100, Qwen3.6-35B-A3B-UD-Q4_K_M.gguf)

Warm JIT (the one-time compile is excluded). Prefill is chunked (256-token
chunks, the scheduler's cap); decode is single-token steps.

| context | prefill tok/s | decode tok/s |
|---------|--------------|--------------|
| 4K      | 152.6        | 17.2         |
| 16K     | 150.1        | 16.2         |
| 64K     | 142.1        | 12.3         |

Prefill is MoE-bound but the flash kernels keep it ~150 tok/s; decode is
~12-17 tok/s (MoE expert routing dominates). The first request after startup
pays no recompile (the runner warms both JIT graphs at init).

## Known limits

- Prefill throughput (~150 tok/s) and decode (~15 tok/s) are the practical
  ceiling for interactive use; a 4K-token prompt prefills in ~27 s.
- Multi-request batching is not supported (max_running_req=1).
- The Copilot `maxInputTokens` setting is a client-side control: raising it
  sends more context, which raises prefill time.
