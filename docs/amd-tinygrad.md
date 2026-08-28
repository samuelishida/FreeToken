# FreeToken on AMD via tinygrad

`--device tinygrad` runs the model through the **tinygrad fork's** inference
stack on AMD (RX 7900 XTX / gfx1100). The tinygrad AMD backend talks directly
to the kernel driver (kfd/hsa/amdgpu_drm ioctls) — **no ROCm userspace**.
FreeToken's engine (scheduler, sampler, OpenAI API) stays; model execution
goes through tinygrad's `Transformer`.

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
./scripts/serve-tinygrad.sh --model /path/to/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
# then point Copilot / any OpenAI client at http://127.0.0.1:1920/v1
```

The profiling scripts also take `--model` via launch args (no env vars):

```bash
.venv-rocm/bin/python scripts/bench-tinygrad.py --model /path/to/model.gguf --kernels
.venv-rocm/bin/python scripts/vram-tinygrad.py --model /path/to/model.gguf
```

## Benchmarks (RX 7900 XTX, gfx1100, Qwen3.6-35B-A3B-UD-Q4_K_M.gguf)

Warm JIT (the one-time compile is excluded). Prefill is chunked (256-token
chunks, the scheduler's cap); decode is single-token steps.

| context  | prefill tok/s | decode tok/s |
|----------|--------------|--------------|
| 4K       | 152.0        | 16.5         |
| 16K      | 149.1        | 15.9         |
| 64K      | 141.1        | 12.3         |
| 128K     | 133.2        | 9.5          |

## Inc 3 (KV Q8) results — 2026-08-28

| context  | prefill tok/s | decode tok/s | decode vs fp16-KV |
|----------|--------------|--------------|-------------------|
| 4K       | 153.5        | 18.2         | +10%  |
| 16K      | 150.6        | 17.7         | +11%  |
| 64K      | 143.2        | 15.4         | +25%  |
| 128K     | 135.1        | 15.3         | +61%  |

- kv: 2.68 → 1.51 GB; headroom 0.39 → 1.59 GB (gate PASSES after
  `allocator.free_cache()` post-warmup in the runner).
- decode graph: 57 kernels (54 + the quantize path).

## Inc 4/Inc 5 (decode GEMM) — 2026-08-28

- Inc 4 (`4b5f8188d`..`b319c90d6`): qsum precomputed in q8_quantize; the Q4_K/Q5_K
  decode dp4a kernel reads it (bit-identical, -8 dp4a/group/output). Decode flat
  at 4K (18.1) — the dp4a chain wasn't the bottleneck.
- Inc 5 (`58584a056`): tokens=1 GEMMs ride the WMMA kernel (activation zero-padded
  to the 16-row tile, padded rows sliced off); the q8 activation quantize leaves
  the decode path entirely.

## Final result (Inc 5) — kernel-opt plan complete

| ctx  | prefill (base → final) | decode (base → final) |
|------|------------------------|------------------------|
| 4K   | 152.0 → 153.8          | 16.5 → 18.2 (+10%)     |
| 16K  | 149.1 → 150.8          | 15.9 → 17.9 (+13%)     |
| 64K  | 141.1 → 143.2          | 12.3 → 15.0 (+22%)     |
| 128K | 133.2 → 134.3          | 9.5 → 15.7 (+65%)      |

- 128K VRAM: 25.39 → 24.17 GB used; headroom 0.36 → **1.59 GB (gate ≥ 1 GB ✓)**.
- Decode gains came from KV Q8 (8× less KV bandwidth per token). Decode GEMMs
  (dp4a → WMMA) were flat: decode is weight-bandwidth bound (~350 GB/s effective
  vs the card's 960 GB/s — the gap is the MoE expert read pattern, outside the
  kernel scope).
- Quality: e2e " Paris", prefix reuse and state reset pass in every increment.

Prefill is MoE-bound but the flash kernels keep it ~150 tok/s; decode is
~9-17 tok/s (MoE expert routing dominates). The first request after startup
pays no recompile (the runner warms both JIT graphs at init).

## Baseline (Inc 1, kernel-opt plan)

Measured with `scripts/bench-tinygrad.py --kernels` and
`scripts/vram-tinygrad.py` (2026-08-28, driver 25.75 GB card).

### VRAM at 128K context

| item      | size     |
|-----------|----------|
| weights   | 22.13 GB |
| kv        | 2.68 GB  |
| gdn_state | 0.03 GB  (Inc 2: fp16 state, was 0.06 fp32) |
| remainder | 0.51 GB  (activations + JIT graphs) |
| **free**  | **1.59 GB — GATE PASSES** (Inc 3 KV Q8 +1.17 GB; free_cache pós-warmup +0.64) |

The gate failure is the motivation for Inc 3 (KV Q8): quantizing the KV
cache to 8-bit restores ~1.3 GB, bringing headroom to ~1.6 GB.

### Decode graph

- **54 kernels** per decode step (captured JIT graph).
- Decode is latency/bandwidth-bound: 22 GB of weights per token at ~960 GB/s
  gives a ~43 tok/s floor; measured 9.5-16.5 tok/s → kernel-level overhead.

### Startup/warmup anatomy (DEBUG=2 trace)

- Model load: 20.61 GB H2D copy in ~65 s (~318 MB/s) — one-time.
- Eager warmup pass: ~92 s GPU, dominated by slow **"batched" MoE/GDN GEMM
  kernels** (e.g. `batched 512` at ~700 ms / ~2 TFLOPS / 17 GB/s each, ~280
  launches). These run only in the eager phase (JIT calls 1-2); the captured
  replay path is fast, so steady-state tok/s is unaffected.
- JIT capture compiles a second kernel set (symbolic start_pos/n_toks
  variants — different cache keys from the bound eager ones). First-ever run
  at a new max_context pays ~45 comgr compiles at ~30-60 s each (~25-45 min);
  they are cached in `~/.cache/tinygrad/cache.db` (table
  `compile_hip_gfx1100_22`) afterwards. Warm restart: ~5-6 min total.
- Practical consequence: the first 128K start on a machine is slow; keep the
  tinygrad cache warm (do not run with a different max_context casually —
  each new max_context recompiles the whole graph).

## Decode overhead breakdown (Inc 1, 2026-08-28)

`scripts/decode-overhead-tinygrad.py --model ... --ctx 4096 --steps 30` (4K, JIT quente):

| componente | mediana ms |
|---|---|
| full forward_batch | 58.6 |
|  ├─ H2D do token (input) | ~13  |
|  ├─ dispatch JIT (Python) | ~1.5 |
|  ├─ GPU (7 batched n_calls) | ~32-40 |
|  └─ D2H logits (608 KB, sync incluso) | ~1 |

- CORREÇÃO do diagnóstico anterior: o decode é **GPU-bound pelos GEMMs MoE de
  experts** (batched 64/128/256/512 → 1.3/4.4/8.6/17.5 ms no trace), a ~130 GB/s
  efetivos vs 960 do cartão. Os kernels customizados (WMMA/dp4a do decode) não
  cobrem o matmul de experts (tokens=8, gather `weight[sel]` + matmul genérico).
- O "39.2 ms de dispatch" anotado antes era o sync de GPU escondido (medição
  async) — o dispatch Python real é ~1.5 ms.
- O H2D do token (13 ms/passo, custo fixo) é a segunda maior alavanca.

## Rangeify fixes (variable-T + start_pos simbólico) — 2026-08-28

O `get_kernel_graph` do fork (o gerador de grafo de kernel via rangeify)
crashava em 2 pontos com extensões simbólicas:
- `IndexError` no `ended` comprehension (indexing.py:217) — o
  `broadcast_axes` devolve índices no rank completo mas o `range_map[c][0]`
  é pós-EXPAND/merge (o broadcast_rngs derruba o `nleft`; o bitcast merge
  derruba o trailing) — crash no bench @16384;
- `REDUCE has no ranges` (indexing.py:115) — o REDUCE não registrado —
  17 testes unit MTP-mock.

Fixes (fork `16daecea1`): a conversão do REDUCE sintetiza os ranges do shape
do src quando não registrado; o `ended` guarda os índices OOB; o kv quantize
shape-generic (`kv_q8_quantize_batched`) + o gate `_q8_kv` com fallback fp16.
Resultados: **16K bench 151.5/25.8 tok/s** (era: crash); o sweep unit
134→111 falhas (23 a mais passando, 0 regressão); MTP 27 ✓, ollama 42 ✓.
Learnings: `.agents/learnings/`.

## Known limits

- Prefill throughput (~150 tok/s) and decode (~15 tok/s) are the practical
  ceiling for interactive use; a 4K-token prompt prefills in ~27 s.
- Multi-request batching is not supported (max_running_req=1).
- The Copilot `maxInputTokens` setting is a client-side control: raising it
  sends more context, which raises prefill time.
