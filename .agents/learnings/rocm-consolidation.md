# ROCm Consolidation

## Context

Consolidated portable ROCm, GGUF, Qwen3.5, native K-quant, sampling, graph-safety, benchmark, and JIT-cache work across FreeToken PRs into `help/pr-132`.

## Hardest decision

Kept incumbent production routes and made new GGUF MoE routing opt-in: `fused_gguf.py` must prove packed expert layout, quant support, route reduction, and parity before serving promotion. This preserves fail-closed behavior across non-gfx1100 targets.

## Alternatives rejected

- Promote every open ROCm PR wholesale — several were gfx1100-specific, exploratory, or source-fork-only changes without matching correctness evidence.
- Treat unit tests and one live kernel parity run as serving proof — real Qwen GGUF repeated A/B, physical-target coverage, and profiler traces remain required.
- Reuse one generic JIT cache key — target, backend, compiler/toolchain, source, and flags must identify compiled artifacts to prevent stale cross-target binaries.

## Least confident

Qwen GGUF end-to-end serving and repeated A/B speedup remain unverified on available hardware; native mixed Q5_K/Q6_K MoE kernel tests pass, but this does not establish model-level correctness or promotion.

## Reuse

Read before changing `python/freetoken/moe/fused_gguf.py`, `python/freetoken/kernel/utils.py`, `benchmarks/bench_rocm_matrix.py`, or ROCm target routing. Keep candidates opt-in until manifests prove matching model, runtime, toolchain, JIT, route, completion, and replay identity.

## 2026-09-05 — Mixed Qwen GGUF on ROCm

### Context

Validated Qwen3.5-35B-A3B Q4_K_S on gfx1100 through CPU Q4_0 conversion and native mixed-quant GPU offload.

### Hardest decision

Carry source-native Q4_K/Q5_K/Q6_K/Q8_0 types and padded row strides through GPU MoE dispatch, while converting CPU expert sources one tensor at a time to Q4_0. This preserves GPU fidelity and matches the CPU executor's single packed contract.

### Alternatives rejected

- Force all native banks to Q4_K — terminal Q8_0 gate/up rows and mixed down rows would be rejected or mis-strided.
- Install CUDA-only `flashlib` in ROCm venv — adds incompatible dependencies; in-tree Triton full-fetch LRU already supplies required cache semantics.
- Treat separate CPU/GPU sampled outputs as parity — `Thinking` versus `1` proves only finite HTTP smoke, not matching model streams.

### Least confident

Model-level native/reference parity, graph replay, other physical ROCm targets, and performance promotion remain unproven; gfx1100 smoke only proves startup, route execution, finite output, and exact one-token completion.

### Reuse

Read before changing `python/freetoken/models/qwen3_5_moe/gguf.py`, `python/freetoken/moe/fused_gguf.py`, `python/freetoken/moe/offload_kernels.py`, or CPU/hybrid bank selection. Preserve exact quant metadata and fail-closed promotion gates.
