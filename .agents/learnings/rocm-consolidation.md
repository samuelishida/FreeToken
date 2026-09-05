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
