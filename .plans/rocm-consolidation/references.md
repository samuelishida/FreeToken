# ROCm and GGUF Source Matrix

Snapshot date: 2026-09-04. Open-list source: [FlashML ROCm pull requests](https://github.com/FlashML-org/FreeToken/pulls?q=is%3Apr+rocm+is%3Aopen).

| PR | Role | Plan treatment |
|---|---|---|
| [#23](https://github.com/FlashML-org/FreeToken/pull/23) | Early gfx1100-1103 ROCm bring-up | Historical input only; superseded by #132. |
| [#131](https://github.com/FlashML-org/FreeToken/pull/131) | Generic GGUF types/readers/layers plus model adapters | Port generic substrate in Inc 6; Qwen3.5 adapter in Inc 7; other model adapters remain separate candidates. |
| [#132](https://github.com/FlashML-org/FreeToken/pull/132) | RDNA3/RDNA4 foundation, HIP extensions, arch detection, Triton compatibility | Inc 1 baseline. |
| [#133](https://github.com/FlashML-org/FreeToken/pull/133) | TVM-FFI index/store HIP portability | Inc 3, stacked after foundation. |
| [#134](https://github.com/FlashML-org/FreeToken/pull/134) | CUDA-only optional backend gating | Inc 4. |
| [#135](https://github.com/FlashML-org/FreeToken/pull/135) | RCCL tensor-parallel communication | Inc 5, real multi-GPU proof required. |
| [#136](https://github.com/FlashML-org/FreeToken/pull/136) | Native GGUF ROCm build and Q4_0 kernels | Inc 8; Q4_0 first, other quant types separate. |
| [#137](https://github.com/FlashML-org/FreeToken/pull/137) | Earlier AMD serving bring-up | Mine non-overlapping portable fixes only; no separate landing. |
| [#217](https://github.com/FlashML-org/FreeToken/pull/217) | Source-fork ROCm + Qwen3.5 GGUF + performance experiments | Inc 7/9/10/14-17 selective ports; current router path is `moe/fused.py`; do not merge source branch wholesale. |
| [#241](https://github.com/FlashML-org/FreeToken/pull/241) | gfx1150 build/JIT/Triton/attention hardening | Inc 2 portable fixes; validate before target declaration. |
| [#260](https://github.com/FlashML-org/FreeToken/pull/260) | gfx1151 validation and fallback/build evidence | Inc 13 matrix input; selective code only after fresh evidence. |
| [#316](https://github.com/FlashML-org/FreeToken/pull/316) | HIP graph-capture-safe expert copies | Inc 11. |
| [#378](https://github.com/FlashML-org/FreeToken/pull/378) | CPU/Hybrid MoE graph replay safety | Inc 12; retain fail-closed handshake. |

## Rejected wholesale ports

- Source branch `origin/feat/amd-rocm-gfx1100-support` contains unrelated PLE,
  Qwen4, benchmark, paper, and gfx1100 candidate work. It is evidence/source,
  not an implementation base.
- `gfx1100`-named or rotated-wave kernels are not architecture-portable by
  naming. Candidate code must first pass generic capability selection,
  independent numerical tests, and served A/B gates.
- Compile-only and synthetic ABI results are retained as matrix evidence, not
  advertised serving support.
