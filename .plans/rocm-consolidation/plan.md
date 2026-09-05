# ROCm Consolidation and Performance Plan

## Context

FreeToken has an open ROCm stack spread across PRs #23, #132-#137, #217,
#241, #260, #316, and #378, plus shared GGUF work in #131. The current
checkout is `help/pr-132`, cloned from `samuelishida/FreeToken`, with
`FlashML-org/FreeToken` available as `upstream`. The source fork contains
useful ROCm, GGUF, Qwen3.5, benchmark, and performance work, but also large
unrelated changes and gfx1100-specific experiments.

Outcome: produce a sequence of small, reviewable PRs that lands proven
portable ROCm and GGUF work, validates RDNA3/RDNA4/RDNA3.5 coverage, and
promotes performance changes only when independent correctness and end-to-end
A/B evidence supports them.

This is a planning artifact. It does not modify runtime code, install the
shared virtual environment, push branches, or open/comment on GitHub PRs.

## Architectural decisions

- Decision: use current `upstream/main` as implementation base and port
  selected commits from open PRs/source fork. Rationale: source fork diverges
  by thousands of lines and deletes/reworks parts of #132; wholesale merging
  would hide conflicts and provenance. Source: `upstream/main` ref,
  `origin/feat/amd-rocm-gfx1100-support`, and `CONTRIBUTING.md:34-40`.
- Decision: land #132 foundation before follow-up portability fixes. Rationale:
  #133 and #136 are explicitly stacked on #132, while #137/#23 overlap its
  bring-up. Source: [PR #132](https://github.com/FlashML-org/FreeToken/pull/132)
  and [PR #136](https://github.com/FlashML-org/FreeToken/pull/136).
- Decision: centralize ROCm target parsing and use architecture-family
  capability checks; never encode `gfx1100` in a production kernel name or
  silently alias an unsupported target. Rationale: the same code must serve
  gfx1100-1103, gfx1150-1151, and gfx1200-1201. Source: PR #132/#136 target
  policy and `python/freetoken/models/gguf/config.py:17-21`.
- Decision: keep forced modes fail-loud and fallback modes explicit. A
  requested candidate does not count as observed execution. Rationale:
  runtime fallback can make a benchmark or correctness claim false. Source:
  `python/freetoken/layers/gguf.py:43-65` and source-fork ROCm learnings.
- Decision: separate generic GGUF metadata/reader/model loading from native
  ROCm kernels. Rationale: #131 proves broad GGUF tables and loaders on CUDA,
  while #136 proves only Q4_0 native ROCm coverage. Source: current
  `python/freetoken/layers/gguf.py:32-65`, [PR #131](https://github.com/FlashML-org/FreeToken/pull/131),
  and [PR #136](https://github.com/FlashML-org/FreeToken/pull/136).
- Decision: preserve incumbent defaults until a candidate wins numerical,
  serving, and performance gates. Rationale: source-fork Qwen experiments
  include slower or degenerate candidates despite direct-kernel success.
  Source: `.agents/learnings/qwen-moe-rocm-base-speed.md` on
  `origin/feat/amd-rocm-gfx1100-support`.
- Decision: benchmark non-MTP/base decode first; keep teacher-forced replay
  and sampled generation as separate claims. Rationale: matching token paths
  are required for attribution, and sampled streams diverge. Source:
  `benchmarks/README.md:3-31` and source-fork ROCm learnings.
- Decision: treat graph safety as correctness work, not an optimization
  toggle. Enable HIP capture paths only after a real capture/instantiate/replay
  probe; otherwise route to an explicit safe path. Source: [PR #316](https://github.com/FlashML-org/FreeToken/pull/316)
  and [PR #378](https://github.com/FlashML-org/FreeToken/pull/378).

## Assumptions and answers from code

- Decision: implementation starts from a fresh branch based on
  `upstream/main`; existing `help/pr-132` remains the staging checkout.
  Source: current `git status --short --branch` and remotes.
- Decision: ROCm CI/build support must coexist with CUDA. Source:
  `upstream/main:AGENTS.md:39-51` and `setup.py:21-64`, which currently make
  `CUDA_HOME` and `cudart` mandatory.
- Decision: tests mirror the protected subsystem and compare against an
  independent reference, CPU mirror, or round trip. Source:
  `upstream/main:tests/README.md:3-18,73-100`.
- Decision: each implementation increment is one upstreamable PR with its own
  regression test and, for performance work, same-model/same-settings A/B
  evidence. Source: `upstream/main:CONTRIBUTING.md:34-40`.
- Answered from code: current GGUF registry has Gemma4 only. Source:
  `python/freetoken/models/register.py:103-109` and
  `python/freetoken/models/gguf/config.py:17-21`.
- Answered from code: current native GGUF path exposes Q4_0, Q8_0, and Q6_K
  groups, with a reference dequant fallback. Source:
  `python/freetoken/layers/gguf.py:32-65`.
- Answered from code: benchmark commands already cover served MoE decode,
  expert loading, offload copies, and bandwidth. Source:
  `benchmarks/README.md:6-31`.
- Decision: every verification increment starts with a Python/dependency/Torch
  backend preflight. Current linked `./.venv` resolves to sibling
  `../FreeToken/.venv` and lacks Torch, so missing dependencies block
  verification; do not install or recreate shared environments from this plan.
  Source: current `readlink -f .venv` and import preflight.
- Open question resolved by default: RCCL support is planned, but multi-GPU
  promotion requires hardware evidence; single-GPU ROCm must remain unaffected.
  Source: [PR #135](https://github.com/FlashML-org/FreeToken/pull/135) and
  `CONTRIBUTING.md:36-40`.

## Risks accepted

- ROCm/PyTorch/Triton/HIP version skew: pin compatible ranges, report exact
  versions, and fail with actionable diagnostics; accept no universal image
  guarantee until matrix evidence exists.
- Source-fork conflict and attribution drift: port by behavior/file contract,
  preserve upstream history, and record source PR/commit in each PR body.
- Kernel numerical drift: require CPU/PyTorch reference tolerances before any
  end-to-end or speed claim.
- Graph capture regressions: retain eager and safe-copy fallbacks; fail closed
  when capture capability is not proven.
- Perf noise and cache effects: use fresh-process runs, medians/variance, fixed
  model/config/token protocol, and separate warm/cold results.
- Hardware availability: mark gfx1150/1151 and multi-GPU gates as pending
  until real hardware is available; do not infer support from compilation.

## Increment DAG

- Inc 1 — RDNA ROCm foundation (L) — depends on: none — unblocks: 2, 5, 6
- Inc 2 — Portable HIP hardening (M) — depends on: 1 — unblocks: 3, 4, 6, 8, 11, 12
- Inc 3 — TVM-FFI HIP JIT (S) — depends on: 2 — unblocks: 8
- Inc 4 — Optional backend gating (S) — depends on: 2 — unblocks: none
- Inc 5 — RCCL communication path (M) — depends on: 1 — unblocks: none
- Inc 6 — Generic GGUF substrate (L) — depends on: 2 — unblocks: 7, 8, 9, 10
- Inc 7 — Qwen3.5 GGUF adapter (L) — depends on: 6 — unblocks: 9, 10, 13
- Inc 8 — Native ROCm Q4_0 GGUF (M) — depends on: 3, 6 — unblocks: 9, 10, 13
- Inc 9 — Portable K-quant native path (L) — depends on: 7, 8 — unblocks: 10, 13
- Inc 10 — Qwen GGUF runtime correctness (M) — depends on: 7, 8, 9 — unblocks: 13
- Inc 11 — HIP graph-safe expert copies (M) — depends on: 2 — unblocks: 13
- Inc 12 — CPU/Hybrid graph replay safety (L) — depends on: 2 — unblocks: 13
- Inc 13 — Cross-arch serving matrix (M) — depends on: 9, 10, 11, 12 — unblocks: 14
- Inc 14 — ROCm benchmark and provenance gates (M) — depends on: 13 — unblocks: 15, 16, 17
- Inc 15 — Guarded ROCm fused router (M) — depends on: 14 — unblocks: none
- Inc 16 — Guarded native GGUF MoE kernels (L) — depends on: 9, 10, 14 — unblocks: none
- Inc 17 — ROCm profiler and JIT hygiene (S) — depends on: 14 — unblocks: none

Increments 3, 4, and 6 can proceed in parallel after Inc 2 where their branch
conflicts are absent. Inc 5 can proceed after Inc 1. Increments 11 and 12
can proceed in parallel after Inc 2. Inc 15, Inc 16, and Inc 17 are
independent PRs and must not be merged as a bundle.

## Increments

### Inc 1 — RDNA ROCm foundation (L)

**Depends on:** none
**Unblocks:** 2, 5, 6
**Status:** done.
**Done criteria:** #132 behavior is rebased onto current `upstream/main`, builds through HIP and CUDA paths, and has independent architecture/build tests without gfx1100-only runtime assumptions.

#### Files to touch

##### `setup.py`
- What changes: select CUDA or HIP runtime/include/library configuration without making `CUDA_HOME` mandatory on ROCm.
- Function(s): retain `setup(...)`; add backend-specific runtime path/toolchain helpers.
- Data shapes: backend `{cuda, rocm}` plus compiler/runtime paths; extension list stays stable.
- Integration points: PyTorch C++ extension build and `python/freetoken/kernel/_toolchain.py`.
- Error paths: missing HIP/CUDA toolchain names exact missing variables and abort before compile.

##### `pyproject.toml`, `docs/install.md`
- What changes: document supported ROCm setup and keep CUDA dependency/test behavior unchanged; record exact tested Torch/ROCm/Triton/HIP tuples instead of declaring a broad ROCm range before Inc 13 evidence.
- Function(s): dependency/build metadata only.
- Data shapes: compatible Torch/ROCm/Triton version ranges.
- Integration points: `uv` editable install and CI environment.
- Error paths: unsupported platform reports remediation; no silent CPU-only success for a requested ROCm build.

##### `.github/workflows/unit-nvidia.yml` (new)
- What changes: add a hosted CUDA compile/unit gate for foundation changes; keep it safe for fork pull requests and free of self-hosted secrets.
- Function(s): path-filtered `pull_request` and manual workflow; compile extensions, then run non-slow unit tests.
- Data shapes: Torch/CUDA tuple, extension names, pytest exit status.
- Integration points: `setup.py`, future native GGUF compile checks, PR status.
- Error paths: CUDA/ROCm mismatch or compile failure fails the workflow; GPU-dependent tests may skip only with their existing reason.

##### `python/freetoken/kernel/_toolchain.py`, `backend.py`, `utils.py`, `python/freetoken/utils/arch.py`
- What changes: add shared HIP detection, visible-device `gcnArchName` lookup, override parsing, and launch-argument policy.
- Function(s): backend/arch helpers return normalized target strings and capability flags.
- Data shapes: `gfxNNNN` target, family `{rdna3, rdna3.5, rdna4}`, and explicit feature booleans.
- Integration points: native extensions, Triton launch kwargs, JIT loaders.
- Error paths: malformed/unsupported overrides fail before JIT; absence of a GPU uses existing CUDA/CPU behavior.

##### `python/freetoken/kernel/csrc/include/freetoken/hip_compat.h`, `utils.cuh`, `pinned_tensor.cpp`, `csrc/cpu_moe/cpu_moe_ext.cpp`, `csrc/jit/fast_index_copy.cuh`
- What changes: replace CUDA-only API assumptions with guarded HIP/CUDA equivalents for pinned memory, CPU-MoE graph hooks, and fast index copy.
- Function(s): preserve existing extension ABI and launch signatures.
- Data shapes: device pointers, host-pinned pointers, and stream handles remain opaque backend-native values.
- Integration points: `freetoken.kernel.pinned`, CPU MoE executor, JIT fast-index path.
- Error paths: unsupported graph/API capability disables only dependent path and reports state.

##### `python/freetoken/kernel/triton/activation.py`, `e4m3_compat.py`, `norm.py`
- What changes: use backend-portable Triton operations and guarded ROCm launch kwargs.
- Function(s): preserve existing tensor signatures and output dtype/shape.
- Data shapes: bf16/fp16/fp8 tensors and masks unchanged.
- Integration points: model attention/MLP kernels.
- Error paths: unsupported fp8 capability selects existing safe implementation.

##### `tests/kernels/test_e4m3_compat.py`, `test_pinned_tensor.py`, `test_rocm_launch_kwargs.py`, `test_norm.py` (new), `tests/utils/test_rocm_arch.py`
- What changes: add independent parsing, allocator, launch-filter, fused-add/norm, and architecture cases.
- Function(s): tests exercise public helpers and extension-facing behavior, not branch constants.
- Data shapes: valid/invalid gfx names, visible-device output, and CUDA/ROCm keyword sets.
- Integration points: pytest GPU skips and CPU parser tests.
- Error paths: verify unsupported target and missing runtime fail loudly.

#### Edge cases

- Multiple GPUs with different visible architecture names: use first visible device only when policy requires it, never hardcode host GPU.
- ROCm 7.x and CUDA import/build paths must not import each other's libraries.
- `FREETOKEN_ROCM_ARCH` must override discovery only when valid.
- No native graph feature may be enabled merely because `torch.version.hip` exists.

#### Verification

- Run: `uv run pytest tests/kernels/test_e4m3_compat.py tests/kernels/test_pinned_tensor.py tests/kernels/test_rocm_launch_kwargs.py tests/kernels/test_norm.py tests/utils/test_rocm_arch.py`.
- Run: `python setup.py build_ext --inplace` once under CUDA and once under ROCm when hardware/toolchains exist.
- Tests to add/update: listed architecture/build tests plus CUDA regression smoke.
- Done: build logs identify backend, target, compiler, and Torch version; no test/benchmark claim is made for unavailable hardware.

### Inc 2 — Portable HIP hardening (M)

**Depends on:** 1
**Unblocks:** 3, 4, 6, 8, 11, 12
**Status:** done.
**Done criteria:** non-overlapping ROCm fixes from #241/#137 compile and run on declared families without changing CUDA behavior or importing gfx1100-specific kernels; attention tile changes land only when reproduced by a focused regression.

#### Files to touch

##### `python/freetoken/kernel/triton/activation.py`, `attention.py`
- What changes: remove HIP-invalid raw NVIDIA inline PTX and gate PDL/launch options by backend; port the #241 GQA tile floor only if a focused gfx1150/1151 regression reproduces it.
- Function(s): preserve Triton op signatures; add backend branch only around implementation details.
- Data shapes: activation tensors, split attention tiles, and launch metadata remain shape-compatible.
- Integration points: prefill/decode attention and Triton activation/norm callers.
- Error paths: unsupported tile/feature falls back to existing safe kernel or raises in forced mode.

##### `python/freetoken/kernel/csrc/include/freetoken/utils.cuh`, `python/freetoken/kernel/csrc/jit/fast_index_copy.cuh`, `python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp`, `python/freetoken/kernel/utils.py`, `setup.py`, `pyproject.toml`, `.gitignore`
- What changes: apply only HIP fixes not already supplied by Inc 1; GGUF-specific compiler flags and dispatch-header changes belong to Inc 8. Ignore generated HIP sources.
- Function(s): centralize compile flags; do not pass `--expt-relaxed-constexpr`, `-ccbin`, or CUDA-only flags to `hipcc`.
- Data shapes: shuffle masks use backend-required width; target list remains normalized.
- Integration points: pinned tensor, CPU MoE, and non-GGUF JIT extensions.
- Error paths: missing HIP compiler/runtime produces actionable build diagnostics.

##### `tests/kernels/test_rocm_launch_kwargs.py`, `tests/kernels/test_triton_attention.py`, `tests/utils/test_rocm_arch.py`
- What changes: cover no-PTX HIP source policy, tile/PDL decisions, HIP compile flags, and architecture/launch behavior; GGUF dispatch is tested in Inc 8.
- Function(s): use source/config inspection and independent output checks.
- Data shapes: gfx1100, gfx1150/1151, gfx1200/1201 where available.
- Integration points: fresh Triton/JIT cache and native non-GGUF build smoke.
- Error paths: assert CUDA flags remain on CUDA and are absent on HIP.

#### Edge cases

- `hipcc` may expose CUDA-like include trees but still reject CUDA flags; GGUF-specific handling is deferred to Inc 8.
- Triton versions may lack a backend-specific intrinsic; use existing safe op, not raw PTX.
- Attention tile changes must cover both decode GQA and split prefill.

#### Verification

- Run: focused ROCm kernel/arch tests plus `uv run pytest tests/ -m "not slow"` on CUDA.
- Run: fresh `TORCH_EXTENSIONS_DIR` and fresh Triton/JIT cache build on each available ROCm family.
- Done: build, Triton smoke, and one served request per available family; report exact versions and unsupported families separately.

**Evidence:** Inc 1 already supplied all non-overlapping #241/#137 activation, PDL,
HIP runtime, fast-copy, and compiler-flag fixes. Focused gfx1100 Triton attention
and ROCm tests passed with fresh caches: `50 passed` in
`/tmp/hawk-implement-plan-check-inc2.log`. Decode GQA groups 4, 5, 6, and 7,
split/non-split extend, sinks, and sliding-window paths all passed; no additional
attention tile change was ported because no gfx1150/1151 regression was reproducible
on available gfx1100 hardware. gfx1150/1151/gfx1200/1201 and CUDA remain hardware-
pending; CUDA build was unavailable because `nvcc` is absent.

### Inc 3 — TVM-FFI HIP JIT (S)

**Depends on:** 2
**Unblocks:** 8
**Status:** done.
**Done criteria:** TVM-FFI index/store JIT kernels execute on ROCm for supported integer/split/masked paths while NVIDIA PTX behavior remains unchanged.

#### Files to touch

##### `python/freetoken/kernel/csrc/include/freetoken/utils.cuh`, `python/freetoken/kernel/csrc/jit/index.cu`, `store.cu`
- What changes: make device/runtime types use `kDLROCM`, `kWarpThreads`, backend-safe widths, and split/masked indexing.
- Function(s): preserve index/store ABI and tensor stride contracts.
- Data shapes: contiguous and strided tensors, 32/64-bit indices, split/masked slices.
- Integration points: TVM-FFI JIT loader and fast-index callers.
- Error paths: unsupported dtype/rank/target returns existing explicit error.

##### `tests/kernels/test_jit_index_store.py`
- What changes: compare JIT results to PyTorch reference for all supported paths.
- Function(s): test warm-cache and fresh-build behavior.
- Data shapes: empty, singleton, non-contiguous, and boundary-sized tensors.
- Integration points: HIP and CUDA test markers.
- Error paths: invalid masks and unsupported target fail rather than produce output.

#### Edge cases

- ROCm warp width differs from NVIDIA assumptions.
- First-call compilation must not reuse stale CUDA JIT artifacts.
- Masked zero-length slices must preserve shape and dtype.

#### Verification

- Run: `uv run pytest tests/kernels/test_jit_index_store.py` on ROCm and CUDA.
- Done: focused tests pass with fresh caches; six warm-path cases and all split/masked cases are recorded.

**Evidence:** Index/store JIT ran against ROCm gfx1100 with fresh Triton and
Torch extension caches; `7 passed` in
`/tmp/hawk-implement-plan-check-inc3.log`. Index/store matchers now accept both
`kDLCUDA` and `kDLROCM`; empty index requests return without launching a zero-block
kernel. CUDA execution remains unrun because no CUDA toolkit/device is available.

### Inc 4 — Optional backend gating (S)

**Depends on:** 2
**Unblocks:** none
**Status:** done.
**Done criteria:** CUDA-only optional packages are never probed or selected on ROCm, while CUDA retains existing probes and forced unsupported modes fail explicitly.

#### Files to touch

##### `python/freetoken/kernel/backend.py`
- What changes: expose `is_rocm`, return false for CUDA-only package probes on ROCm, and avoid driver-CUDA probing there.
- Function(s): `is_flashinfer_installed()`, `is_sgl_kernel_installed()`, `is_triton_kernels_installed()`, `driver_cuda_version()`.
- Data shapes: booleans and `int | None` remain stable.
- Integration points: engine backend selection and MoE dispatch.
- Error paths: broken optional package stays unavailable; no import side effect.

##### `python/freetoken/moe/nvfp4_backends.py`
- What changes: route ROCm to Triton and raise when a caller forces CUDA-only NVFP4 backend.
- Function(s): backend resolver returns explicit `{triton, cuda, unsupported}` state.
- Data shapes: model dtype/layout and device capability are preserved.
- Integration points: MoE model setup.
- Error paths: no silent CUDA fallback on ROCm.

##### `tests/moe/test_nvfp4_backends.py`
- What changes: extend existing backend tests to cover probe suppression on ROCm and probe retention on CUDA.
- Function(s): monkeypatch only module discovery; assert live resolver result.
- Data shapes: installed/missing package matrix.
- Integration points: model backend selection.
- Error paths: forced CUDA-only request raises expected exception.

#### Edge cases

- `find_spec` failures from broken parent packages must remain “unavailable”.
- ROCm with a package accidentally installed must not load CUDA code.
- CUDA driver version remains available on CUDA.

#### Verification

- Run: `uv run pytest tests/moe/test_nvfp4_backends.py` and backend import smoke on ROCm/CUDA.
- Done: no CUDA-only package probe appears in ROCm trace; single-GPU ROCm server starts through Triton.

**Evidence:** ROCm backend-selection/probe tests pass `4 passed` in
`/tmp/hawk-implement-plan-check-inc4.log`; live ROCm probe smoke passed in
`/tmp/hawk-implement-plan-check-inc4-smoke.log`. flashinfer, sgl_kernel,
triton_kernels, and CUDA-driver probing are suppressed on HIP; NVFP4 auto and
forced supported mode resolve to Triton, while forced Marlin/flashinfer fail
explicitly. CUDA probe retention is unit-tested but no CUDA runtime is available.

### Inc 5 — RCCL communication path (M)

**Depends on:** 1
**Unblocks:** none
**Status:** done.
**Done criteria:** multi-GPU ROCm tensor-parallel communication uses RCCL when explicitly configured; single-GPU and CUDA paths remain unchanged.

#### Files to touch

##### `python/freetoken/kernel/pynccl.py`
- What changes: reject PyNCCL before JIT/link on ROCm; keep custom PyNCCL extension CUDA-only. ROCm uses PyTorch's RCCL-backed process group.
- Function(s): guard `_load_nccl_module()` by backend; preserve CUDA communicator creation, all-reduce, broadcast, and teardown signatures.
- Data shapes: ranks, device IDs, streams, and tensor dtypes remain explicit.
- Integration points: CUDA PyNCCL and ROCm process-group selection.
- Error paths: ROCm never attempts `libnccl`; CUDA missing library, invalid rank, and mixed-device topology fail before serving.

##### `python/freetoken/engine/engine.py`
- What changes: select Gloo for TP1/control traffic, CUDA PyNCCL for CUDA TP>1 when enabled, and PyTorch `backend="nccl"` for ROCm TP>1 so PyTorch routes to RCCL.
- Function(s): `_init_communication(config)` selection path only; no hidden multi-GPU activation.
- Data shapes: `tp_size=1` bypasses communicator; `tp_size>1` requires backend.
- Integration points: model load and scheduler startup.
- Error paths: missing RCCL-backed PyTorch backend, invalid rank, or unsupported topology yields actionable setup error.

##### `tests/engine/test_rocm_communication.py` (new)
- What changes: assert ROCm selects PyTorch `backend="nccl"`, PyNCCL is never loaded, CUDA retains PyNCCL, and TP1 creates no device communicator; skip real multi-GPU tests when unavailable.
- Function(s): test backend selection, rank setup, teardown, and loader guard with fake process groups/reference reduction.
- Data shapes: one/two-rank tensors and stream handles.
- Integration points: engine startup.
- Error paths: missing RCCL and invalid topology.

#### Edge cases

- TP1 must not initialize a device communicator.
- ROCm must not load PyNCCL or link `libnccl`; TP>1 must use PyTorch's RCCL-backed `nccl` process group.
- ROCm and CUDA libraries must never be mixed in one process.
- Real two-GPU evidence is required before claiming serving support.

#### Verification

- Run: `uv run pytest tests/engine/test_rocm_communication.py`.
- Done: TP1 ROCm/CUDA smoke passes and TP2 backend selection is unit-tested; real all-reduce/serving is reported only with RCCL hardware and exact topology.

**Evidence:** `tests/engine/test_rocm_communication.py` passes `6 passed` in
`/tmp/hawk-implement-plan-check-inc5.log`. ROCm TP2 now selects PyTorch
`backend="nccl"` (RCCL on ROCm), bypasses PyNCCL, and creates Gloo only for
control traffic; CUDA TP2 retains PyNCCL and TP1 creates no device communicator.
PyNCCL loader and initializer reject ROCm before JIT/link. Real two-GPU RCCL
all-reduce and serving remain hardware-pending.

### Inc 6 — Generic GGUF substrate (L)

**Depends on:** 2
**Unblocks:** 7, 8, 9, 10
**Status:** done.
**Done criteria:** GGUF type metadata, shard reading, tokenizer handling, and quantized layer contracts support #131’s proven generic cases without claiming unvalidated ROCm native kernels.

#### Files to touch

##### `python/freetoken/models/gguf/dequant.py`, `reader.py`, `config.py`, `tokenizer.py`
- What changes: carry complete known GGML type/block tables, validate shards/tensor rows, and map architecture metadata. Special-token atomic registration belongs to Inc 10.
- Function(s): `row_bytes`, type-set dispatch, metadata-only reader, shard resolver, architecture lookup, tokenizer encode/decode helpers.
- Data shapes: GGUF metadata dict, tensor descriptors, shard offsets, quant type IDs, and token IDs.
- Integration points: registry/config loading and model weight iterators.
- Error paths: unknown type, missing shard, inconsistent row bytes, duplicate tensor, and invalid metadata raise named errors.

##### `python/freetoken/layers/gguf.py`
- What changes: add merged-linear and LM-head contracts needed by mixed quant/model loaders.
- Function(s): `fused_mul_mat_gguf`, `GGUFLinear`, `GGUFMergedLinear`, `GGUFLMHead`, embedding helpers.
- Data shapes: packed `[out, row_bytes]` weights, input `[tokens, in_features]`, output `[tokens, out_features]`.
- Integration points: native kernel dispatch or explicit reference dequant.
- Error paths: unsupported quant/device/shape raises; no uninitialized `torch.empty` result.

##### `python/freetoken/kernel/csrc/gguf/gguf_kernel.cu`, `moe_vec.cuh`
- What changes: preserve/extend CUDA dispatch for types whose independent reference tests pass; keep HIP-specific additions for Inc 8/9.
- Function(s): dequant/matvec dispatch and unsupported-type switch.
- Data shapes: GGML block rows and output dimensions.
- Integration points: `python/freetoken/kernel/gguf.py`.
- Error paths: switch default throws before returning a buffer.

##### `tests/models/test_gguf_type_tables.py`, `test_gguf_shards.py`, `tests/models/test_gguf_dispatch.py`, `tests/moe/test_cpu_moe_kquant.py`
- What changes: derive expected block sizes from GGML definitions/round trips and test shard/type failures.
- Function(s): independent CPU/PyTorch reference comparisons.
- Data shapes: every declared type, malformed shard tables, mixed row widths.
- Integration points: live registry/config and layer dispatch.
- Error paths: assert explicit errors for unsupported types.

#### Edge cases

- I-quants may be metadata-supported before native matmul support; reference/fallback policy must be explicit.
- Mixed quant types in one expert bank cannot use one stride; decline setup loudly.
- Fused rows require equal input dimension and row bytes.

#### Verification

- Run: `uv run pytest tests/models/test_gguf_type_tables.py tests/models/test_gguf_shards.py tests/models/test_gguf_dispatch.py tests/moe/test_cpu_moe_kquant.py`.
- Done: table tests cover all declared types; unsupported native path fails before allocation; CUDA Gemma4 GGUF regression passes.

**Evidence:** Added complete gguf-py-compatible block/type metadata through current
types `F32`–`Q1_0`, a Q8_0 reference dequant, ordered split-GGUF resolution, duplicate
name and packed-row validation, and explicit merged-linear/LM-head contracts. Native
dispatch remains limited to existing proven Q4_0/Q8_0/Q6_K routes; mixed quantized
merged rows reject with a stride error. Focused gate passed `26 passed` in
`/tmp/hawk-implement-plan-check-inc6.log`; Gemma4 GGUF rope regression passed `1
passed, 1 skipped` in `/tmp/hawk-implement-plan-check-inc6-gemma.log`. Shared ROCm
environment lacks flashlib, so metadata tests use scoped import stubs; no CUDA
Gemma4 fixture was available. Native GGUF ROCm work remains Inc 8/9.

### Inc 7 — Qwen3.5 GGUF adapter (L)

**Depends on:** 6
**Unblocks:** 9, 10, 13
**Status:** done.
**Done criteria:** Qwen3.5 MoE GGUF loads and performs TP1 reference decode with correct GDN/config/expert-bank contracts, without ROCm-specific kernel assumptions.

#### Files to touch

##### `python/freetoken/models/qwen3_5_moe/gguf.py`, `gguf_experts.py`, `model.py`
- What changes: parse `qwen35moe.*` metadata, deinterleave GDN tensors, map tied/untied heads, iterate dense and expert weights, and select GGUF layers.
- Function(s): `parse_gguf_config`, `iter_gguf_weights`, `load_gguf_expert_sources`, `convert_qwen35moe_to_gguf`.
- Data shapes: config metadata, packed row tensors, GDN state tensors, expert source descriptors, TP1 model.
- Integration points: model registry, weight loader, expert bank provider, GGUF layers.
- Error paths: TP>1, MTP, unsupported expert quant mix, missing required metadata, and bad permutation fail loudly.

##### `python/freetoken/models/register.py`, `python/freetoken/models/weight.py`, `python/freetoken/moe/expert_banks.py`
- What changes: register Qwen35 GGUF and expose a GGUF expert source/bank contract.
- Function(s): registry `ModelSpec`, GGUF source iterator, provider selection.
- Data shapes: lazy tensor source records with quant type, row bytes, offset, and layer/expert identity.
- Integration points: model setup and CPU/GPU/hybrid MoE backends.
- Error paths: registry architecture mismatch and bank-stride mismatch stop setup.

##### `tests/models/test_qwen35moe_gguf.py`, `test_qwen35moe_gguf_deint.py`, `tests/models/test_gguf_tokenizer_specials.py`, `tests/models/test_qwen35moe_moe.py`
- What changes: test metadata/config, GDN permutation, packed shapes, special tokens, and expert budgets against independent references.
- Function(s): fixture-level tests plus optional local-checkpoint tests.
- Data shapes: minimal synthetic GGUF metadata and packed rows; real Qwen fixture when available.
- Integration points: live registry and expert bank setup.
- Error paths: malformed metadata/permutation and unsupported quant mix.

#### Edge cases

- `A=-exp(A_log)` conversion must not double-log or change sign.
- Packed row permutations differ between dense and expert tensors.
- Tokenizer control/user-defined IDs must preserve original atomic encoding.

#### Verification

- Run: focused model/MoE tests above, then a real TP1 Qwen3.5 GGUF load with finite logits and exact completion count.
- Done: loader tests pass without ROCm; real GPU result records model file hash, tokenizer behavior, and unsupported feature gates.

**Evidence:** Ported generic Qwen3.5 GGUF metadata/config conversion, dense/GDN/head
weight mapping, mixed Q4_K/Q5_K/Q6_K/Q8_0 expert-bank contracts, atomic control-token
registration, chat-template resolution, and allocation-free resident budget helper.
Focused tests passed `38 passed` in `/tmp/hawk-implement-plan-check-inc8-static.log`
(includes all Inc7 tests). Real fixture `Qwen_Qwen3.5-35B-A3B-Q4_K_S.gguf` loaded on
gfx1100 through both CPU Q4_0 conversion and GPU native mixed-quant offload; one-token
HTTP 200 finite-output smoke passed. Backend-specific outputs are not parity evidence.

### Inc 8 — Native ROCm Q4_0 GGUF (M)

**Depends on:** 3, 6
**Unblocks:** 9, 10, 13
**Status:** done.
**Done criteria:** native Q4_0 GGUF build and matvec path works on declared wave32 RDNA3/RDNA4 targets, with staged sources, fresh-cache behavior, and explicit fallback for other quant types.

#### Files to touch

##### `python/freetoken/kernel/gguf.py`, `python/freetoken/kernel/utils.py`
- What changes: stage CUDA GGUF sources before HIPify, cache by source/Torch/HIP/target identity, share ROCm target parser, and bridge `FREETOKEN_ROCM_ARCH` to `PYTORCH_ROCM_ARCH`.
- Function(s): GGUF module builder/loader and ROCm compile/link flag helpers.
- Data shapes: module cache key includes source hash, compiler versions, target, and quant capability.
- Integration points: `GGUFLinear` dispatch and JIT extension cache.
- Error paths: non-wave32 or undeclared target fails before compile; forced native mode never silently falls back.

##### `python/freetoken/kernel/csrc/gguf/dispatch.h`
- What changes: guard wave32 ABI and use HIP-safe shuffle-mask width.
- Function(s): device dispatch helpers retain CUDA behavior.
- Data shapes: Q4_0 blocks, warp masks, output rows.
- Integration points: HIPified GGUF kernels.
- Error paths: unsupported wavefront size produces compile-time/runtime diagnostic.

##### `tests/kernels/test_gguf_rocm.py`, `tests/utils/test_rocm_arch.py`, `docs/install.md`, `README.md`
- What changes: test architecture bridge, Q4_0 dequant/matvec/MoE reference, and document supported targets/flags.
- Function(s): fresh build plus reference numerical checks.
- Data shapes: gfx1100-1103 and gfx1200-1201; other targets rejected until declared.
- Integration points: native GGUF loader and install docs.
- Error paths: missing headers/toolchain and unsupported arch.

#### Edge cases

- HIPify generated source must not persist across incompatible source/Torch/HIP versions.
- `gfx1100` and `gfx1201` must share family policy but may not share unverified tuning.
- Q5/Q6/I-quant requests must use explicit fallback or fail in forced-native mode.

#### Verification

- Run: `uv run pytest tests/kernels/test_gguf_rocm.py tests/utils/test_rocm_arch.py`.
- Run: fresh `TORCH_EXTENSIONS_DIR` native build and Q4_0 CPU/reference parity on each available target.
- Done: six tests and fresh-build evidence pass on supported RDNA targets; CUDA GGUF smoke remains green.

**Evidence:** HIP source staging and generic HIPified GGUF headers are checked in;
`dispatch.h` uses 64-bit ROCm shuffle masks and `gguf_kernel.cu` selects HIP guard,
stream, dequant, MMVQ, MMQ, and MoE headers without changing CUDA includes. Fresh
`TORCH_EXTENSIONS_DIR=/tmp/freetoken-gguf-ext-inc8` build passed under ROCm 7.2.1,
then native Q4_0 matvec and MoE finite-output gates passed; static architecture and
fallback gates passed `7 passed, 2 deselected` in `/tmp/hawk-implement-plan-check-inc8.log`,
and live native gates passed `2 passed, 7 deselected` using the fresh cache. Only
gfx1100 hardware is present; gfx1103/RDNA4 and CUDA runtime smoke remain unverified.

### Inc 9 — Portable K-quant native path (L)

**Depends on:** 7, 8
**Unblocks:** 10, 13
**Status:** done.
**Done criteria:** Q4_K/Q5_K/Q6_K/Q8_0 native GGUF paths are architecture-portable and independently validated; unsupported combinations retain explicit fallback.

#### Files to touch

##### `python/freetoken/kernel/csrc/gguf/ggml-common_hip.h`, `dequantize_hip.cuh`, `mmq_hip.cuh`, `mmvq_hip.cuh`, `vecdotq_hip.cuh`
- What changes: port generic K-quant dense dequant/matvec/vector-dot primitives from source fork, excluding gfx1100-only candidate kernels. Native MoE expert implementation belongs to Inc 16.
- Function(s): preserve block decode and matvec ABI; separate architecture capability from algorithm selection.
- Data shapes: Q4_K/Q5_K/Q6_K/Q8_0 blocks, dense `[tokens, hidden]` inputs, and dense output rows.
- Integration points: dense GGUF dispatch and linear/embedding layers.
- Error paths: alignment, block-size, and target capability failures select reference path or raise when forced.

##### `python/freetoken/kernel/csrc/gguf/gguf_kernel.cu`
- What changes: wire HIP/CUDA conditional includes and dense Q4_K/Q5_K/Q6_K/Q8_0 operators into active dispatch; add no expert-kernel implementation here.
- Function(s): module symbols `ggml_dequantize`, `ggml_mul_mat_vec_a8`, and `ggml_mul_mat_a8`; preserve output shapes and dtype contracts.
- Data shapes: GGML block rows, `[tokens, in_features]` inputs, and `[tokens, out_features]` outputs.
- Integration points: `python/freetoken/kernel/gguf.py` and dense GGUF layers.
- Error paths: unsupported quant type or target fails before returning a buffer.

##### `python/freetoken/kernel/gguf.py`, `python/freetoken/layers/gguf.py`
- What changes: register dense K-quant capability matrix without activating native MoE expert dispatch.
- Function(s): dispatch policy and provider metadata.
- Data shapes: `{quant_type, row_bytes, in_features, out_features, arch_family}`.
- Integration points: dense GGUF layers; Qwen expert metadata remains owned by Inc 7/16.
- Error paths: per-layer quant mismatch, unsupported target, and invalid alignment fail before execution.

##### `tests/kernels/test_gguf_rocm.py` (extend Inc 8), `tests/kernels/test_gguf_linear.py` (new), `benchmarks/bench_gguf_linear.py` (new/selective port)
- What changes: extend ROCm matrix and compare dense native outputs with CPU/PyTorch dequant or llama-compatible reference; add linear microbenchmark only as diagnostic. MoE tests/benchmarks belong to Inc 16.
- Function(s): deterministic direct-kernel checks and target matrix.
- Data shapes: decode and prefill batches, odd dimensions, and Q4K/Q5K/Q6K/Q8 rows.
- Integration points: JIT cache and dense GGUF layers.
- Error paths: unsupported candidate must be visible in trace and never counted as native.

#### Edge cases

- Grouped prefill, rotated-wave, and gfx1100-specific source candidates are not promoted by synthetic ABI success.
- Q4K/Q5K/Q6K native direct speed is not an end-to-end win.
- MoE headers, expert-bank layout, and native expert dispatch are deferred to Inc 16.
- Native output must be checked before adding any new default route.

#### Verification

- Run: direct dense numerical matrix on every available ROCm family plus CUDA GGUF regression.
- Done: all supported dense type/shape cases pass tolerance; native default changes only when Inc 14 proves serving benefit; no native MoE claim is made here.

**Evidence:** Generic HIP K-quant headers and active conditional includes provide
Q4_K/Q5_K/Q6_K/Q8_0 dequant/matvec symbols; no gfx1100 candidate kernel is selected
by dense dispatch. Fresh-cache gfx1100 gates passed `8 passed` in
`/tmp/hawk-implement-plan-check-inc9.log`, including independent Q8_1-activation
reference parity for all four quant types. Other ROCm families and CUDA runtime remain
hardware-pending; native MoE promotion remains deferred to Inc 16/Inc 14 gates.

### Inc 10 — Qwen GGUF runtime correctness (M)

**Depends on:** 7, 8, 9
**Unblocks:** 13
**Status:** live Qwen GGUF load/decode passed on gfx1100; native/reference parity and repeated A/B promotion pending.
**Done criteria:** Qwen3.5 GGUF serving preserves tokenizer, tool-call, last-token, penalty, and finite-logit correctness across native/reference routes.

#### Files to touch

##### `python/freetoken/models/gguf/tokenizer.py`, `python/freetoken/models/qwen3_5_moe/gguf.py`, `python/freetoken/models/register.py`
- What changes: port only proven Qwen architecture dispatch and output-head/penalty semantics from source fork; special-token atomic registration is validated here and is separate from Inc 6 metadata parsing.
- Function(s): tokenizer registration, GGUF model conversion, registry parser selection.
- Data shapes: token IDs, logits `[batch, vocab]`, last-token index, penalty state.
- Integration points: server tokenizer/parser and sampling.
- Error paths: missing tokenizer metadata, bad output shape, and unsupported Qwen feature fail explicitly.

##### `python/freetoken/core.py`, `python/freetoken/engine/sample.py`, `python/freetoken/models/qwen3_5_moe/model.py`, `python/freetoken/scheduler/scheduler.py`, `python/freetoken/server/args.py`, `python/freetoken/server/generation.py`, `python/freetoken/server/openai_api.py`
- What changes: port only proven last-token and penalty plumbing from source fork; re-check target first and make this a no-op when current code already contains behavior.
- Function(s): last-token gather, penalty application, request-to-sampler plumbing, and output-shape validation.
- Data shapes: logits `[batch, vocab]`, last-token indices, penalty state, streamed completion tokens.
- Integration points: Qwen3.5 GGUF model, scheduler, and OpenAI-compatible server.
- Error paths: invalid gather/penalty state or non-finite logits fails visibly; no silent semantic change.

##### `python/freetoken/server/function_call_parser.py`, `tests/server/test_parser_auto_selection.py`
- What changes: change parser selection only if target re-check reproduces a regression; current Qwen3.5 architecture mapping is already covered, so expected result may be no code change.
- Function(s): parser resolver and architecture-specific parser entry point.
- Data shapes: streamed message/tool-call tokens and model architecture metadata.
- Integration points: registry and OpenAI-compatible server.
- Error paths: parser mismatch is visible; no generic fallback that changes output semantics.

##### `tests/models/test_qwen35moe_gguf.py`, `tests/models/test_gguf_tokenizer_specials.py` (new), `tests/server/test_parser_auto_selection.py`, `tests/server/test_function_call_parser.py`, `tests/engine/test_sample.py` (new/selective)
- What changes: add finite-logit, last-token gather, penalty, tokenizer atomicity, tool-call, parser-selection, and exact completion tests.
- Function(s): test live registry and server call pattern.
- Data shapes: greedy and sampled requests, control/user-defined tokens, one-token decode.
- Integration points: native and reference GGUF layers.
- Error paths: repeated/empty output, invalid tool token, and non-finite logits.

#### Edge cases

- No MTP/speculative decode in acceptance; base decode only.
- Teacher-forced replay and sampled generation are separate protocols.
- A direct kernel pass does not prove served output correctness.
- Tokenizer control/user-defined IDs must preserve original atomic encoding; parser auto-selection must remain architecture-specific.
- Qwen3.5 parser mapping already present in target is a no-op unless a regression test disproves it.

#### Verification

- Run: model/server focused tests; real Qwen GGUF request with finite logits, non-empty output, exact token count, and parser trace.
- Done: native and reference routes agree within declared numerical/output gates; no sampled-only claim is used to validate replay.

**Evidence:** Qwen3.5 prefill last-token gather was already present in the adapter and is
covered by `tests/models/test_qwen35moe_last_token.py`. Added neutral sampling penalty
fields, prompt-boundary tracking, generated-suffix presence/frequency application, engine
plumbing, and OpenAI propagation. Focused gate passed `62 passed` in
`/tmp/hawk-implement-plan-check-inc10.log`, including parser auto-selection and tool-call
regressions. Real fixture `Qwen_Qwen3.5-35B-A3B-Q4_K_S.gguf` (sha256
`c889bb0b997a0e22d0477bd00a427e2b0e923c2f7eec3bea21091354a7ffb5a7`) loaded on gfx1100:
CPU Q4_0 conversion and GPU native mixed-quant offload both completed prefill warmup and
returned finite one-token HTTP 200 completions. CPU returned `Thinking`; GPU returned `1`;
these are separate backend/quantization streams, not parity evidence. Exact content parity,
longer completion, and native/reference agreement remain pending.

### Inc 11 — HIP graph-safe expert copies (M)

**Depends on:** 2
**Unblocks:** 13
**Status:** done.
**Done criteria:** during HIP graph capture, expert-cache copies avoid indirect host pointer tables; eager ROCm/CUDA fast paths remain intact.

#### Files to touch

##### `python/freetoken/moe/offload_cache.py`
- What changes: detect HIP graph capture and use legacy per-bank copy for capture only; retain fused copy for eager paths.
- Function(s): `OffloadMoeCache.copy_missing` and capture predicate.
- Data shapes: bank IDs, slot fingerprints, pinned host buffers, device destinations.
- Integration points: CPU/offload/hybrid MoE executor and graph replay.
- Error paths: invalid bank/slot/fingerprint fails before copy; no indirect pointer dereference during capture.

##### `tests/moe/test_fused_copy.py`
- What changes: test capture/eager routing and copy correctness with changing routes/fingerprints.
- Function(s): fake capture context plus independent tensor-copy reference.
- Data shapes: Q4K/Q5K rows, 40 layers, 256 experts, top-k 8, varying miss sets.
- Integration points: offload cache and graph capture hooks.
- Error paths: stale fingerprints and empty miss set.

#### Edge cases

- HIP capture must not observe host pointer-table mutation.
- CUDA fused path must remain unchanged.
- Synthetic copy latency is not serving throughput evidence.

#### Verification

- Run: `uv run pytest tests/moe/test_fused_copy.py`.
- Done: capture path passes changing-route replay; eager path retains fused behavior; report direct-copy latency separately from tok/s.

**Evidence:** `_hip_graph_capture_active()` gates only HIP capture; CUDA and eager
ROCm continue using fused multi-bank copies. Capture routing unit test passed `1
passed, 4 deselected` in `/tmp/hawk-implement-plan-check-inc11.log`; existing fused
copy parity passed `4 passed, 1 deselected` in
`/tmp/hawk-implement-plan-check-inc11-fused.log`. The route test uses mocked capture
state; full graph replay on non-gfx1100 HIP hardware remains Inc 13 hardware evidence.

### Inc 12 — CPU/Hybrid graph replay safety (L)

**Depends on:** 2
**Unblocks:** 13
**Status:** done.
**Done criteria:** CPU/Hybrid MoE graph replay enables only after real HIP signal-memory capture/instantiate/replay probing and never silently reuses stale graph state.

#### Files to touch

##### `python/freetoken/engine/engine.py`, `python/freetoken/moe/cpu_executor.py`, `python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp`
- What changes: add executor-owned signal/node storage, dynamic per-layer capacity, and fail-closed handshake for HIP graph batch-memory nodes; preserve CUDA module-level memop path.
- Function(s): graph capability probe, executor setup, replay synchronization, CPU MoE submit/sync.
- Data shapes: per-layer capacities, signal handles, graph node IDs, batch sizes, expert routes.
- Integration points: engine graph capture/replay and CPU/Hybrid scheduler.
- Error paths: probe failure, unsupported HIP version, capacity overflow, stale signal, or replay mismatch disables only unsafe graph path.

##### `tests/engine/test_cpu_moe_graph_safety.py`, `tests/moe/test_cpu_moe.py`, `tests/moe/test_cpu_moe_q4_0.py`
- What changes: reproduce eager-correct/replay-wrong case and assert fail-closed behavior across batch sizes and quant types.
- Function(s): fake/native capability matrix plus CPU reference output.
- Data shapes: bf16, MXFP4, DS-FP4, GGUF Q4_0; 19 graph batch sizes and dynamic capacity.
- Integration points: engine and CPU MoE executor.
- Error paths: HIP 7.2 compatibility and unavailable signal APIs.

#### Edge cases

- Batch sizes above initial 16-slot capacity must allocate/declare more storage.
- CUDA graph routing must not inherit HIP handshake state.
- Eager output correctness is insufficient; replay output must match.

#### Verification

- Run: focused CPU/Hybrid graph tests on ROCm and CUDA CPU-MoE regression.
- Done: real replay passes all supported cases; unsupported ROCm versions fail closed with a visible reason; no graph claim from eager-only execution.

**Evidence:** Rebuilt `_cpu_moe` under ROCm 7.2.1/Torch `2.10.0+git8514f05`; native
`memops_probe` executed and returned `False` because current HIP headers/runtime do
not expose required graph batch-memory APIs. Engine and direct executor therefore
disable capture and retain eager CPU/Hybrid execution. New graph-safety matrix passed
`14 passed` in `/tmp/hawk-implement-plan-check-inc12.log`; existing CPU-MoE graph
regressions correctly skipped `4 skipped, 21 deselected` in
`/tmp/hawk-implement-plan-check-inc12-moe-graph.log`. CUDA module-level memops remain
separate and were covered by the matrix. Real HIP graph replay remains pending on a
HIP version exposing signal-memory graph nodes; no eager-only replay claim made.

### Inc 13 — Cross-arch serving matrix (M)

**Depends on:** 9, 10, 11, 12
**Unblocks:** 14
**Status:** gfx1100 CPU and GPU served; gfx1100 graph replay and other physical targets pending.
**Done criteria:** declared ROCm support is backed by fresh build, graph/eager, native/reference GGUF, and served-request evidence across available gfx1100-1103, gfx1150-1151, and gfx1200-1201 targets.

#### Files to touch

##### `python/freetoken/utils/arch.py`, `python/freetoken/kernel/aot.py`, `aot_models.py`, `python/freetoken/kernel/gguf.py`
- What changes: make capability declarations and AOT/JIT target selection share one normalized matrix.
- Function(s): target parser, family classifier, native capability lookup, AOT manifest selector.
- Data shapes: target -> `{family, wave_size, native_gguf_types, graph_features}`.
- Integration points: JIT/AOT loading, model setup, docs.
- Error paths: target absent from matrix is unsupported; no accidental nearest-arch substitution.

##### `docs/amd-rocm-gfx1151.md`, `docs/install.md`, `docs/models.md`
- What changes: document tested vs compile-only targets, required versions, supported GGUF quant/fallback paths, and known graph limitations.
- Function(s): documentation only.
- Data shapes: hardware/version/command/evidence table.
- Integration points: install and model selection.
- Error paths: docs must not call compile success serving support.

##### `tests/utils/test_rocm_arch.py`, `tests/kernels/test_gguf_rocm.py`, `tests/e2e/test_cache_rebuild.py`
- What changes: validate matrix parsing, unsupported-target failures, native GGUF capability, and small real-server cache rebuild where model fixture exists.
- Function(s): live registry/engine tests and e2e gate.
- Data shapes: target strings, backend capability records, small model path.
- Integration points: engine startup and server rebuild.
- Error paths: missing fixture/hardware skips with reason, never passes as support evidence.

#### Edge cases

- gfx1150/1151 support is distinct from gfx1100 and gfx1201 until both have native evidence.
- ROCm version changes invalidate JIT cache identity.
- Graph disabled is a valid explicit result, not a test failure when unsupported.

#### Verification

- Run: architecture tests, `FREETOKEN_REBUILD_TEST_MODEL=<small model dir> uv run pytest tests/e2e/test_cache_rebuild.py`, and one served request per available GPU.
- Done: support table labels each cell `tested`, `compile-only`, or `unsupported`; no unverified cell is advertised.

**Evidence:** Added closed `RocmArchCapability` matrix for RDNA3 gfx1100-1103,
RDNA3.5 gfx1150-1151, and RDNA4 gfx1200-1201; runtime suffixes normalize to exact
targets and unknown targets fail before GGUF dispatch. Updated installation/model
documentation with explicit compile-only status. Focused gate passed `19 passed, 1
skipped` in `/tmp/hawk-implement-plan-check-inc13.log`; cache-rebuild e2e skipped
because no test model was supplied. Available hardware is gfx1100 only, so served
requests, graph/eager comparison, and native/reference matrix cells for other targets
remain pending. Available gfx1100 served the real Qwen fixture through CPU Q4_0 and GPU
native mixed-quant offload; GPU used the in-tree Triton full-fetch LRU fallback because
CUDA-only `flashlib.kernels.slot_cache` is absent. gfx1150/1151/gfx1200/1201 and real HIP
graph replay remain pending.

### Inc 14 — ROCm benchmark and provenance gates (M)

**Depends on:** 13
**Unblocks:** 15, 16, 17
**Status:** code-complete; real repeated A/B gate pending.
**Done criteria:** every optimization candidate has reproducible same-model A/B evidence, runtime route proof, and correctness/completion gates.

#### Files to touch

##### `benchmarks/bench_rocm_matrix.py` (new), `benchmarks/check_decode_gate.py` (new; selective port), `benchmarks/bench_decode_replay.py` (new; selective port), `benchmarks/bench_decode_moe.py`, `benchmarks/bench_gguf_moe_kernels.py` (new; selective port)
- What changes: reuse source-fork replay/gate machinery where compatible, then add machine-readable manifest/provenance and candidate-vs-incumbent protocol around existing benchmarks. Files marked new are absent from current upstream.
- Function(s): manifest writer, route/capture counter collector, median/variance summarizer, teacher-forced replay runner.
- Data shapes: manifest schema 1: `{"workload":{"model_sha256","prompt_sha256","token_count","mtp","flags"},"runtime":{"commit","dirty_diff","gpu","driver","torch","rocm","hip","triton","jit_sha","env_digest"},"observed":{"backend","quant","graph_mode","cache_hits","fetches","fallbacks","finite_logits","completion_count"},"timing":{"lane","repeats","median_tok_s","spread"}}`.
- Integration points: served decode, direct kernels, offload/cache telemetry.
- Error paths: missing route/counter/output evidence marks result invalid rather than filling defaults.

##### `.github/workflows/unit-rocm.yml` (new; manual/self-hosted only), `.github/workflows/unit-nvidia.yml` (extend Inc 1 gate), `benchmarks/README.md`, `docs/reproducibility.md`
- What changes: add manual/self-hosted ROCm and hosted NVIDIA compile/unit gates; document exact commands, warmup/fresh-process policy, sampled vs teacher-forced gates, and A/B reporting format.
- Function(s): manual/self-hosted ROCm gate and hosted NVIDIA compile/unit gate; protocol documentation.
- Data shapes: JSON manifest and summary table.
- Integration points: PR body and review evidence.
- Error paths: incomplete run is labeled incomplete; ROCm workflow has no `pull_request` trigger, runs no fork code on self-hosted machines, takes target/model as inputs, and never hardcodes gfx1100 as final support.

#### Edge cases

- Requested backend/quant does not prove observed backend/quant.
- Warm cache, cold cache, graph replay, and eager runs are separate populations.
- Torch profiler CPU ranges are not additive wall time.
- Manifest helper can be prototyped while Inc 2-13 proceed; final benchmark-gate merge waits for Inc 13 serving fields.

#### Verification

- Run: existing benchmark scripts plus new manifest on incumbent `upstream/main` and candidate with identical settings; invoke ROCm workflow manually on available hardware and keep NVIDIA workflow safe for pull requests.
- Done: repeated runs produce valid manifests and distinguish sampled streams from paired replay; no optimization PR proceeds without this artifact.

**Evidence:** Added manifest, replay, and candidate-gate helpers with workload hashes,
runtime/toolchain/JIT identity, observed route/counter/completion fields, timing lanes,
and median-based comparison. Added reproducibility documentation and manual/self-hosted
ROCm workflow with safe upstream checkout. Focused metadata/gate tests passed `10 passed`
in `/tmp/hawk-implement-plan-check-inc14.log`. No same-model incumbent/candidate
repeated serving run was available, so no optimization promotion claim is made.

### Inc 15 — Guarded ROCm fused router (M)

**Depends on:** 14
**Unblocks:** none
**Status:** code-complete; incumbent retained; A/B promotion pending.
**Done criteria:** fused ROCm router is merged only if route/output parity holds and end-to-end non-MTP base decode beats incumbent beyond measured noise; otherwise legacy router remains default and negative evidence is recorded.

#### Files to touch

##### `python/freetoken/moe/fused.py`, `python/freetoken/layers/moe.py`
- What changes: profile `fused_topk` and, only when justified, optimize existing in-repo router behind explicit capability/feature selection; do not re-port source-fork #98 wholesale because current upstream already has `moe/fused.py`.
- Function(s): `fused_topk`, `_torch_fused_topk`, top-k indices/weights, and fused MoE handoff.
- Data shapes: logits `[tokens, experts]`, top-k IDs/weights, capacity/routing masks.
- Integration points: Triton MoE path and CPU/reference router.
- Error paths: unsupported target, route mismatch, non-finite weights, or forced candidate failure raises/uses explicit fallback.

##### `tests/moe/test_fused_moe.py`, `benchmarks/bench_decode_moe.py`
- What changes: extend existing fused-MoE tests to compare routes/outputs with CPU/PyTorch reference and expose candidate route in benchmark manifest.
- Function(s): deterministic routing test and served A/B.
- Data shapes: empty/singleton/batched tokens, ties, masked experts, top-k boundaries.
- Integration points: Inc 14 evidence protocol.
- Error paths: degenerate repeated output or route divergence blocks promotion.

#### Edge cases

- Direct-kernel speed is not serving speed.
- Candidate must not activate for unvalidated gfx target.
- No MTP/speculative decode in speed gate.

#### Verification

- Run: router tests, exact output/completion gate, then at least three fresh-process A/B medians on same Qwen GGUF/model settings.
- Done: promote only on independent parity plus reproducible end-to-end win; otherwise keep feature off and document the negative result.

**Evidence:** Added ROCm-only route spy coverage for in-repo Triton routing across
single, small-batch, and batched token shapes; current upstream implementation already
uses this portable router, so no redundant source-fork CUDA package path was added.
ROCm live route test and same-model repeated A/B serving evidence remain pending on a
usable model fixture; incumbent default remains unchanged.

### Inc 16 — Guarded native GGUF MoE kernels (L)

**Depends on:** 9, 10, 14
**Unblocks:** none
**Status:** code-complete; gfx1100 native served smoke passed; performance promotion pending.
**Done criteria:** portable native Qwen GGUF expert kernels are merged only if multi-quant numerical parity and served decode improve over incumbent; gfx1100-only candidates remain opt-in experiments.

#### Files to touch

##### `python/freetoken/moe/fused_gguf.py`, `python/freetoken/kernel/csrc/gguf/gguf_kernel.cu`, `moe_hip.cuh`, `moe_vec_hip.cuh`, `vecdotq_hip.cuh`
- What changes: after Inc 9 dense ownership is complete, port generic packed expert gate/up/down kernels and wire HIP/CUDA MoE symbols into GGUF dispatch; connect them to quant capability records.
- Function(s): fused expert matvec and reduction; no hardcoded arch symbol in public dispatch.
- Data shapes: mixed Q4_K/Q5_K/Q6_K/Q8_0 expert rows, route IDs, token batches.
- Integration points: Qwen expert bank and GGUF layer dispatch.
- Error paths: alignment/quant mismatch/unsupported target uses explicit reference route or fails in forced mode.

##### `tests/kernels/test_gguf_moe.py` (new/selective), `benchmarks/bench_gguf_moe_kernels.py` (new/selective)
- What changes: CPU reduction, dispatch/id-space, mixed-type, direct numerical, and end-to-end expert evidence gates; dense linear tests remain Inc 9's ownership.
- Function(s): independent reference comparisons and route trace assertions.
- Data shapes: decode/prefill, odd hidden dimensions, per-layer quant variation.
- Integration points: Inc 14 manifest and Qwen serving.
- Error paths: no native claim when fallback/candidate route is observed.

#### Edge cases

- Existing source-fork fused gate/up candidate has no promotion authority if served output degenerates.
- Grouped prefill or rotated-wave path requires real model evidence, not synthetic launch success.
- Native kernels must remain optional for architectures without direct validation.
- Source-fork gfx1100-only candidates remain opt-in experiments and cannot satisfy portable MoE promotion.

#### Verification

- Run: all direct/reference tests, fresh JIT build, finite-logit/completion gate, and three-plus A/B served runs per target.
- Done: only portable, numerically correct, reproducibly faster path changes default; otherwise retain legacy MMVQ/reference behavior.

**Evidence:** Added guarded packed GGUF MoE gate/up/down routing, reusable work buffers,
explicit raw/slot ID spaces, mixed Q5_K/Q6_K strided dispatch, and weighted Triton
route reduction. Contract, grouped-route, and mixed-quant tests pass `8 passed` in
`/tmp/hawk-implement-plan-check-inc16.log`; gfx1100 live Q5_K, Q6_K, and Q8_0 native
output matches dequantized/Q8_1 reference within the BF16 reduction gate with finite
results. Candidate fused gate/up and ID-aware paths remain opt-in. Real Qwen native GPU
offload loaded mixed Q4_K/Q5_K/Q6_K/Q8_0 banks and served two finite one-token requests;
no incumbent/candidate A/B run exists, so no default/performance promotion is claimed.

### Inc 17 — ROCm profiler and JIT hygiene (S)

**Depends on:** 14
**Unblocks:** none
**Status:** code-complete; optional hardware trace pending.
**Done criteria:** ROCm profiling and JIT-cache artifacts identify the observed route and stale locks cannot poison fresh runs; no runtime behavior change is accepted from this increment.

#### Files to touch

##### `python/freetoken/kernel/utils.py`
- What changes: selectively port source-fork profiler/cache identity and lock diagnostics from #217; preserve source/version invalidation and never delete shared caches.
- Function(s): cache-key construction, extension-cache diagnostics, lock/error reporting.
- Data shapes: source hash, Torch/ROCm/HIP/Triton versions, target, JIT SHA, cache path, lock owner/state.
- Integration points: Inc 8/9 GGUF JIT and Inc 14 provenance manifest.
- Error paths: stale lock, cache mismatch, or missing compiler reports actionable state and uses isolated cache identity.

##### `benchmarks/profile_decode_rocm.py` (new/selective port), `benchmarks/README.md`
- What changes: add optional ROCm trace collection for decode lanes and document profiler warmup, fresh-process, and route-counter requirements.
- Function(s): trace capture and manifest attachment only; no automatic performance promotion.
- Data shapes: kernel/CPU ranges, lane, repeats, route, graph mode, cache hits/fetches.
- Integration points: Inc 14 manifest and same-model A/B protocol.
- Error paths: unavailable profiler or incomplete trace is labeled incomplete, not treated as zero overhead.

##### `tests/kernels/test_kernel_cache_version.py` (existing), `tests/utils/test_decode_benchmark_metadata.py` (new/selective)
- What changes: assert cache identity changes with source/toolchain/target and manifests reject missing identity or route fields.
- Function(s): deterministic cache-key and metadata-schema tests.
- Data shapes: old/new source hashes, stale lock records, complete/incomplete manifests.
- Integration points: JIT loader and benchmark gate.
- Error paths: stale or incomplete artifacts fail validation without mutating shared cache state.

#### Edge cases

- Use unique `TORCH_EXTENSIONS_DIR` per run; never remove or overwrite a shared sibling cache.
- Profiler CPU ranges overlap GPU work and are not summed into wall time.
- A fresh cache proves hygiene, not kernel correctness or speed.

#### Verification

- Run: cache-key/schema tests, fresh-process build with unique cache, and optional ROCm profile on available hardware.
- Done: stale-lock and source/toolchain invalidation cases pass; profile/manifest records observed route; no speed claim is made without Inc 14 A/B gates.

**Evidence:** Added source/toolchain/backend/target-derived JIT identity injected into
compile flags, backend-tag cache rejection, scoped stale-lock diagnostics/cleanup, and
explicit ROCm trace wrapper with incomplete-artifact status. Cache/version and profiler
tests pass `20 passed` in `/tmp/hawk-implement-plan-check-inc17.log`. No `rocprofv3`
trace was captured in this run; no profiler or speed claim is made.

## Cross-cutting verification

- Environment preflight: resolve Python/venv, import Torch, identify CUDA vs
  ROCm, and record compiler/runtime versions before running a gate; missing
  dependencies are `blocked`, not a reason to create a second large venv.
- Before every PR: `git diff --check`, focused tests, and `uv run pytest tests/ -m "not slow"` where environment supports it.
- Build matrix: CUDA regression plus ROCm versions/targets actually available; record skips and reasons.
- Runtime acceptance: server boots, native/reference route is observable, logits are finite, output is non-empty, completion count is exact, and graph state/capture result is recorded.
- Performance acceptance: same model file/tokenizer, prompt/settings, quant/KV/MTP flags, GPU selection, fresh-process policy, and at least three runs; report median and spread, not peak.
- Reproducibility identity: commit and dirty diff, model/fixture hashes, GPU/driver, Torch/ROCm/HIP/Triton versions, JIT source/cache identity, environment digest, route counters, cache hit/fetch counts.
- Rollback: every candidate has an explicit feature/capability gate and incumbent path; failed evidence disables candidate without deleting reference code.
- Human handoff: do not push or open/comment on PRs from agent workflow; contributor reviews every line and supplies real hardware evidence.

## Standards / common-mistakes referenced

- `upstream/main:AGENTS.md:1-71` — binding AI policy, repository layout, development commands, test placement, A/B requirement, and no agent push/PR actions.
- `upstream/main:CONTRIBUTING.md:20-75` — one change per PR, hardware/checkpoint/command evidence, performance proof, bug tests, and commit style.
- `upstream/main:tests/README.md:3-100` — subsystem test placement and independent-reference expectations.
- `benchmarks/README.md:1-31` — existing decode/load/copy/bandwidth benchmark surfaces.
- `origin/feat/amd-rocm-gfx1100-support:.agents/learnings/index.yml` and selected learning files — fail-closed dispatch, graph/replay timing, negative candidate handling, and base-speed scope.
- No `.agents/standards/` or `.agents/common-mistakes/` directory exists in this checkout or `upstream/main`; binding guidance above is used instead.

## Open questions (CONSIDER from review)

- Confirm with FlashML maintainers whether RCCL/TP belongs in this contribution series or should remain a separate multi-GPU PR.
- Confirm which real GPUs are available for Inc 13: RX 7900 XTX gfx1100, gfx1150/1151, and R9700 gfx1201.
- Decide whether Qwen3Moe/DeepSeek-V4 GGUF adapters from #131 deserve separate model-specific PRs after generic substrate lands; they are not required for Qwen3.5 ROCm serving.
- Set exact statistical threshold for a performance promotion after Inc 14 establishes variance; proposed default is >=5% median end-to-end gain with no correctness/latency regression.

## Out of scope

- Wholesale merge/cherry-pick of `samuelishida/FreeToken` or source branch `feat/amd-rocm-gfx1100-support`.
- #23/#137 as separate code landings; their bring-up is superseded/overlapped by #132 and #241.
- PLE/SSD tiering, Qwen3.8, tinygrad, MTP/speculative decode, and unrelated source-fork features.
- NVIDIA NVFP4/Marlin optimization as ROCm work.
- Promoting gfx1100-only rotated-wave, grouped-prefill, b10434, or fused candidates from microbenchmarks alone.
- Claiming support from compile success, HTTP 200, server boot, or a requested-but-unobserved backend.
- Agent-created commits, pushes, PRs, issue comments, or reviewer replies.
