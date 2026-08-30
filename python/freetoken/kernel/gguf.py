"""Borrowed llama.cpp GGUF dequant/GEMM CUDA kernels, JIT-compiled on first use.

The ``.cu``/``.cuh`` under ``csrc/gguf/`` are vendored verbatim from sgl-kernel
(``csrc/quantization/gguf/``), which are themselves ports of llama.cpp. We compile
them through ``torch.utils.cpp_extension.load`` (the same toolchain sglang/vllm use)
into a torch-op module and expose the handful of ops the GGUF path needs. This is a
separate, torch-native extension that sits alongside FreeToken's tvm-ffi kernels.

All ops keep the weight in its native GGUF block layout (packed ``uint8`` rows) and
dequantize *inside* the kernel -- no bf16 copy of the weight is ever materialized.
"""

from __future__ import annotations

import functools
import os
import pathlib
import shutil

import torch

_CSRC = pathlib.Path(__file__).parent / "csrc" / "gguf"

# ROCm builds of the vendored ggml kernels can launch successfully yet never
# return on gfx1100 for K-quants (the first affected operation is usually the
# Q5_K token embedding).  Keep packed bytes as the source of truth, but use the
# bounded pure-Torch decoder for formats whose layout is implemented locally.
# IQ2/IQ3 remain on their dedicated ggml kernels until their decoder is ported.
_ROCM_TORCH_DEQUANT_TYPES = frozenset({0, 1, 2, 8, 12, 13, 14, 30})
_ROCM_DEQUANT_CHUNK_ROWS = 4096


def _rocm_torch_fallback_enabled(quant_type: int) -> bool:
    if torch.version.hip is None or int(quant_type) not in _ROCM_TORCH_DEQUANT_TYPES:
        return False
    # Keep conservative Torch fallback as library default for older ROCm/GFX
    # combinations. Qwen3.8 production route opts into validated vendored
    # GGUF GEMV/dequant kernels; users can force Torch fallback with env=1.
    return os.environ.get("FREETOKEN_ROCM_GGUF_TORCH_FALLBACK", "1").strip().lower() in (
        "1", "true", "yes", "on"
    )


def _rocm_dequant_rows(
    weight: torch.Tensor,
    quant_type: int,
    m: int,
    n: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Decode standard GGUF rows without entering known-stalling HIP kernels."""
    from freetoken.models.gguf.dequant import dequantize

    if m < 0 or n < 0:
        raise ValueError(f"GGUF dimensions must be non-negative, got m={m}, n={n}")
    if weight.dtype != torch.uint8:
        raise TypeError(f"packed GGUF weight must be uint8, got {weight.dtype}")
    expected = int(m) * int(n)
    if expected == 0:
        return torch.empty((m, n), dtype=dtype, device=weight.device)
    flat = dequantize(weight.reshape(-1), int(quant_type), dtype)
    if flat.numel() != expected:
        raise ValueError(
            f"GGUF packed shape/type decode produced {flat.numel()} values; expected {expected} "
            f"for [{m}, {n}] type {quant_type}"
        )
    return flat.reshape(m, n)


def _rocm_matmul_fallback(
    weight: torch.Tensor,
    x: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    """Chunked W4/5/6/8A16 matmul; bounds temporary dense-weight memory."""
    if x.ndim != 2:
        raise ValueError(f"GGUF matmul input must be rank-2, got shape {tuple(x.shape)}")
    rows, n = int(row), int(x.shape[1])
    if weight.shape[0] != rows:
        raise ValueError(f"GGUF row mismatch: weight has {weight.shape[0]}, requested {rows}")
    out = torch.empty((x.shape[0], rows), dtype=x.dtype, device=x.device)
    for start in range(0, rows, _ROCM_DEQUANT_CHUNK_ROWS):
        stop = min(rows, start + _ROCM_DEQUANT_CHUNK_ROWS)
        dense = _rocm_dequant_rows(weight[start:stop], quant_type, stop - start, n, x.dtype)
        out[:, start:stop] = x @ dense.T
    return out


def _host_compiler() -> str | None:
    """A host compiler nvcc + libtorch headers accept.

    The system default gcc can be too new for the torch headers (gcc 16 hard-errors),
    and on this toolchain even nvcc+gcc-13 trips a non-conformant ``typename
    decltype`` in ``List_inl.h`` once ``torch::Tensor`` is instantiated -- but nvcc
    with ``clang++`` as host compiles it cleanly. So prefer clang++, then fall back
    to an older gcc. Override with ``FREETOKEN_GGUF_HOST_CXX``.
    """
    override = os.environ.get("FREETOKEN_GGUF_HOST_CXX")
    if override:
        return override
    for cxx in ("clang++", "g++-13", "g++-14", "g++-15"):
        if shutil.which(cxx):
            return cxx
    return None


def _c_compiler_for(cxx: str) -> str:
    base = os.path.basename(cxx)
    if "clang" in base:
        return shutil.which("clang") or "clang"
    cc = base.replace("g++", "gcc")
    return shutil.which(cc) or cc

@functools.cache
def _module():
    from torch.utils.cpp_extension import load

    if torch.version.hip is not None:
        # ROCm: hipcc (torch.utils.cpp_extension picks it up), pass the HIP defines so
        # the kernels compile their HIP branches; drop the CUDA-only -ccbin/flag logic.
        # Explicit --offload-arch (plus PYTORCH_ROCM_ARCH) prevents torch from auto-
        # emitting ~14 gfx arches, which would multiply build time per arch.
        gfx = os.getenv("FREETOKEN_KERNEL_CACHE_GFX", "gfx1100")
        os.environ.setdefault("PYTORCH_ROCM_ARCH", gfx)
        extra_cuda_cflags = [
            "-O3", f"--offload-arch={gfx}", "-DUSE_HIP=1", "-DUSE_ROCM=1",
        ]
        os.environ.pop("CXX", None)
        os.environ.pop("CC", None)
    else:
        extra_cuda_cflags = ["-O3", "--expt-relaxed-constexpr"]
        host_cxx = _host_compiler()
        if host_cxx is not None:
            # Point both nvcc's host pass (-ccbin) and torch's C++ compile (CXX) at a
            # libtorch/nvcc-compatible compiler. Force (not setdefault): the system
            # default (CXX unset -> g++) can be a gcc too new for the torch headers.
            cxx_path = shutil.which(host_cxx) or host_cxx
            extra_cuda_cflags += ["-ccbin", cxx_path]
            os.environ["CXX"] = cxx_path
            os.environ["CC"] = _c_compiler_for(cxx_path)

    # gguf_kernel.cu carries its own PYBIND11_MODULE (appended at the end), so a
    # plain `load` of the single source compiles + binds the ggml_* ops.
    return load(
        name="freetoken_gguf_kernels",
        sources=[str(_CSRC / "gguf_kernel.cu")],
        extra_include_paths=[str(_CSRC)],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
    )


# ---- thin typed wrappers (signatures mirror sgl_kernel.quantization.gguf) ----


def ggml_dequantize(
    weight: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Dequantize a packed GGUF weight ``[m, row_bytes]`` to a dense ``[m, n]`` tensor."""
    # The vendored IQ4_NL kernel consumes complete 256-value superblocks.  Its
    # block launcher rounds up smaller widths and writes past the requested
    # output, which is especially dangerous for Qwen PLE's 160-wide rows.
    # Keep this guard before extension loading so CPU-only callers fail fast too;
    # PLE rows use freetoken.kernel.triton.ple_iq4_nl instead.
    if int(quant_type) == 20 and (int(n) <= 0 or int(n) % 256):
        raise ValueError(
            "generic GGUF IQ4_NL dequantization requires n > 0 and n % 256 == 0; "
            f"got n={n}. Use the dedicated five-block PLE dequant helper for 160-wide rows."
        )
    if _rocm_torch_fallback_enabled(quant_type):
        return _rocm_dequant_rows(weight, int(quant_type), int(m), int(n),
                                  dtype or torch.float16)
    return _module().ggml_dequantize(weight, quant_type, m, n, dtype)


def ggml_mul_mat_vec_a8(
    weight: torch.Tensor, x: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    """MMVQ: small-batch GEMV with on-the-fly dequant. ``row`` = output features."""
    if _rocm_torch_fallback_enabled(quant_type):
        return _rocm_matmul_fallback(weight, x, int(quant_type), int(row))
    return _module().ggml_mul_mat_vec_a8(weight, x, quant_type, row)


def ggml_mul_mat_a8(
    weight: torch.Tensor, x: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    """MMQ: large-batch quantized matmul. ``row`` = output features."""
    if _rocm_torch_fallback_enabled(quant_type):
        return _rocm_matmul_fallback(weight, x, int(quant_type), int(row))
    return _module().ggml_mul_mat_a8(weight, x, quant_type, row)


def ggml_moe_a8(
    x: torch.Tensor,
    weight: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    """MMQ grouped expert matmul over stacked experts ``weight[E, row, *]``."""
    return _module().ggml_moe_a8(
        x, weight, sorted_token_ids, expert_ids, num_tokens_post_padded,
        quant_type, row, top_k, tokens,
    )


def ggml_moe_a8_vec(
    x: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: int,
    tokens: int,
) -> torch.Tensor:
    """MMVQ grouped expert GEMV over stacked experts ``weight[E, row, *]``."""
    return _module().ggml_moe_a8_vec(x, weight, topk_ids, top_k, quant_type, row, tokens)


def ggml_moe_get_block_size(quant_type: int) -> int:
    return _module().ggml_moe_get_block_size(quant_type)


__all__ = [
    "ggml_dequantize",
    "ggml_mul_mat_vec_a8",
    "ggml_mul_mat_a8",
    "ggml_moe_a8",
    "ggml_moe_a8_vec",
    "ggml_moe_get_block_size",
]
