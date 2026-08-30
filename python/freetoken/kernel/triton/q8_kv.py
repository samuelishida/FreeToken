"""Per-token symmetric int8 K/V cache store for QSA."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _q8_store_kernel(
    k_ptr,
    v_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    indices_ptr,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_cache_token,
    stride_cache_head,
    stride_scale_token,
    stride_scale_head,
    stride_indices,
    HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    dim = tl.arange(0, BLOCK_D)
    mask = dim < HEAD_DIM
    dst = tl.load(indices_ptr + token * stride_indices).to(tl.int64)

    k = tl.load(
        k_ptr + token * stride_kt + head * stride_kh + dim,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    v = tl.load(
        v_ptr + token * stride_vt + head * stride_vh + dim,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    k_scale = tl.maximum(tl.max(tl.abs(k), axis=0) / 127.0, 1.0e-8)
    v_scale = tl.maximum(tl.max(tl.abs(v), axis=0) / 127.0, 1.0e-8)
    k_scaled = k / k_scale
    v_scaled = v / v_scale
    k_q = tl.where(mask, tl.where(k_scaled >= 0, tl.floor(k_scaled + 0.5), tl.ceil(k_scaled - 0.5)), 0.0).to(tl.int8)
    v_q = tl.where(mask, tl.where(v_scaled >= 0, tl.floor(v_scaled + 0.5), tl.ceil(v_scaled - 0.5)), 0.0).to(tl.int8)

    tl.store(
        k_cache_ptr + dst * stride_cache_token + head * stride_cache_head + dim,
        k_q,
        mask=mask,
    )
    tl.store(
        v_cache_ptr + dst * stride_cache_token + head * stride_cache_head + dim,
        v_q,
        mask=mask,
    )
    tl.store(
        k_scale_ptr + dst * stride_scale_token + head * stride_scale_head,
        k_scale.to(k_scale_ptr.dtype.element_ty),
    )
    tl.store(
        v_scale_ptr + dst * stride_scale_token + head * stride_scale_head,
        v_scale.to(v_scale_ptr.dtype.element_ty),
    )


def q8_store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """Quantize/scatter ``[tokens, heads, dim]`` K/V rows into int8 cache."""
    if k.ndim != 3 or v.shape != k.shape:
        raise ValueError("Q8 KV store expects matching [tokens, heads, dim] inputs")
    if k_cache.ndim != 3 or v_cache.shape != k_cache.shape:
        raise ValueError("Q8 KV store expects matching cache tensors")
    if k_cache.dtype != v_cache.dtype or k_cache.dtype != torch.int8:
        raise ValueError("Q8 KV cache must use int8 storage")
    if k_scales.shape != k_cache.shape[:2] or v_scales.shape != k_cache.shape[:2]:
        raise ValueError("Q8 KV scale shape must match cache token/head dimensions")
    if indices.shape != (k.shape[0],):
        raise ValueError("Q8 KV indices must have one location per token")
    if k.shape[1] != k_cache.shape[1] or k.shape[2] != k_cache.shape[2]:
        raise ValueError("Q8 KV input/cache geometry mismatch")
    if k.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"unsupported Q8 KV input dtype {k.dtype}")
    if not k.is_cuda:
        raise ValueError("Q8 KV store requires CUDA/HIP tensors")
    _q8_store_kernel[(k.shape[0], k.shape[1])](
        k,
        v,
        k_cache,
        v_cache,
        k_scales,
        v_scales,
        indices,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_scales.stride(0),
        k_scales.stride(1),
        indices.stride(0),
        HEAD_DIM=k.shape[2],
        NUM_HEADS=k.shape[1],
        BLOCK_D=triton.next_power_of_2(k.shape[2]),
        num_warps=4,
        num_stages=1,
    )


__all__ = ["q8_store_cache"]
