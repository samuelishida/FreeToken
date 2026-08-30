"""Native-GGUF quantized layers: weights stay in their packed block layout and are
dequantized *inside* the borrowed llama.cpp kernels (no bf16 copy ever materialized).

Mirrors vLLM/sglang's ``GGUFLinearMethod`` / ``GGUFEmbeddingMethod`` dispatch, ported
onto FreeToken's ``BaseOP``. FreeToken keeps fused projections (qkv, gate_up) as a
single tensor: because Q4_0/K-quants pack each *output row* independently over the
input dim, the loader can concatenate the per-shard packed rows along dim 0 (they
share an input dim, hence the same ``row_bytes``), so a fused layer is still one
``[out, row_bytes]`` qweight -- no per-shard padding bookkeeping needed.

TP is assumed to be 1 (the gemma4 GGUF path restricts to TP=1, like the HF path).
"""

from __future__ import annotations

import torch
import os
from freetoken.utils import init_logger

from freetoken.models.gguf.dequant import (
    BLOCK_SHAPE,
    GGML_BF16,
    GGML_F16,
    GGML_F32,
    GGML_IQ2_XS,
    GGML_IQ3_XXS,
    GGML_IQ4_NL,
    GGML_NAME,
    GGML_Q4_0,
    GGML_Q4_K,
    GGML_Q5_K,
    GGML_Q6_K,
    GGML_Q8_0,
    row_bytes,
)

from .base import BaseOP

logger = init_logger(__name__)

# ggml type groups for kernel dispatch (subset we build kernels for).
_UNQUANTIZED = {GGML_F32, GGML_F16, GGML_BF16}
# standard + k-quants: both an MMVQ (small-batch GEMV) and MMQ (large-batch) kernel exist.
_MMVQ = {
    GGML_Q4_0, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0,
    GGML_IQ2_XS, GGML_IQ3_XXS, GGML_IQ4_NL,
}
_MMQ = {GGML_Q4_0, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0}
_DEQUANT = {GGML_Q4_0, GGML_Q8_0, GGML_Q6_K}
_VECTOR_ONLY = {GGML_IQ2_XS, GGML_IQ3_XXS, GGML_IQ4_NL}

# Below this token count, the MMVQ GEMV kernel wins (matches vLLM's heuristic).
_MMVQ_SAFE = 6
_TRITON_DECODE_LOGGED: set[int] = set()
_TRITON_DECODE_FAILURE_LOGGED: set[int] = set()
_GGUF_CALL_LOGGED: set[int] = set()


def fused_mul_mat_gguf(x: torch.Tensor, qweight: torch.Tensor, qweight_type: int) -> torch.Tensor:
    """y = x @ dequant(qweight).T, dispatched by batch size and quant type."""
    from freetoken.kernel.gguf import (
        ggml_dequantize,
        ggml_mul_mat_a8,
        ggml_mul_mat_vec_a8,
    )

    out_features = qweight.shape[0]
    if int(qweight_type) not in _GGUF_CALL_LOGGED:
        logger.info(
            "GGUF linear decode call: type=%s rows=%s width=%s device=%s dtype=%s",
            int(qweight_type), int(out_features), int(x.shape[1]), x.device, x.dtype,
        )
        _GGUF_CALL_LOGGED.add(int(qweight_type))
    if x.shape[0] == 0:
        return x.new_empty((0, out_features))
    if qweight_type in _UNQUANTIZED:
        return x @ qweight.T
    # ROCm gfx1100's vendored GGUF MMVQ path can stall on standard K-quants.
    # Decode one token through the bounded Triton packed GEMV instead. This
    # keeps weights packed, avoids multi-gigabyte dequant buffers, and falls
    # through to the known-correct Torch decoder when Triton is unavailable.
    if (
        torch.version.hip is not None
        and x.shape[0] == 1
        and qweight_type in (GGML_Q8_0, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K)
        and os.environ.get("FREETOKEN_DISABLE_GGUF_TRITON_DECODE", "").strip().lower()
        not in ("1", "true", "yes", "on")
    ):
        try:
            from freetoken.kernel.triton.qwen4exp_quant import fused_gguf_decode_standard
            output = fused_gguf_decode_standard(
                x,
                qweight,
                ggml_type=int(qweight_type),
                out_features=int(out_features),
            )
            if int(qweight_type) not in _TRITON_DECODE_LOGGED:
                logger.info(
                    "GGUF Triton standard GEMV active: type=%s rows=%s width=%s",
                    int(qweight_type), int(out_features), int(x.shape[1]),
                )
                _TRITON_DECODE_LOGGED.add(int(qweight_type))
            return output
        except Exception as exc:
            # Optional optimization. Existing dispatch below remains parity
            # fallback; qwen38 script forces safe Torch fallback on ROCm.
            if int(qweight_type) not in _TRITON_DECODE_FAILURE_LOGGED:
                logger.warning(
                    "GGUF Triton standard GEMV unavailable: type=%s rows=%s width=%s: %s",
                    int(qweight_type), int(out_features), int(x.shape[1]), exc,
                )
                _TRITON_DECODE_FAILURE_LOGGED.add(int(qweight_type))
            pass
    # IQ formats have vendored MMVQ kernels but no MMQ kernels.
    if qweight_type in _VECTOR_ONLY or (x.shape[0] <= _MMVQ_SAFE and qweight_type in _MMVQ):
        return ggml_mul_mat_vec_a8(qweight, x, qweight_type, out_features)
    if qweight_type in _MMQ:
        return ggml_mul_mat_a8(qweight, x, qweight_type, out_features)
    if qweight_type in _DEQUANT:
        block, type_size = BLOCK_SHAPE[qweight_type]
        in_features = qweight.shape[1] // type_size * block
        weight = ggml_dequantize(qweight, qweight_type, out_features, in_features, x.dtype)
        return x @ weight.T
    raise NotImplementedError(f"unsupported GGUF type {GGML_NAME.get(qweight_type, qweight_type)}")


class GGUFLinear(BaseOP):
    """Linear whose weight is a native GGUF block-quantized ``[out, row_bytes]`` tensor."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant_type: int,
        has_bias: bool = False,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self._quant_type = quant_type
        self.qweight = torch.empty(out_features, row_bytes(in_features, quant_type), dtype=torch.uint8)
        self.bias = torch.empty(out_features) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = fused_mul_mat_gguf(x, self.qweight, self._quant_type)
        if self.bias is not None:
            out = out + self.bias
        return out


class GGUFEmbedding(BaseOP):
    """Vocab embedding stored as a native GGUF block-quantized table.

    The full table is never dequantized: only the looked-up rows are gathered (in
    packed form) and dequantized per lookup, matching vLLM's ``_apply_gguf_embedding``.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        quant_type: int,
        embed_scale: float | None = None,
    ):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._quant_type = quant_type
        self.qweight = torch.empty(
            num_embeddings, row_bytes(embedding_dim, quant_type), dtype=torch.uint8
        )
        self._embed_scale = embed_scale
        self._embed_scale_t: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.gguf import ggml_dequantize

        flat = x.flatten()
        rows = self.qweight.index_select(0, flat)  # [n, row_bytes] packed
        y = ggml_dequantize(rows, self._quant_type, flat.shape[0], self.embedding_dim, torch.bfloat16)
        y = y.view(*x.shape, self.embedding_dim)
        if self._embed_scale is not None:
            if self._embed_scale_t is None:
                self._embed_scale_t = torch.tensor(self._embed_scale, dtype=y.dtype, device=y.device)
            y = y * self._embed_scale_t
        return y


__all__ = ["GGUFLinear", "GGUFEmbedding", "fused_mul_mat_gguf"]
