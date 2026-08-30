"""Safe five-block IQ4_NL dequantization for packed Qwen PLE rows.

Qwen GGUF PLE rows contain five ordinary IQ4_NL blocks (90 bytes, 160 values),
not one GGML 256-value superblock.  This module keeps the packed representation
through the storage tiers and decodes only requested rows.  The CPU path uses
the independent scalar/vector oracle; CUDA/HIP uses a small Triton gather kernel.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from freetoken.models.gguf.dequant import (
    IQ4_NL_KVALUES,
    PLE_IQ4_NL_BLOCK_BYTES,
    PLE_IQ4_NL_BLOCKS_PER_ROW,
    PLE_IQ4_NL_ROW_BYTES,
    PLE_IQ4_NL_ROW_VALUES,
    dequant_iq4_nl,
)


class PLEDequantUnavailable(RuntimeError):
    """Raised when a packed PLE row reaches an unsupported accelerator path."""


_BLOCK_VALUES = 32
_KVALUES = IQ4_NL_KVALUES


try:  # Triton is optional for CPU-only installs.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised by CPU-only environments
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _ple_iq4_nl_kernel(
        packed_ptr,
        scale_ptr,
        out_ptr,
        kvalues_ptr,
        packed_stride,
        out_stride,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        value = tl.arange(0, BLOCK)
        mask = value < 160
        block = value // 32
        within = value % 32
        q_index = within % 16
        byte_offset = block * 18 + 2 + q_index
        q = tl.load(packed_ptr + row * packed_stride + byte_offset, mask=mask, other=0).to(tl.int32)
        q = tl.where(within >= 16, q >> 4, q & 0xF)
        levels = tl.load(kvalues_ptr + q, mask=mask, other=0).to(tl.float32)
        scale = tl.load(scale_ptr + row * 5 + block, mask=mask, other=0.0)
        tl.store(out_ptr + row * out_stride + value, scale * levels, mask=mask)


@lru_cache(maxsize=16)
def _kvalues(device_index: int, device_type: str) -> torch.Tensor:
    device = torch.device(device_type, device_index) if device_type != "cpu" else torch.device("cpu")
    return torch.tensor(_KVALUES, dtype=torch.int8, device=device)


def _validate_input(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed PLE rows must be uint8, got {packed.dtype}")
    if packed.ndim != 2 or packed.shape[1] not in (PLE_IQ4_NL_ROW_BYTES, 96):
        raise ValueError(
            "packed PLE rows must have shape [N,90] or aligned [N,96], "
            f"got {tuple(packed.shape)}"
        )
    return packed.contiguous()


def _torch_dequant(packed: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Vectorized fallback. Safe on CPU and useful as a deterministic test oracle."""
    raw = packed[:, :PLE_IQ4_NL_ROW_BYTES].reshape(-1, PLE_IQ4_NL_BLOCK_BYTES)
    scales = raw[:, :2].contiguous().view(torch.float16).to(torch.float32)
    qbytes = raw[:, 2:]
    low = (qbytes & 0x0F).to(torch.int16)
    high = (qbytes >> 4).to(torch.int16)
    levels = torch.cat((low, high), dim=1).to(torch.long)
    lut = torch.tensor(_KVALUES, dtype=torch.float32, device=packed.device)
    values = lut.index_select(0, levels.reshape(-1)).reshape(-1, _BLOCK_VALUES)
    values = (values * scales).reshape(-1, PLE_IQ4_NL_ROW_VALUES)
    return values.to(out_dtype)


def dequant_iq4_nl_rows(
    packed: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode [N,90] or aligned [N,96] packed rows into [N,160]."""
    packed = _validate_input(packed)
    rows = packed.shape[0]
    if out is not None and (out.shape != (rows, PLE_IQ4_NL_ROW_VALUES) or out.dtype != out_dtype):
        raise ValueError(
            f"PLE output must be {(rows, PLE_IQ4_NL_ROW_VALUES)} {out_dtype}, got "
            f"{tuple(out.shape)} {out.dtype}"
        )
    if rows == 0:
        result = torch.empty((0, PLE_IQ4_NL_ROW_VALUES), dtype=out_dtype, device=packed.device)
    elif packed.device.type == "cpu":
        result = _torch_dequant(packed, out_dtype)
    elif packed.device.type == "cuda":
        if triton is None:
            raise PLEDequantUnavailable("Triton is required for CUDA/HIP PLE dequantization")
        # Scales are ordinary fp16 values at each 18-byte block boundary.  Keep
        # extraction on-device, then let Triton gather nibble values and write
        # exactly 160 outputs per row.
        raw = packed[:, :PLE_IQ4_NL_ROW_BYTES].reshape(rows, PLE_IQ4_NL_BLOCKS_PER_ROW, PLE_IQ4_NL_BLOCK_BYTES)
        scales = raw[:, :, :2].contiguous().view(torch.float16).to(torch.float32)
        result = torch.empty((rows, PLE_IQ4_NL_ROW_VALUES), dtype=out_dtype, device=packed.device)
        try:
            _ple_iq4_nl_kernel[(rows,)](
                packed,
                scales,
                result,
                _kvalues(packed.device.index or 0, packed.device.type),
                packed.stride(0),
                result.stride(0),
                BLOCK=256,
                num_warps=4,
            )
        except Exception as exc:
            raise PLEDequantUnavailable(
                f"Triton five-block IQ4_NL PLE kernel unavailable on {packed.device}: {exc}"
            ) from exc
    else:
        raise PLEDequantUnavailable(f"PLE dequantization unsupported on {packed.device.type}")
    if out is not None:
        out.copy_(result)
        return out
    return result


def gather_dequant_iq4_nl_rows(
    packed_slots: torch.Tensor,
    row_ids: torch.Tensor | None = None,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode packed slots, optionally selecting rows before dequantization."""
    packed_slots = _validate_input(packed_slots)
    if row_ids is not None:
        packed_slots = packed_slots.index_select(0, row_ids.reshape(-1).to(torch.long))
    return dequant_iq4_nl_rows(packed_slots, out_dtype=out_dtype, out=out)


__all__ = [
    "PLEDequantUnavailable",
    "dequant_iq4_nl_rows",
    "gather_dequant_iq4_nl_rows",
]
