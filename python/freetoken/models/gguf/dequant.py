"""GGML block-quant dequantization in pure torch (the formats this repo's GGUF
checkpoints use: Q4_0, Q6_K, plus trivial F32/F16/BF16).

This is the *reference / CPU* path, NOT the engine's hot path: GGUF weights stay
packed and are dequantized inside the borrowed ggml CUDA kernels (see
``freetoken.kernel.gguf``). These routines are used only to (a) materialize the few
dense F32/F16 tensors at load (norms, scales, router) via :func:`dequantize`, and
(b) cross-check the CUDA kernels in tests. The ``BLOCK_SHAPE`` table and
:func:`row_bytes` are the type metadata the packed (kernel) path also relies on.

Each ``dequant_*`` takes the raw little-endian bytes as a ``uint8`` tensor whose
final axis spans whole blocks, and returns the values in *storage order* (ggml's
fastest axis first); the caller reshapes to the torch shape (``dims[::-1]``). The
math mirrors ``ggml-quants.c``.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

import torch

# ggml_type enum values (subset present in these checkpoints).
GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_0 = 2
GGML_Q8_0 = 8
GGML_Q4_K = 12
GGML_Q5_K = 13
GGML_Q6_K = 14
GGML_IQ2_XS = 17
GGML_IQ3_XXS = 18
GGML_IQ4_NL = 20
GGML_BF16 = 30

# Qwen4Exp PLE uses five ordinary IQ4_NL blocks per row rather than ggml's
# 256-value (eight-block) superblock.  Keep this geometry explicit: the
# generic CUDA/HIP IQ4_NL kernel is not a valid decoder for these rows.
PLE_IQ4_NL_BLOCK_VALUES = 32
PLE_IQ4_NL_BLOCK_BYTES = 18
PLE_IQ4_NL_BLOCKS_PER_ROW = 5
PLE_IQ4_NL_ROW_BYTES = (
    PLE_IQ4_NL_BLOCK_BYTES * PLE_IQ4_NL_BLOCKS_PER_ROW
)
PLE_IQ4_NL_ROW_VALUES = (
    PLE_IQ4_NL_BLOCK_VALUES * PLE_IQ4_NL_BLOCKS_PER_ROW
)

# Canonical non-linear IQ4 lookup values from ggml-common.h.  Tuple form keeps
# oracle definition device-independent; dequant_iq4_nl creates a tensor on
# input device for indexing.
IQ4_NL_KVALUES = (
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113,
)


@functools.lru_cache(maxsize=1)
def _iq_tables() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load compact IQ lookup tables from vendored ggml sources.

    IQ2/3 use irregular codebooks; reproducing them algorithmically would risk
    changing GGUF bytes.  Keep one source of truth by extracting tables from the
    exact ggml-common.h shipped with this checkout, then cache CPU tensors.  Only
    512*8 + 256*4 bytes are retained; callers move indexed rows to their device.
    """
    header = Path(__file__).resolve().parents[2] / "kernel" / "csrc" / "gguf" / "ggml-common.h"
    try:
        source = header.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - only broken source distributions
        raise RuntimeError(f"IQ lookup tables unavailable: cannot read {header}") from exc

    def values(name: str, count: int) -> list[int]:
        match = re.search(
            rf"{re.escape(name)}\[{count}\]\s*=\s*\{{(.*?)\}};",
            source,
            re.DOTALL,
        )
        if match is None:
            raise RuntimeError(f"IQ lookup table {name} missing from {header}")
        found = [
            int(token, 0)
            for token in re.findall(r"0x[0-9a-fA-F]+|\b\d+\b", match.group(1))
        ]
        if len(found) != count:
            raise RuntimeError(f"IQ lookup table {name} has {len(found)} entries, expected {count}")
        return found

    def unpack_words(words: list[int], width: int) -> torch.Tensor:
        raw = bytearray()
        for word in words:
            raw.extend(int(word).to_bytes(width, "little"))
        return torch.tensor(list(raw), dtype=torch.uint8)

    iq2 = unpack_words(values("iq2xs_grid", 512), 8).reshape(512, 8)
    iq3 = unpack_words(values("iq3xxs_grid", 256), 4).reshape(256, 4)
    signs = torch.tensor(values("ksigns_iq2xs", 128), dtype=torch.uint8)
    sign_bits = torch.arange(8, dtype=torch.uint8)
    signs = ((signs[:, None] & (1 << sign_bits)) != 0).to(torch.float32)
    signs.mul_(-2.0).add_(1.0)
    return iq2, iq3, signs

# (block numel, bytes per block) per ggml type.
BLOCK_SHAPE: dict[int, tuple[int, int]] = {
    GGML_F32: (1, 4),
    GGML_F16: (1, 2),
    GGML_BF16: (1, 2),
    GGML_Q4_0: (32, 18),
    GGML_Q8_0: (32, 34),
    GGML_Q4_K: (256, 144),
    GGML_Q5_K: (256, 176),
    GGML_Q6_K: (256, 210),
    # Packed geometry only. Numerical IQ dequantization lives in vendored HIP.
    GGML_IQ2_XS: (256, 74),
    GGML_IQ3_XXS: (256, 98),
    GGML_IQ4_NL: (32, 18),
}

GGML_NAME = {
    GGML_F32: "F32",
    GGML_F16: "F16",
    GGML_BF16: "BF16",
    GGML_Q4_0: "Q4_0",
    GGML_Q8_0: "Q8_0",
    GGML_Q4_K: "Q4_K",
    GGML_Q5_K: "Q5_K",
    GGML_Q6_K: "Q6_K",
    GGML_IQ2_XS: "IQ2_XS",
    GGML_IQ3_XXS: "IQ3_XXS",
    GGML_IQ4_NL: "IQ4_NL",
}


def row_bytes(numel: int, ggml_type: int) -> int:
    """Packed byte length of one row of ``numel`` elements in ``ggml_type`` blocks.

    Single source of truth for the ``numel // block * type_size`` math shared by the
    packed-weight ops (``GGUFLinear``/``GGUFEmbedding``) and the expert bank loaders.
    """
    block, type_size = BLOCK_SHAPE[ggml_type]
    assert numel % block == 0, (
        f"{numel} not a multiple of block {block} for {GGML_NAME.get(ggml_type, ggml_type)}"
    )
    return numel // block * type_size


def _f16_scales(raw: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """Reinterpret bytes ``[lo:hi]`` (2 per block) of each block row as fp16 -> fp32 [N,1]."""
    return raw[:, lo:hi].contiguous().view(torch.float16).to(torch.float32)


def _normalize_iq4_nl_ple_rows(
    packed: torch.Tensor, rows: int | None
) -> tuple[torch.Tensor, int]:
    """Validate and reshape native five-block PLE bytes to ``[rows, 90]``.

    PLE readers expose either row-shaped bytes or one flat byte span.  Use
    ``reshape`` instead of ``view`` so non-contiguous row tensors are accepted
    without making callers materialize their own copy.  Width validation occurs
    before any dequantization or extension dispatch.
    """
    if not isinstance(packed, torch.Tensor):
        raise ValueError("IQ4_NL PLE input must be a torch.Tensor")
    if packed.dtype != torch.uint8:
        raise ValueError(
            f"IQ4_NL PLE input must have dtype torch.uint8, got {packed.dtype}"
        )
    if rows is not None:
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            raise ValueError(f"IQ4_NL PLE rows must be a non-negative integer, got {rows!r}")

    if packed.ndim == 1:
        byte_count = packed.numel()
        if byte_count % PLE_IQ4_NL_ROW_BYTES:
            raise ValueError(
                "IQ4_NL PLE flattened input has non-integral rows: "
                f"{byte_count} bytes is not divisible by {PLE_IQ4_NL_ROW_BYTES}"
            )
        inferred_rows = byte_count // PLE_IQ4_NL_ROW_BYTES
        if rows is not None and rows != inferred_rows:
            raise ValueError(
                f"IQ4_NL PLE row count {rows} does not match {inferred_rows} packed rows"
            )
        return packed.reshape(inferred_rows, PLE_IQ4_NL_ROW_BYTES), inferred_rows

    if packed.ndim != 2 or packed.shape[1] != PLE_IQ4_NL_ROW_BYTES:
        width = packed.shape[-1] if packed.ndim else 0
        raise ValueError(
            "IQ4_NL PLE input must have shape [rows, 90] or flat length "
            f"multiple of 90, got shape {tuple(packed.shape)} (width {width})"
        )
    inferred_rows = packed.shape[0]
    if rows is not None and rows != inferred_rows:
        raise ValueError(
            f"IQ4_NL PLE row count {rows} does not match {inferred_rows} packed rows"
        )
    return packed.reshape(inferred_rows, PLE_IQ4_NL_ROW_BYTES), inferred_rows


def dequant_iq4_nl(
    packed: torch.Tensor,
    *,
    rows: int | None = None,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode Qwen4Exp's five-block IQ4_NL PLE rows using pure Torch.

    Each 18-byte block stores little-endian fp16 scale followed by 16 bytes of
    nibbles.  Low nibbles produce values 0..15, then high nibbles produce
    values 16..31, matching ``dequantize_block_iq4_nl``.  Unlike the generic
    GGUF kernel, this routine always allocates exactly ``[rows, 160]`` and is
    safe for CPU-only tests as well as tensors on an accelerator device.
    """
    supported_dtypes = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
    if out_dtype not in supported_dtypes:
        raise ValueError(
            "IQ4_NL PLE output dtype must be floating-point (float16, bfloat16, "
            f"float32, or float64), got {out_dtype}"
        )

    packed, row_count = _normalize_iq4_nl_ple_rows(packed, rows)
    if row_count == 0:
        return torch.empty(
            (0, PLE_IQ4_NL_ROW_VALUES), dtype=out_dtype, device=packed.device
        )

    # Flatten blocks only after geometry validation.  ``contiguous`` is needed
    # for the byte reinterpretation of fp16 scales and is bounded to input size.
    blocks = packed.reshape(row_count * PLE_IQ4_NL_BLOCKS_PER_ROW,
                            PLE_IQ4_NL_BLOCK_BYTES)
    scales = _f16_scales(blocks, 0, 2)
    nibbles = blocks[:, 2:]
    kvalues = torch.as_tensor(IQ4_NL_KVALUES, dtype=torch.float32,
                              device=packed.device)
    low = kvalues[(nibbles & 0x0F).to(torch.long)]
    high = kvalues[(nibbles >> 4).to(torch.long)]
    values = torch.cat((low, high), dim=1) * scales
    return values.reshape(row_count, PLE_IQ4_NL_ROW_VALUES).to(out_dtype)


def dequant_iq4_nl_blocks(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Decode ordinary 32-value IQ4_NL blocks (18 bytes each).

    GGUF expert rows contain many independent blocks; this differs from the
    five-block/160-value PLE row helper above only in its row geometry.
    """
    raw = raw.reshape(-1, 18).contiguous()
    if raw.shape[0] == 0:
        return torch.empty((0,), dtype=out_dtype, device=raw.device)
    scales = raw[:, :2].view(torch.float16).to(torch.float32)
    nibbles = raw[:, 2:]
    kvalues = torch.as_tensor(IQ4_NL_KVALUES, dtype=torch.float32, device=raw.device)
    low = kvalues[(nibbles & 0x0F).to(torch.long)]
    high = kvalues[(nibbles >> 4).to(torch.long)]
    return (torch.cat((low, high), dim=1) * scales).reshape(-1).to(out_dtype)


def dequant_iq2_xs(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Decode standard 256-value IQ2_XS blocks with Torch operations.

    This is used only by the ROCm safety path.  It preserves ggml storage order
    (eight 32-value groups, each four 8-value codebook vectors) and never calls
    the gfx1100-unstable native IQ kernel.
    """
    raw = raw.reshape(-1, 74).contiguous()
    n = raw.shape[0]
    if n == 0:
        return torch.empty((0,), dtype=out_dtype, device=raw.device)
    grid_table, _, sign_table = _iq_tables()
    device = raw.device
    q2 = raw[:, 2:66].view(torch.uint16).reshape(n, 8, 4)
    q2_i = q2.to(torch.int64)
    scale_bytes = raw[:, 66:74]
    grid = grid_table.to(device=device).index_select(0, (q2_i & 0x1FF).reshape(-1))
    grid = grid.reshape(n, 8, 4, 8).to(torch.float32)
    signs = sign_table.to(device=device).index_select(0, (q2_i >> 9).reshape(-1))
    signs = signs.reshape(n, 8, 4, 8)
    il = torch.arange(4, device=device, dtype=torch.long).view(1, 1, 4)
    subscale = ((scale_bytes.to(torch.long).view(n, 8, 1) >> (4 * (il // 2))) & 0xF)
    scale = raw[:, :2].view(torch.float16).to(torch.float32).view(n, 1, 1)
    scale = scale * (0.5 + subscale.to(torch.float32)) * 0.25
    return (grid * signs * scale.unsqueeze(-1)).reshape(-1).to(out_dtype)


def dequant_iq3_xxs(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Decode standard 256-value IQ3_XXS blocks with Torch operations."""
    raw = raw.reshape(-1, 98).contiguous()
    n = raw.shape[0]
    if n == 0:
        return torch.empty((0,), dtype=out_dtype, device=raw.device)
    _, grid_table, sign_table = _iq_tables()
    device = raw.device
    q3 = raw[:, 2:66].reshape(n, 8, 8)
    gas = raw[:, 66:98].view(torch.uint16).reshape(n, 8, 2)
    idx1 = q3[:, :, 0::2].reshape(n, 8, 4)
    idx2 = q3[:, :, 1::2].reshape(n, 8, 4)
    grid_cpu = grid_table.to(device=device)
    grid1 = grid_cpu.index_select(0, idx1.to(torch.long).reshape(-1)).reshape(n, 8, 4, 4)
    grid2 = grid_cpu.index_select(0, idx2.to(torch.long).reshape(-1)).reshape(n, 8, 4, 4)
    grid = torch.cat((grid1, grid2), dim=-1).to(torch.float32)
    aux = gas[:, :, 0].to(torch.int64) | (gas[:, :, 1].to(torch.int64) << 16)
    il = torch.arange(4, device=device, dtype=torch.long).view(1, 1, 4)
    signs = sign_table.to(device=device).index_select(
        0, ((aux[:, :, None] >> (7 * il)) & 0x7F).reshape(-1)
    ).reshape(n, 8, 4, 8)
    scale = raw[:, :2].view(torch.float16).to(torch.float32).view(n, 1, 1, 1)
    scale = scale * (0.5 + (aux >> 28).to(torch.float32)).unsqueeze(-1).unsqueeze(-1) * 0.5
    return (grid * signs * scale).reshape(-1).to(out_dtype)


def dequant_q4_0(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q4_0: per 32-elem block = fp16 scale ``d`` + 16 packed nibbles; ``w = d*(q-8)``.

    Byte ``j`` of the 16 holds element ``j`` in its low nibble and ``j+16`` in its high
    nibble, so storage order within the block is ``[lo0..lo15, hi0..hi15]``.
    """
    raw = raw.reshape(-1, 18)
    d = _f16_scales(raw, 0, 2)  # [N,1]
    qs = raw[:, 2:18]  # [N,16] uint8
    lo = (qs & 0x0F).to(torch.float32)
    hi = (qs >> 4).to(torch.float32)
    q = torch.cat([lo, hi], dim=1)  # [N,32]
    return ((q - 8.0) * d).reshape(-1).to(out_dtype)


def dequant_q6_k(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q6_K: 256-elem super-block = 128B low nibbles + 64B high 2-bits + 16 int8
    sub-scales + fp16 ``d``. Direct vectorization of ggml's two-half loop."""
    raw = raw.reshape(-1, 210)
    n = raw.shape[0]
    ql = raw[:, 0:128]  # [n,128]
    qh = raw[:, 128:192]  # [n,64]
    sc = raw[:, 192:208].view(torch.int8).to(torch.float32)  # [n,16]
    d = _f16_scales(raw, 208, 210)  # [n,1]

    y = torch.empty((n, 256), dtype=torch.float32, device=raw.device)
    # l in 0..15 -> is=0; l in 16..31 -> is=1 (per ggml: is = l/16).
    is_idx = (torch.arange(32, device=raw.device) // 16)  # [32] in {0,1}
    for h in range(2):  # two 128-elem halves of the super-block
        qlh = ql[:, h * 64:(h + 1) * 64]  # [n,64]
        qhh = qh[:, h * 32:(h + 1) * 32]  # [n,32]
        sch = sc[:, h * 8:(h + 1) * 8]  # [n,8]
        a = qlh[:, 0:32].to(torch.int32)  # ql[l]
        b = qlh[:, 32:64].to(torch.int32)  # ql[l+32]
        hb = qhh.to(torch.int32)  # qh[l]
        q1 = ((a & 0x0F) | (((hb >> 0) & 3) << 4)) - 32
        q2 = ((b & 0x0F) | (((hb >> 2) & 3) << 4)) - 32
        q3 = ((a >> 4) | (((hb >> 4) & 3) << 4)) - 32
        q4 = ((b >> 4) | (((hb >> 6) & 3) << 4)) - 32
        s1 = sch.index_select(1, is_idx + 0).to(torch.float32)
        s2 = sch.index_select(1, is_idx + 2).to(torch.float32)
        s3 = sch.index_select(1, is_idx + 4).to(torch.float32)
        s4 = sch.index_select(1, is_idx + 6).to(torch.float32)
        base = h * 128
        y[:, base + 0:base + 32] = d * s1 * q1.to(torch.float32)
        y[:, base + 32:base + 64] = d * s2 * q2.to(torch.float32)
        y[:, base + 64:base + 96] = d * s3 * q3.to(torch.float32)
        y[:, base + 96:base + 128] = d * s4 * q4.to(torch.float32)
    return y.reshape(-1).to(out_dtype)


def dequant_q8_0(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q8_0: fp16 scale followed by 32 signed int8 values."""
    raw = raw.reshape(-1, 34)
    scale = _f16_scales(raw, 0, 2)
    values = raw[:, 2:].view(torch.int8).to(torch.float32)
    return (values * scale).reshape(-1).to(out_dtype)


def dequant_q4_k(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q4_K: vectorized 256-value K-quant decoder.

    Layout follows ``block_q4_K``: fp16 ``(d, dmin)``, 12 packed scale/min
    bytes, then 128 low/high nibble bytes.  Keeping this in Torch gives HIP a
    safe fallback when the optional vendored kernel is unavailable.
    """
    raw = raw.reshape(-1, 144)
    n = raw.shape[0]
    dm = raw[:, :4].view(torch.float16).to(torch.float32)
    dall, dmin = dm[:, 0], dm[:, 1]
    scales = raw[:, 4:16]
    qs = raw[:, 16:].to(torch.int32)

    def _scale_min(j: int) -> tuple[torch.Tensor, torch.Tensor]:
        if j < 4:
            d = scales[:, j] & 0x3F
            m = scales[:, j + 4] & 0x3F
        else:
            d = (scales[:, j + 4] & 0x0F) | ((scales[:, j - 4] >> 6) << 4)
            m = (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)
        return d.to(torch.float32), m.to(torch.float32)

    out = torch.empty((n, 256), dtype=torch.float32, device=raw.device)
    for il in range(4):
        q = qs[:, 32 * il:32 * il + 32]
        d0, m0 = _scale_min(2 * il)
        d1, m1 = _scale_min(2 * il + 1)
        lo = (q & 0x0F).to(torch.float32)
        hi = (q >> 4).to(torch.float32)
        out[:, 64 * il:64 * il + 32] = lo * (dall * d0)[:, None] - (dmin * m0)[:, None]
        out[:, 64 * il + 32:64 * il + 64] = hi * (dall * d1)[:, None] - (dmin * m1)[:, None]
    return out.reshape(-1).to(out_dtype)


def quantize_q8_0(w: torch.Tensor) -> torch.Tensor:
    """Quantize dense rows to packed Q8_0 blocks (``half d`` + 32 int8).

    ``w``'s last dim must be a multiple of 32 (the Q8_0 block); returns the packed
    ``[..., n/32*34]`` uint8 layout the ggml Q8_0 kernels read. Used to re-quantize
    K-quant expert banks to a uniform 8-bit type (Q8_0 >= Q5_K/Q6_K precision, so no
    quality loss) when the offload cache needs a single per-bank format.
    """
    n = w.shape[-1]
    assert n % 32 == 0, f"Q8_0 quantize needs last dim % 32 == 0, got {n}"
    wq = w.float().view(*w.shape[:-1], n // 32, 32)
    d = wq.abs().amax(dim=-1, keepdim=True).clamp(min=1e-9) / 127.0
    q = torch.round(wq / d).to(torch.int8)
    dh = d.to(torch.float16).view(torch.uint8)  # [..., n//32, 2]
    packed = torch.cat([dh, q.view(torch.uint8)], dim=-1)  # [..., n//32, 34]
    return packed.reshape(*w.shape[:-1], (n // 32) * 34).contiguous()


def dequant_q5_k(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q5_K: 256-elem super-block = half2 dm (dall, dmin), 12B 6-bit scale/min, 32B
    qh high-bits, 128B qs low nibbles. Mirrors ggml's dequantize_block_q5_K."""
    raw = raw.reshape(-1, 176)
    n = raw.shape[0]
    dm = raw[:, 0:4].view(torch.float16).to(torch.float32)  # [n,2]
    dall, dmin = dm[:, 0], dm[:, 1]
    scales = raw[:, 4:16]
    qh = raw[:, 16:48].to(torch.int32)
    qs = raw[:, 48:176].to(torch.int32)

    def _sm(j):
        if j < 4:
            d = scales[:, j] & 63
            m = scales[:, j + 4] & 63
        else:
            d = (scales[:, j + 4] & 0xF) | ((scales[:, j - 4] >> 6) << 4)
            m = (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)
        return d.to(torch.float32), m.to(torch.float32)

    y = torch.zeros(n, 256, dtype=torch.float32, device=raw.device)
    for il in range(4):
        s0, m0 = _sm(2 * il)
        s1, m1 = _sm(2 * il + 1)
        d0, M0 = dall * s0, dmin * m0
        d1, M1 = dall * s1, dmin * m1
        bit0 = 1 << (2 * il)
        bit1 = bit0 << 1
        ql = qs[:, 32 * il:32 * il + 32]
        ql0, ql1 = ql[:, 0::2], ql[:, 1::2]
        h0, h1 = qh[:, 0::2], qh[:, 1::2]
        v0 = (ql0 & 0xF) + ((h0 & bit0) != 0).to(torch.float32) * 16
        v1 = (ql1 & 0xF) + ((h1 & bit0) != 0).to(torch.float32) * 16
        even = torch.stack([v0, v1], dim=-1).reshape(n, 32)  # [v0[0],v1[0],v0[1],...]
        y[:, 64 * il:64 * il + 32] = even * d0.unsqueeze(1) - M0.unsqueeze(1)
        w0 = (ql0 >> 4) + ((h0 & bit1) != 0).to(torch.float32) * 16
        w1 = (ql1 >> 4) + ((h1 & bit1) != 0).to(torch.float32) * 16
        odd = torch.stack([w0, w1], dim=-1).reshape(n, 32)
        y[:, 64 * il + 32:64 * il + 64] = odd * d1.unsqueeze(1) - M1.unsqueeze(1)
    return y.reshape(-1).to(out_dtype)


_DEQUANT = {
    GGML_Q4_0: dequant_q4_0,
    GGML_Q4_K: dequant_q4_k,
    GGML_Q5_K: dequant_q5_k,
    GGML_Q6_K: dequant_q6_k,
    GGML_Q8_0: dequant_q8_0,
    GGML_IQ2_XS: dequant_iq2_xs,
    GGML_IQ3_XXS: dequant_iq3_xxs,
    GGML_IQ4_NL: dequant_iq4_nl_blocks,
}


def dequantize(raw: torch.Tensor, ggml_type: int, out_dtype: torch.dtype) -> torch.Tensor:
    """Dequantize ``raw`` (uint8) of any supported ggml type to flat ``out_dtype``."""
    if ggml_type == GGML_F32:
        return raw.view(torch.float32).to(out_dtype)
    if ggml_type == GGML_F16:
        return raw.view(torch.float16).to(out_dtype)
    if ggml_type == GGML_BF16:
        return raw.view(torch.bfloat16).to(out_dtype)
    fn = _DEQUANT.get(ggml_type)
    if fn is None:
        raise NotImplementedError(
            f"dequant for ggml type {GGML_NAME.get(ggml_type, ggml_type)} not implemented"
        )
    return fn(raw, out_dtype)


__all__ = [
    "GGML_F32",
    "GGML_F16",
    "GGML_BF16",
    "GGML_Q4_0",
    "GGML_Q8_0",
    "GGML_Q4_K",
    "GGML_Q5_K",
    "GGML_Q6_K",
    "GGML_IQ2_XS",
    "GGML_IQ3_XXS",
    "GGML_IQ4_NL",
    "PLE_IQ4_NL_BLOCK_VALUES",
    "PLE_IQ4_NL_BLOCK_BYTES",
    "PLE_IQ4_NL_BLOCKS_PER_ROW",
    "PLE_IQ4_NL_ROW_BYTES",
    "PLE_IQ4_NL_ROW_VALUES",
    "IQ4_NL_KVALUES",
    "GGML_NAME",
    "BLOCK_SHAPE",
    "row_bytes",
    "dequant_q4_0",
    "dequant_q4_k",
    "dequant_iq4_nl",
    "dequant_iq4_nl_blocks",
    "dequant_iq2_xs",
    "dequant_iq3_xxs",
    "dequant_q8_0",
    "dequant_q5_k",
    "dequant_q6_k",
    "quantize_q8_0",
    "dequantize",
]
