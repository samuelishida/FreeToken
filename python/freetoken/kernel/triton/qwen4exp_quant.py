"""Native GGUF quant dispatch for Qwen4-Exp on CUDA-compatible ROCm.

The vendored ggml IQ MoE kernel can hang on gfx1100.  Decode therefore uses a
small Triton packed GEMV: one launch handles all selected expert rows, reads IQ
bytes in-place, and never materializes a dense expert slab.  Prefill continues
to use the Torch grouped reference path until a higher-throughput Triton GEMM
kernel is available.
"""
from __future__ import annotations

import functools

import torch

try:  # Triton is optional for CPU tooling and source distributions.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised only without Triton installed
    triton = None
    tl = None


if tl is not None:

    @triton.jit
    def _q8_0_gemv_kernel(
        x_ptr, w_ptr, out_ptr, sx0, sw0, so1, tokens, out_features, num_blocks,
        BLOCK_OUT: tl.constexpr,
    ):
        """Q8_0 W8A16 GEMV. One program owns a tile of output rows."""
        pid = tl.program_id(0)
        rows = pid * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        valid = rows < out_features
        acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
        j = tl.arange(0, 32)
        for block in tl.range(0, num_blocks):
            base = w_ptr + rows * sw0 + block * 34
            lo = tl.load(base, mask=valid, other=0).to(tl.uint16)
            hi = tl.load(base + 1, mask=valid, other=0).to(tl.uint16)
            d = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            raw = tl.load(base[:, None] + 2 + j[None, :], mask=valid[:, None], other=0).to(tl.int32)
            q = tl.where(raw > 127, raw - 256, raw).to(tl.float32)
            xv = tl.load(x_ptr + block * 32 + j, mask=block * 32 + j < num_blocks * 32, other=0.0)
            acc += d * tl.sum(q * xv[None, :], axis=1)
        tl.store(out_ptr + rows * so1, acc.to(tl.bfloat16), mask=valid)


    @triton.jit
    def _q4_k_gemv_kernel(
        x_ptr, w_ptr, out_ptr, sx0, sw0, so1, tokens, out_features, num_blocks,
        BLOCK_OUT: tl.constexpr,
    ):
        """Q4_K W4A16 GEMV, preserving ggml's scale/min packing."""
        pid = tl.program_id(0)
        rows = pid * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        valid = rows < out_features
        acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
        j = tl.arange(0, 32)
        for block in tl.range(0, num_blocks):
            base = w_ptr + rows * sw0 + block * 144
            d0 = tl.load(base + 0, mask=valid, other=0).to(tl.uint16)
            d1 = tl.load(base + 1, mask=valid, other=0).to(tl.uint16)
            dmin0 = tl.load(base + 2, mask=valid, other=0).to(tl.uint16)
            dmin1 = tl.load(base + 3, mask=valid, other=0).to(tl.uint16)
            dall = (d0 | (d1 << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            dmin = (dmin0 | (dmin1 << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            for il in tl.static_range(0, 4):
                # q4_K scale/min selector, expanded from ggml's _scale_min.
                if 2 * il < 4:
                    sd0 = tl.load(base + 4 + 2 * il, mask=valid, other=0).to(tl.int32) & 63
                    sm0 = tl.load(base + 4 + 2 * il + 4, mask=valid, other=0).to(tl.int32) & 63
                    sd1 = tl.load(base + 4 + 2 * il + 1, mask=valid, other=0).to(tl.int32) & 63
                    sm1 = tl.load(base + 4 + 2 * il + 1 + 4, mask=valid, other=0).to(tl.int32) & 63
                else:
                    j0 = 2 * il
                    j1 = j0 + 1
                    sd0 = (tl.load(base + 4 + j0 + 4, mask=valid, other=0).to(tl.int32) & 15) | ((tl.load(base + 4 + j0 - 4, mask=valid, other=0).to(tl.int32) >> 6) << 4)
                    sm0 = (tl.load(base + 4 + j0 + 4, mask=valid, other=0).to(tl.int32) >> 4) | ((tl.load(base + 4 + j0, mask=valid, other=0).to(tl.int32) >> 6) << 4)
                    sd1 = (tl.load(base + 4 + j1 + 4, mask=valid, other=0).to(tl.int32) & 15) | ((tl.load(base + 4 + j1 - 4, mask=valid, other=0).to(tl.int32) >> 6) << 4)
                    sm1 = (tl.load(base + 4 + j1 + 4, mask=valid, other=0).to(tl.int32) >> 4) | ((tl.load(base + 4 + j1, mask=valid, other=0).to(tl.int32) >> 6) << 4)
                q = tl.load(base[:, None] + 16 + il * 32 + j[None, :], mask=valid[:, None], other=0).to(tl.int32)
                lo = (q & 15).to(tl.float32) * (dall * sd0.to(tl.float32))[:, None] - (dmin * sm0.to(tl.float32))[:, None]
                hi = (q >> 4).to(tl.float32) * (dall * sd1.to(tl.float32))[:, None] - (dmin * sm1.to(tl.float32))[:, None]
                xlo = tl.load(x_ptr + block * 256 + il * 64 + j, mask=block * 256 + il * 64 + j < num_blocks * 256, other=0.0)
                xhi = tl.load(x_ptr + block * 256 + il * 64 + 32 + j, mask=block * 256 + il * 64 + 32 + j < num_blocks * 256, other=0.0)
                acc += tl.sum(lo * xlo[None, :], axis=1) + tl.sum(hi * xhi[None, :], axis=1)
        tl.store(out_ptr + rows * so1, acc.to(tl.bfloat16), mask=valid)


    @triton.jit
    def _q5_k_gemv_kernel(
        x_ptr, w_ptr, out_ptr, sx0, sw0, so1, tokens, out_features, num_blocks,
        BLOCK_OUT: tl.constexpr,
    ):
        """Q5_K W5A16 GEMV with ggml interleaved high-bit layout."""
        pid = tl.program_id(0)
        rows = pid * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        valid = rows < out_features
        acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
        j = tl.arange(0, 32)
        pair = j // 2
        odd = j & 1
        for block in tl.range(0, num_blocks):
            base = w_ptr + rows * sw0 + block * 176
            d0 = tl.load(base + 0, mask=valid, other=0).to(tl.uint16)
            d1 = tl.load(base + 1, mask=valid, other=0).to(tl.uint16)
            m0 = tl.load(base + 2, mask=valid, other=0).to(tl.uint16)
            m1 = tl.load(base + 3, mask=valid, other=0).to(tl.uint16)
            dall = (d0 | (d1 << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            dmin = (m0 | (m1 << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            for il in tl.static_range(0, 4):
                sj0 = 2 * il
                sj1 = sj0 + 1
                if sj0 < 4:
                    sd0 = tl.load(base + 4 + sj0, mask=valid, other=0).to(tl.int32) & 63
                    sm0 = tl.load(base + 4 + sj0 + 4, mask=valid, other=0).to(tl.int32) & 63
                    sd1 = tl.load(base + 4 + sj1, mask=valid, other=0).to(tl.int32) & 63
                    sm1 = tl.load(base + 4 + sj1 + 4, mask=valid, other=0).to(tl.int32) & 63
                else:
                    sd0 = (tl.load(base + 4 + sj0 + 4, mask=valid, other=0).to(tl.int32) & 15) | ((tl.load(base + 4 + sj0 - 4, mask=valid, other=0).to(tl.int32) >> 6) << 4)
                    sm0 = (tl.load(base + 4 + sj0 + 4, mask=valid, other=0).to(tl.int32) >> 4) | ((tl.load(base + 4 + sj0, mask=valid, other=0).to(tl.int32) >> 6) << 4)
                    sd1 = (tl.load(base + 4 + sj1 + 4, mask=valid, other=0).to(tl.int32) & 15) | ((tl.load(base + 4 + sj1 - 4, mask=valid, other=0).to(tl.int32) >> 6) << 4)
                    sm1 = (tl.load(base + 4 + sj1 + 4, mask=valid, other=0).to(tl.int32) >> 4) | ((tl.load(base + 4 + sj1, mask=valid, other=0).to(tl.int32) >> 6) << 4)
                # q5_K stores low/high output lanes interleaved in each byte
                # pair: byte 2*p is even lane, byte 2*p+1 odd lane. The high
                # two-bit payload follows the same pair ordering.
                byte_idx = 2 * pair + odd
                raw = tl.load(base[:, None] + 48 + il * 32 + byte_idx[None, :], mask=valid[:, None], other=0).to(tl.int32)
                high = tl.load(base[:, None] + 16 + byte_idx[None, :], mask=valid[:, None], other=0).to(tl.int32)
                bit0 = 1 << (2 * il)
                bit1 = 1 << (2 * il + 1)
                vlo = (raw & 15) + ((high & bit0) != 0).to(tl.int32) * 16
                vhi = (raw >> 4) + ((high & bit1) != 0).to(tl.int32) * 16
                xlo = tl.load(x_ptr + block * 256 + il * 64 + j, mask=block * 256 + il * 64 + j < num_blocks * 256, other=0.0)
                xhi = tl.load(x_ptr + block * 256 + il * 64 + 32 + j, mask=block * 256 + il * 64 + 32 + j < num_blocks * 256, other=0.0)
                acc += tl.sum((vlo.to(tl.float32) * (dall * sd0.to(tl.float32))[:, None] - (dmin * sm0.to(tl.float32))[:, None]) * xlo[None, :], axis=1)
                acc += tl.sum((vhi.to(tl.float32) * (dall * sd1.to(tl.float32))[:, None] - (dmin * sm1.to(tl.float32))[:, None]) * xhi[None, :], axis=1)
        tl.store(out_ptr + rows * so1, acc.to(tl.bfloat16), mask=valid)


    @triton.jit
    def _q6_k_gemv_kernel(
        x_ptr, w_ptr, out_ptr, sx0, sw0, so1, tokens, out_features, num_blocks,
        BLOCK_OUT: tl.constexpr,
    ):
        """Q6_K W6A16 GEMV with two 128-value halves."""
        pid = tl.program_id(0)
        rows = pid * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        valid = rows < out_features
        acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
        j = tl.arange(0, 32)
        for block in tl.range(0, num_blocks):
            base = w_ptr + rows * sw0 + block * 210
            d0 = tl.load(base + 208, mask=valid, other=0).to(tl.uint16)
            d1 = tl.load(base + 209, mask=valid, other=0).to(tl.uint16)
            d = (d0 | (d1 << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            for half in tl.static_range(0, 2):
                for sub in tl.static_range(0, 4):
                    ql_off = half * 64 + (sub & 1) * 32
                    qh_off = half * 32
                    ql = tl.load(base[:, None] + ql_off + j[None, :], mask=valid[:, None], other=0).to(tl.int32)
                    qh = tl.load(base[:, None] + 128 + qh_off + j[None, :], mask=valid[:, None], other=0).to(tl.int32)
                    if sub == 0:
                        q = (ql & 15) | ((qh & 3) << 4)
                        sc_off = half * 8 + (j // 16)
                    elif sub == 1:
                        q = ((ql & 15) | (((qh >> 2) & 3) << 4))
                        sc_off = half * 8 + 2 + (j // 16)
                    elif sub == 2:
                        q = (ql >> 4) | (((qh >> 4) & 3) << 4)
                        sc_off = half * 8 + 4 + (j // 16)
                    else:
                        q = (ql >> 4) | (((qh >> 6) & 3) << 4)
                        sc_off = half * 8 + 6 + (j // 16)
                    scale = tl.load(base[:, None] + 192 + sc_off[None, :], mask=valid[:, None], other=0).to(tl.int8).to(tl.float32)
                    xv = tl.load(x_ptr + block * 256 + half * 128 + sub * 32 + j, mask=block * 256 + half * 128 + sub * 32 + j < num_blocks * 256, other=0.0)
                    acc += tl.sum((q.to(tl.float32) - 32.0) * (d[:, None] * scale) * xv[None, :], axis=1)
        tl.store(out_ptr + rows * so1, acc.to(tl.bfloat16), mask=valid)

    @triton.jit
    def _iq2_xs_decode_kernel(
        x_ptr, w_ptr, ids_ptr, out_ptr, grid_ptr, signs_ptr,
        sx0, sx1, sw0, sw1, sw2, si0, si1, so0, so1,
        tokens, top_k, out_features, num_blocks,
        BLOCK_OUT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        route = pid // tl.cdiv(out_features, BLOCK_OUT)
        tile = pid % tl.cdiv(out_features, BLOCK_OUT)
        token = route // top_k
        choice = route - token * top_k
        rows = tile * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        valid_rows = rows < out_features
        expert = tl.load(ids_ptr + token * si0 + choice * si1).to(tl.int64)
        x_base = x_ptr + token * sx0
        w_base = w_ptr + expert * sw0 + rows * sw1
        acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
        eight = tl.arange(0, 8)
        for block in tl.range(0, num_blocks):
                block_base = w_base + block * sw2
                d_lo = tl.load(block_base + 0, mask=valid_rows, other=0).to(tl.uint16)
                d_hi = tl.load(block_base + 1, mask=valid_rows, other=0).to(tl.uint16)
                d_bits = d_lo | (d_hi << 8)
                d = d_bits.to(tl.float16, bitcast=True).to(tl.float32)
                for group in tl.static_range(0, 8):
                    scale_byte = tl.load(block_base + 66 + group, mask=valid_rows, other=0).to(tl.uint16)
                    scale_lo = 0.5 + (scale_byte & 0xF).to(tl.float32)
                    scale_hi = 0.5 + (scale_byte >> 4).to(tl.float32)
                    for part in tl.static_range(0, 4):
                        qbase = block_base + 2 + (group * 4 + part) * 2
                        q_lo = tl.load(qbase + 0, mask=valid_rows, other=0).to(tl.uint16)
                        q_hi = tl.load(qbase + 1, mask=valid_rows, other=0).to(tl.uint16)
                        q = q_lo | (q_hi << 8)
                        grid_id = q & 511
                        sign_id = q >> 9
                        grid = tl.load(grid_ptr + grid_id[:, None] * 8 + eight[None, :])
                        signs = tl.load(signs_ptr + sign_id[:, None] * 8 + eight[None, :])
                        pos = block * 256 + group * 32 + part * 8
                        values = tl.load(x_base + pos + eight, mask=pos + eight < num_blocks * 256, other=0.0)
                        dot = tl.sum(values[None, :] * grid.to(tl.float32) * signs, axis=1)
                        acc += d * 0.25 * (scale_lo if part < 2 else scale_hi) * dot
        tl.store(out_ptr + route * so0 + rows * so1, acc.to(tl.bfloat16), mask=valid_rows)


    @triton.jit
    def _iq3_xxs_decode_kernel(
        x_ptr, w_ptr, ids_ptr, out_ptr, grid_ptr, signs_ptr,
        sx0, sx1, sw0, sw1, sw2, si0, si1, so0, so1,
        tokens, top_k, out_features, num_blocks,
        BLOCK_OUT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        route = pid // tl.cdiv(out_features, BLOCK_OUT)
        tile = pid % tl.cdiv(out_features, BLOCK_OUT)
        token = route // top_k
        choice = route - token * top_k
        rows = tile * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        valid_rows = rows < out_features
        expert = tl.load(ids_ptr + token * si0 + choice * si1).to(tl.int64)
        x_base = x_ptr + token * sx0
        w_base = w_ptr + expert * sw0 + rows * sw1
        acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
        four = tl.arange(0, 4)
        for block in tl.range(0, num_blocks):
                block_base = w_base + block * sw2
                d_lo = tl.load(block_base + 0, mask=valid_rows, other=0).to(tl.uint16)
                d_hi = tl.load(block_base + 1, mask=valid_rows, other=0).to(tl.uint16)
                d = (d_lo | (d_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
                for group in tl.static_range(0, 8):
                    qbase = block_base + 2 + group * 8
                    # Promote q3 indices before multiplying by codebook stride;
                    # uint8 arithmetic would wrap indices above 63.
                    q0 = tl.load(qbase + 0, mask=valid_rows, other=0).to(tl.int32)
                    q1 = tl.load(qbase + 1, mask=valid_rows, other=0).to(tl.int32)
                    q2 = tl.load(qbase + 2, mask=valid_rows, other=0).to(tl.int32)
                    q3 = tl.load(qbase + 3, mask=valid_rows, other=0).to(tl.int32)
                    q4 = tl.load(qbase + 4, mask=valid_rows, other=0).to(tl.int32)
                    q5 = tl.load(qbase + 5, mask=valid_rows, other=0).to(tl.int32)
                    q6 = tl.load(qbase + 6, mask=valid_rows, other=0).to(tl.int32)
                    q7 = tl.load(qbase + 7, mask=valid_rows, other=0).to(tl.int32)
                    # Keep auxiliary payload unsigned: high scale nibble lives
                    # above signed int32 range for many valid IQ3 blocks.
                    gas0 = tl.load(block_base + 66 + group * 4 + 0, mask=valid_rows, other=0).to(tl.uint32)
                    gas1 = tl.load(block_base + 66 + group * 4 + 1, mask=valid_rows, other=0).to(tl.uint32)
                    gas2 = tl.load(block_base + 66 + group * 4 + 2, mask=valid_rows, other=0).to(tl.uint32)
                    gas3 = tl.load(block_base + 66 + group * 4 + 3, mask=valid_rows, other=0).to(tl.uint32)
                    aux0 = gas0 | (gas1 << 8)
                    aux1 = gas2 | (gas3 << 8)
                    aux = aux0 | (aux1 << 16)
                    sum_group = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
                    for pair in tl.static_range(0, 4):
                        q_a = tl.where(pair == 0, q0, tl.where(pair == 1, q2, tl.where(pair == 2, q4, q6)))
                        q_b = tl.where(pair == 0, q1, tl.where(pair == 1, q3, tl.where(pair == 2, q5, q7)))
                        grid_a = tl.load(grid_ptr + q_a[:, None] * 4 + four[None, :])
                        grid_b = tl.load(grid_ptr + q_b[:, None] * 4 + four[None, :])
                        sign_id = ((aux >> (7 * pair)) & 127).to(tl.int32)
                        signs_a = tl.load(
                            signs_ptr + sign_id[:, None] * 8 + four[None, :]
                        )
                        signs_b = tl.load(
                            signs_ptr + sign_id[:, None] * 8 + 4 + four[None, :]
                        )
                        pos = block * 256 + group * 32 + pair * 8
                        xa = tl.load(x_base + pos + four, mask=pos + four < num_blocks * 256, other=0.0)
                        xb = tl.load(x_base + pos + 4 + four, mask=pos + 4 + four < num_blocks * 256, other=0.0)
                        sum_group += tl.sum(xa[None, :] * grid_a.to(tl.float32) * signs_a, axis=1)
                        sum_group += tl.sum(xb[None, :] * grid_b.to(tl.float32) * signs_b, axis=1)
                    acc += d * 0.5 * (0.5 + (aux >> 28).to(tl.float32)) * sum_group
        tl.store(out_ptr + route * so0 + rows * so1, acc.to(tl.bfloat16), mask=valid_rows)


    @triton.jit
    def _iq4_nl_decode_kernel(
        x_ptr, w_ptr, ids_ptr, out_ptr, values_ptr,
        sx0, sx1, sw0, sw1, sw2, si0, si1, so0, so1,
        tokens, top_k, out_features, num_blocks,
        BLOCK_OUT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        route = pid // tl.cdiv(out_features, BLOCK_OUT)
        tile = pid % tl.cdiv(out_features, BLOCK_OUT)
        token = route // top_k
        choice = route - token * top_k
        rows = tile * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        valid_rows = rows < out_features
        expert = tl.load(ids_ptr + token * si0 + choice * si1).to(tl.int64)
        x_base = x_ptr + token * sx0
        w_base = w_ptr + expert * sw0 + rows * sw1
        acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
        sixteen = tl.arange(0, 16)
        for block in tl.range(0, num_blocks):
                block_base = w_base + block * sw2
                raw = tl.load(block_base[:, None] + 2 + sixteen[None, :], mask=valid_rows[:, None], other=0).to(tl.int32)
                lo = tl.load(values_ptr + (raw & 0xF))
                hi = tl.load(values_ptr + (raw >> 4))
                xlo = tl.load(x_base + block * 32 + sixteen, mask=block * 32 + sixteen < num_blocks * 32, other=0.0)
                xhi = tl.load(x_base + block * 32 + 16 + sixteen, mask=block * 32 + 16 + sixteen < num_blocks * 32, other=0.0)
                dot = tl.sum(xlo[None, :] * lo.to(tl.float32), axis=1)
                dot += tl.sum(xhi[None, :] * hi.to(tl.float32), axis=1)
                d_lo = tl.load(block_base, mask=valid_rows, other=0).to(tl.uint16)
                d_hi = tl.load(block_base + 1, mask=valid_rows, other=0).to(tl.uint16)
                d = (d_lo | (d_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
                acc += d * dot
        tl.store(out_ptr + route * so0 + rows * so1, acc.to(tl.bfloat16), mask=valid_rows)


@functools.lru_cache(maxsize=8)
def _tables(device_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from freetoken.models.gguf.dequant import _iq_tables

    iq2, iq3, signs2 = _iq_tables()
    # ggml's ksigns64 stores eight 0x00/0xff bytes per entry; zero is +1.
    import re
    from pathlib import Path

    header = Path(__file__).resolve().parents[1] / "csrc" / "gguf" / "ggml-common.h"
    source = header.read_text(encoding="utf-8")
    match = re.search(r"ksigns64\[128\]\s*=\s*\{(.*?)\};", source, re.DOTALL)
    if match is None:
        raise RuntimeError("ksigns64 table missing")
    words = [int(token, 16) for token in re.findall(r"0x[0-9a-fA-F]+", match.group(1))]
    signs3 = torch.tensor(
        [[1.0 if byte == 0 else -1.0 for byte in int(word).to_bytes(8, "little")] for word in words],
        dtype=torch.float32,
    )
    dev = torch.device("cuda", device_index)
    kvalues = torch.tensor(
        (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113),
        dtype=torch.float32,
    )
    return tuple(t.to(dev) for t in (iq2, iq3, signs2, signs3, kvalues))


def qwen4_gguf_grouped_available(device: torch.device | None = None) -> bool:
    """Return whether dedicated grouped IQ kernels are safe for ``device``.

    The generic vendored IQ kernels are unsafe on gfx1100 and this checkout does
    not ship a dedicated grouped IQ implementation yet.  Keep capability
    probing explicit so callers select the bounded batched-Torch fallback rather
    than accidentally launching the unsafe generic kernel.  A future AOT/Triton
    implementation can replace this body without changing MoE dispatch.
    """
    if torch.version.hip is None:
        return False
    if device is None:
        device = torch.device("cuda")
    if device.type != "cuda":
        return False
    try:
        name = torch.cuda.get_device_name(device).lower()
    except Exception:  # noqa: BLE001 -- capability probe must be non-fatal
        return False
    # The dedicated Triton GEMV is explicitly designed for RDNA3.  Keep an env
    # escape hatch for deployments that need to force the Torch parity path.
    import os
    if os.environ.get("FREETOKEN_DISABLE_QWEN4_TRITON", "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    return triton is not None and tl is not None


def fused_qwen4_gguf_decode(
    hidden_states: torch.Tensor,
    packed: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    ggml_type: int,
    out_features: int,
) -> torch.Tensor:
    """Decode one packed IQ bank for small decode batches via Triton GEMV."""
    if triton is None or tl is None:
        raise RuntimeError("Triton is unavailable")
    if hidden_states.ndim != 2 or packed.ndim != 3 or topk_ids.ndim != 2:
        raise ValueError("Qwen4 Triton decode expects [tokens,K], [experts,O,bytes], [tokens,top_k]")
    tokens, width = map(int, hidden_states.shape)
    top_k = int(topk_ids.shape[1])
    rows = int(packed.shape[1])
    row_bytes = int(packed.shape[2])
    if rows != int(out_features) or top_k <= 0 or tokens <= 0:
        raise ValueError("invalid Qwen4 Triton decode geometry")
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError("Qwen4 Triton decode currently requires bfloat16 activations")
    block, type_size = ({17: (256, 74), 18: (256, 98), 20: (32, 18)}[int(ggml_type)])
    if width % block or row_bytes != width // block * type_size:
        raise ValueError("packed IQ decode row geometry mismatch")
    iq2, iq3, signs2, signs3, kvalues = _tables(torch.cuda.current_device())
    out = torch.empty((tokens * top_k, rows), device=hidden_states.device, dtype=hidden_states.dtype)
    if int(ggml_type) == 17:
        kernel = _iq2_xs_decode_kernel
        table_args = (iq2, signs2)
    elif int(ggml_type) == 18:
        kernel = _iq3_xxs_decode_kernel
        # IQ3_XXS uses same 7-bit sign lookup as IQ2_XS (ksigns_iq2xs).
        table_args = (iq3, signs2)
    else:
        kernel = _iq4_nl_decode_kernel
        table_args = (kvalues,)
    # Larger output tiles reduce launch count dramatically on RDNA3 (Qwen
    # decode emits 10 routes, with 640/2560 output rows). Keep conservative
    # default for older Triton/ROCm builds; production script selects 64 after
    # parity and warm-kernel checks. Invalid overrides fall back safely.
    import os
    # IQ2/IQ3 gate/up are register-heavy at 64 lanes; IQ4 down benefits from
    # 64-lane tiles. Select per format unless operator overrides.
    default_block = "64" if int(ggml_type) == 20 else "32"
    try:
        block_out = int(os.environ.get("FREETOKEN_QWEN4_TRITON_BLOCK_OUT", default_block))
    except ValueError:
        block_out = 16
    if block_out not in (16, 32, 64):
        block_out = 16
    grid = (tokens * top_k * triton.cdiv(rows, block_out),)
    kernel[grid](
        hidden_states, packed, topk_ids, out,
        *table_args,
        hidden_states.stride(0), hidden_states.stride(1),
        packed.stride(0), packed.stride(1), row_bytes // (width // block),
        topk_ids.stride(0), topk_ids.stride(1), out.stride(0), out.stride(1),
        tokens, top_k, rows, width // block,
        BLOCK_OUT=block_out,
        num_warps=4,
    )
    return out


def fused_qwen4_gguf_grouped(
    hidden_states: torch.Tensor,
    gate_q: torch.Tensor,
    up_q: torch.Tensor,
    down_q: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    layer_id: int,
    *,
    scratch_mib: int = 128,
) -> torch.Tensor:
    """Route-fused IQ GEMV for Qwen4 GGUF prefill.

    Each route is an independent packed-row GEMV, so this avoids materializing
    dense weights for every unique expert. It uses same kernels as decode but
    accepts arbitrary token batches; output stays route-sized until final
    weighted reduction. ``scratch_mib`` is retained for dispatch compatibility:
    GEMV workspace is bounded by output activations and does not depend on it.
    """
    if hidden_states.ndim != 2 or topk_ids.ndim != 2 or topk_weights.ndim != 2:
        raise ValueError("Qwen4 Triton grouped GEMV expects 2D activations/routes/weights")
    if hidden_states.shape[0] != topk_ids.shape[0] or topk_ids.shape != topk_weights.shape:
        raise ValueError("Qwen4 Triton grouped GEMV route geometry mismatch")
    if hidden_states.shape[0] <= 0 or topk_ids.shape[1] <= 0:
        return hidden_states.new_zeros((hidden_states.shape[0], hidden_states.shape[1]))

    from freetoken.models.gguf.dequant import GGML_IQ2_XS, GGML_IQ3_XXS, GGML_IQ4_NL
    gate_type = GGML_IQ3_XXS if int(layer_id) == 2 else GGML_IQ2_XS
    gate_out = fused_qwen4_gguf_decode(
        hidden_states, gate_q, topk_ids, ggml_type=gate_type,
        out_features=int(gate_q.shape[1]),
    )
    up_out = fused_qwen4_gguf_decode(
        hidden_states, up_q, topk_ids, ggml_type=gate_type,
        out_features=int(up_q.shape[1]),
    )
    if activation in ("silu", "swish"):
        inter = torch.nn.functional.silu(gate_out) * up_out
    elif activation == "gelu_tanh":
        inter = torch.nn.functional.gelu(gate_out, approximate="tanh") * up_out
    elif activation == "gelu":
        inter = torch.nn.functional.gelu(gate_out) * up_out
    else:
        raise ValueError(f"unsupported Qwen4 Triton activation {activation!r}")
    down_out = fused_qwen4_gguf_decode(
        inter, down_q, topk_ids.reshape(-1, 1),
        ggml_type=GGML_IQ4_NL, out_features=int(down_q.shape[1]),
    )
    route_weights = topk_weights.reshape(-1, 1).to(down_out.dtype)
    return (down_out * route_weights).view(
        hidden_states.shape[0], topk_ids.shape[1], -1
    ).sum(dim=1)


def fused_gguf_decode_standard(
    hidden_states: torch.Tensor,
    packed: torch.Tensor,
    *,
    ggml_type: int,
    out_features: int,
) -> torch.Tensor:
    """Decode standard K/Q8 GGUF rows with Triton GEMV.

    Decode-only path for small batches. Prefill keeps existing MMQ/Torch
    dispatch where larger GEMM amortizes dequant work.
    """
    if triton is None or tl is None:
        raise RuntimeError("Triton is unavailable")
    if hidden_states.ndim != 2 or packed.ndim != 2:
        raise ValueError("standard GGUF decode expects [tokens,width] and [rows,row_bytes]")
    tokens, width = map(int, hidden_states.shape)
    rows, row_bytes = map(int, packed.shape)
    if tokens != 1 or rows != int(out_features):
        raise ValueError("standard GGUF decode geometry is outside small-batch GEMV path")
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError("standard GGUF Triton decode requires bfloat16 activations")
    shapes = {8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210)}
    try:
        block, type_size = shapes[int(ggml_type)]
    except KeyError as exc:
        raise ValueError(f"unsupported standard GGUF Triton type {ggml_type}") from exc
    if width % block or row_bytes != width // block * type_size:
        raise ValueError("standard GGUF decode row geometry mismatch")
    out = torch.empty((tokens, rows), device=hidden_states.device, dtype=hidden_states.dtype)
    kernel = {8: _q8_0_gemv_kernel, 12: _q4_k_gemv_kernel,
              13: _q5_k_gemv_kernel, 14: _q6_k_gemv_kernel}[int(ggml_type)]
    import os
    # RX7900 XTX measurements favor 16 lanes for Q8/Q6/Q4-K and 64 for Q5-K.
    # Keep operator override, but avoid one generic tile forcing slower register
    # pressure across unlike GGUF row formats.
    default_standard_block = "64" if int(ggml_type) == 13 else "16"
    try:
        standard_block = int(
            os.environ.get("FREETOKEN_GGUF_TRITON_BLOCK_OUT", default_standard_block)
        )
    except ValueError:
        standard_block = int(default_standard_block)
    if standard_block not in (16, 32, 64):
        standard_block = int(default_standard_block)
    kernel[(triton.cdiv(rows, standard_block),)](
        hidden_states, packed, out,
        hidden_states.stride(0), packed.stride(0), out.stride(1),
        tokens, rows, width // block,
        BLOCK_OUT=standard_block, num_warps=4,
    )
    return out


SUPPORTED_TYPES = frozenset({0, 8, 12, 13, 14, 17, 18, 20, 30})


def _validate_generic_iq4_width(ggml_type: int, width: int) -> None:
    if ggml_type == 20 and (width <= 0 or width % 256):
        raise ValueError(
            f"generic IQ4_NL projection requires width > 0 and divisible by 256, got {width}; "
            "PLE rows must use the dedicated five-block helper"
        )


def quant_linear(x: torch.Tensor, packed: torch.Tensor, ggml_type: int, out_features: int) -> torch.Tensor:
    if ggml_type not in SUPPORTED_TYPES: raise ValueError(f"unsupported Qwen4-Exp GGUF type {ggml_type}")
    if ggml_type in (0, 30):
        dtype = torch.float32 if ggml_type == 0 else torch.bfloat16
        return x @ packed.view(dtype).reshape(out_features, -1).to(x.dtype).T
    if not x.is_cuda: raise RuntimeError("packed GGUF quant dispatch requires HIP/CUDA tensor")
    if ggml_type in (0, 30):
        dense = packed.view(torch.float32 if ggml_type == 0 else torch.bfloat16).reshape(
            packed.shape[0], out_features, -1).to(x.dtype)
        return torch.einsum("ki,koi->ko", x, dense)
    if ggml_type in SUPPORTED_TYPES:
        # IQ formats are decoded by the existing GGUF HIP block decoder.  Keep
        # expansion local to this projection; never expand an expert bank.
        from freetoken.kernel.gguf import ggml_dequantize
        block, type_size = {8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210),
                            17: (256, 74), 18: (256, 98), 20: (32, 18)}[ggml_type]
        in_features = packed.shape[1] * block // type_size
        _validate_generic_iq4_width(ggml_type, in_features)
        dense = ggml_dequantize(packed, int(ggml_type), int(out_features), int(in_features), x.dtype)
        return x @ dense.T
    from freetoken.kernel.gguf import ggml_mul_mat_a8, ggml_mul_mat_vec_a8
    return (ggml_mul_mat_vec_a8 if x.shape[0] <= 6 else ggml_mul_mat_a8)(packed, x, ggml_type, out_features)


def expert_quant_linear(x: torch.Tensor, packed: torch.Tensor, topk_ids: torch.Tensor,
                        ggml_type: int, out_features: int, top_k: int = 10) -> torch.Tensor:
    if ggml_type not in SUPPORTED_TYPES: raise ValueError(f"unsupported Qwen4-Exp GGUF type {ggml_type}")
    if not x.is_cuda: raise RuntimeError("packed GGUF MoE dispatch requires HIP/CUDA tensor")
    if ggml_type in SUPPORTED_TYPES:
        # Keep selected-slab semantics explicit. The vendored fused MoE entry
        # point requires full-bank stride/ID ownership; using it with a compact
        # remapped slab corrupts memory on ROCm. IQ decode therefore expands only
        # routed rows and contracts them with one batched GEMM.
        from freetoken.kernel.gguf import ggml_dequantize
        block, type_size = {8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210),
                            17: (256, 74), 18: (256, 98), 20: (32, 18)}[ggml_type]
        in_features = packed.shape[-1] * block // type_size
        _validate_generic_iq4_width(ggml_type, in_features)
        dense = ggml_dequantize(packed.reshape(-1, packed.shape[-1]), int(ggml_type),
                                int(packed.shape[0] * out_features), int(in_features), x.dtype)
        dense = dense.reshape(packed.shape[0], out_features, in_features)
        return torch.einsum("ki,koi->ko", x, dense)
    raise ValueError(f"unsupported selected-expert GGUF type {ggml_type}")


__all__ = [
    "SUPPORTED_TYPES",
    "quant_linear",
    "expert_quant_linear",
    "qwen4_gguf_grouped_available",
    "fused_gguf_decode_standard",
    "fused_qwen4_gguf_decode",
    "fused_qwen4_gguf_grouped",
]
