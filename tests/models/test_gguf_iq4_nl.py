from __future__ import annotations

import struct

import pytest
import torch

from freetoken.models.gguf.dequant import (
    IQ4_NL_KVALUES,
    PLE_IQ4_NL_BLOCK_BYTES,
    PLE_IQ4_NL_BLOCKS_PER_ROW,
    PLE_IQ4_NL_ROW_BYTES,
    PLE_IQ4_NL_ROW_VALUES,
    dequant_iq4_nl,
)


def _pack_row(scales: list[float], nibbles: list[list[int]]) -> torch.Tensor:
    assert len(scales) == PLE_IQ4_NL_BLOCKS_PER_ROW
    assert len(nibbles) == PLE_IQ4_NL_BLOCKS_PER_ROW
    raw = bytearray()
    for scale, block in zip(scales, nibbles):
        assert len(block) == PLE_IQ4_NL_BLOCK_BYTES - 2
        assert all(0 <= value < 256 for value in block)
        raw.extend(struct.pack("<e", scale))
        raw.extend(block)
    assert len(raw) == PLE_IQ4_NL_ROW_BYTES
    return torch.tensor(list(raw), dtype=torch.uint8)


def _scalar_decode(raw: torch.Tensor) -> list[float]:
    values: list[float] = []
    raw = raw.reshape(PLE_IQ4_NL_BLOCKS_PER_ROW, PLE_IQ4_NL_BLOCK_BYTES)
    for block in raw.tolist():
        scale = struct.unpack("<e", bytes(block[:2]))[0]
        low = [IQ4_NL_KVALUES[value & 0xF] * scale for value in block[2:]]
        high = [IQ4_NL_KVALUES[value >> 4] * scale for value in block[2:]]
        values.extend(low + high)
    return values


def test_five_block_oracle_matches_scalar_storage_order() -> None:
    row = _pack_row(
        [1.0, 0.5, -2.0, 3.25, 0.125],
        [
            list(range(16)),
            [0xF0 | i for i in range(16)],
            [((15 - i) << 4) | i for i in range(16)],
            [0x55] * 16,
            [0xA3] * 16,
        ],
    )

    actual = dequant_iq4_nl(row)
    expected = torch.tensor(_scalar_decode(row), dtype=torch.float32).reshape(1, -1)
    assert actual.shape == (1, PLE_IQ4_NL_ROW_VALUES)
    assert torch.equal(actual, expected)
    # First block makes nibble and byte order visible without relying on scale.
    assert actual[0, :16].tolist() == list(IQ4_NL_KVALUES)
    assert actual[0, 16:32].tolist() == [IQ4_NL_KVALUES[0]] * 16


def test_oracle_accepts_flat_and_noncontiguous_rows() -> None:
    rows = torch.stack(
        [
            _pack_row([1.0] * 5, [[i] * 16 for i in range(5)]),
            _pack_row([0.5] * 5, [[15 - i] * 16 for i in range(5)]),
        ]
    )
    expected = dequant_iq4_nl(rows)

    flat = rows.flatten()
    assert flat.is_contiguous()
    torch.testing.assert_close(dequant_iq4_nl(flat, rows=2), expected)

    # Every other byte in a wider tensor produces a valid, non-contiguous [N,90]
    # view.  Oracle must not require callers to make this copy first.
    wider = torch.zeros((rows.shape[0], rows.shape[1] * 2), dtype=torch.uint8)
    wider[:, ::2] = rows
    noncontiguous = wider[:, ::2]
    assert not noncontiguous.is_contiguous()
    torch.testing.assert_close(dequant_iq4_nl(noncontiguous), expected)


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_oracle_output_dtype_and_empty_geometry(out_dtype: torch.dtype) -> None:
    empty = torch.empty((0, PLE_IQ4_NL_ROW_BYTES), dtype=torch.uint8)
    output = dequant_iq4_nl(empty, out_dtype=out_dtype)
    assert output.shape == (0, PLE_IQ4_NL_ROW_VALUES)
    assert output.dtype == out_dtype

    row = _pack_row([1.0] * 5, [[0] * 16] * 5)
    output = dequant_iq4_nl(row, out_dtype=out_dtype)
    assert output.shape == (1, PLE_IQ4_NL_ROW_VALUES)
    assert output.dtype == out_dtype


@pytest.mark.parametrize(
    "packed",
    [
        torch.empty(89, dtype=torch.uint8),
        torch.empty(91, dtype=torch.uint8),
        torch.empty((1, 89), dtype=torch.uint8),
        torch.empty((1, 91), dtype=torch.uint8),
        torch.empty((1, 2, 90), dtype=torch.uint8),
    ],
)
def test_oracle_rejects_malformed_ple_geometry(packed: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="IQ4_NL PLE"):
        dequant_iq4_nl(packed)


def test_oracle_rejects_dtype_and_row_count_mismatch() -> None:
    with pytest.raises(ValueError, match="dtype torch.uint8"):
        dequant_iq4_nl(torch.zeros(PLE_IQ4_NL_ROW_BYTES, dtype=torch.float32))
    with pytest.raises(ValueError, match="row count"):
        dequant_iq4_nl(torch.zeros((2, PLE_IQ4_NL_ROW_BYTES), dtype=torch.uint8), rows=1)
    with pytest.raises(ValueError, match="output dtype"):
        dequant_iq4_nl(torch.zeros(PLE_IQ4_NL_ROW_BYTES, dtype=torch.uint8), out_dtype=torch.int32)
