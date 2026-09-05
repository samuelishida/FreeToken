from __future__ import annotations

import pytest
import torch


def test_declared_ggml_table_matches_gguf_py():
    import gguf

    from freetoken.models.gguf.dequant import BLOCK_SHAPE, GGML_NAME, GGML_TYPE_INFO

    expected = {int(key): tuple(value) for key, value in gguf.GGML_QUANT_SIZES.items()}
    assert BLOCK_SHAPE == expected
    assert {key: value[2] for key, value in GGML_TYPE_INFO.items()} == GGML_NAME


def test_q8_0_reference_dequant_matches_signed_int8_scale():
    from freetoken.models.gguf.dequant import GGML_Q8_0, dequantize

    raw = torch.zeros(34, dtype=torch.uint8)
    raw[:2] = torch.tensor([2.0], dtype=torch.float16).view(torch.uint8)
    raw[2:] = torch.arange(-16, 16, dtype=torch.int8).view(torch.uint8)
    expected = torch.arange(-16, 16, dtype=torch.float32) * 2.0
    torch.testing.assert_close(dequantize(raw, GGML_Q8_0, torch.float32), expected)


def test_row_bytes_rejects_unknown_and_partial_blocks():
    from freetoken.models.gguf.dequant import GGML_Q4_0, row_bytes

    with pytest.raises(ValueError, match="unknown GGML type 999"):
        row_bytes(32, 999)
    with pytest.raises(AssertionError, match="not a multiple of block"):
        row_bytes(31, GGML_Q4_0)


@pytest.mark.parametrize("ggml_type", [2, 10, 12, 14, 16, 20, 24, 30, 39, 40, 41])
def test_every_declared_type_has_stable_row_geometry(ggml_type):
    from freetoken.models.gguf.dequant import BLOCK_SHAPE, row_bytes

    block, size = BLOCK_SHAPE[ggml_type]
    assert row_bytes(block * 3, ggml_type) == size * 3
