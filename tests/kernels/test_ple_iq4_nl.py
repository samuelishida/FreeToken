from __future__ import annotations

import pytest
import torch

from freetoken.kernel import gguf
from freetoken.kernel.triton.ple_iq4_nl import (
    PLEDequantUnavailable,
    dequant_iq4_nl_rows,
)
from freetoken.models.gguf.dequant import dequant_iq4_nl


def test_helper_accepts_native_and_aligned_rows_and_matches_oracle() -> None:
    packed = torch.randint(0, 256, (5, 90), dtype=torch.uint8)
    for offset in range(0, 90, 18):
        packed[:, offset] = 0
        packed[:, offset + 1] = 0x3C
    aligned = torch.zeros((5, 96), dtype=torch.uint8)
    aligned[:, :90] = packed
    expected = dequant_iq4_nl(packed, out_dtype=torch.float32)
    torch.testing.assert_close(dequant_iq4_nl_rows(packed, out_dtype=torch.float32), expected)
    torch.testing.assert_close(dequant_iq4_nl_rows(aligned, out_dtype=torch.float32), expected)


def test_helper_empty_and_out_buffer() -> None:
    packed = torch.randint(0, 256, (2, 90), dtype=torch.uint8)
    out = torch.empty((2, 160), dtype=torch.float16)
    result = dequant_iq4_nl_rows(packed, out_dtype=torch.float16, out=out)
    assert result is out
    assert result.shape == (2, 160)
    assert dequant_iq4_nl_rows(torch.empty((0, 96), dtype=torch.uint8)).shape == (0, 160)


@pytest.mark.parametrize("shape", [(1, 89), (1, 91), (1, 95), (1, 97), (1, 3, 90)])
def test_helper_rejects_invalid_packed_geometry(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="90.*96"):
        dequant_iq4_nl_rows(torch.zeros(shape, dtype=torch.uint8))


@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.parametrize("n", [160, 32])
def test_generic_python_wrapper_rejects_unsafe_iq4_width(monkeypatch, m: int, n: int) -> None:
    monkeypatch.setattr(gguf, "_module", lambda: pytest.fail("extension must not load"))
    with pytest.raises(ValueError, match="n % 256"):
        gguf.ggml_dequantize(torch.empty((m, 90), dtype=torch.uint8), 20, m, n)


@pytest.mark.parametrize("n", [256, 512])
def test_generic_python_wrapper_preserves_supported_widths(monkeypatch, n: int) -> None:
    class FakeModule:
        def ggml_dequantize(self, weight, quant_type, m, width, dtype):
            return torch.empty((m, width), dtype=dtype or torch.float16)

    calls = []
    fake = FakeModule()

    def module():
        calls.append(True)
        return fake

    monkeypatch.setattr(gguf, "_module", module)
    result = gguf.ggml_dequantize(torch.empty((1, n // 32 * 18), dtype=torch.uint8), 20, 1, n)
    assert result.shape == (1, n)
    assert calls == [True]


def test_gpu_triton_parity_when_accelerator_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP unavailable")
    packed = torch.randint(0, 256, (3, 96), dtype=torch.uint8, device="cuda")
    for offset in range(0, 90, 18):
        packed[:, offset] = 0
        packed[:, offset + 1] = 0x3C
    result = dequant_iq4_nl_rows(packed, out_dtype=torch.float32)
    reference = dequant_iq4_nl(packed[:, :90], out_dtype=torch.float32)
    torch.cuda.synchronize()
    torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_gpu_missing_triton_is_actionable(monkeypatch) -> None:
    import freetoken.kernel.triton.ple_iq4_nl as ple

    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP unavailable")
    monkeypatch.setattr(ple, "triton", None)
    with pytest.raises(PLEDequantUnavailable, match="Triton"):
        dequant_iq4_nl_rows(torch.zeros((1, 90), dtype=torch.uint8, device="cuda"))
