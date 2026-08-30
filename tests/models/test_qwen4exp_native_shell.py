from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.models.qwen4_exp.state import Qwen4ExpState, assert_batch_one


def test_state_rejects_overflow_and_double_begin():
    state = Qwen4ExpState(); state.reserve(8); state.begin(4); state.advance(3)
    with pytest.raises(RuntimeError, match="already owns"):
        state.begin(5)
    with pytest.raises(ValueError, match="overflow"):
        state.advance(6)
    state.reset(); assert state.position == 0 and not state.live


def test_state_rejects_unreserved_and_bad_batch():
    with pytest.raises(RuntimeError, match="not reserved"):
        Qwen4ExpState().begin()
    assert_batch_one(torch.zeros((1, 2), dtype=torch.int32))
    with pytest.raises(ValueError, match="requires input_ids"):
        assert_batch_one(torch.zeros((2, 2), dtype=torch.int32))


def test_engine_adapter_requires_hip(monkeypatch):
    from freetoken.models.qwen4_exp.engine import Qwen4ExpRocmEngine
    monkeypatch.setattr(torch.version, "hip", None)
    cfg = SimpleNamespace(max_seq_len=128)
    with pytest.raises(RuntimeError, match="ROCm/HIP"):
        Qwen4ExpRocmEngine(cfg)
