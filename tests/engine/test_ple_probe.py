from types import SimpleNamespace

import pytest
import torch

from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM


def test_validate_ple_runtime_reports_geometry_without_sync(monkeypatch):
    model = object.__new__(Qwen4ExpForCausalLM)
    model._config = SimpleNamespace(model_path="model.gguf")
    model._ple_table = SimpleNamespace(
        num_rows=45, _device=torch.device("cpu"), _report={"backend": "fake", "mode": "paged"},
        lookup=lambda ids: torch.zeros((ids.shape[0], 160), dtype=torch.bfloat16),
    )
    result = model.validate_ple_runtime()
    assert result["state"] == "ok"
    assert result["row_bytes"] == 90 and result["row_values"] == 160


def test_validate_ple_runtime_fails_actionably():
    model = object.__new__(Qwen4ExpForCausalLM)
    model._config = SimpleNamespace(model_path="model.gguf")
    model._ple_table = SimpleNamespace(
        num_rows=45, _device=torch.device("cpu"), _report={"backend": "fake", "mode": "paged"},
        lookup=lambda ids: torch.zeros((ids.shape[0], 159)),
    )
    with pytest.raises(RuntimeError, match="Qwen4Exp PLE probe failed"):
        model.validate_ple_runtime()
