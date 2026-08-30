from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


TARGET = Path("/home/smk/models/Qwen3.8-Flash-Next-GGUF/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf")


def _tool():
    path = Path(__file__).parents[2] / "scripts" / "inspect-qwen38-gguf.py"
    spec = importlib.util.spec_from_file_location("inspect_qwen38_gguf", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.needs_weights
@pytest.mark.skipif(not TARGET.is_file(), reason="Qwen3.8 GGUF not installed")
def test_target_manifest_contract(tmp_path):
    tool = _tool()
    manifest = tool.inspect_gguf(TARGET)
    tool.verify_manifest(manifest)
    assert manifest["architecture"] == "qwen4exp"
    assert manifest["ple"]["shard"] == 2
    assert manifest["ple"]["shape"] == [320001536, 160]
    assert manifest["metadata"]["qwen4exp.ple.layers"] == [1]
    out = tmp_path / "manifest.json"
    out.write_text(json.dumps(manifest))
    assert json.loads(out.read_text())["types"] == [0, 8, 12, 13, 14, 17, 18, 20, 30]


def test_manifest_rejects_missing_type():
    tool = _tool()
    with pytest.raises(ValueError, match="missing expected types"):
        tool.verify_manifest({"architecture": "qwen4exp", "shards": [{"index": 1}, {"index": 2}, {"index": 3}], "types": []})
