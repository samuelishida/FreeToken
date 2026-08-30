import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ollama-tg" / "tg-fork"))
from tinygrad.llm.qwen4exp_spec import Qwen4ExpTextSpec

def good():
  return {"text_config": {"model_type":"qwen4_exp_text", "num_hidden_layers":48, "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"]*12, "hidden_size":2560, "hc_count":4, "hc_lowrank":320, "num_attention_heads":24, "num_key_value_heads":2, "head_dim":256, "partial_rotary_dim":64, "indexer_n_heads":4, "indexer_kv_heads":1, "indexer_head_dim":128, "indexer_compress_ratio":4, "indexer_budget":2048, "num_experts":512, "num_experts_per_tok":10, "ple":{"layer_ids":[3,7],"ngram_size":3,"embed_dim":160,"rows":320001536,"row_width":160,"parts":128,"dtype":"F8_E4M3FN"}}}

def test_production_geometry():
  s = Qwen4ExpTextSpec.from_hf_config(good()); assert s.hidden_size == 2560 and s.block_topk == 512

@pytest.mark.parametrize("change, message", [(lambda c: c["text_config"].update({"indexer_kv_heads":2}), "indexer_kv_heads"), (lambda c: c["text_config"].update({"indexer_budget":2050}), "divisible"), (lambda c: c["text_config"]["ple"].update({"dtype":"int4"}), "dtype")])
def test_rejects_bad_contract(change, message):
  c = good(); change(c)
  with pytest.raises(ValueError, match=message): Qwen4ExpTextSpec.from_hf_config(c)

def test_requires_text_config():
  with pytest.raises(ValueError, match="text_config"): Qwen4ExpTextSpec.from_hf_config({"model_type":"qwen4_exp"})
