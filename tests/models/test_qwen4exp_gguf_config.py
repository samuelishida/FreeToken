import os

import pytest

from freetoken.models.gguf.config import build_gguf_shim
from freetoken.models.register import _load_attr, get_model_spec


TARGET = os.environ.get("FREETOKEN_QWEN38_GGUF")


@pytest.mark.skipif(TARGET is None, reason="set FREETOKEN_QWEN38_GGUF for real checkpoint gate")
def test_real_qwen4exp_gguf_config():
    shim = build_gguf_shim(TARGET)
    assert shim.architectures == ["Qwen4ExpGGUFForCausalLM"]
    assert shim.vocab_size == 248320
    assert not shim.tie_word_embeddings
    spec = get_model_spec(shim.architectures[0])
    config = _load_attr(spec.module, spec.parse_config)(shim)
    assert config.num_layers == 48
    assert config.num_experts == 512 and config.num_experts_per_tok == 10
    assert tuple(config.attention_groups[1].layer_ids) == tuple(range(3, 48, 4))
    assert config.hidden_size == 2560
    assert config.num_qo_heads == 24 and config.num_kv_heads == 2
    assert config.head_dim == 256
    assert config.expert_quant == config.attn_quant == config.dense_quant == "gguf"
    assert config.lm_head_quant == config.moe_weight_format == "gguf"
    args = config.qwen4_args
    assert (args.hc_count, args.hc_lowrank) == (4, 320)
    assert args.ple_layer_ids == (1,)
    assert (args.index_n_heads, args.index_kv_heads, args.index_head_dim) == (4, 1, 128)
    assert (args.index_budget, args.index_ratio) == (2048, 4)
    assert {state.name for state in config.slot_states} == {"ple_conv", "ple_ngram_ctx"}
