"""Opt-in real Qwen3.8 GGUF smoke gate.

The full checkpoint is intentionally never pulled by normal CI. Set both
FREETOKEN_QWEN38_GGUF and FREETOKEN_QWEN38_E2E=1 on an AMD host to run it.
"""
import os

import numpy as np
import pytest


TARGET = os.environ.get("FREETOKEN_QWEN38_GGUF")


@pytest.mark.needs_weights
@pytest.mark.skipif(not TARGET or not os.environ.get("FREETOKEN_QWEN38_E2E"),
                    reason="set FREETOKEN_QWEN38_GGUF and FREETOKEN_QWEN38_E2E=1")
def test_qwen38_gguf_constructs_and_resets():
    from tinygrad import Tensor
    from tinygrad.llm.model import Transformer

    model, kv = Transformer.from_gguf(TARGET, max_context=128)
    assert kv == {}
    model.blocks = model.blocks[:1]
    model.output_weight = None
    hidden = model.forward(Tensor([[248044, 100]], dtype="int32"), 0).realize().numpy()
    assert hidden.shape == (1, 2, 2560)
    assert hidden.dtype.kind == "f" and np.isfinite(hidden).all()
    model.reset_state()
    assert model.position == 0
