"""Explicit test-only primitive oracle for Qwen4 GGUF expert parity.

Normal layer dispatch never imports this module. Keeping slow reference
execution out of serving code prevents accidental fallback to a Python expert
loop when a HIP kernel is unavailable.
"""

from __future__ import annotations

import torch

from freetoken.layers.activation import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul
from freetoken.models.gguf.dequant import GGML_IQ2_XS, GGML_IQ3_XXS, GGML_IQ4_NL
from freetoken.moe.fused_qwen4_gguf import _apply_activation, _selected_dense

_ACT = {
    "silu": silu_and_mul,
    "swish": silu_and_mul,
    "gelu": gelu_and_mul,
    "gelu_tanh": gelu_tanh_and_mul,
}


def primitive_experts_qwen4_gguf(
    hidden_states, gate_q, up_q, down_q, topk_weights, topk_ids, activation, layer_id
):
    """Slow per-expert reference; callable only through ``oracle=True``."""
    if activation not in _ACT:
        raise ValueError(f"unsupported Qwen4Exp activation {activation!r}")
    tokens, hidden = hidden_states.shape
    top_k = topk_ids.shape[1]
    intermediate = gate_q.shape[1]
    gate_type = GGML_IQ3_XXS if layer_id == 2 else GGML_IQ2_XS
    ids = topk_ids.reshape(-1)
    x = hidden_states[:, None, :].expand(tokens, top_k, hidden).reshape(-1, hidden)
    out = torch.zeros_like(x)
    valid = ids >= 0
    # Oracle-only host list. Production dispatch never reaches this module.
    for expert_id in torch.unique(ids[valid]).detach().cpu().tolist():
        positions = torch.nonzero(valid & (ids == int(expert_id)), as_tuple=False).flatten()
        selected = torch.tensor([expert_id], dtype=torch.long, device=gate_q.device)
        x_selected = x.index_select(0, positions)
        gate_w = _selected_dense(
            gate_q, selected, gate_type, intermediate, hidden, hidden_states.dtype
        )[0]
        up_w = _selected_dense(
            up_q, selected, gate_type, intermediate, hidden, hidden_states.dtype
        )[0]
        inter = _apply_activation(
            activation, torch.cat((x_selected @ gate_w.T, x_selected @ up_w.T), dim=-1)
        )
        down_w = _selected_dense(
            down_q, selected, GGML_IQ4_NL, hidden, intermediate, hidden_states.dtype
        )[0]
        down = inter @ down_w.T
        weights = topk_weights.reshape(-1).index_select(0, positions).to(down.dtype)
        out.index_copy_(0, positions, down * weights.unsqueeze(-1))
    return out.view(tokens, top_k, hidden).sum(dim=1)


__all__ = ["primitive_experts_qwen4_gguf"]
