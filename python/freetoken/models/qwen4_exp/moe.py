from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING
import os
import time

import torch
from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add, shared_gate_sigmoid
from freetoken.layers.moe import make_moe_layer
from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE
from freetoken.utils import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen4ExpMoE(Qwen3_5MoE):
    """Qwen3_5MoE with the shared-expert gate on triton instead of gemv + sigmoid + mul + add.

    Same weights, same state dict. The gate reduction stays ahead of the routed experts, which may write into ``hidden_states`` in place.
    """

    def __init__(self, config: ModelConfig, layer_id: int | None = None) -> None:
        if getattr(config, "expert_quant", "none") != "fp8_block":
            super().__init__(config, layer_id=layer_id)
            return
        # Qwen3.8's block-fp8 checkpoint quantizes only the routed experts; the shared
        # expert stays bf16, so hide expert_quant from _SharedExpert's fp8 branch and
        # rebuild the routed experts with the fp8_block bank layout.
        super().__init__(replace(config, expert_quant="none"), layer_id=layer_id)
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            weight_format="fp8_block",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        observe = getattr(self, "_status_observer", None)
        profile = (
            os.environ.get("FREETOKEN_QWEN38_COMPONENT_TIMING", "").strip().lower()
            in ("1", "true", "yes", "on")
        ) and getattr(getattr(self, "experts", None), "layer_id", None) == 4 and hidden_states.is_cuda
        t0 = time.perf_counter() if profile else 0.0
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if observe is not None:
            observe("moe_router_started")
        router_logits = self.gate.forward(hidden_states)
        if profile:
            torch.cuda.synchronize(hidden_states.device)
            logger.info("Qwen3.8 component layer=4 router_ms=%.2f", (time.perf_counter()-t0)*1000)
        if observe is not None:
            observe("moe_router_done")
        shared = self.shared_expert.forward(hidden_states)
        if torch.version.hip is not None:
            gate = torch.sigmoid(hidden_states.float() @ self.shared_expert_gate.weight.view(-1).float())
        else:
            gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
        if observe is not None:
            observe("moe_shared_done")
        if profile:
            torch.cuda.synchronize(hidden_states.device)
            logger.info("Qwen3.8 component layer=4 shared_ms=%.2f", (time.perf_counter()-t0)*1000)
        self.experts._status_observer = observe
        routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
        if profile:
            torch.cuda.synchronize(hidden_states.device)
            logger.info("Qwen3.8 component layer=4 routed_ms=%.2f", (time.perf_counter()-t0)*1000)
        if observe is not None:
            observe("moe_routed_done")
        if torch.version.hip is not None:
            result = (routed + shared * gate.to(shared.dtype).unsqueeze(-1)).view(num_tokens, hidden_dim)
            if profile:
                torch.cuda.synchronize(hidden_states.device)
                logger.info("Qwen3.8 component layer=4 moe_total_ms=%.2f", (time.perf_counter()-t0)*1000)
            return result
        return shared_gate_mul_add(routed, shared, gate).view(num_tokens, hidden_dim)


__all__ = ["Qwen4ExpMoE"]
