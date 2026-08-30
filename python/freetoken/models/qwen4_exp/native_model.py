"""Native Qwen4-Exp GGUF model assembly."""
from __future__ import annotations

import torch

from .native_gdn import Qwen4ExpGDN
from .native_moe import Qwen4ExpExpertCache, Qwen4ExpMoE
from .native_ops import FinalHyperConnection, HyperConnection, PackedEmbedding, PackedLinear, rms
from .native_ple import Qwen4ExpPLE
from .qsa import Qwen4ExpQSA


class Qwen4ExpLayer:
    def __init__(self, source, layer_idx: int, device: torch.device, rope_theta: float):
        p = f"blk.{layer_idx}."
        self.layer_idx = layer_idx
        self.attn_gr = HyperConnection(source, p + "hc_attn_", device)
        self.ffn_gr = HyperConnection(source, p + "hc_ffn_", device)
        self.attn = (Qwen4ExpQSA(source, layer_idx, device, rope_theta) if layer_idx % 4 == 3
                     else Qwen4ExpGDN(source, layer_idx, device))
        self.moe = Qwen4ExpMoE(source, layer_idx, device)

    def reset(self):
        self.attn.reset(); self.moe.reset()

    def __call__(self, streams: torch.Tensor) -> torch.Tensor:
        branch_input = self.attn_gr.read(streams)
        branch = self.attn(branch_input)
        streams = self.attn_gr.write(streams, branch)
        branch_input = self.ffn_gr.read(streams)
        branch = self.moe(branch_input)
        return self.ffn_gr.write(streams, branch)


class Qwen4ExpNativeModel:
    def __init__(self, source, config, device: torch.device):
        self.source, self.config, self.device = source, config, device
        self.max_seq_len = min(int(config.max_seq_len), 32768)
        metadata = source.metadata
        theta = float(metadata.get("qwen4exp.rope.freq_base", 1e7))
        self.embedding = PackedEmbedding(source, "token_embd.weight", device)
        self.lm_head = PackedLinear(source, "output.weight", device)
        self.ple = Qwen4ExpPLE(source, 1, device, max_rows=16384)
        self.expert_cache = Qwen4ExpExpertCache(4 << 30)
        self.layers = [Qwen4ExpLayer(source, i, device, theta) for i in range(48)]
        for layer in self.layers: layer.moe.gpu_cache = self.expert_cache
        self.final = FinalHyperConnection(source, device)
        self.position = 0

    def reset(self):
        self.position = 0; self.ple.reset()
        for layer in self.layers: layer.reset()

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Qwen4-Exp native model requires input_ids [1,T]")
        if start_pos < 0 or start_pos + input_ids.shape[1] > self.max_seq_len:
            raise ValueError("Qwen4-Exp context exceeds native cache plan")
        ids = input_ids.to(device=self.device, dtype=torch.long)
        streams = self.embedding(ids).unsqueeze(-2).expand(-1, -1, 4, -1).contiguous()
        for layer in self.layers:
            # PLE is injected before attention in block 1 (GGUF metadata is
            # zero-based: ``ple.layers=[1]``).  Its rows are still fetched once
            # per request chunk, never materialized as a table.
            if layer.layer_idx == 1:
                streams = streams + self.ple(ids, streams).reshape(1, ids.shape[1], 4, 2560)
            streams = layer(streams)
            if not torch.isfinite(streams).all():
                raise FloatingPointError(f"non-finite Qwen4-Exp state at layer {layer.layer_idx}, position {start_pos}")
        hidden = self.final(streams)
        self.position = start_pos + ids.shape[1]
        return self.lm_head(hidden[:, -1])


__all__ = ["Qwen4ExpNativeModel"]
