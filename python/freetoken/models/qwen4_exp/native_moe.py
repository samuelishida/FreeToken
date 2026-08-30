"""Bounded selected-expert Qwen4-Exp MoE."""
from __future__ import annotations

from collections import OrderedDict
import os

import torch
import torch.nn.functional as F

from freetoken.kernel.triton.qwen4exp_quant import expert_quant_linear, quant_linear
from .native_ops import PackedLinear


class Qwen4ExpExpertCache:
    def __init__(self, limit_bytes: int):
        self.limit_bytes = int(limit_bytes); self.items = OrderedDict(); self.bytes = 0

    def get(self, key):
        value = self.items.get(key)
        if value is not None: self.items.move_to_end(key)
        return value

    def put(self, key, value):
        size = value.numel() * value.element_size()
        while self.items and self.bytes + size > self.limit_bytes:
            _, old = self.items.popitem(last=False); self.bytes -= old.numel() * old.element_size()
        if size <= self.limit_bytes: self.items[key] = value; self.bytes += size


class Qwen4ExpMoE:
    def __init__(self, source, layer_idx: int, device: torch.device, cache: Qwen4ExpExpertCache | None = None):
        p = f"blk.{layer_idx}."
        self.source, self.layer_idx, self.device = source, layer_idx, device
        self.router = PackedLinear(source, p + "ffn_gate_inp.weight", device)
        self.shared_gate = PackedLinear(source, p + "ffn_gate_shexp.weight", device)
        self.shared_up = PackedLinear(source, p + "ffn_up_shexp.weight", device)
        self.shared_down = PackedLinear(source, p + "ffn_down_shexp.weight", device)
        self.shared_router = source.read_tensor(p + "ffn_gate_inp_shexp.weight", device=device).float()
        self.gate_name, self.up_name, self.down_name = (p + n for n in (
            "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"))
        self.gate_type = source.locate(self.gate_name).ggml_type
        self.up_type = source.locate(self.up_name).ggml_type
        self.down_type = source.locate(self.down_name).ggml_type
        self.gpu_cache = cache or Qwen4ExpExpertCache(
            int(float(os.environ.get("QWEN4EXP_GPU_EXPERT_CACHE_GIB", "4")) * (1 << 30)))
        self.reset()

    def reset(self):
        self.expert_reads = 0

    def _rows(self, name: str, expert: int, rows: int) -> torch.Tensor:
        key = (name, expert)
        packed = self.gpu_cache.get(key)
        if packed is not None: return packed
        packed = self.source.read_rows(name, range(expert * rows, (expert + 1) * rows), device=self.device)
        if packed.numel() * packed.element_size() <= self.gpu_cache.limit_bytes:
            self.gpu_cache.put(key, packed)
        self.expert_reads += 1
        return packed

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[0] != 1 or x.shape[-1] != 2560:
            raise ValueError("Qwen4-Exp MoE expects [1,T,2560]")
        flat = x.reshape(-1, 2560)
        scores = torch.sigmoid(self.router(flat).float())
        probs, ids = torch.topk(scores, 10, dim=-1, sorted=True)
        probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-12)
        outputs = []
        for token in range(flat.shape[0]):
            inp = flat[token:token + 1]
            experts = [int(expert) for expert in ids[token].tolist()]
            gate_pack = torch.stack([self._rows(self.gate_name, expert, 640) for expert in experts])
            up_pack = torch.stack([self._rows(self.up_name, expert, 640) for expert in experts])
            gate = expert_quant_linear(inp.expand(10, -1), gate_pack, ids[token:token + 1],
                                       self.gate_type, 640).reshape(10, 640)
            up = expert_quant_linear(inp.expand(10, -1), up_pack, ids[token:token + 1],
                                     self.up_type, 640).reshape(10, 640)
            down_pack = torch.stack([self._rows(self.down_name, expert, 2560) for expert in experts])
            down = expert_quant_linear((F.silu(gate) * up), down_pack, ids[token:token + 1],
                                       self.down_type, 2560).reshape(10, 2560)
            routed = down.mul(probs[token].to(x.dtype).unsqueeze(-1)).sum(0)
            shared = self.shared_down(F.silu(self.shared_gate(inp)) * self.shared_up(inp))[0]
            shared = shared * torch.sigmoid(inp @ self.shared_router.to(inp.dtype))
            outputs.append(routed + shared)
        return torch.stack(outputs).reshape_as(x)


__all__ = ["Qwen4ExpMoE", "Qwen4ExpExpertCache"]
