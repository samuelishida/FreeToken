"""Device Qwen4-Exp GatedDeltaNet recurrence."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .native_ops import PackedLinear, rms


class Qwen4ExpGDN:
    def __init__(self, source, layer_idx: int, device: torch.device):
        p = f"blk.{layer_idx}."
        self.qkv = PackedLinear(source, p + "attn_qkv.weight", device)
        self.gate = PackedLinear(source, p + "attn_gate.weight", device)
        self.alpha = PackedLinear(source, p + "ssm_alpha.weight", device)
        self.beta = PackedLinear(source, p + "ssm_beta.weight", device)
        self.out = PackedLinear(source, p + "ssm_out.weight", device)
        self.conv = source.read_tensor(p + "ssm_conv1d.weight", device=device).float()
        self.dt = source.read_tensor(p + "ssm_dt.bias", device=device).float()
        self.a = source.read_tensor(p + "ssm_a", device=device).float()
        self.norm = source.read_tensor(p + "ssm_norm.weight", device=device).float()
        self.state = torch.zeros((48, 128, 128), device=device, dtype=torch.float32)
        self.conv_state = torch.zeros((3, 10240), device=device, dtype=torch.float32)

    def reset(self):
        self.state.zero_(); self.conv_state.zero_()

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[0] != 1 or x.shape[-1] != 2560:
            raise ValueError("Qwen4-Exp GDN expects [1,T,2560]")
        raw = self.qkv(x).float()
        gate = self.gate(x).float().reshape(1, x.shape[1], 48, 128)
        beta = torch.sigmoid(self.beta(x).float()).reshape(1, x.shape[1], 48)
        alpha = (F.softplus(self.alpha(x).float() + self.dt) * self.a).reshape(1, x.shape[1], 48)
        outputs = []
        for t in range(x.shape[1]):
            window = torch.cat((self.conv_state, raw[0, t:t + 1]), dim=0)
            self.conv_state = window[1:]
            mixed = torch.sum(window * self.conv.T, dim=0)
            mixed = F.silu(mixed)
            q, k, v = mixed[:2048].reshape(16, 128), mixed[2048:4096].reshape(16, 128), mixed[4096:].reshape(48, 128)
            q = F.normalize(q.float(), dim=-1).repeat_interleave(3, dim=0)
            k = F.normalize(k.float(), dim=-1).repeat_interleave(3, dim=0)
            h = self.state * torch.exp(torch.clamp(alpha[0, t], min=-30.0, max=30.0))[:, None, None]
            delta = (v - torch.sum(h * k[:, None, :], dim=-1)) * beta[0, t, :, None]
            self.state = h + delta[:, :, None] * k[:, None, :]
            outputs.append(torch.sum(self.state * q[:, :, None], dim=-1))
        core = torch.stack(outputs, dim=0).unsqueeze(0)
        core = rms(core, self.norm)
        return self.out((core * F.silu(gate)).reshape(1, x.shape[1], 6144))


__all__ = ["Qwen4ExpGDN"]
