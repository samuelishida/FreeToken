"""Small native Qwen4-Exp building blocks.

Weights remain in GGUF layout.  ``PackedLinear`` is the only place that knows
whether a projection is dense, K-quantized, or IQ-quantized.
"""
from __future__ import annotations

import torch

from freetoken.kernel.triton.qwen4exp_quant import quant_linear


class PackedLinear:
    def __init__(self, source, name: str, device: torch.device):
        loc = source.locate(name)
        self.name, self.ggml_type = name, loc.ggml_type
        self.out_features = int(loc.shape[0])
        self.in_features = int(loc.shape[-1])
        self.weight = source.read_tensor(name, device=device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1]
        return quant_linear(x.reshape(-1, x.shape[-1]), self.weight,
                            self.ggml_type, self.out_features).reshape(*shape, self.out_features)


class PackedEmbedding:
    def __init__(self, source, name: str, device: torch.device):
        loc = source.locate(name)
        self.ggml_type, self.rows, self.dim = loc.ggml_type, loc.shape[0], loc.shape[1]
        self.weight = source.read_tensor(name, device=device)

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.gguf import ggml_dequantize
        flat = ids.reshape(-1).to(torch.long)
        packed = self.weight.index_select(0, flat)
        if self.ggml_type in (0, 30):
            dtype = torch.float32 if self.ggml_type == 0 else torch.bfloat16
            out = packed.view(dtype).reshape(flat.numel(), self.dim).to(torch.float32)
        else:
            if self.ggml_type == 20 and self.dim % 256:
                raise ValueError(
                    f"generic IQ4_NL embedding requires width divisible by 256, got {self.dim}; "
                    "PLE rows must use the dedicated five-block helper"
                )
            out = ggml_dequantize(packed, self.ggml_type, flat.numel(), self.dim, torch.float32)
        return out.reshape(*ids.shape, self.dim)


def rms(x: torch.Tensor, weight: torch.Tensor, *, grouped: bool = False, eps: float = 1e-6):
    if grouped:
        stream_shape = x.shape[-2:] if x.ndim >= 2 and x.shape[-2:] == (4, weight.numel() // 4) else None
        if stream_shape is not None: x = x.reshape(*x.shape[:-2], weight.numel())
        if x.shape[-1] != weight.numel(): raise ValueError("grouped RMS width mismatch")
        groups = x.reshape(*x.shape[:-1], 4, weight.numel() // 4)
        w = weight.reshape(4, -1)
        out = (groups * torch.rsqrt(groups.float().square().mean(-1, keepdim=True) + eps)
               * (w + 1.0)).reshape_as(x)
        return out.reshape(*out.shape[:-1], 4, weight.numel() // 4) if stream_shape is not None else out
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps) * weight


class HyperConnection:
    def __init__(self, source, prefix: str, device: torch.device):
        self.norm = source.read_tensor(prefix + "norm.weight", device=device).reshape(4, 2560)
        self.down = PackedLinear(source, prefix + "down.weight", device)
        self.up = PackedLinear(source, prefix + "up.weight", device)
        self.inject = source.read_tensor(prefix + "inject.weight", device=device)

    def read(self, streams: torch.Tensor) -> torch.Tensor:
        n = rms(streams, self.norm.reshape(-1), grouped=True)
        low = torch.nn.functional.silu(self.down(n.reshape(*n.shape[:-2], 10240)) / 4.0)
        gate = torch.sigmoid(self.up(low)).reshape(*n.shape)
        return (gate * n).mean(-2)

    def write(self, streams: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        n = rms(streams, self.norm.reshape(-1), grouped=True)
        flat = n.reshape(*n.shape[:-2], 10240)
        inject = 2.0 * torch.sigmoid(self.inject_linear(flat) / 4.0)
        return streams + inject.unsqueeze(-1) * branch.unsqueeze(-2)

    def inject_linear(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.inject.to(x.dtype).T


class FinalHyperConnection:
    def __init__(self, source, device: torch.device):
        self.norm = source.read_tensor("output_hc_norm.weight", device=device).reshape(4, 2560)
        self.down = PackedLinear(source, "output_hc_down.weight", device)
        self.up = PackedLinear(source, "output_hc_up.weight", device)

    def __call__(self, streams: torch.Tensor) -> torch.Tensor:
        n = rms(streams, self.norm.reshape(-1), grouped=True)
        low = torch.nn.functional.silu(self.down(n.reshape(*n.shape[:-2], 10240)) / 4.0)
        return (torch.sigmoid(self.up(low)).reshape_as(n) * n).mean(-2)


__all__ = ["PackedLinear", "PackedEmbedding", "HyperConnection", "FinalHyperConnection", "rms"]
