"""GGUF PLE/Engram row paging and device projection."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .native_ops import PackedLinear, rms

_MULT = (23703573157769, 20109073645365, 8052911324071)
_PRIME = (20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
          20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171)
_OFFSET = (0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
           160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275)
EOS = 248044


class Qwen4ExpPLE:
    def __init__(self, source, layer_idx: int, device: torch.device, max_rows: int = 4096):
        if layer_idx != 1: raise ValueError("Qwen4-Exp GGUF has one PLE layer at blk.1")
        p = "blk.1."
        self.source, self.device, self.max_rows = source, device, int(max_rows)
        self.table = source.locate("per_layer_token_embd.weight")
        self.key = PackedLinear(source, p + "ple_key.weight", device)
        self.value = PackedLinear(source, p + "ple_value.weight", device)
        self.norm_key = source.read_tensor(p + "ple_norm_key.weight", device=device).float()
        self.norm_query = source.read_tensor(p + "ple_norm_query.weight", device=device).float()
        self.norm_conv = source.read_tensor(p + "ple_norm_conv.weight", device=device).float()
        self.conv = source.read_tensor(p + "ple_conv1d.weight", device=device).float()
        self.history: list[int] = []
        self.conv_state = torch.zeros((9, 10240), device=device, dtype=torch.float32)

    def reset(self):
        self.history.clear(); self.conv_state.zero_()

    def row_ids(self, tokens: torch.Tensor) -> torch.Tensor:
        values = tokens.reshape(-1).to(torch.int64)
        rows = []
        for value in values.unbind():
            current = int(value.item())
            prev = self.history[-1] if self.history else EOS
            prev2 = self.history[-2] if len(self.history) > 1 and self.history[-1] != EOS else EOS
            out = []
            for head in range(16):
                n = 2 if head < 8 else 3
                mixed = current * _MULT[0]
                mixed ^= prev * _MULT[1]
                if n == 3: mixed ^= prev2 * _MULT[2]
                out.append((mixed % _PRIME[head] + _OFFSET[head]) % self.table.rows)
            rows.append(out); self.history.append(current)
        return torch.tensor(rows, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def __call__(self, tokens: torch.Tensor, streams: torch.Tensor) -> torch.Tensor:
        ids = self.row_ids(tokens)
        chunk_tokens = max(1, self.max_rows // 16)
        outputs = []
        from freetoken.kernel.triton.ple_iq4_nl import dequant_iq4_nl_rows
        for start in range(0, tokens.numel(), chunk_tokens):
            end = min(tokens.numel(), start + chunk_tokens)
            # GGUF row paging is the deliberate host/SSD boundary. IDs are copied
            # as scalars only; projections and convolution remain on HIP.
            packed = self.source.read_rows("per_layer_token_embd.weight", ids[start:end].reshape(-1).tolist(), device=self.device)
            if self.table.ggml_type != 20:
                raise ValueError(
                    f"Qwen4-Exp native PLE requires IQ4_NL rows, got ggml type {self.table.ggml_type}"
                )
            values = dequant_iq4_nl_rows(packed, out_dtype=torch.float32)
            values = values.reshape(1, end - start, 16, 160).reshape(1, end - start, 2560)
            local_streams = streams[:, start:end]
            key = rms(self.key(values), self.norm_key, grouped=True).reshape_as(local_streams)
            value = self.value(values)
            query = rms(local_streams.reshape(1, end - start, 10240), self.norm_query, grouped=True).reshape_as(local_streams)
            dot = (query * key).sum(-1, keepdim=True) / (2560.0 ** 0.5)
            gate = torch.sign(dot) * torch.sqrt(torch.clamp(dot.abs(), min=1e-6))
            flat = (torch.sigmoid(gate) * value.unsqueeze(-2)).reshape(1, end - start, 10240)
            normalized = rms(flat, self.norm_conv, grouped=True)
            window = torch.cat((self.conv_state, normalized.reshape(-1, 10240)), dim=0)
            local_out = []
            for i in range(end - start):
                conv = sum(window[9 + i - (3 - tap) * 3] * self.conv[:, tap] for tap in range(4))
                local_out.append(flat[0, i] + F.silu(conv))
            self.conv_state = window[-9:].detach()
            outputs.extend(local_out)
        return torch.stack(outputs).reshape(1, tokens.numel(), 10240)


__all__ = ["Qwen4ExpPLE", "EOS"]
