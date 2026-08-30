"""Stateful device QSA: compressed block routing plus gathered GQA."""
from __future__ import annotations

import math
import torch

from .native_ops import PackedLinear, rms


def _rope(x: torch.Tensor, start: int, dim: int, theta: float = 1e7) -> torch.Tensor:
    if dim <= 0: return x
    positions = torch.arange(start, start + x.shape[1], device=x.device, dtype=torch.float32)
    inv = theta ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    angle = positions[:, None] * inv[None]
    cos, sin = angle.cos(), angle.sin()
    left, right = x[..., :dim // 2], x[..., dim // 2:dim]
    rotated = torch.cat((left * cos.reshape(1, -1, 1, dim // 2) - right * sin.reshape(1, -1, 1, dim // 2),
                         right * cos.reshape(1, -1, 1, dim // 2) + left * sin.reshape(1, -1, 1, dim // 2)), dim=-1)
    return torch.cat((rotated, x[..., dim:]), dim=-1)


class Qwen4ExpQSA:
    def __init__(self, source, layer_idx: int, device: torch.device, rope_theta: float = 1e7):
        p = f"blk.{layer_idx}."
        self.q = PackedLinear(source, p + "attn_q.weight", device)
        self.k = PackedLinear(source, p + "attn_k.weight", device)
        self.v = PackedLinear(source, p + "attn_v.weight", device)
        self.o = PackedLinear(source, p + "attn_output.weight", device)
        self.iq = PackedLinear(source, p + "indexer.q_proj.weight", device)
        self.ik = PackedLinear(source, p + "indexer.k_proj.weight", device)
        self.qnorm = source.read_tensor(p + "attn_q_norm.weight", device=device).float()
        self.knorm = source.read_tensor(p + "attn_k_norm.weight", device=device).float()
        self.iqnorm = source.read_tensor(p + "indexer.q_norm.weight", device=device).float()
        self.iknorm = source.read_tensor(p + "indexer.k_norm.weight", device=device).float()
        self.rope_theta = float(rope_theta)
        self.reset()

    def reset(self):
        self.keys = []; self.values = []; self.raw_index = []; self.blocks = []

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        t = x.shape[1]
        qraw = self.q(x).reshape(1, t, 24, 2, 256)
        q, gate = qraw[..., 0, :], qraw[..., 1, :]
        k = self.k(x).reshape(1, t, 2, 256)
        v = self.v(x).reshape(1, t, 2, 256)
        q, k = _rope(rms(q, self.qnorm), start_pos, 64, self.rope_theta), _rope(rms(k, self.knorm), start_pos, 64, self.rope_theta)
        iq = self.iq(x).reshape(1, t, 4, 128)
        ik = self.ik(x).reshape(1, t, 128)
        iq = _rope(rms(iq, self.iqnorm), start_pos, 64, self.rope_theta)
        ik = rms(ik, self.iknorm)
        self.keys.append(k[0]); self.values.append(v[0])
        self.raw_index.extend(ik[0].unbind(0))
        while len(self.raw_index) >= 4:
            group = torch.stack(self.raw_index[:4]).mean(0)
            pos = len(self.blocks) * 4
            group = group * torch.rsqrt(group.square().mean() + 1e-6)
            group = _rope(group.reshape(1, 1, 1, 128), pos, 64, self.rope_theta).reshape(128)
            self.blocks.append(group); del self.raw_index[:4]
        all_k, all_v = torch.cat(self.keys), torch.cat(self.values)
        block_tensor = torch.stack(self.blocks) if self.blocks else all_k.new_empty((0, 128))
        outputs = []
        for row in range(t):
            position = start_pos + row
            visible_blocks = (position + 1) // 4
            if visible_blocks <= 512:
                selected = torch.arange(visible_blocks, device=x.device, dtype=torch.long)
            else:
                scores = torch.relu(iq[0, row] @ block_tensor[:visible_blocks].T).sum(0) / math.sqrt(128)
                selected = torch.topk(scores, min(512, visible_blocks), sorted=False).indices
            selected = (selected[:, None] * 4 + torch.arange(4, device=x.device)).reshape(-1)
            tail_start = visible_blocks * 4
            tail_count = (position + 1) - tail_start
            if tail_count:
                selected = torch.cat((selected, tail_start + torch.arange(min(3, tail_count), device=x.device)))
            selected = selected[selected <= position]
            if selected.numel() == 0:
                outputs.append(torch.zeros((24, 256), device=x.device, dtype=x.dtype)); continue
            kk, vv = all_k.index_select(0, selected), all_v.index_select(0, selected)
            grouped = []
            for head in range(24):
                kv = head // 12
                score = (q[0, row, head] @ kk[:, kv].T) / math.sqrt(256)
                grouped.append(torch.softmax(score.float(), -1) @ vv[:, kv].float())
            outputs.append(torch.stack(grouped).to(x.dtype))
        attended = torch.stack(outputs).unsqueeze(0) * torch.sigmoid(gate)
        return self.o(attended.reshape(1, t, 6144))


__all__ = ["Qwen4ExpQSA"]
