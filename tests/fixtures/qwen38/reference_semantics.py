import torch

def _norm(x, w, eps=1e-6):
  z = x.float(); return (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + eps)) * (w.float() + 1)

def gr_read(R, weights):
  assert R.ndim == 3 and R.shape[1:] == (4, 2560)
  n = _norm(R, weights["norm"])
  low = torch.nn.functional.silu((n.reshape(R.shape[0], -1) @ weights["down"].float().T) / 4)
  gate = torch.sigmoid(low @ weights["up"].float().T).reshape(R.shape[0], 4, 2560)
  return (gate * n).mean(1)

def gr_write(R, y, weights):
  assert R.ndim == 3 and y.shape == (R.shape[0], 2560)
  n = _norm(R, weights["norm"]); inj = 2 * torch.sigmoid((n.reshape(R.shape[0], -1) @ weights["inject"].float().T) / 4)
  return R + inj.unsqueeze(-1) * y.unsqueeze(1)

def compress_raw_index_keys(keys, start_pos):
  assert keys.ndim == 2 and keys.shape[1] == 128 and start_pos % 4 == 0
  n = keys.shape[0] // 4
  pooled = keys[:n * 4].float().reshape(n, 4, 128).mean(1)
  pooled = pooled * torch.rsqrt(pooled.square().mean(-1, keepdim=True) + 1e-6)
  rope_dim = 64
  half = rope_dim // 2
  inv = 10000 ** (-torch.arange(half, dtype=torch.float32) * 2 / rope_dim)
  for row in range(n):
    angle = (start_pos + row * 4) * inv
    x1, x2 = pooled[row, :half].clone(), pooled[row, half:rope_dim].clone()
    pooled[row, :half], pooled[row, half:rope_dim] = x1 * angle.cos() - x2 * angle.sin(), x2 * angle.cos() + x1 * angle.sin()
  return pooled

def select_qsa_blocks(q, compressed, visible_len):
  assert q.ndim == 3 and q.shape[1:] == (4, 128)
  scores = torch.relu(torch.einsum("thd,bd->thb", q.float(), compressed.float())).sum(1) / (128 ** .5)
  rows = []
  for row in range(q.shape[0]):
    query_visible = min(visible_len, visible_len - q.shape[0] + row + 1)
    complete = min(compressed.shape[0], query_visible // 4)
    count = min(512, complete)
    ids = scores[row, :complete].topk(count).indices.sort().values if count else torch.empty(0, dtype=torch.long)
    blocks = (ids[:, None] * 4 + torch.arange(4)).reshape(-1)
    tail = torch.arange(complete * 4, min(query_visible, complete * 4 + 3))
    rows.append(torch.cat((blocks, tail)))
  width = max((len(row) for row in rows), default=0)
  out = torch.full((q.shape[0], width), -1, dtype=torch.long)
  for row, ids in enumerate(rows): out[row, :len(ids)] = ids
  return out
