# Qwen38 Grouped MoE OOM

## Context

Fixed ROCm Qwen3.8 prefill OOM in the grouped GGUF expert fallback. A 128 MiB
selected-expert budget still produced a 3.17 GiB temporary on routed projection.

## Hardest decision

Replace route-wide batched projections with bounded per-group `torch.mm` using
shared selected-weight views. This trades a small expert-group loop for strict
memory bounds and preserves route order via device-side positions.

## Alternatives rejected

- Increase GPU memory or allocator limits — does not remove route-weight
  replication and fails on 24 GiB cards.
- Increase scratch budget — makes the accidental temporary larger.
- Re-enable primitive per-expert oracle — correctness-only path and too slow for
  serving.

## Least confident

Dedicated future fused HIP kernels may use a different workspace contract; keep
their capability gate disabled until their allocation behavior is measured on
supported ROCm targets.

## Reuse

Read before changing `python/freetoken/moe/fused_qwen4_gguf.py` or Qwen GGUF
prefill routing. Never index selected expert weights by every route unless the
operator explicitly proves allocation-free behavior.
