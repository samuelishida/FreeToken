from __future__ import annotations

import torch

from freetoken.moe.fused_qwen4_gguf import (
    _groups_per_chunk,
    _batched_torch_qwen4_gguf,
    stable_group_routes,
)
from freetoken.moe.offload_cache import OffloadMoeCache


def test_stable_group_routes_preserves_route_order_without_host_list():
    ids = torch.tensor([[3, 1, 3, -1], [2, 1, 2, 3]], dtype=torch.int32)
    unique, positions, groups, valid = stable_group_routes(ids)
    assert unique.tolist() == [1, 2, 3]
    assert positions.tolist() == [1, 5, 4, 6, 0, 2, 7]
    assert groups.tolist() == [0, 0, 1, 1, 2, 2, 2]
    assert valid.tolist() == [True, True, True, False, True, True, True, True]
    assert unique.device == ids.device
    assert positions.device == ids.device


def test_selected_scratch_budget_is_at_least_one_group_and_bounded():
    assert _groups_per_chunk(1, 2560, 5120, 2) == 1
    assert _groups_per_chunk(128, 2560, 5120, 2) == 1
    assert _groups_per_chunk(1024, 2560, 5120, 2) == 10


def test_batched_qwen4_path_is_finite_and_handles_invalid_routes():
    # Minimal valid IQ geometry: gate/up IQ2 rows use 256 values; down IQ4
    # rows use 32-value blocks. Zero scales make deterministic zero output.
    experts, hidden, intermediate = 3, 256, 256
    gate = torch.zeros((experts, intermediate, 74), dtype=torch.uint8)
    up = torch.zeros_like(gate)
    down = torch.zeros((experts, hidden, 144), dtype=torch.uint8)
    x = torch.randn((4, hidden))
    ids = torch.tensor([[0, 1], [2, 0], [1, -1], [2, 2]], dtype=torch.int32)
    weights = torch.tensor([[0.5, 0.5], [0.2, 0.8], [1.0, 0.0], [0.3, 0.7]])
    output = _batched_torch_qwen4_gguf(
        x, gate, up, down, weights, ids, "silu", 0, scratch_mib=1
    )
    assert output.shape == (4, hidden)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) == 0


def test_batched_qwen4_path_does_not_replicate_weights_per_route(monkeypatch):
    """Route projections must share each selected expert weight, not bmm-copy it."""
    experts, hidden, intermediate = 2, 256, 256
    gate = torch.zeros((experts, intermediate, 74), dtype=torch.uint8)
    up = torch.zeros_like(gate)
    down = torch.zeros((experts, hidden, 144), dtype=torch.uint8)
    x = torch.randn((8, hidden))
    ids = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]], dtype=torch.int32)
    weights = torch.full((8, 2), 0.5)

    def no_route_weight_replication(*_args, **_kwargs):
        raise AssertionError("route-sized bmm weight replication is forbidden")

    monkeypatch.setattr(torch, "bmm", no_route_weight_replication)
    output = _batched_torch_qwen4_gguf(
        x, gate, up, down, weights, ids, "silu", 0, scratch_mib=1
    )
    assert output.shape == (8, hidden)
    assert torch.isfinite(output).all()


def test_rocm_safe_slot_assignment_is_device_vectorized_and_rewrites_routes():
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=4,
        cache_size=4,
        device=torch.device("cpu"),
        quant_format="gguf_qwen4",
    )
    routes = torch.tensor([[2, 1], [2, 3]], dtype=torch.int32)
    cache._ensure_experts_rocm_safe(0, routes)
    assert routes.tolist() == [[1, 0], [1, 2]]
    assert cache.num_indices.item() == 3
    assert cache._pending_num_indices_host == 3
    assert cache.slot_for_id[0].tolist() == [-1, 0, 1, 2]
