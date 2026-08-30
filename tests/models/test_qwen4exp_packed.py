from __future__ import annotations

import pytest

from freetoken.models.qwen4_exp.packed import (
    PackedCache, PackedExpertHotCache, Qwen4ExpPackedSource, _runs,
)


def test_runs_split_ranges():
    assert _runs((1, 2, 3, 8, 9)) == ((1, 4), (8, 10))
    assert _runs(()) == ()


def test_cache_is_bounded_and_reports_eviction():
    cache = PackedCache(4)
    import torch
    cache.put(("a", (0,)), torch.zeros(2, dtype=torch.uint8))
    cache.put(("b", (0,)), torch.zeros(2, dtype=torch.uint8))
    assert cache.report()["resident_bytes"] == 4
    cache.put(("c", (0,)), torch.zeros(2, dtype=torch.uint8))
    assert cache.report()["evictions"] == 1
    with pytest.raises(MemoryError): cache.put(("x",), torch.zeros(5, dtype=torch.uint8))


def test_expert_hot_cache_keeps_packed_bytes_and_prefers_probationary_eviction():
    import torch

    cache = PackedExpertHotCache(12)
    value = {"gate": torch.zeros(2, dtype=torch.uint8), "up": torch.ones(2, dtype=torch.uint8)}
    cache.put((0, 0), value)
    cache.put((1, 0), value)
    assert cache.get((0, 0)) is not None
    cache.put((2, 0), value)
    cache.put((3, 0), value)
    report = cache.report()
    assert report["resident_bytes"] == 12
    assert report["evictions"] == 1
    assert report["protected"] == 1


def test_qwen4_cache_accepts_layer_specific_iq_row_widths():
    import torch
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=2, num_experts=4, cache_size=8, device=torch.device("cpu"),
        quant_format="gguf_qwen4", prefill_overlap=False,
    )
    sources = {
        "gate": [torch.zeros(4, 2, 740, dtype=torch.uint8), torch.zeros(4, 2, 980, dtype=torch.uint8)],
        "up": [torch.zeros(4, 2, 740, dtype=torch.uint8), torch.zeros(4, 2, 980, dtype=torch.uint8)],
        "down": [torch.zeros(4, 3, 18, dtype=torch.uint8), torch.zeros(4, 3, 18, dtype=torch.uint8)],
    }
    cache.set_bank_sources(sources, layer_residency=["pageable", "pageable"])
    assert [view.shape for view in cache.bank_views(layer_id=0)] == [(8, 2, 740), (8, 2, 740), (8, 3, 18)]
    assert [view.shape for view in cache.bank_views(layer_id=1)] == [(8, 2, 980), (8, 2, 980), (8, 3, 18)]


def test_qwen4_expert_dispatch_uses_grouped_packed_kernels(monkeypatch):
    import torch
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)

    layer = OffloadMoELayer(
        layer_id=2, num_experts=4, top_k=2, hidden_size=8, intermediate_size=16
    )
    calls = []

    def fake_grouped(x, weight, ids, top_k, quant_type, row, tokens):
        calls.append((quant_type, row, tokens, tuple(weight.shape), tuple(ids.shape)))
        return torch.zeros((tokens * top_k, row), dtype=x.dtype)

    monkeypatch.setattr("freetoken.kernel.gguf.ggml_moe_a8_vec", fake_grouped)
    monkeypatch.setattr("freetoken.layers.activation.silu_and_mul", lambda x: x[..., :16])
    cache = type("Cache", (), {"quant_format": "gguf_qwen4"})()
    hidden = torch.randn(3, 8)
    weights = torch.full((3, 2), 0.5)
    ids = torch.tensor([[0, 1], [2, 3], [1, 0]], dtype=torch.int32)
    views = (
        torch.empty(4, 16, 980, dtype=torch.uint8),
        torch.empty(4, 16, 980, dtype=torch.uint8),
        torch.empty(4, 8, 360, dtype=torch.uint8),
    )

    out = layer._expert_gemm(
        cache, hidden, weights, ids, views=views, n=None, alphas=None, is_prefill=False
    )

    assert out.shape == (3, 8)
    assert [call[:3] for call in calls] == [(18, 16, 3), (18, 16, 3), (20, 8, 6)]
    assert all(call[3][0] == 4 for call in calls)


def test_qwen4_prefill_deduplicates_routes_before_lru(monkeypatch):
    import torch
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    layer = OffloadMoELayer(
        layer_id=0, num_experts=4, top_k=2, hidden_size=8, intermediate_size=16
    )
    raw_ids = torch.tensor([[0, 1], [0, 2], [1, 0]], dtype=torch.int32)
    weights = torch.full((3, 2), 0.5)
    calls = {}
    monkeypatch.setattr(
        "freetoken.layers.moe.fused_topk",
        lambda **_: (weights, raw_ids.clone()),
    )
    cache = type("Cache", (), {})()
    cache.quant_format = "gguf_qwen4"
    cache._pageable_gpu_layers = {0}
    cache.slot_for_id = torch.tensor([[10, 11, 12, -1]], dtype=torch.int32)
    cache.ensure_experts = lambda layer_id, ids: calls.setdefault("ids", ids.clone())
    cache.copy_missing = lambda: calls.setdefault("copied", True)
    cache.bank_views = lambda **_: ()
    cache.alphas_for_slots = lambda *_: None
    layer.offload_cache = cache

    def fake_gemm(_cache, hidden, _weights, ids, **_):
        calls["remapped"] = ids.clone()
        return hidden

    monkeypatch.setattr(
        layer,
        "_expert_gemm", fake_gemm,
    )

    out = layer.prefill_forward(torch.randn(3, 8), torch.randn(3, 4))
    assert out.shape == (3, 8)
    assert calls["ids"].tolist() == [0, 1, 2]
    assert calls["remapped"].tolist() == [[10, 11], [10, 12], [11, 10]]
    assert calls["copied"] is True


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="needs CUDA")
def test_qwen4_pageable_scatter_writes_strided_cache_tails():
    import torch

    from freetoken.moe.offload_kernels import scatter_pageable_qwen4_rows

    widths = (5, 5, 4)
    sources = tuple(
        torch.arange(2 * 3 * width, dtype=torch.uint8, device="cuda").reshape(2, 3, width)
        for width in widths
    )
    caches = (
        torch.full((8, 3, 7), 255, dtype=torch.uint8, device="cuda"),
        torch.full((8, 3, 7), 255, dtype=torch.uint8, device="cuda"),
        torch.full((8, 3, 5), 255, dtype=torch.uint8, device="cuda"),
    )
    destinations = (caches[0][..., :5], caches[1][..., :5], caches[2][..., :4])
    ids = torch.tensor([4, 6], dtype=torch.int32, device="cuda")

    scatter_pageable_qwen4_rows(sources, destinations, ids)
    torch.cuda.synchronize()

    for source, destination, width in zip(sources, destinations, widths):
        torch.testing.assert_close(destination[4], source[0])
        torch.testing.assert_close(destination[6], source[1])
        assert bool(torch.all(destination[0] == 255))
        assert destination.shape[-1] == width


@pytest.mark.needs_weights
@pytest.mark.skipif(not __import__("pathlib").Path("/home/smk/models/Qwen3.8-Flash-Next-GGUF/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf").is_file(), reason="Qwen3.8 GGUF not installed")
def test_target_source_reads_bounded_rows():
    path = "/home/smk/models/Qwen3.8-Flash-Next-GGUF/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf"
    source = Qwen4ExpPackedSource(path, cache_bytes=1 << 20)
    try:
        rows = source.read_rows("blk.2.ffn_gate_exps.weight", [0, 2, 1], device="cpu")
        assert rows.shape == (3, 980 * 1)
        assert source.report()["read_bytes"] == 3 * 980
    finally: source.close()


@pytest.mark.needs_weights
@pytest.mark.skipif(not __import__("pathlib").Path("/home/smk/models/Qwen3.8-Flash-Next-GGUF/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf").is_file(), reason="Qwen3.8 GGUF not installed")
def test_target_expert_banks_are_file_backed_and_preserve_types():
    from freetoken.models.gguf.config import build_gguf_shim
    from freetoken.models.qwen4_exp.gguf import load_gguf_expert_sources, parse_gguf_config

    path = "/home/smk/models/Qwen3.8-Flash-Next-GGUF/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf"
    config = parse_gguf_config(build_gguf_shim(path))
    banks = load_gguf_expert_sources(path, config)
    assert banks["gate"][0].shape == (512, 640, 740)
    assert banks["gate"][2].shape == (512, 640, 980)
    assert banks["up"][0].shape == (512, 640, 740)
    assert banks["down"][0].shape == (512, 2560, 360)
