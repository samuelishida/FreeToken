from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.models.gguf.dequant import (
    PLE_IQ4_NL_ROW_BYTES,
    PLE_IQ4_NL_ROW_VALUES,
    dequant_iq4_nl,
)
from freetoken.models.qwen4_exp import ple_store
from freetoken.models.qwen4_exp.ple_gguf import PagePlan, PackedPagedPLETable, PackedResidentPLETable


def _fake_source(monkeypatch, tmp_path, rows=46):
    source = tmp_path / "source.gguf"
    source.write_bytes(bytes(range(256)) * ((rows * 90 + 255) // 256))
    header = SimpleNamespace(rows=rows, shard_path=str(source), data_offset=0)
    monkeypatch.setattr(ple_store, "_source", lambda _model: ((), header))
    monkeypatch.setattr(ple_store, "source_fingerprint", lambda _model: "fixture-fingerprint")
    return source


def _args(store, **overrides):
    values = dict(ple_store=str(store), ple_store_build="never", ple_ram_cache_mib=1,
                  ple_gpu_cache_mib=0, ple_staging_mib=1, ple_io="buffered",
                  ple_io_depth=2, ple_prefetch=True)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_sidecar_has_exact_45_row_pages_and_validates_after_rename(monkeypatch, tmp_path):
    source = _fake_source(monkeypatch, tmp_path)
    store = tmp_path / "model.ftple"
    ple_store.build_store(str(source), store)

    header = ple_store.validate_store(store, str(source))
    assert header["rows"] == 46
    assert header["source_kind"] == "GGUF"
    assert header["quant_type"] == "IQ4_NL"
    assert header["logical_row_width"] == 160
    assert header["packed_row_bytes"] == 90
    assert header["total_data_pages"] == 2
    assert store.stat().st_size == ple_store.HEADER_BYTES + 2 * ple_store.PAGE_BYTES
    raw = store.read_bytes()
    source_bytes = source.read_bytes()
    assert raw[ple_store.HEADER_BYTES:ple_store.HEADER_BYTES + 45 * 90] == source_bytes[:45 * 90]
    assert raw[ple_store.HEADER_BYTES + 45 * 90:ple_store.HEADER_BYTES + ple_store.PAGE_BYTES] == b"\0" * 46
    assert raw[ple_store.HEADER_BYTES + ple_store.PAGE_BYTES:ple_store.HEADER_BYTES + ple_store.PAGE_BYTES + 90] == source_bytes[45 * 90:46 * 90]


def test_sidecar_page_rows_keep_iq4_nl_geometry_and_decode(monkeypatch, tmp_path):
    source = _fake_source(monkeypatch, tmp_path)
    store = tmp_path / "model.ftple"
    ple_store.build_store(str(source), store)

    sidecar = store.read_bytes()
    # First page contains exactly 45 native rows; trailing 46 bytes are page
    # padding and must not become part of an IQ4_NL row.
    first_page = torch.tensor(
        list(sidecar[ple_store.HEADER_BYTES:ple_store.HEADER_BYTES + ple_store.PAGE_BYTES]),
        dtype=torch.uint8,
    )
    rows = first_page[:45 * PLE_IQ4_NL_ROW_BYTES].reshape(45, PLE_IQ4_NL_ROW_BYTES)
    assert rows.shape == (45, PLE_IQ4_NL_ROW_BYTES)
    assert torch.equal(rows[0], torch.tensor(list(source.read_bytes()[:90]), dtype=torch.uint8))
    assert torch.equal(
        rows[-1], torch.tensor(list(source.read_bytes()[44 * 90:45 * 90]), dtype=torch.uint8)
    )
    decoded = dequant_iq4_nl(rows[[0, 0, 44]])
    assert decoded.shape == (3, PLE_IQ4_NL_ROW_VALUES)

    second_page = sidecar[ple_store.HEADER_BYTES + ple_store.PAGE_BYTES:]
    final_row = torch.tensor(list(second_page[:PLE_IQ4_NL_ROW_BYTES]), dtype=torch.uint8)
    assert torch.equal(final_row, torch.tensor(list(source.read_bytes()[45 * 90:46 * 90]), dtype=torch.uint8))
    # A sidecar's row mapping is byte identity, including final partial page.
    assert len(second_page) == ple_store.PAGE_BYTES
    assert second_page[PLE_IQ4_NL_ROW_BYTES:] == b"\0" * (ple_store.PAGE_BYTES - PLE_IQ4_NL_ROW_BYTES)


def test_prefetch_and_late_lookup_join_one_read_per_page(monkeypatch, tmp_path):
    source = _fake_source(monkeypatch, tmp_path)
    store = tmp_path / "model.ftple"
    ple_store.build_store(str(source), store)
    table = PackedPagedPLETable(str(source), _args(store))
    monkeypatch.setattr(table, "_dequant", lambda packed, _ids, _out=None: packed.clone())
    ids = torch.tensor([[0, 45, 0]], dtype=torch.int64)
    try:
        table.prefetch(ids)
        rows = table.lookup(ids)
        assert rows.shape == (3, 90)
        assert table.report()["ssd_read_ops"] == 2
        table.lookup(ids)
        report = table.report()
        assert report["ssd_read_ops"] == 2
        assert report["ram_page_hits"] >= 2
        assert "prefetch_plan_us" in report
        assert "ram_gather_us" in report
    finally:
        table.close()


def test_page_plan_stably_deduplicates_rows_groups_pages_and_restores_order():
    plan = PagePlan.build([45, 0, 45, 44, 90, 0])
    assert plan.unique_rows == (45, 0, 44, 90)
    assert plan.inverse.tolist() == [0, 1, 0, 2, 3, 1]
    assert plan.pages == ((1, ((0, 0),)), (0, ((1, 0), (2, 44))), (2, ((3, 0),)))
    assert plan.sorted_pages == (0, 1, 2)
    assert plan.page_spans == ((0, 2),)


def test_prefetch_accepts_multiple_batches_with_bounded_coordinator(monkeypatch, tmp_path):
    source = _fake_source(monkeypatch, tmp_path, rows=136)
    store = tmp_path / "model.ftple"
    ple_store.build_store(str(source), store)
    table = PackedPagedPLETable(str(source), _args(store, ple_io_depth=2))
    monkeypatch.setattr(table, "_dequant", lambda packed, _ids, _out=None: packed.clone())
    first = torch.tensor([[0]], dtype=torch.int64)
    second = torch.tensor([[45]], dtype=torch.int64)
    try:
        table.prefetch(first)
        table.prefetch(second)
        # Lookup joins page futures, rather than waiting for whole prefetch
        # batches; both accepted batches still issue one read each.
        assert table.lookup(first).shape == (1, 90)
        assert table.lookup(second).shape == (1, 90)
        report = table.report()
        assert report["prefetch_requests"] == 2
        assert report["prefetch_queue_capacity"] == 2
        assert report["prefetch_workers"] == 2
        assert report["ssd_read_ops"] == 2
    finally:
        table.close()


def test_gpu_cache_defaults_on_for_nonzero_budget_and_zero_is_valid(monkeypatch, tmp_path):
    source = _fake_source(monkeypatch, tmp_path)
    store = tmp_path / "model.ftple"
    ple_store.build_store(str(source), store)
    enabled = PackedPagedPLETable(str(source), _args(store, ple_gpu_cache_mib=1))
    disabled = PackedPagedPLETable(str(source), _args(store, ple_gpu_cache_mib=0))
    try:
        assert enabled._batched_cache is True
        assert enabled.report()["gpu_budget_bytes"] > 0
        assert disabled._batched_cache is True
        assert disabled.report()["gpu_budget_bytes"] == 0
    finally:
        enabled.close()
        disabled.close()


def test_2q_promotes_demand_hit_and_evicts_probationary_first(monkeypatch, tmp_path):
    source = _fake_source(monkeypatch, tmp_path)
    store = tmp_path / "model.ftple"
    ple_store.build_store(str(source), store)
    table = PackedPagedPLETable(str(source), _args(store, ple_cache_policy="2q"))
    value = torch.zeros(ple_store.PAGE_BYTES, dtype=torch.uint8)
    try:
        table.capacity = 2
        with table._cache_lock:
            table._prefetch_pages.add(0)
            table._admit_page_locked(0, value)
            assert 0 in table._probationary
            table._admit_page_locked(1, value)
            table._touch_page_locked(0, demand=True)
            assert 0 in table._protected
            table._admit_page_locked(2, value)
            assert set(table._pages) == {0, 2}
            assert 1 not in table._pages
    finally:
        table.close()


def test_prefetch_flag_disables_submission(monkeypatch, tmp_path):
    source = _fake_source(monkeypatch, tmp_path)
    store = tmp_path / "model.ftple"
    ple_store.build_store(str(source), store)
    table = PackedPagedPLETable(str(source), _args(store, ple_prefetch=False))
    try:
        table.prefetch(torch.tensor([[0]], dtype=torch.int64))
        assert table.report()["prefetch_requests"] == 0
    finally:
        table.close()


def test_resident_and_paged_keep_identical_packed_rows(monkeypatch, tmp_path):
    source = _fake_source(monkeypatch, tmp_path)
    store = tmp_path / "model.ftple"
    ple_store.build_store(str(source), store)
    args = _args(store, ple_ram_cache_mib=1)
    paged = PackedPagedPLETable(str(source), args)
    resident = PackedResidentPLETable(str(source), args)
    ids = [0, 44, 45]
    try:
        assert torch.equal(paged._packed_cpu(ids), resident._packed_cpu(ids))
    finally:
        paged.close()
        resident.close()
