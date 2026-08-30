from types import SimpleNamespace

from freetoken.message import StatusReply
from freetoken.server.api_server import cache_geometry
from freetoken.server.stats import StatsTracker


def test_cache_geometry_merges_live_ple_report_without_claiming_dense_source():
    config = SimpleNamespace(
        served_model_name="qwen", max_seq_len=128, page_size=1, moe_cache_size=0,
        moe_cache_rate=None, moe_cache_policy="lru",
        model_config=SimpleNamespace(
            has_linear_attention=False, has_swa_attention=False, is_moe=False,
            num_experts=0, num_moe_layers=0, dsv4_args=None,
        ),
    )
    state = SimpleNamespace(
        config=config, stats=StatsTracker(), last_rebuild=None, cache_pools={"num_pages": 8},
        unit_bytes={}, cache_budget_bytes=0, storage_report={"ple": {
            "mode": "paged", "backend": "ftple-paged", "num_rows": 45,
            "ram_budget_bytes": 4096, "gpu_budget_bytes": 96, "staging_budget_bytes": 96,
            "cache_policy": "2q", "batched_cache": True, "fused_dequant": True,
            "ssd_read_ops": 1, "prefetch_plan_us": 4, "qsa_allocated_columns": 256,
        }}, gpus=[], free_vram_bytes=0, cache_floors=None,
        ple_probe={"state": "ok"}, ple_probe_timeout_s=300.0, engine=None,
        swa_full_tokens_ratio=0.0,
    )
    state.stats.observe_status(StatusReply(2, "dequant_started", 1, 1.0, {
        "ssd_read_ops": 3, "dequant_calls": 1, "prefetch_plan_us": 9,
        "qsa_allocated_columns": 512,
    }))
    geo = cache_geometry(state)
    assert geo["ple_mode"] == "paged"
    assert geo["ple_counters"]["ssd_read_ops"] == 3
    assert geo["ple_counters"]["prefetch_plan_us"] == 9
    assert geo["ple_counters"]["qsa_allocated_columns"] == 512
    assert geo["ple_cache_policy"] == "2q"
    assert geo["ple_batched_cache"] and geo["ple_fused_dequant"]
    assert geo["ple_probe"]["state"] == "ok"
    assert geo["storage"]["ple"]["mode"] == "paged"
