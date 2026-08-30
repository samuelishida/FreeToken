from types import SimpleNamespace

from freetoken.message import StatusReply
from freetoken.server.stats import StatsTracker, build_stats


def test_status_events_are_monotonic_and_counters_cumulative():
    stats = StatsTracker()
    stats.observe_status(StatusReply(7, "forward_started", 2, 10.0, {"lookup_calls": 1}))
    stats.observe_status(StatusReply(7, "submitted", 1, 9.0, {"lookup_calls": 0}))
    stats.observe_status(StatusReply(7, "ple_lookup", 3, 11.0, {"lookup_calls": 2}))
    snap = stats.status_snapshot()
    assert snap["request_stage"] == [{"uid": 7, "stage": "ple_lookup", "seq": 3, "timestamp": 11.0, "error": None}]
    assert snap["ple_counters"]["lookup_calls"] == 2


def test_build_stats_exposes_live_status():
    config = SimpleNamespace(
        served_model_name="test", max_seq_len=128, page_size=1,
        model_config=SimpleNamespace(has_linear_attention=False, has_swa_attention=False, is_moe=False),
    )
    state = SimpleNamespace(config=config, stats=StatsTracker(), ready_at=None, gpus=[])
    state.stats.observe_status(StatusReply(1, "first_token", 4, 3.0, {"dequant_calls": 1}))
    doc = build_stats(state, 0, 0)
    assert doc["status"]["request_stage"][0]["stage"] == "first_token"
    assert doc["status"]["ple_counters"]["dequant_calls"] == 1
