import sys
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

from profile_decode_rocm import build_report, load_trace  # noqa: E402


def _payload():
    return {
        "clock_correlations": [{"host_ns": 0, "device_ns": 0}],
        "observed": {"route": "legacy", "graph_mode": "eager"},
        "lane": "teacher_forced_replay",
        "tokens": [{"token": 1, "start_ns": 0, "end_ns": 100}],
        "kernels": [
            {"kind": "kernel", "name": "decode", "start_ns": 10, "end_ns": 60},
            {"kind": "kernel", "name": "ensure_experts", "start_ns": 20, "end_ns": 40},
        ],
        "copies": [{"kind": "copy", "start_ns": 50, "end_ns": 80}],
    }


def test_profile_report_keeps_disjoint_gpu_ledger_and_route_identity():
    report = build_report(_payload())
    assert report["status"] == "complete"
    assert report["observed"]["route"] == "legacy"
    assert report["ledgers"][0]["gpu_ns"] == 70
    assert report["ledgers"][0]["unattributed_ns"] == 30
    assert report["warm_offload"]["status"] == "measured"


def test_profile_report_marks_missing_trace_evidence_incomplete():
    payload = _payload()
    payload.pop("clock_correlations")
    payload["observed"].pop("route")
    report = build_report(payload)
    assert report["status"] == "incomplete"
    assert "clock_correlations" in report["missing"]
    assert "observed.route" in report["missing"]


def test_load_trace_accepts_event_lists(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text("[{\"kind\": \"kernel\"}]", encoding="utf-8")
    assert load_trace(path) == {"events": [{"kind": "kernel"}]}


@pytest.mark.parametrize("value", [None, "bad", 3])
def test_load_trace_rejects_non_objects(tmp_path, value):
    path = tmp_path / "trace.json"
    import json

    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="object or event list"):
        load_trace(path)
