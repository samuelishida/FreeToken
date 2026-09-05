"""Reject incomplete or incomparable ROCm A/B evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from bench_decode_replay import validate_replay_record
from bench_rocm_matrix import load_manifests, validate_manifest


def evaluate_gate(
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    *,
    min_runs: int = 3,
    min_gain: float = 0.05,
) -> dict[str, Any]:
    reasons: list[str] = []
    for label, rows in (("candidate", candidate), ("baseline", baseline)):
        for index, row in enumerate(rows):
            reasons.extend(f"{label}[{index}]: {reason}" for reason in validate_manifest(row))
            timing = row.get("timing")
            if isinstance(timing, dict) and timing.get("lane") == "teacher_forced_replay":
                reasons.extend(
                    f"{label}[{index}]: {reason}" for reason in validate_replay_record(row)
                )
    if len(candidate) < min_runs:
        reasons.append(f"candidate needs >= {min_runs} runs, got {len(candidate)}")
    if len(baseline) < min_runs:
        reasons.append(f"baseline needs >= {min_runs} runs, got {len(baseline)}")

    def identity(row: dict[str, Any]) -> tuple[Any, ...]:
        work = row.get("workload", {})
        observed = row.get("observed", {})
        timing = row.get("timing", {})
        return (
            work.get("model_sha256"), work.get("prompt_sha256"), work.get("token_count"),
            work.get("mtp"), json.dumps(work.get("flags"), sort_keys=True), observed.get("quant"),
            observed.get("graph_mode"), timing.get("lane"),
        )

    all_rows = candidate + baseline
    identities = {identity(row) for row in all_rows if isinstance(row, dict)}
    if len(identities) > 1:
        reasons.append("candidate and baseline do not share one workload/quant/graph/lane identity")
    candidate_speeds = [row.get("timing", {}).get("median_tok_s") for row in candidate]
    baseline_speeds = [row.get("timing", {}).get("median_tok_s") for row in baseline]
    candidate_speeds = [float(value) for value in candidate_speeds if isinstance(value, (int, float))]
    baseline_speeds = [float(value) for value in baseline_speeds if isinstance(value, (int, float))]
    candidate_median = statistics.median(candidate_speeds) if candidate_speeds else None
    baseline_median = statistics.median(baseline_speeds) if baseline_speeds else None
    gain = (
        (candidate_median / baseline_median) - 1.0
        if candidate_median is not None and baseline_median not in (None, 0)
        else None
    )
    if gain is None or gain < min_gain:
        reasons.append(
            f"candidate gain {gain if gain is not None else 'unavailable'} is below {min_gain:.1%}"
        )
    result = {
        "schema": "freetoken-rocm-gate-v1",
        "gate": not reasons,
        "min_runs": min_runs,
        "min_gain": min_gain,
        "candidate_runs": len(candidate),
        "baseline_runs": len(baseline),
        "candidate_median_tok_s": candidate_median,
        "baseline_median_tok_s": baseline_median,
        "gain": gain,
        "reasons": reasons,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--min-runs", type=int, default=3)
    parser.add_argument("--min-gain", type=float, default=0.05)
    parser.add_argument("--json", dest="json_out")
    args = parser.parse_args(argv)
    try:
        result = evaluate_gate(
            load_manifests(args.candidate), load_manifests(args.baseline),
            min_runs=args.min_runs, min_gain=args.min_gain,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        result = {"schema": "freetoken-rocm-gate-v1", "gate": False, "reasons": [str(exc)]}
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    print(encoded)
    if args.json_out:
        Path(args.json_out).write_text(encoded + "\n", encoding="utf-8")
    return 0 if result["gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
