"""Teacher-forced replay identity helpers for ROCm comparisons.

Serving adapters may differ, but replay rows must carry same prompt/continuation identity and
route evidence. Sampled streams never enter this lane.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

LANE = "teacher_forced_replay"


def ids_sha256(ids: Iterable[int]) -> str:
    return hashlib.sha256(json.dumps([int(value) for value in ids], separators=(",", ":")).encode()).hexdigest()


def build_replay_record(
    manifest: dict[str, Any],
    *,
    prompt_ids: Iterable[int],
    continuation_ids: Iterable[int],
    route_digest: str,
    route_hash_status: str = "matched",
) -> dict[str, Any]:
    """Attach forced-ID and route identity to one already validated timing manifest."""
    prompt = [int(value) for value in prompt_ids]
    continuation = [int(value) for value in continuation_ids]
    if not prompt or not continuation:
        raise ValueError("replay prompt and continuation IDs must be non-empty")
    if not route_digest:
        raise ValueError("replay route digest is required")
    result = dict(manifest)
    result["timing"] = {**dict(manifest.get("timing", {})), "lane": LANE}
    result["replay"] = {
        "forced": True,
        "prompt_ids_sha256": ids_sha256(prompt),
        "continuation_ids_sha256": ids_sha256(continuation),
        "route_digest": route_digest,
        "route_hash_status": route_hash_status,
    }
    return result


def validate_replay_record(value: object) -> list[str]:
    if not isinstance(value, dict):
        return ["replay record is not an object"]
    replay = value.get("replay")
    timing = value.get("timing")
    problems: list[str] = []
    if not isinstance(timing, dict) or timing.get("lane") != LANE:
        problems.append("timing.lane must be teacher_forced_replay")
    if not isinstance(replay, dict) or replay.get("forced") is not True:
        problems.append("replay.forced must be true")
    if not isinstance(replay, dict) or not replay.get("prompt_ids_sha256"):
        problems.append("replay prompt identity is missing")
    if not isinstance(replay, dict) or not replay.get("continuation_ids_sha256"):
        problems.append("replay continuation identity is missing")
    if not isinstance(replay, dict) or replay.get("route_hash_status") != "matched":
        problems.append("replay route hashes are not matched")
    return problems
