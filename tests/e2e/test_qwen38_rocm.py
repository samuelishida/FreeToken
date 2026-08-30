"""Opt-in production ROCm/GGUF HTTP acceptance gate.

The server is started separately with ``scripts/serve-qwen38-rocm.sh``. CI skips this
module unless a real endpoint is explicitly selected.
"""
import json
import os
import time
import urllib.request

import pytest


BASE = os.environ.get("FREETOKEN_QWEN38_URL", "http://127.0.0.1:1922").rstrip("/")
ENABLED = bool(os.environ.get("FREETOKEN_QWEN38_ROCM_E2E"))


def _get(path: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=timeout) as response:
        return json.loads(response.read())


def _stream(prompt: str, model: str, timeout: float | None = None) -> tuple[bool, bool]:
    # Resolve from same environment used by production launcher. Keep explicit
    # override for CI smoke tests and cold-compile experiments.
    if timeout is None:
        timeout = float(os.environ.get("FREETOKEN_QWEN38_REQUEST_TIMEOUT_S", "3600"))
    payload = json.dumps({
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1, "temperature": 0, "stream": True,
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=payload,
        headers={"content-type": "application/json"}, method="POST",
    )
    first_data = first_token = False
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            first_data = True
            body = raw[6:].strip()
            if body == b"[DONE]":
                break
            event = json.loads(body)
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                if (delta.get("content") or delta.get("reasoning_content")
                        or choice.get("finish_reason") is not None):
                    first_token = True
    return first_data, first_token


@pytest.mark.needs_weights
@pytest.mark.skipif(not ENABLED, reason="set FREETOKEN_QWEN38_ROCM_E2E=1 for live ROCm acceptance")
def test_qwen38_rocm_first_token_and_live_ple_status():
    deadline = time.monotonic() + float(os.environ.get("FREETOKEN_QWEN38_STARTUP_TIMEOUT", "300"))
    health = {}
    while time.monotonic() < deadline:
        try:
            health = _get("/health")
            if health.get("status") == "ok":
                break
        except OSError:
            pass
        time.sleep(1)
    assert health.get("status") == "ok", health
    model = (_get("/v1/models").get("data") or [{}])[0].get("id")
    assert model
    cases = [1, 3, 4, 5, 1024, 2047, 2048, 2049, 2051, 24000]
    if not os.environ.get("FREETOKEN_QWEN38_LONG_CASES"):
        cases = [1]
    for target in cases:
        first_data, first_token = _stream("x " * target, model)
        assert first_data, f"no SSE data for prompt target {target}"
        assert first_token, f"no token for prompt target {target}"
    status = _get("/v1/cache/status")
    stats = _get("/v1/stats")
    assert status["state"] == "serving"
    assert (status["geometry"].get("ple_probe") or {}).get("state") in ("ok", "skipped")
    assert stats["requests"]["active"] == 0
