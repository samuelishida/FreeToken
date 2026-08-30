#!/usr/bin/env python3
"""Benchmark Qwen3.8 GGUF through FreeToken's production HTTP Engine.

The server must already be running with ``scripts/serve-qwen38-rocm.sh``. This
keeps results on standard ROCm Engine path: no native wrapper or direct model call.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


# Keep 24K in default matrix: it catches long-prefill regressions while still
# leaving 32K as a separate upper-context gate.
TARGETS = {1024: 20.0, 24000: 15.0, 32768: 15.0}


def _get(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def _prompt_for_target(tokenizer, target: int, *, seed: int = 0) -> tuple[str, int]:
    """Build deterministic non-repeating text near target tokens.

    Repeating one token produces artificial radix-prefix reuse between chunks and
    hides cold PLE/QSA work. Hex sequence terms keep each token window distinct
    while remaining reproducible across cold/warm/thrash runs.
    """
    def text_for(terms: int) -> str:
        return " ".join(f"ftbench_{seed:08x}_{index:08x}" for index in range(terms))

    lo, hi = 1, max(2, target * 2)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tokenizer.encode(text_for(mid), add_special_tokens=False)) <= target:
            lo = mid
        else:
            hi = mid - 1
    text = text_for(lo)
    return text, len(tokenizer.encode(text, add_special_tokens=False))


def _stream_case(url: str, model: str, prompt: str, total_tokens: int, timeout: float) -> dict:
    payload = {
        # FreeToken reserves one scheduler slot for the terminal sample; request one
        # extra so returned token count reaches requested measurement length.
        "model": model, "prompt": prompt, "max_tokens": total_tokens + 1,
        "temperature": 0.0, "stream": True,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        url + "/v1/completions", data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"}, method="POST",
    )
    received: list[float] = []
    usage: dict = {}
    started = time.perf_counter()
    first_data_s = None
    first_token_s = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                if not raw.startswith(b"data: "):
                    continue
                if first_data_s is None:
                    first_data_s = time.perf_counter() - started
                body = raw[6:].strip()
                if body == b"[DONE]":
                    break
                event = json.loads(body)
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                # Heartbeat chunks intentionally carry an empty text field and
                # finish_reason=None. They are not generated tokens; counting
                # them made 15-second keepalives look like decode throughput.
                # Control tokens can also detokenize empty, so final usage remains
                # authoritative for completion_tokens.
                if choices and choices[0].get("finish_reason") is None and choices[0].get("text", ""):
                    if first_token_s is None:
                        first_token_s = time.perf_counter() - started
                    received.append(time.perf_counter())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"benchmark request failed ({exc.code}): {exc.read().decode(errors='replace')}") from exc
    completion_count = int(usage.get("completion_tokens") or len(received))
    if completion_count < total_tokens:
        raise RuntimeError(f"server returned {completion_count} tokens, expected {total_tokens}")
    # Eight warmup tokens; measured intervals are steady decode steps.
    samples = [received[i] - received[i - 1] for i in range(9, len(received))]
    median = statistics.median(samples)
    return {
        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": completion_count,
        "warmup_tokens": 8, "samples": len(samples), "median_ms": median * 1000.0,
        "p95_ms": sorted(samples)[max(0, int(len(samples) * 0.95) - 1)] * 1000.0,
        "median_tok_s": 1.0 / median, "wall_s": time.perf_counter() - started,
        "first_data_s": first_data_s, "first_token_s": first_token_s,
        "finite": True,
    }


def _counter_delta(before: dict, after: dict) -> dict:
    """Monotonic integer delta; topology strings stay in the surrounding snapshot."""
    result = {}
    for key, value in after.items():
        if isinstance(value, int) and isinstance(before.get(key), int):
            result[key] = value - before[key]
    return result


def _ple_counters(status: dict) -> dict:
    return dict(status.get("geometry", {}).get("ple_counters", {}) or {})


def _prefill_case(base: str, model: str, prompt: str, timeout: float) -> dict:
    """One-token stream used to measure TTFT and capture live PLE progress snapshots."""
    before = _get(base + "/v1/cache/status", 15.0)
    payload = {
        "model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0.0,
        "stream": True, "ignore_eos": True, "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        base + "/v1/completions", data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"}, method="POST",
    )
    started = time.perf_counter()
    first_data_s = first_token_s = None
    progress = []
    usage = {}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                if not raw.startswith(b"data: "):
                    continue
                now = time.perf_counter()
                if first_data_s is None:
                    first_data_s = now - started
                body = raw[6:].strip()
                if body == b"[DONE]":
                    break
                event = json.loads(body)
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if (choices and choices[0].get("finish_reason") is None
                        and choices[0].get("text", "") and first_token_s is None):
                    first_token_s = now - started
                # Heartbeats arrive during long prefill. Poll status at each SSE data event;
                # this stays outside serving process and cannot alter scheduler timing.
                progress.append({"elapsed_s": now - started, "ple": _ple_counters(_get(base + "/v1/cache/status", 15.0))})
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"prefill benchmark request failed ({exc.code}): {exc.read().decode(errors='replace')}") from exc
    if first_token_s is None:
        raise RuntimeError("prefill benchmark did not receive generated token")
    after = _get(base + "/v1/cache/status", 15.0)
    prompt_tokens = usage.get("prompt_tokens")
    return {
        "prompt_tokens": prompt_tokens, "ttft_s": first_token_s,
        "first_data_s": first_data_s, "prefill_tok_s": (prompt_tokens / first_token_s if prompt_tokens else None),
        "wall_s": time.perf_counter() - started,
        "ple_counter_delta": _counter_delta(_ple_counters(before), _ple_counters(after)),
        "progress": progress, "before": before.get("geometry", {}).get("storage", {}),
        "after": after.get("geometry", {}).get("storage", {}), "finite": True,
    }


def run_prefill_case(base: str, model: str, tokenizer, context: int, samples: int, timeout: float) -> dict:
    # Context-specific namespace prevents radix-prefix reuse across matrix cells.
    prompt, local_tokens = _prompt_for_target(tokenizer, context, seed=context)
    runs = [_prefill_case(base, model, prompt, timeout) for _ in range(samples)]
    ttft = [run["ttft_s"] for run in runs]
    return {
        "context_target": context, "prompt_tokens_local": local_tokens, "runs": runs,
        "ttft_p50_s": statistics.median(ttft),
        "ttft_p95_s": sorted(ttft)[max(0, int(len(ttft) * .95) - 1)],
        "prefill_tok_s_p50": statistics.median(
            run["prefill_tok_s"] for run in runs if run["prefill_tok_s"] is not None
        ),
    }


def run_case(base: str, model: str, tokenizer, context: int, samples: int, timeout: float) -> dict:
    if context not in TARGETS:
        raise ValueError(f"unsupported benchmark context {context}; use {sorted(TARGETS)}")
    # Include one interval before first measured sample: warmup + samples + 1
    # generated events are needed to time ``samples`` decode steps.
    # Leave one context slot beyond warmup + measured events. The scheduler reserves
    # one terminal sample, so an exact max-context request can return one fewer event.
    prompt_target = context - samples - 10
    if prompt_target < 1:
        raise ValueError("context must leave room for warmup and measured output")
    prompt, prompt_tokens = _prompt_for_target(tokenizer, prompt_target, seed=context)
    result = _stream_case(base, model, prompt, samples + 9, timeout)
    result.update({"context_target": context, "prompt_target": prompt_target,
                   "prompt_tokens_local": prompt_tokens, "target_tok_s": TARGETS[context]})
    return result


def prerequisites(base: str, model: str, model_path: str) -> tuple[dict, str]:
    card = _get(base + "/v1/models", 15.0)
    entries = card.get("data", [])
    names = [item.get("id") for item in entries]
    selected = model
    if selected not in names:
        selected = next(
            (item.get("id") for item in entries if item.get("root") == model_path),
            selected,
        )
    if selected not in names and len(names) == 1:
        selected = names[0]
    return ({"model": selected, "requested_model": model,
             "served_model_present": selected in names, "url": base,
             "platform": platform.platform(), "backend": "standard-http-engine"}, selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("FT_QWEN38_URL", "http://127.0.0.1:1922"))
    parser.add_argument("--model", default=os.environ.get("FT_QWEN38_SERVED_MODEL"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--ctx", default="1024,24000,32768")
    parser.add_argument("--prefill-ctx", default="", help="comma-separated prompt lengths for TTFT/prefill evidence")
    parser.add_argument("--prefill-samples", type=int, default=1)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--expert-host-cache-mib", type=int, default=None)
    parser.add_argument("--ple-page-cache-mib", type=int, default=None)
    parser.add_argument("--ple-row-cache-mib", type=int, default=None)
    parser.add_argument("--prefetch-depth", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=None)
    # Must match serve's resolved one-hour request deadline. Operators can use
    # a shorter value for local smoke tests, but never inherit stale 30-minute
    # watchdog behavior from older benchmark revisions.
    parser.add_argument(
        "--timeout", type=float,
        default=float(os.environ.get("FT_QWEN38_REQUEST_TIMEOUT_S", "3600")),
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    from freetoken.utils import load_tokenizer

    base = args.url.rstrip("/")
    # Tokenizer registration logs during construction. Keep benchmark stdout a
    # machine-readable JSON document; operators can still see setup logs on stderr.
    with contextlib.redirect_stdout(sys.stderr):
        tokenizer = load_tokenizer(args.model_path)
    requested_model = args.model or Path(args.model_path).name
    prereq, model = prerequisites(base, requested_model, args.model_path)
    report = {
        "format": "qwen38-rocm-http-bench-v4", "prerequisites": prereq,
        "cases": [], "prefill_cases": [],
        "requested_controls": {
            key: value for key, value in {
                "expert_host_cache_mib": args.expert_host_cache_mib,
                "ple_page_cache_mib": args.ple_page_cache_mib,
                "ple_row_cache_mib": args.ple_row_cache_mib,
                "prefetch_depth": args.prefetch_depth, "chunk": args.chunk,
            }.items() if value is not None
        },
    }
    for context in (int(value) for value in args.ctx.split(",")):
        report["cases"].append(run_case(base, model, tokenizer, context, args.samples, args.timeout))
    if args.prefill_samples < 1:
        raise ValueError("--prefill-samples must be >= 1")
    for context in (int(value) for value in args.prefill_ctx.split(",") if value):
        report["prefill_cases"].append(
            run_prefill_case(base, model, tokenizer, context, args.prefill_samples, args.timeout)
        )
    # Keep benchmark evidence self-contained: these control-plane reads are
    # non-blocking and expose actual VRAM/cache geometry after measured cases.
    for endpoint in ("/v1/cache/status", "/v1/stats"):
        try:
            report[endpoint.rsplit("/", 1)[-1].replace("/", "_")] = _get(base + endpoint, 15.0)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            report.setdefault("runtime_errors", []).append(f"{endpoint}: {exc}")
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
    print(encoded, end="")
    if args.strict:
        if not report["prerequisites"]["served_model_present"]:
            raise SystemExit("strict benchmark prerequisite failed: served model missing")
        if any(case["median_tok_s"] < case["target_tok_s"] for case in report["cases"]):
            raise SystemExit("strict benchmark throughput gate failed")


if __name__ == "__main__":
    main()
