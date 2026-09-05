# ROCm reproducibility

Performance claims use same-model, same-prompt, same-token-count comparisons.
MTP/speculative decode stays off. Sampled output, greedy correctness, and
teacher-forced replay are separate lanes and cannot be combined.

Each accepted JSON manifest uses `freetoken-rocm-manifest-v1`:

```json
{
  "workload": {
    "model_sha256": "...",
    "prompt_sha256": "...",
    "token_count": 512,
    "mtp": "off",
    "flags": {}
  },
  "runtime": {
    "commit": "...",
    "dirty_diff": "...",
    "gpu": "...",
    "driver": "...",
    "torch": "...",
    "rocm": "...",
    "hip": "...",
    "triton": "...",
    "jit_sha": "...",
    "env_digest": "..."
  },
  "observed": {
    "backend": "rocm",
    "quant": "Q4_K",
    "graph_mode": "eager",
    "route": "legacy",
    "cache_hits": 0,
    "fetches": 0,
    "fallbacks": 0,
    "finite_logits": true,
    "completion_count": 512
  },
  "timing": {
    "lane": "teacher_forced_replay",
    "repeats": 3,
    "median_tok_s": 100.0,
    "spread": 2.0
  }
}
```

Create manifests with `benchmarks/bench_rocm_matrix.py`, then validate a
candidate against incumbent evidence:

```bash
PYTHONPATH=benchmarks python benchmarks/check_decode_gate.py \
  --candidate candidate.jsonl --baseline incumbent.jsonl \
  --min-runs 3 --min-gain 0.05 --json gate.json
```

Gate failure is correct when route counters, finite logits, exact completion,
full hashes, lane identity, or repeat count is missing. A direct kernel timing
does not satisfy served decode evidence.
