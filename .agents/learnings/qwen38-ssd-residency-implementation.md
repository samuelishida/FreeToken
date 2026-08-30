# Qwen38 SSD Residency Implementation

## Context

Implemented `.plans/qwen38-ssd-residency/plan.md`: one-hour request lifecycle,
explicit Qwen expert residency, grouped ROCm MoE dispatch, bounded `.ftple`
PLE paging, cache identity, scheduler fairness, and topology reporting.

## Hardest decision

Keep legacy direct-CLI defaults compatible while making the ROCm serve script
opt into packed PLE GPU caching and grouped MoE. This preserves existing tests
and callers while production Qwen launches get the intended fast path.

## Alternatives rejected

- Silent `ram` to `mmap` fallback — violates the requested loud host-budget
  contract; use explicit `auto` or `mmap` on machines that cannot hold experts.
- Full PLE mmap/residency — defeats bounded SSD paging and makes cache pressure
  invisible; retain sidecar pages plus RAM/pinned/GPU layers.
- Device-wide synchronization for prefetch — stalls decode; late lookups join
  bounded in-flight page reads instead.

## Least confident

Live 24K/32K production throughput still needs a controlled restart with new
flags. Existing long-running PIDs predate these changes and report legacy
topology, 1800-second timeout, and no packed PLE GPU cache.

## Reuse

Read before changing `engine/engine.py`, `models/qwen4_exp/ple_gguf.py`,
`models/qwen4_exp/packed.py`, `scheduler/scheduler.py`, or
`scripts/serve-qwen38-rocm.sh`; restart the service before interpreting live
status telemetry.
