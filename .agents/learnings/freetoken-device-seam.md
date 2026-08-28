# freetoken-device-seam

## Context
FreeToken's engine is CUDA-only on `main`; the tinygrad path added a second
device branch (`Engine._init_tinygrad`) reusing the runner-owned seam shape
(stub KV/attn/graph, CPU sampler, runner owns model state).

## Hardest decision
The scheduler's `batch.input_ids` is the EXTEND (the prefill chunk or the
single decode token), NOT the full sequence — the runner must accumulate the
prompt in its own buffer and copy each chunk at its global position
(`cached_len`), then slice symbolically. Slicing `input_ids[cached_len:device_len]`
on a 1-token decode batch returns an empty array.

## Alternatives rejected
- None — this was discovered by debugging the serve path (empty-reshape crash).

## Least confident
- `pin_memory=True` on `torch.empty`/`torch.tensor` raises "No CUDA GPUs are
  available" on the CPU-only tinygrad path (guarded in scheduler.py
  `_make_positions`/`_make_input_tuple`/`_make_write_tuple` and sample.py
  `make_device_tensor`); `torch.tensor(pin_memory=True)` works in a fresh
  process but fails inside the server — the guard is `device.type == "cuda"`.
- The engine's `max_seq_len` must be the runner's max_context (rounded to a
  multiple of 128), not the model's native max position, or the scheduler
  accepts prompts the runner cannot hold; `/v1/models` must report the same.
- `sync_all_ranks` needs a no-op guard when `tp_cpu_group` is None (tinygrad
  path has no process group).

## Reuse
`python/freetoken/engine/engine.py` (`_init_tinygrad`), `scheduler/scheduler.py`,
`server/openai_api.py` (`_model_context_length`), `server/args.py` (`--device`).
Read before adding another non-CUDA device or touching the scheduler's
batch/position plumbing.
