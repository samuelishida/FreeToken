# tinygrad-runner-jit

## Context
FreeToken's AMD path (`--device tinygrad`) wraps the tinygrad fork's
`Transformer` in a `TinygradModelRunner` (engine/tinygrad_runner.py). The
runner maps FreeToken batches to `model.logits()` calls with two TinyJit
specializations.

## Hardest decision
The runner must mirror `Transformer.generate()`'s JIT pattern exactly:
`start_pos` and the prefill token count are UOp variables bound at the CALL
SITE (outside the JIT'd function), or the JIT's variable bookkeeping breaks on
the second decode call (`TypeError: _f() missing 1 required positional
argument: 'start_pos'`). Also: `Transformer.forward` returns a SAMPLED TOKEN,
not logits — the runner needs a public `logits(tokens, start_pos)` method
(added to the fork as `_forward_hidden(tokens, start_pos, Tensor([1.0]))[2]`).

## Alternatives rejected
- Calling `_forward_hidden(...)[2]` directly — private method, fragile; a
  public `logits()` in the fork is cleaner.
- Padding prefill chunks to a multiple of 32 — pollutes the KV/attention; the
  symbolic-slice approach (persistent buffer + UOp variable) lets the AMD
  flash kernel pad the query tile internally and slice garbage rows off.
- One JIT for prefill and decode — the decode call (n_toks=1) breaks the
  captured var_vals; two JITs (like the model's prefill_jit/rollout_jit) are
  required.
- Branching from `main` — main's tokenizer/GGUF loader doesn't support
  qwen35moe (KeyError); branch from `feat/amd-rocm-gfx1100-support`.

## Least confident
- The warmup runs each JIT twice (TinyJit: 1st call eager, 2nd capture, 3rd
  exec) with a 256-token chunk, but a prompt whose LAST chunk is not 256
  tokens (e.g. 44) recompiles the prefill JIT for that shape (~25 s one-time).
- The persistent prompt buffer is reused across requests; a fresh request with
  a SHORTER prompt leaves stale tokens beyond its length (harmless — never
  read — but verify with a long-then-short sequence).
- VRAM at 128K context (fp16 KV ~2.7 GB + 22 GB weights) not measured; the
  serve script defaults to 32K conservatively.

## Reuse
`python/freetoken/engine/tinygrad_runner.py`, the fork's
`tinygrad/llm/model.py` (logits method), and any future tinygrad-backed
engine. Read before touching the runner's JIT/warmup or the fork's model API.
