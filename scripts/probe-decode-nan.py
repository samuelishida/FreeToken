"""Decode-NaN diagnosis probe (decode-nan-captured-jit plan, Inc 1).

Reproduces the long-context decode failure on the tinygrad path and classifies
it: the runner's captured decode JIT produces 100% NaN logits at large
start_pos while a fresh eager graph at the same position is finite
(.plans/decode-nan-captured-jit/evidence.md). Two modes:

  --mode capture-diag  (default)  capture a stats-instrumented TinyJit at the
                      runner's warmup pattern (BEFORE the prefill — the capture
                      call executes the graph), then prefill n-prompt tokens and
                      exec the captured graph at the large position. The lazy
                      per-subpart stats (post-attention / post-FFN / post-layer,
                      one row each) attribute the FIRST NaN to a block class and
                      sub-component in ONE GPU pass.
  --mode ab           alternate fresh-eager and runner-jit decode steps; reports
                      finiteness deltas between the two paths.

Usage:
  .venv-rocm/bin/python scripts/probe-decode-nan.py --model <gguf> [--n-prompt 16384] [--mode capture-diag]
"""
import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "python")
torch.set_grad_enabled(False)

from freetoken.core import Batch, Req, SamplingParams  # noqa: E402
from freetoken.distributed import DistributedInfo  # noqa: E402
from freetoken.engine import Engine, EngineConfig  # noqa: E402


def report(tag, lg):
    fin = np.isfinite(lg)
    mn = f"{lg[fin].min():.2f}" if fin.any() else "-"
    mx = f"{lg[fin].max():.2f}" if fin.any() else "-"
    print(f"  {tag} finite%={100.0 * fin.mean():.2f} nan={int(np.isnan(lg).sum())} "
          f"inf={int(np.isinf(lg).sum())} min={mn} max={mx} argmax={int(lg[0].argmax())}",
          flush=True)
    return fin.all()


def forward(eng, full_ids, extend, cached_len, phase, tag):
    """One runner forward through the engine (uses the runner's captured jits)."""
    req = Req(
        input_ids=full_ids, table_idx=0, cached_len=cached_len,
        output_len=max(1, len(extend)), uid=0,
        sampling_params=SamplingParams(
            temperature=1.0, top_k=20, top_p=0.95),  # the server default
        cache_handle=None,
    )
    batch = Batch(reqs=[req], phase=phase)
    batch.input_ids = extend
    batch.positions = torch.arange(cached_len, cached_len + len(extend), dtype=torch.int32)
    batch.padded_reqs = batch.reqs
    sample_args = eng.sampler.prepare(batch)
    lg = eng.tinygrad_runner.forward(batch)
    lg_np = lg.detach().numpy().astype(np.float32)
    ok = report(tag, lg_np)
    if not ok:
        return 0
    nxt = eng.sampler.sample_cpu(lg, sample_args, batch)
    return int(nxt[0].item())


def prefill_toks(eng, ids):
    """Chunked 256-token real prefill; returns elapsed seconds."""
    cached, t1 = 0, time.monotonic()
    while cached < len(ids):
        e = min(cached + 256, len(ids))
        forward(eng, ids[:e], ids[cached:e], cached, "prefill", f"prefill-{e}")
        if e % 8192 == 0 or e == len(ids):
            print(f"  prefill cached={e} el={time.monotonic() - t1:.0f}s", flush=True)
        cached = e
    return time.monotonic() - t1


def _tag(i: int, n_blocks: int, per_block: int = 1) -> str:
    if per_block == 3:  # post-attention, post-FFN, post-layer
        b, part = i // 3, i % 3
        return f"L{b}-{'attn' if part == 0 else ('ffn' if part == 1 else 'out')}"
    return f"L{i}"


def capture_diag(eng, n_prompt, args):
    """Stats-instrumented capture at the runner's warmup pattern, exec at large sp."""
    from tinygrad.engine.jit import TinyJit
    m = eng.tinygrad_runner.model
    rn = eng.tinygrad_runner

    # 1. capture (BEFORE prefill — the capture call executes; the state garbage it
    #    writes is fully inside the region the real prefill overwrites, so keep
    #    capture_sp < n_prompt).
    capture_sp = args.capture_sp
    m._per_layer_debug = True
    if args.sub_stats:
        m._sub_stats = True
    j = TinyJit(lambda t, sp: m.logits(t, sp))

    def _call(sp):
        return j(rn._Tensor(np.array([[11]], dtype=np.int32)), rn._v_sp.bind(sp))

    _call(capture_sp)  # eager
    _call(capture_sp)  # capture
    assert j.captured is not None, "diag jit failed to capture"
    print(f"diag jit captured at sp={capture_sp} (before prefill)", flush=True)
    m._per_layer_debug = False

    # 2. real prefill rebuilds the state the capture call clobbered
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1000, 90000, (n_prompt,), dtype=torch.int32, generator=g)
    prefill_toks(eng, ids)

    # 3. exec the captured graph at the large position: same pattern as the runner
    print(f"exec captured diag jit at sp={n_prompt + 1}", flush=True)
    m._per_layer_debug = True
    lg = _call(n_prompt + 1)
    m._per_layer_debug = False
    st = None
    stats = getattr(m, "_dbg_stats", None)
    if stats:
        from tinygrad import Tensor
        st = Tensor.stack(*stats)
        Tensor.realize(lg, st)
        st_np = st.numpy().astype(np.float32)
        first_bad = None
        for i in range(st_np.shape[0]):
            nnan, ninf, amax = int(st_np[i, 0]), int(st_np[i, 1]), float(st_np[i, 2])
            bad = nnan or ninf
            if bad and first_bad is None:
                first_bad = i
            if bad or (first_bad is not None and i // 3 <= first_bad // 3):
                print(f"  {_tag(i, len(m.blk))} nan={nnan} inf={ninf} absmax={amax:.1f}"
                      f"{' <<< FIRST BAD' if i == first_bad else ''}", flush=True)
        print(f"first_bad_stat={first_bad} -> {_tag(first_bad, len(m.blk)) if first_bad is not None else 'NONE (all finite)'}",
              flush=True)
        cls = [type(b).__name__ for b in m.blk]
        if first_bad is not None:
            print(f"first_bad block class={cls[first_bad // 3]}", flush=True)
    lg_np = lg.numpy().astype(np.float32)
    report(f"captured-exec@{n_prompt + 1}", lg_np)
    if not np.isfinite(lg_np).all():
        dump = f"{args.out_dir}/nan-logits-captured-{n_prompt}.npy"
        np.save(dump, lg_np)
        print(f"NON-FINITE captured-exec at {n_prompt + 1}; dumped {dump}",
              flush=True)
        print("DIAGNOSIS: captured-exec reproduces NaN (see stats above).", flush=True)
    else:
        print("DIAGNOSIS: captured-exec FINITE — captured-vs-eager delta (Inc 2 gate) passed upstream.",
              flush=True)


def ab(eng, n_prompt, n_gen):
    """Alternate fresh-eager (even steps) and runner-captured (odd steps) decodes."""
    from tinygrad.engine.jit import TinyJit
    m = eng.tinygrad_runner.model
    rn = eng.tinygrad_runner
    ids = torch.randint(1000, 90000, (n_prompt,), dtype=torch.int32,
                        generator=torch.Generator().manual_seed(0))
    prefill_toks(eng, ids)
    gen: list[int] = []
    finite = []
    for i in range(n_gen):
        tok_in = gen[-1] if gen else 11
        cached = len(ids) + len(gen) - 1
        if i % 2 == 0:  # fresh-eager pass (TinyJit #1 = eager, var binding resolved)
            j = TinyJit(lambda t, sp: m.logits(t, sp))
            lg = j(rn._Tensor(np.array([[tok_in]], dtype=np.int32)), rn._v_sp.bind(cached))
            lg = lg.realize().numpy().astype(np.float32)
            ok = report(f"fresh-eager sp={cached}", lg)
            finite.append(ok)
            tok = int(lg[0].argmax())
        else:           # the runner's captured decode JIT
            full = torch.cat([ids, torch.tensor(gen, dtype=torch.int32)])
            tok = forward(eng, full, torch.tensor([tok_in], dtype=torch.int32), cached,
                          "decode", f"runner-jit sp={cached}")
            finite.append(False)
        gen.append(tok)
    print(f"gen: {gen[:16]}", flush=True)
    print(f"finiteness by step {[(i, f) for i, f in enumerate(finite)]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-prompt", type=int, default=16384)
    ap.add_argument("--n-gen", type=int, default=8)
    ap.add_argument("--mode", choices=["capture-diag", "ab"], default="capture-diag")
    ap.add_argument("--capture-sp", type=int, default=256,
                    help="the position the diag jit is captured at (before prefill). "
                         "M1 test: capture at a LARGE sp — if exec at runtime sp is then "
                         "finite, the captured graph has capture-position-dependent baking.")
    ap.add_argument("--sub-stats", action="store_true",
                    help="also collect post-attn/post-FFN sub stats (bigger capture graph; "
                         "default OFF after two machine freezes — try per-layer first)")
    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--variant", choices=["fused", "moe-generic", "moe-tensorq8", "flash-off"], default="fused",
                    help="kernel-substitution diagnostic: route one component to its "
                         "generic path in the capture+exec (fused = no substitution).")
    ap.add_argument("--out-dir", default="probe-logs",
                    help="Persistent dir for logs/dumps (NOT /tmp — /tmp is wiped on reboot).")
    a = ap.parse_args()
    import os
    os.makedirs(a.out_dir, exist_ok=True)

    ml = a.max_seq_len or ((a.n_prompt + a.n_gen + 1024 + 127) // 128) * 128
    eng = Engine(EngineConfig(
        model_path=a.model, tp_info=DistributedInfo(0, 1), dtype=torch.float16,
        device="tinygrad", max_running_req=1, max_seq_len_override=ml))
    if a.variant != "fused":
        import tinygrad.llm.kernels.amd as amod
        import tinygrad.llm.model as gmod
        for b in eng.tinygrad_runner.model.blk:
            if a.variant == "moe-generic":
                for e in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
                    if hasattr(b, e):
                        getattr(b, e)._packed = False  # route MoE to the weight[sel] generic path
            elif a.variant == "moe-tensorq8":
                pass  # keep ExpertWeights fused; only the activation-quantize kernel is swapped below
            elif a.variant == "flash-off" and hasattr(b, "use_flash"):
                b.use_flash = False  # attention falls back to kv_q8_dequant + SDPA
        if a.variant == "moe-tensorq8":
            amod._MOE_TENSOR_Q8 = True  # pure-tensor activation quantize inside expert_linear,
                                        # the fused expert GEMM kernel itself stays engaged
        print(f"variant={a.variant} applied to all blocks", flush=True)
    if a.mode == "capture-diag":
        capture_diag(eng, a.n_prompt, a)
    else:
        ab(eng, a.n_prompt, a.n_gen)


if __name__ == "__main__":
    main()