# Qwen38 grouped MoE route-weight OOM

Symptom: ROCm Qwen3.8 prefill crashes in `fused_qwen4_gguf.py` at
`gate_w.index_select(0, local_groups)` with a multi-GiB CUDA allocation, even
when `qwen38_moe_scratch_mib` limits selected expert dequantization.

Cause: indexing selected `[groups, intermediate, hidden]` weights by every route
materializes one full expert weight per routed token. Scratch limits selected
experts, not this route-expanded projection temporary.

Fix: keep one shared weight view per bounded expert group and run `torch.mm`
over that group's token rows. Never use route-sized `bmm` weight replication;
test grouped prefill with `torch.bmm` forbidden or allocation telemetry.
