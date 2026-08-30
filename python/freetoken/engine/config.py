from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from freetoken.distributed import DistributedInfo
from freetoken.models.register import _load_attr, get_model_spec
from freetoken.utils import cached_load_hf_config

if TYPE_CHECKING:
    from freetoken.models import ModelConfig


# One finite default shared by CLI, direct EngineConfig users, request deadlines,
# and startup metadata.  A timeout is a safety bound, never an admission or
# performance mechanism.
DEFAULT_REQUEST_TIMEOUT_S = 3600.0


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    # Compute device: cuda (default) or tinygrad (AMD no-ROCm path via the
    # tinygrad fork's direct kfd/hsa backend).
    device: str = "cuda"
    max_running_req: int = 4
    # Requests beyond max_running_req are held by the frontend only.  Zero is
    # the compatibility sentinel: serial engines retain their historical
    # bounded 4 * max_running_req admission budget.
    max_pending_requests: int = 0
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    # NVFP4 routed-expert GEMM backend (--nvfp4-backend): auto|marlin|flashinfer|triton.
    nvfp4_backend: str = "triton"
    # Expert-bank host load (--expert-load): auto|serial|parallel. "auto" reads scattered
    # experts in parallel but falls back to serial when free RAM can't cover the banks + the
    # parallel reader's extra (non-reclaimable) whole-shard buffer; "serial" forces the
    # low-memory reclaimable read; "parallel" forces the fast read.
    expert_load: str = "auto"
    moe_cache_size: int = 0
    moe_cache_rate: float | None = None
    moe_cache_auto: bool = False
    kv_reserve_tokens: int = 8192  # KV floor for --moe-cache-auto; small by design (MoE-priority)
    # Independent KV storage dtype. ``auto`` follows compute dtype; ``q8`` is currently
    # supported by QSA only and keeps QSA index slabs in the 16-bit compute dtype.
    kv_cache_dtype: str = "auto"
    # Optional lower KV floor retried when the primary auto-cache floor cannot fit.
    kv_reserve_fallback_tokens: int | None = None
    moe_cache_policy: str = "lru"
    moe_prefill_overlap: bool = True
    # Prefill hit/miss split: serve cache-resident experts D2D during prefill
    # prefetch instead of re-streaming the full layer over PCIe. Needs CUDA >= 12.8
    # (cudaMemcpyBatchAsync); no-op unless moe_cache_size > 2 * num_experts.
    moe_prefill_hit_d2d: bool = False
    moe_collect_stats: bool = False  # capture decode miss-rate counters into the cuda graph
    # CPU MoE backend (--moe-backend cpu): number of CPU worker threads computing
    # the decode experts. 0 = auto (physical cores). Ignored by other backends.
    moe_cpu_threads: int = 0
    # Hybrid CPU/GPU decode (--moe-backend offload only): which MoE layers decode on
    # the CPU executor instead of the GPU offload/PCIe path. Spec is an explicit id
    # list ("3,7,11"), a count ("8" -> 8 layers evenly strided across depth), or a
    # fraction ("0.5"). None/"" = all layers on GPU (plain offload). --moe-backend cpu
    # already means all layers on CPU and ignores this.
    moe_cpu_layers: str | None = None
    # Hybrid MoE backend (--moe-backend hybrid): max experts fetched over PCIe per
    # (layer, decode step); the rest of that step's misses are computed on the CPU.
    # -1 (default) = auto: fetch the benched pcie_bw/cpu_bw fraction of each step's
    # misses so the PCIe fetch and the CPU compute finish together (perfect overlap);
    # falls back to a fixed cap of 1 without a usable `ft bench bw` profile.
    moe_hybrid_max_fetch: int = -1
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    # Hybrid GDN models default to the HybridRadixCache (cross-request GDN-state prefix reuse);
    # `--cache-type naive` opts out. linear_state_cache_ratio sizes the GDN snapshot cache as
    # ceil(ratio * max_running_req) extra slots.
    linear_state_cache_ratio: float = 2.0
    # Window/full ratio for the SWA radix cache (`--cache-type radix` on SWA models) and the DSV4
    # window tier: the DEFAULT window-pool size = max(working-set floor, ratio x full-pool tokens).
    # < 1.0 trades retained window-prefix capacity for memory savings; must be in (0, 1]. It is the
    # DSV4 window/full ratio directly. Used only when swa_num_pages_override is None (a runtime
    # rebuild can pin an absolute window instead).
    swa_full_tokens_ratio: float = 0.2
    # Absolute window-pool size in the pool's own pages (usable, dummy excluded); None -> use the
    # ratio default above. A runtime cache rebuild sets this (num_swa_pages) to pin the window
    # regardless of the full anchor; the ratio is the startup default and the fallback.
    swa_num_pages_override: int | None = None
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    # KV capacity in tokens; resolved into num_page_override by _adjust_config once page_size
    # is final. Mutually exclusive with num_page_override.
    num_token_override: int | None = None
    ple_store: str | None = None
    ple_store_build: str = "auto"
    ple_ram_cache_mib: int | str = 512
    # Packed IQ4_NL hot rows are separate from 4 KiB page cache.  ``0`` keeps
    # direct EngineConfig callers on legacy page-only behavior; ROCm launcher
    # opts into ``auto``.
    ple_row_cache_mib: int | str = 0
    qwen38_host_cache_mib: int | str = 0
    qwen38_expert_host_cache_mib: int | str = 0
    ple_gpu_cache_mib: int = 128
    ple_staging_mib: int = 32
    ple_io: str = "auto"
    ple_io_depth: int | str = 64
    # Deprecated tinygrad-only knobs retained for config compatibility.
    ple_ram_gib: float = 0.0
    ple_workers: int = 2
    ple_prefetch: bool = True
    ple_mode: str = "auto"
    ple_cache_policy: str = "lru"
    ple_prefetch_depth: int | str = 64
    # Packed GPU-row cache is production default whenever a non-zero GPU budget exists.
    # BooleanOptionalAction still exposes --no-ple-batched-cache for rollback/debugging.
    # CLI/script enables this for Qwen ROCm; false remains a supported rollback
    # for generic callers and legacy tests.
    ple_batched_cache: bool = False
    ple_fused_dequant: bool = False
    qwen38_qsa_prefill_live_width: bool = False
    # Qwen3.8 GGUF grouped routed-expert execution. The grouped path is the
    # production default; disabling it selects the bounded batched-Torch path,
    # never the primitive per-expert oracle.
    qwen38_moe_grouped: bool = True
    # Selected packed-row scratch budget. It is charged against the MoE cache
    # budget by engine setup and never permits full-bank dequantization.
    qwen38_moe_scratch_mib: int = 128
    qwen38_prefill_adaptive: bool = False
    # Qwen3.8 GGUF routed expert host source. ``ram`` is production default:
    # packed anonymous CPU RAM; ``mmap`` is explicit compatibility mode;
    # ``auto`` selects ram only when metadata preflight confirms headroom.
    qwen38_expert_residency: str = "ram"
    # Total wall-clock budget for one OpenAI generation, including rendering, admission,
    # prefill/decode, and final response assembly. Finite by design: zero never means infinite.
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    # SSE keepalive interval during long prefill/PLE reads. Must remain below request timeout.
    sse_heartbeat_s: float = 15.0
    # Parent-side deadline for GPU/PLE startup probe. Kept separate from request watchdog.
    ple_probe_timeout_s: float = 300.0

    def __post_init__(self) -> None:
        """Fail before model allocation, including direct programmatic construction."""
        try:
            timeout = float(self.request_timeout_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("request_timeout_s must be a positive finite number") from exc
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("request_timeout_s must be a positive finite number")
        try:
            heartbeat = float(self.sse_heartbeat_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("sse_heartbeat_s must be a positive finite number") from exc
        if heartbeat <= 0 or not math.isfinite(heartbeat):
            raise ValueError("sse_heartbeat_s must be a positive finite number")
        if heartbeat >= timeout:
            raise ValueError("sse_heartbeat_s must be shorter than request_timeout_s")
        try:
            pending = int(self.max_pending_requests)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_pending_requests must be a non-negative integer") from exc
        if pending < 0 or pending != self.max_pending_requests:
            raise ValueError("max_pending_requests must be a non-negative integer")
        if self.qwen38_expert_residency not in ("ram", "mmap", "auto", "auto-tier"):
            raise ValueError(
                "qwen38_expert_residency must be one of: ram, mmap, auto, auto-tier"
            )
        from freetoken.engine.host_tier import parse_budget

        for name in (
            "ple_ram_cache_mib", "ple_row_cache_mib", "qwen38_host_cache_mib",
            "qwen38_expert_host_cache_mib",
        ):
            try:
                parse_budget(getattr(self, name), name=name)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
        for name, minimum in (("ple_io_depth", 1), ("ple_prefetch_depth", 0)):
            value = getattr(self, name)
            if isinstance(value, str) and value in ("auto", "adaptive"):
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be auto or an integer in [0, 256]") from exc
            if parsed < minimum or parsed > 256 or parsed != value:
                bound = f"{minimum}, 256"
                raise ValueError(f"{name} must be an integer in [{bound}]")
        if not isinstance(self.qwen38_moe_grouped, bool):
            raise ValueError("qwen38_moe_grouped must be a boolean")
        if not isinstance(self.qwen38_prefill_adaptive, bool):
            raise ValueError("qwen38_prefill_adaptive must be a boolean")
        try:
            scratch_mib = int(self.qwen38_moe_scratch_mib)
        except (TypeError, ValueError) as exc:
            raise ValueError("qwen38_moe_scratch_mib must be a positive integer") from exc
        if scratch_mib <= 0 or scratch_mib != self.qwen38_moe_scratch_mib:
            raise ValueError("qwen38_moe_scratch_mib must be a positive integer")

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        spec = get_model_spec(self.hf_config.architectures[0])
        parse_config = _load_attr(spec.module, spec.parse_config)
        return parse_config(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
