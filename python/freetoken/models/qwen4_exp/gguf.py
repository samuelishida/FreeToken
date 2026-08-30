"""FreeToken config adapter for embedded Qwen4-Exp GGUF checkpoints."""
from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING

import torch
from freetoken.utils import init_logger

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)

from .config import Qwen4ExpArgs, ple_slot_states
from .model import Qwen4ExpForCausalLM

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim
    from freetoken.engine.config import EngineConfig
    from .ple_gguf import _BasePLETable


logger = init_logger(__name__)


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    m = shim.metadata
    p = "qwen4exp."

    def g(key: str):
        value = m.get(p + key)
        if value is None:
            raise ValueError(f"missing GGUF metadata key {p + key}")
        return value

    layers = int(g("block_count"))
    interval = int(g("full_attention_interval"))
    full_ids = tuple(i for i in range(layers) if (i + 1) % interval == 0)
    linear_ids = tuple(i for i in range(layers) if (i + 1) % interval != 0)

    hidden = int(g("embedding_length"))
    head_dim = int(g("attention.key_length"))
    index_head_dim = int(g("attention.indexer.key_length"))
    index_heads = int(g("attention.indexer.head_count"))
    index_ratio = int(g("attention.compress_ratios")[full_ids[0]])
    if index_ratio <= 1:
        raise ValueError(f"Qwen4Exp GGUF requires compressed QSA ratio > 1, got {index_ratio}")
    rotary = RotaryConfig(
        head_dim,
        int(g("rope.dimension_count")),
        int(g("context_length")),
        float(g("rope.freq_base")),
        None,
    )

    qwen4_args = Qwen4ExpArgs(
        hidden_size=hidden,
        hc_count=int(g("hyper_connection.count")),
        hc_lowrank=int(g("hyper_connection.low_rank")),
        ple_layer_ids=tuple(int(i) for i in g("ple.layers")),
        ple_embed_dim=hidden,
        ple_conv_kernel_size=int(g("ple.conv_kernel")),
        ngram_size=int(g("ple.ngram_size")),
        heads_per_ngram=int(g("ple.heads_per_ngram")),
        ngram_vocab_size_base=int(g("ple.head_vocab_sizes")[0]),
        make_ngram_vocab_size_divisible_by=1,
        split_ngram_parts=128,
        ngram_boundary_token_id=int(g("ple.eos_token_id")),
        index_n_heads=index_heads,
        index_kv_heads=1,
        index_head_dim=index_head_dim,
        index_budget=int(g("attention.indexer.top_k")),
        index_ratio=index_ratio,
    )

    if qwen4_args.ple_layer_ids != (1,):
        raise ValueError(
            "Qwen4Exp GGUF adapter currently supports PLE only at zero-based layer 1, "
            f"got {qwen4_args.ple_layer_ids}"
        )

    groups = (
        LinearGatedDeltaGroupConfig(
            "linear",
            linear_ids,
            int(g("ssm.group_count")),
            int(g("ssm.time_step_rank")),
            int(g("ssm.state_size")),
            int(g("ssm.state_size")),
            int(g("ssm.conv_kernel")),
            "silu",
        ),
        FullAttentionGroupConfig(
            "qsa",
            full_ids,
            int(g("attention.head_count_kv")),
            head_dim,
            rotary,
            index_head_dim=index_head_dim,
            num_index_layers=len(full_ids),
            index_ratio=index_ratio,
        ),
    )
    return ModelConfig(
        num_layers=layers, num_qo_heads=int(g("attention.head_count")),
        num_kv_heads=int(g("attention.head_count_kv")), head_dim=head_dim,
        hidden_size=hidden, vocab_size=int(shim.vocab_size),
        intermediate_size=0, hidden_act="silu", rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")),
        tie_word_embeddings=bool(shim.tie_word_embeddings), rotary_config=rotary,
        num_experts=int(g("expert_count")), num_experts_per_tok=int(g("expert_used_count")),
        moe_intermediate_size=int(g("expert_feed_forward_length")),
        shared_expert_intermediate_size=int(g("expert_shared_feed_forward_length")),
        norm_topk_prob=True,
        model_type="qwen4_exp",
        architectures=list(shim.architectures),
        moe_enabled=True,
        expert_quant="gguf",
        attn_quant="gguf",
        dense_quant="gguf",
        lm_head_quant="gguf",
        moe_weight_format="gguf",
        attention_groups=groups,
        qwen4_args=qwen4_args,
        slot_states=ple_slot_states(qwen4_args),
        # Native GGUF routed experts are disk/host banks; constructing this
        # config must never allocate resident BF16 expert tensors.
        moe_backend="offload",
        single_stream_only=True,
    )


def is_gguf_model(config) -> bool:
    return getattr(config, "moe_weight_format", None) in ("gguf", "gguf_qwen4")


_EXPERT_RESIDENCIES = frozenset(("ram", "mmap", "auto", "auto-tier"))


def _gguf_source_fingerprint(model_path: str, headers, shards) -> str:
    """Return stable identity from GGUF metadata/header ranges, without payload IO."""
    digest = hashlib.sha256()
    for shard in shards:
        stat = os.stat(shard.path)
        digest.update(os.fsencode(os.path.realpath(shard.path)))
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    for header in headers:
        if "_exps.weight" not in header.name:
            continue
        digest.update(
            f"{header.name}|{header.shard_index}|{header.data_offset}|{header.nbytes}|"
            f"{header.ggml_type}|{header.shape}".encode()
        )
    return digest.hexdigest()


def _qwen4_expert_headers(model_path: str, config):
    from freetoken.models.gguf.reader import load_gguf_headers

    _metadata, shards, headers = load_gguf_headers(model_path)
    by_name = {header.name: header for header in headers}
    required = []
    for layer in range(int(config.num_moe_layers)):
        prefix = f"blk.{layer}."
        required.extend(
            by_name.get(prefix + suffix)
            for suffix in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
        )
    missing = [f"blk.{layer}.{suffix}" for layer in range(int(config.num_moe_layers)) for suffix in
               ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
               if f"blk.{layer}.{suffix}" not in by_name]
    if missing:
        raise ValueError(f"Qwen4Exp GGUF missing routed expert tensors: {missing[:4]}")
    return shards, tuple(required), _gguf_source_fingerprint(model_path, headers, shards)


def estimate_gguf_expert_host_bytes(model_path: str, config):
    """Estimate packed expert-bank bytes from GGUF headers only.

    No tensor payload, mmap, or destination tensor is opened/allocated.  The
    returned object is consumed by engine preflight before model construction.
    """
    from freetoken.moe.expert_banks import ExpertHostFootprint

    shards, headers, fingerprint = _qwen4_expert_headers(model_path, config)
    per_layer = []
    per_bank: dict[str, int] = {"gate": 0, "up": 0, "down": 0}
    alignment = 1
    for layer in range(int(config.num_moe_layers)):
        values = {}
        for bank, suffix in (
            ("gate", "ffn_gate_exps.weight"),
            ("up", "ffn_up_exps.weight"),
            ("down", "ffn_down_exps.weight"),
        ):
            header = headers[layer * 3 + (0 if bank == "gate" else 1 if bank == "up" else 2)]
            if header.row_bytes is None or header.rows != int(config.num_experts) * int(
                config.moe_intermediate_size if bank != "down" else config.hidden_size
            ):
                raise ValueError(
                    f"{header.name}: unexpected packed expert geometry "
                    f"shape={header.shape} rows={header.rows} row_bytes={header.row_bytes}"
                )
            # GGUF's tensor payload and row stride are the source of truth.  Keep
            # the bank quantized and preserve exact source alignment.
            values[bank] = int(header.nbytes)
            per_bank[bank] += int(header.nbytes)
            alignment = max(alignment, int(header.row_bytes))
        per_layer.append(values)
    total = sum(per_bank.values())
    # Runtime headroom is deliberately explicit and never supplied by swap.
    headroom = max(4 << 30, total // 10)
    return ExpertHostFootprint(
        quant_format="gguf_qwen4",
        per_bank_bytes=per_bank,
        per_layer_bytes=tuple(per_layer),
        total_packed_bytes=total,
        alignment=alignment,
        source_fingerprint=fingerprint,
        headroom_bytes=headroom,
    )


def _host_memory_available() -> int | None:
    from freetoken.engine.host_tier import cgroup_memory_limit_bytes

    try:
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    limit = cgroup_memory_limit_bytes()
                    return min(available, limit) if limit is not None else available
    except (OSError, ValueError):
        return None
    return None


def resolve_gguf_expert_residency(model_path: str, config, requested: str, footprint=None):
    """Resolve Qwen GGUF host policy, with explicit auto fallback telemetry."""
    if requested not in _EXPERT_RESIDENCIES:
        raise ValueError(
            f"unsupported qwen38 expert residency {requested!r}; "
            "choose ram, mmap, auto, or auto-tier"
        )
    if requested not in ("auto", "auto-tier"):
        return requested, None
    footprint = footprint or estimate_gguf_expert_host_bytes(model_path, config)
    available = _host_memory_available()
    required = footprint.total_packed_bytes + footprint.headroom_bytes
    if available is None:
        return "auto-tier", "host MemAvailable is unavailable"
    if available < required:
        return "auto-tier", (
            f"host MemAvailable {available / 2**30:.2f} GiB is below packed banks + "
            f"headroom {required / 2**30:.2f} GiB"
        )
    return ("auto-tier" if requested == "auto-tier" else "ram"), None


def load_gguf_expert_sources(
    model_path: str,
    config,
    *,
    layer_sink=None,
    residency: str | None = None,
    return_owner: bool = False,
) -> dict[str, list[torch.Tensor]] | tuple[dict[str, list[torch.Tensor]], object | None]:
    """Load Qwen4Exp routed expert banks under explicit host residency policy.

    ``mmap`` exposes file-backed views for compatibility; ``ram`` copies packed
    bytes into contiguous anonymous CPU tensors and closes source mappings.
    GGUF stores each layer as one packed expert tensor. Layer 2 uses IQ3_XXS
    gate/up; all other layers use IQ2_XS. Down is IQ4_NL everywhere.
    """
    if layer_sink is not None:
        raise NotImplementedError("Qwen4Exp GGUF mmap banks cannot stream into FTW")
    from .packed import Qwen4ExpPackedSource

    # ``None`` intentionally preserves the historical mmap-visible API for
    # callers/tests. Serving passes an explicit policy from EngineConfig.
    requested = "mmap" if residency is None else residency
    if requested not in _EXPERT_RESIDENCIES:
        raise ValueError(
            f"unsupported qwen38 expert residency {requested!r}; choose ram, mmap, auto, or auto-tier"
        )
    footprint = estimate_gguf_expert_host_bytes(model_path, config) if requested == "auto" else None
    requested, fallback_reason = resolve_gguf_expert_residency(
        model_path, config, requested, footprint
    ) if requested == "auto" else (requested, None)
    if fallback_reason:
        logger.warning(
            f"Qwen3.8 GGUF expert residency auto -> auto-tier: {fallback_reason}; "
            "host banks remain file-backed with bounded packed cache"
        )
    source = Qwen4ExpPackedSource(model_path, cache_bytes=0x20000000)
    banks = {"gate": [], "up": [], "down": []}
    E, H, I = config.num_experts, config.hidden_size, config.moe_intermediate_size
    try:
        for layer in range(config.num_moe_layers):
            prefix = f"blk.{layer}."
            gate = source.locate(prefix + "ffn_gate_exps.weight")
            up = source.locate(prefix + "ffn_up_exps.weight")
            down = source.locate(prefix + "ffn_down_exps.weight")
            if gate.ggml_type != up.ggml_type or gate.shape[:2] != (E, I):
                raise ValueError(f"layer {layer}: unexpected Qwen4Exp gate/up geometry")
            if down.ggml_type != 20 or down.shape[:2] != (E, H):
                raise ValueError(f"layer {layer}: unexpected Qwen4Exp down geometry")
            if requested == "ram":
                gate_t = source.materialize_tensor(gate.name, shape=(E, I, gate.row_bytes))
                up_t = source.materialize_tensor(up.name, shape=(E, I, up.row_bytes))
                down_t = source.materialize_tensor(down.name, shape=(E, H, down.row_bytes))
            else:
                gate_t = source.mapped_tensor(gate.name, shape=(E, I, gate.row_bytes))
                up_t = source.mapped_tensor(up.name, shape=(E, I, up.row_bytes))
                down_t = source.mapped_tensor(down.name, shape=(E, H, down.row_bytes))
            banks["gate"].append(gate_t)
            banks["up"].append(up_t)
            banks["down"].append(down_t)
    except BaseException:
        for values in banks.values():
            values.clear()
        if requested == "ram":
            source.close()
        raise
    if requested == "ram":
        # No returned tensor retains an mmap view after materialization. Closing
        # here is required to make host RAM ownership explicit and reclaimable.
        source.close()
    else:
        _LIVE_QWEN4_GGUF_SOURCES.append(source)
    return (banks, None if requested == "ram" else source) if return_owner else banks


_LIVE_QWEN4_GGUF_SOURCES: list[object] = []


def setup_offload_expert_banks(
    model_path: str, model_config, *, device: torch.device, dtype: torch.dtype,
    dummy: bool = False, parallel: bool = False, workers: int = 8,
    chunk: int = 8 << 20, decode_target: str = "gpu", layer_sink=None,
    residency: str | None = None,
):
    """Qwen4Exp GGUF setup; delegate official safetensors to existing provider."""
    from freetoken.moe.expert_banks import ExpertBanks
    if not is_gguf_model(model_config):
        from freetoken.models.qwen3_5_moe.weight import setup_offload_expert_banks as parent
        return parent(model_path, model_config, device=device, dtype=dtype, dummy=dummy,
                      parallel=parallel, workers=workers, chunk=chunk,
                      decode_target=decode_target, layer_sink=layer_sink)
    if dummy:
        raise NotImplementedError("Qwen4Exp GGUF has no synthetic production bank path")
    requested = "mmap" if residency is None else residency
    footprint = estimate_gguf_expert_host_bytes(model_path, model_config)
    resolved, reason = resolve_gguf_expert_residency(
        model_path, model_config, requested, footprint
    )
    if reason:
        logger.warning_rank0(
            f"Qwen3.8 GGUF expert residency auto -> auto-tier: {reason}; "
            "host banks remain file-backed with bounded packed cache"
        )
    sources, owner = load_gguf_expert_sources(
        model_path, model_config, layer_sink=layer_sink, residency=resolved,
        return_owner=True,
    )
    return ExpertBanks(
        "gguf_qwen4", sources,
        layer_residency=["pageable"] * model_config.num_moe_layers,
        source_policy=resolved,
        footprint=footprint,
        owner=owner,
        source_provider=(owner if resolved == "auto-tier" else None),
    )


def convert_qwen4exp_to_gguf(model, config) -> None:
    """Swap main Qwen4Exp dense projections to native GGUF packed layers."""
    from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear
    from freetoken.models.gguf.dequant import (
        GGML_Q4_K, GGML_Q6_K, GGML_Q8_0,
    )

    inner = model.model
    inner.embed_tokens = GGUFEmbedding(config.vocab_size, config.hidden_size, 13)
    model.lm_head = GGUFLinear(config.hidden_size, config.vocab_size, GGML_Q4_K)
    inner.hyper_connection_mixer.input_mix_weight_down = GGUFLinear(
        config.qwen4_args.ple_state_width, config.qwen4_args.hc_lowrank, GGML_Q8_0
    )
    inner.hyper_connection_mixer.input_mix_weight_up = GGUFLinear(
        config.qwen4_args.hc_lowrank, config.qwen4_args.ple_state_width, GGML_Q8_0
    )
    for layer in inner.layers.op_list:
        if layer._is_linear:
            g = layer.linear_attn
            # Layer 2 is the checkpoint's Q6_K GDN input projection; all
            # other GDN input projections are Q5_K.
            g.in_proj_qkvz = GGUFLinear(
                config.hidden_size, g.conv_dim + g.value_dim, 14 if layer._layer_id == 2 else 13
            )
            g.out_proj = GGUFLinear(g.value_dim, config.hidden_size, GGML_Q6_K)
        else:
            attn = layer.self_attn
            attn.q_gate_proj._quant_type = 13
            attn.kv_proj._quant_type = GGML_Q6_K
            attn.o_proj = GGUFLinear(attn.qo_attn_dim, config.hidden_size, 13)
        m = layer.mlp
        m.shared_expert.gate_up_proj = GGUFLinear(
            config.hidden_size, 2 * config.shared_expert_intermediate_size,
            14 if layer._layer_id == 2 else 13,
        )
        m.shared_expert.down_proj = GGUFLinear(
            config.shared_expert_intermediate_size, config.hidden_size, GGML_Q8_0
        )
        for hc in (layer.attn_hyper_connection, layer.mlp_hyper_connection):
            hc.input_mix_weight_down = GGUFLinear(
                config.qwen4_args.ple_state_width,
                hc.lowrank,
                GGML_Q8_0,
            )
            # Inject logits are F32 in target GGUF. Keep this tiny projection
            # dense instead of pretending its row format matches Q8_0 down.
            from freetoken.layers import LinearReplicated
            hc.block_inject_weight = LinearReplicated(
                config.qwen4_args.ple_state_width, hc.hc_count, has_bias=False
            )
            hc.input_mix_weight_up = GGUFLinear(
                hc.lowrank, config.qwen4_args.ple_state_width, GGML_Q8_0
            )
        if layer.ple is not None:
            layer.ple.key_proj = GGUFLinear(config.hidden_size, config.qwen4_args.ple_state_width, GGML_Q8_0)
            layer.ple.value_proj = GGUFLinear(config.hidden_size, config.hidden_size, GGML_Q8_0)


def _to_bf16(t):
    from freetoken.models.gguf.dequant import dequantize

    return dequantize(t.packed().reshape(-1), t.ggml_type, torch.bfloat16).reshape(t.shape)


def _to_fp32(t):
    from freetoken.models.gguf.dequant import dequantize

    return dequantize(t.packed().reshape(-1), t.ggml_type, torch.float32).reshape(t.shape)


def _deint_heads(value: torch.Tensor, num_heads: int) -> torch.Tensor:
    half = num_heads // 2
    perm = [(h // 2) + (h % 2) * half for h in range(num_heads)]
    return value.reshape(num_heads, -1)[perm].reshape(value.shape)


def iter_gguf_weights(model_path: str, device, *, include_moe_experts: bool,
                      include_non_moe: bool):
    """Yield main-model keys from all three GGUF shards; skip PLE/experts."""
    if include_moe_experts or not include_non_moe:
        return
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.models.gguf.reader import load_gguf_metadata

    metadata = load_gguf_metadata(model_path)
    ple = "qwen4exp.ple."
    yield "model.layers.1.ple.ple_embedding.layer_multipliers", torch.tensor(
        metadata[f"{ple}layer_multipliers"], dtype=torch.int64
    )
    yield "model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes", torch.tensor(
        metadata[f"{ple}head_vocab_sizes"], dtype=torch.int64
    )
    yield "model.layers.1.ple.ple_embedding.ngram_heads_offsets", torch.tensor(
        metadata[f"{ple}head_offsets"], dtype=torch.int64
    )

    # Metadata-derived layer pattern is stable for this checkpoint.
    qsa = set(range(3, 48, 4))
    qkv: dict[int, dict[str, torch.Tensor]] = {}
    indexer: dict[int, dict[str, torch.Tensor]] = {}
    gdn: dict[int, dict[str, torch.Tensor]] = {}
    ba: dict[int, dict[str, torch.Tensor]] = {}
    shared: dict[int, dict[str, torch.Tensor]] = {}
    hc: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name == "per_layer_token_embd.weight" or name.endswith("_exps.weight"):
            continue
        if name == "token_embd.weight":
            yield "model.embed_tokens.qweight", t.packed(); continue
        if name == "output.weight":
            yield "lm_head.qweight", t.packed(); continue
        if name == "output_hc_norm.weight":
            yield "model.hyper_connection_mixer.hc_norm.weight", _to_bf16(t); continue
        if name == "output_hc_down.weight":
            yield "model.hyper_connection_mixer.input_mix_weight_down.qweight", t.packed(); continue
        if name == "output_hc_up.weight":
            yield "model.hyper_connection_mixer.input_mix_weight_up.qweight", t.packed(); continue
        if not name.startswith("blk."):
            continue
        layer = int(name.split(".")[1]); suffix = name.split(".", 2)[2]; base = f"model.layers.{layer}"
        if suffix in ("hc_attn_norm.weight", "hc_ffn_norm.weight"):
            yield f"{base}.{'attn' if 'attn' in suffix else 'mlp'}_hyper_connection.hc_norm.weight", _to_bf16(t); continue
        if suffix in ("hc_attn_down.weight", "hc_attn_inject.weight", "hc_attn_up.weight",
                      "hc_ffn_down.weight", "hc_ffn_inject.weight", "hc_ffn_up.weight"):
            group = "attn" if "attn" in suffix else "mlp"
            part = "down" if "_down." in suffix else "inject" if "_inject." in suffix else "up"
            hc.setdefault((layer, group), {})[part] = t.packed() if t.ggml_type == 8 else _to_bf16(t)
            continue
        if suffix == "ssm_conv1d.weight":
            c = _to_fp32(t); c = c.clone(); c[4096:] = _deint_heads(c[4096:], 48)
            yield f"{base}.linear_attn.conv1d.weight", c.unsqueeze(1); continue
        if suffix == "ssm_a":
            yield f"{base}.linear_attn.A_log", torch.log(-_deint_heads(_to_fp32(t), 48)); continue
        if suffix == "ssm_dt.bias":
            yield f"{base}.linear_attn.dt_bias", _deint_heads(_to_fp32(t), 48); continue
        if suffix in ("ssm_norm.weight", "ple_norm_key.weight", "ple_norm_query.weight", "ple_norm_conv.weight",
                      "attn_q_norm.weight", "attn_k_norm.weight", "indexer.q_norm.weight", "indexer.k_norm.weight"):
            names = {
                "ssm_norm.weight": "linear_attn.norm.weight", "ple_norm_key.weight": "ple.norm_key.weight",
                "ple_norm_query.weight": "ple.norm_query.weight", "ple_norm_conv.weight": "ple.norm_conv.weight",
                "attn_q_norm.weight": "self_attn.q_norm.weight", "attn_k_norm.weight": "self_attn.k_norm.weight",
                "indexer.q_norm.weight": "self_attn.indexer.q_layernorm.weight", "indexer.k_norm.weight": "self_attn.indexer.k_layernorm.weight",
            }
            yield f"{base}.{names[suffix]}", _to_bf16(t); continue
        if suffix in ("ssm_alpha.weight", "ssm_beta.weight"):
            ba.setdefault(layer, {})[suffix[4]] = _deint_heads(_to_bf16(t), 48); continue
        if layer in qsa and suffix == "attn_q.weight":
            qkv.setdefault(layer, {})["q"] = t.packed()[:6144]
            qkv[layer]["gate"] = t.packed()[6144:]
            continue
        if layer in qsa and suffix == "attn_gate.weight": qkv.setdefault(layer, {})["gate"] = t.packed(); continue
        if layer in qsa and suffix == "attn_k.weight": qkv.setdefault(layer, {})["k"] = t.packed(); continue
        if layer in qsa and suffix == "attn_v.weight": qkv.setdefault(layer, {})["v"] = t.packed(); continue
        if suffix == "attn_qkv.weight": gdn.setdefault(layer, {})["qkv"] = t.packed(); continue
        if suffix == "attn_gate.weight": gdn.setdefault(layer, {})["z"] = t.packed(); continue
        if suffix == "ssm_out.weight": yield f"{base}.linear_attn.out_proj.qweight", t.packed(); continue
        if suffix == "attn_output.weight": yield f"{base}.self_attn.o_proj.qweight", t.packed(); continue
        if suffix in ("indexer.q_proj.weight", "indexer.k_proj.weight"):
            indexer.setdefault(layer, {})["q" if "q_proj" in suffix else "k"] = _to_bf16(t); continue
        if suffix == "ffn_gate_inp.weight": yield f"{base}.mlp.gate.weight", _to_bf16(t); continue
        if suffix == "ffn_gate_inp_shexp.weight": yield f"{base}.mlp.shared_expert_gate.weight", _to_bf16(t).unsqueeze(0); continue
        if suffix == "ffn_gate_shexp.weight": shared.setdefault(layer, {})["gate"] = t.packed(); continue
        if suffix == "ffn_up_shexp.weight": shared.setdefault(layer, {})["up"] = t.packed(); continue
        if suffix == "ffn_down_shexp.weight": yield f"{base}.mlp.shared_expert.down_proj.qweight", t.packed(); continue
        if suffix == "ple_key.weight": yield f"{base}.ple.key_proj.qweight", t.packed(); continue
        if suffix == "ple_value.weight": yield f"{base}.ple.value_proj.qweight", t.packed(); continue
        if suffix == "ple_conv1d.weight": yield f"{base}.ple.conv1d.weight", _to_fp32(t).unsqueeze(1); continue
        raise ValueError(f"unmapped qwen4exp GGUF tensor: {name}")
        
    # TODO: fused packed tensors are emitted after iteration; all parts share type/row stride.
    for layer, parts in gdn.items():
        if {"qkv", "z"} <= parts.keys(): yield f"model.layers.{layer}.linear_attn.in_proj_qkvz.qweight", torch.cat([parts["qkv"], parts["z"]])
        if {"b", "a"} <= ba.get(layer, {}).keys(): yield f"model.layers.{layer}.linear_attn.in_proj_ba.weight", torch.cat([ba[layer]["b"], ba[layer]["a"]])
    for layer, parts in shared.items():
        if {"gate", "up"} <= parts.keys(): yield f"model.layers.{layer}.mlp.shared_expert.gate_up_proj.qweight", torch.cat([parts["gate"], parts["up"]])
    for layer, parts in qkv.items():
        if layer in qsa and {"q", "gate", "k", "v"} <= parts.keys():
            yield f"model.layers.{layer}.self_attn.q_gate_proj.qweight", torch.cat(
                [parts["q"], parts["gate"]], dim=0
            )
            yield f"model.layers.{layer}.self_attn.kv_proj.qweight", torch.cat(
                [parts["k"], parts["v"]], dim=0
            )
    for layer, parts in indexer.items():
        if {"q", "k"} <= parts.keys():
            yield f"model.layers.{layer}.self_attn.indexer.index_qk_proj.weight", torch.cat([parts["q"], parts["k"]])
    for (layer, group), parts in hc.items():
        if {"down", "inject"} <= parts.keys():
            inj = parts["inject"].to(torch.bfloat16) if parts["inject"].dtype == torch.uint8 else parts["inject"]
            yield f"model.layers.{layer}.{group}_hyper_connection.input_mix_weight_down.qweight", parts["down"]
            yield f"model.layers.{layer}.{group}_hyper_connection.block_inject_weight.weight", inj
        if "up" in parts: yield f"model.layers.{layer}.{group}_hyper_connection.input_mix_weight_up.qweight", parts["up"]


def load_gguf_ple_table(model_path: str, args: "EngineConfig") -> "_BasePLETable":
    from .ple_gguf import DirectGGUFPLETable, PackedPagedPLETable, PackedResidentPLETable

    mode = getattr(args, "ple_mode", "auto")
    if mode == "ssd":
        mode = "paged"
    if mode == "auto": mode = "paged"
    if mode == "paged": return PackedPagedPLETable(model_path, args)
    if mode == "resident": return PackedResidentPLETable(model_path, args)
    if mode == "direct-gguf": return DirectGGUFPLETable(model_path, args)
    raise ValueError(f"unsupported GGUF PLE mode: {mode}")


__all__ = ["Qwen4ExpForCausalLM", "parse_gguf_config", "iter_gguf_weights",
           "convert_qwen4exp_to_gguf", "is_gguf_model", "load_gguf_ple_table",
           "estimate_gguf_expert_host_bytes", "resolve_gguf_expert_residency",
           "load_gguf_expert_sources", "setup_offload_expert_banks"]
