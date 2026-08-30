"""Qwen3.8-Flash-Next decoder stack (text-only).

The residual state is ``R [T, hc_count*hidden]`` end to end: the embedding is repeated over the
``hc_count`` streams, every layer mixes them down to one ``[T, hidden]`` block input and injects
its output back, and the top-level mixer collapses them once before ``lm_head``. There is no
input/post layernorm and no final ``model.norm`` -- the hyper-connection norms are the only ones.

Layer contract (frozen): ``forward(R [T, hc*hidden], batch) -> R' [T, hc*hidden]`` with an
immediate combine::

    R  = R + ple(R, batch)                 # zero-based layer 1 only
    x, s = attn_hc.mix(R); y = (GDN | QSA)(x); R = attn_hc.combine(R, y, s)
    x, s = mlp_hc.mix(R);  y = MoE(x);        R = mlp_hc.combine(R, y, s)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List
import os
import time

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import init_logger, nvtx_annotate

from .attention import Qwen4ExpAttention
from .hc import GatedResidual
from .moe import Qwen4ExpMoE
from .ple import PLELayer

logger = init_logger(__name__)

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig


def build_linear_mixer(config: ModelConfig, layer_id: int) -> BaseOP:
    """GDN mixer of a linear_attention layer (Qwen3.5's GDN with a configurable output gate)."""
    from .gdn import Qwen4ExpGatedDeltaNet

    g = config.linear_attention_group()
    return Qwen4ExpGatedDeltaNet(
        hidden_size=config.hidden_size,
        num_k_heads=g.num_key_heads,
        num_v_heads=g.num_value_heads,
        head_k_dim=g.key_head_dim,
        head_v_dim=g.value_head_dim,
        conv_kernel_size=g.conv_kernel_dim,
        rms_norm_eps=config.rms_norm_eps,
        layer_id=layer_id,
        output_gate=g.output_gate,
        # Qwen3.8's block-fp8 checkpoint keeps the GDN projections bf16 (only the routed
        # experts are quantized), so do not let expert_quant flip them to Fp8Block.
        expert_quant="none" if config.expert_quant == "fp8_block" else config.expert_quant,
        attn_quant=config.attn_quant,
    )


class Qwen4ExpDecoderLayer(BaseOP):
    """One decoder layer over the hyper-connection streams (see the module docstring for the flow)."""

    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            self.linear_attn = build_linear_mixer(config, layer_id)
        else:
            self.self_attn = Qwen4ExpAttention(config, layer_id)
        self.mlp = Qwen4ExpMoE(config, layer_id)
        self.attn_hyper_connection = GatedResidual(config)
        self.mlp_hyper_connection = GatedResidual(config)
        self.ple = (
            PLELayer(config, layer_id) if layer_id in config.qwen4_args.ple_layer_ids else None
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, batch: Batch) -> torch.Tensor:
        observe = getattr(self, "_status_observer", None)
        if self.ple is not None:
            hidden = hidden + self.ple.forward(hidden, batch)
            if observe is not None:
                observe("ple_layer_done")
        block_input, inject = self.attn_hyper_connection.mix(hidden)
        if observe is not None:
            observe("attn_hc_mix_done")
        if self._is_linear:
            block_output = self.linear_attn.forward(block_input)
        else:
            block_output = self.self_attn.forward(block_input, batch)
        if observe is not None:
            observe("attention_done")
        hidden = self.attn_hyper_connection.combine(hidden, block_output, inject)
        if observe is not None:
            observe("attn_hc_combine_done")
        block_input, inject = self.mlp_hyper_connection.mix(hidden)
        if observe is not None:
            observe("mlp_hc_mix_done")
        self.mlp._status_observer = observe
        moe_output = self.mlp.forward(block_input)
        if observe is not None:
            observe("moe_done")
        result = self.mlp_hyper_connection.combine(hidden, moe_output, inject)
        if observe is not None:
            observe("mlp_hc_combine_done")
        return result


class Qwen4ExpModel(BaseOP):
    def __init__(self, config: ModelConfig) -> None:
        self.hc_count = config.qwen4_args.hc_count
        if getattr(config, "dense_quant", "none") == "gguf":
            from freetoken.layers.gguf import GGUFEmbedding
            self.embed_tokens = GGUFEmbedding(config.vocab_size, config.hidden_size, 13)
        else:
            self.embed_tokens = VocabParallelEmbedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
            )
        self.layers = OPList(
            [Qwen4ExpDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False)
        # plain tuple (not an OP child), so it never shows up in the state dict
        self._ple = tuple(layer.ple for layer in self.layers.op_list if layer.ple is not None)

    @property
    def ple_layers(self) -> List[PLELayer]:
        """The PLE layers in decoder order -- the seam the loader attaches table backends to."""
        return list(self._ple)

    def forward(self, input_ids: torch.Tensor, batch: Batch) -> torch.Tensor:
        table = self._ple[0].ple_embedding.table if self._ple else None
        mark_forward = getattr(table, "mark_forward", None)
        emit_stage = None
        if mark_forward is not None:
            # Count entry before embedding/repeat.  A synchronous backend failure here
            # must still be visible as an entered forward in live diagnostics.
            mark_forward()
            emit_stage = getattr(table, "_emit_status", None)
            if emit_stage is not None:
                emit_stage("model_started")
        hidden = self.embed_tokens.forward(input_ids).repeat(1, self.hc_count)
        if mark_forward is not None and emit_stage is not None:
            emit_stage("embedding_done")
        meta = None
        if self._ple:
            from .ple import build_ple_metadata, commit_ngram_context

            meta = build_ple_metadata(batch, self._ple[0].args, input_ids.device)
            if emit_stage is not None:
                emit_stage("ple_meta_done")
        for ple in self._ple:  # gather the pinned-host PLE rows while the early layers run
            ple.start_prefetch(batch, meta)
        if emit_stage is not None and self._ple:
            emit_stage("ple_prefetch_done")
        layer_timing = (
            os.environ.get("FREETOKEN_QWEN38_LAYER_TIMING", "").strip().lower()
            in ("1", "true", "yes", "on")
        ) and hidden.is_cuda
        for layer in self.layers.op_list:
            layer_start = time.perf_counter() if layer_timing else 0.0
            layer._status_observer = emit_stage
            if emit_stage is not None:
                emit_stage(f"layer_{layer._layer_id}_started")
            hidden = layer.forward(hidden, batch)
            if layer_timing:
                torch.cuda.synchronize(hidden.device)
                logger.info(
                    "Qwen3.8 layer timing: layer=%d kind=%s elapsed_ms=%.2f",
                    layer._layer_id,
                    "linear" if layer._is_linear else "qsa",
                    (time.perf_counter() - layer_start) * 1000.0,
                )
            if emit_stage is not None:
                emit_stage(f"layer_{layer._layer_id}_done")
        if meta is not None:
            # single writer: the layers only read the context, so a second PLE layer's
            # prefetch sees the un-rolled window
            commit_ngram_context(meta, getattr(batch, "fla_metadata", None))
        return self.hyper_connection_mixer.mix(hidden)[0]


class Qwen4ExpForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._ple_table = None
        self.model = Qwen4ExpModel(config)
        if getattr(config, "lm_head_quant", "none") == "nvfp4":
            from freetoken.kernel.triton.nvfp4_linear import Nvfp4LMHead

            assert not config.tie_word_embeddings, "NVFP4 lm_head assumes untied embeddings"
            self.lm_head = Nvfp4LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        elif getattr(config, "lm_head_quant", "none") == "gguf":
            from freetoken.layers.gguf import GGUFLinear
            from freetoken.models.gguf.dequant import GGML_Q4_K
            self.lm_head = GGUFLinear(config.hidden_size, config.vocab_size, GGML_Q4_K)
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
        super().__init__()
        from .gguf import convert_qwen4exp_to_gguf, is_gguf_model

        if is_gguf_model(config):
            convert_qwen4exp_to_gguf(self, config)

    def load_host_tables(self, engine_config) -> int:
        """Attach the PLE n-gram table (pinned checkpoint bank, or zeros for dummy weights); returns the pinned host bytes the engine reserves from its pin budget."""
        ple_layers = self.model.ple_layers
        if not ple_layers:
            return 0
        from .ple import PinnedUVATable, ZeroTable, derive_ngram_hash_constants

        if getattr(engine_config, "use_dummy_weight", False):
            # Dummy fill leaves the int64 hash buffers garbage (a zero vocab size divides by
            # zero in the hash), so re-derive the real constants and read a zero table.
            for ple in ple_layers:
                args = ple.args
                mult, sizes, offsets = derive_ngram_hash_constants(
                    vocab_size=self._config.vocab_size,
                    ngram_size=args.ngram_size,
                    num_ngram_heads=args.num_ngram_heads,
                    ngram_vocab_size_base=args.ngram_vocab_size_base,
                    ple_layer_index=ple.ple_index,
                )
                emb = ple.ple_embedding
                emb.layer_multipliers.copy_(torch.tensor(mult, dtype=torch.int64))
                emb.ngram_heads_vocab_sizes.copy_(torch.tensor(sizes, dtype=torch.int64))
                emb.ngram_heads_offsets.copy_(torch.tensor(offsets, dtype=torch.int64))
                emb.attach_table(ZeroTable(offsets[-1] + sizes[-1], args.ngram_head_dim))
            return 0

        if getattr(self._config, "moe_weight_format", None) == "gguf":
            from .gguf import load_gguf_ple_table

            table = load_gguf_ple_table(engine_config.model_path, engine_config)
            self._ple_table = table
            for ple in ple_layers:
                ple.ple_embedding.attach_table(table)
            # Paged `.ftple` RAM cache is pageable. Reserve only genuinely pinned
            # staging against the MoE host-registration quota.
            return int(getattr(table, "pinned_host_bytes", table.host_bytes))

        from .weight import load_ple_table

        table = load_ple_table(engine_config.model_path, self._config.qwen4_args)
        self._ple_table = table  # owns the pinned HostBank; keep it alive
        for ple in ple_layers:
            ple.ple_embedding.attach_table(
                PinnedUVATable(table.bank.tensor, float(table.weight_scale))
            )
        return table.bank.nbytes

    def close_host_tables(self) -> None:
        """Release model-owned PLE file descriptors, workers, and host storage."""
        table = self._ple_table
        if table is None:
            return
        self._ple_table = None
        close = getattr(table, "close", None)
        if close is not None:
            close()

    def validate_ple_runtime(self, device: torch.device | None = None) -> dict:
        """Run one real packed-row PLE lookup through the attached backend.

        This deliberately does not synchronize the device; the Engine owns the single startup
        synchronization so a bad GPU kernel fails before readiness instead of on first request.
        """
        table = self._ple_table
        if table is None:
            return {"state": "skipped", "reason": "no PLE table"}
        rows = int(getattr(table, "num_rows", 0) or 0)
        if rows <= 0:
            raise RuntimeError("PLE table has no rows")
        target = device or getattr(table, "_device", None) or torch.device("cpu")
        # Keep a singleton head dimension: table lookup preserves ``row_ids.shape[:-1]``
        # and appends the 160-value embedding width.
        row_ids = torch.zeros((1, 1), dtype=torch.int64, device=target)
        try:
            values = table.lookup(row_ids)
            if tuple(values.shape) != (1, 160):
                raise RuntimeError(f"PLE probe returned shape {tuple(values.shape)}, expected (1, 160)")
        except Exception as exc:  # noqa: BLE001 - add model/backend context before startup abort
            mode = getattr(table, "_report", {}).get("mode", "unknown")
            raise RuntimeError(
                f"Qwen4Exp PLE probe failed for {getattr(self._config, 'model_path', '<model>')} "
                f"(mode={mode}, quant=IQ4_NL, row_bytes=90, row_values=160): {exc}"
            ) from exc
        return {
            "state": "ok",
            "backend": getattr(table, "_report", {}).get("backend", type(table).__name__),
            "mode": getattr(table, "_report", {}).get("mode"),
            "row_values": 160,
            "row_bytes": 90,
            "device": str(target),
        }

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        return self.lm_head.forward(self.model.forward(batch.input_ids, batch))


__all__ = ["Qwen4ExpDecoderLayer", "Qwen4ExpForCausalLM", "Qwen4ExpModel", "build_linear_mixer"]
