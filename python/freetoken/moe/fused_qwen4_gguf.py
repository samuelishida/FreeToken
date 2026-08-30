"""Grouped native-IQ routed MoE for Qwen4-Exp GGUF.

Qwen4-Exp stores routed gate/up/down banks as packed GGUF rows. The generic IQ
kernel is not safe on gfx1100, so this module keeps banks packed and expands
only selected rows. Route grouping and GEMMs stay device-side: no host expert
list, no one-expert-at-a-time serving loop, and no full-bank dequant.

The optional fused HIP implementation is capability-gated. The vectorized Torch
implementation below is production fallback and correctness baseline for a
future Triton/AOT kernel.
"""

from __future__ import annotations

import threading
import logging
from dataclasses import dataclass

import torch

from freetoken.layers.activation import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul
from freetoken.models.gguf.dequant import GGML_IQ2_XS, GGML_IQ3_XXS, GGML_IQ4_NL

_ACT = {
    "silu": silu_and_mul,
    "swish": silu_and_mul,
    "gelu": gelu_and_mul,
    "gelu_tanh": gelu_tanh_and_mul,
}


def _apply_activation(name: str, values: torch.Tensor) -> torch.Tensor:
    """Apply SwiGLU/GELU on device, with a CPU reference for unit tests."""
    if values.is_cuda:
        return _ACT[name](values)
    gate, up = values.chunk(2, dim=-1)
    if name in ("silu", "swish"):
        return torch.nn.functional.silu(gate) * up
    if name == "gelu_tanh":
        return torch.nn.functional.gelu(gate, approximate="tanh") * up
    return torch.nn.functional.gelu(gate) * up


@dataclass
class _Stats:
    grouped_calls: int = 0
    batched_fallback_calls: int = 0
    oracle_calls: int = 0
    fused_failures: int = 0
    grouped_routes: int = 0
    grouped_experts: int = 0
    max_chunk_experts: int = 0
    triton_decode_calls: int = 0
    triton_decode_success: int = 0
    triton_decode_failures: int = 0


_STATS = _Stats()
_STATS_LOCK = threading.Lock()
_TRITON_FAILURE_LOGGED = False
_TRITON_SUCCESS_LOGGED = False
_GROUPED_FAILURE_LOGGED = False
_GROUPED_SUCCESS_LOGGED = False


def reset_qwen4_gguf_stats() -> None:
    with _STATS_LOCK:
        for name in _Stats.__dataclass_fields__:
            setattr(_STATS, name, 0)


def qwen4_gguf_stats() -> dict[str, int]:
    with _STATS_LOCK:
        return {name: int(getattr(_STATS, name)) for name in _Stats.__dataclass_fields__}


def _count(name: str, amount: int = 1) -> None:
    with _STATS_LOCK:
        setattr(_STATS, name, int(getattr(_STATS, name)) + amount)


def stable_group_routes(route_ids: torch.Tensor) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Return stable device-side route groups.

    Returns ``(unique_ids, sorted_positions, group_index, valid_mask)``. The
    first three tensors are on same device as ``route_ids``. ``group_index``
    maps each sorted route position to its unique expert group. Invalid routes
    (negative IDs) are omitted and remain zero in caller output.
    """
    flat = route_ids.reshape(-1)
    valid = flat >= 0
    valid_positions = torch.nonzero(valid, as_tuple=False).reshape(-1)
    if valid_positions.numel() == 0:
        empty = flat.new_empty((0,), dtype=torch.long)
        return empty, empty, empty, valid

    valid_ids = flat.index_select(0, valid_positions).to(torch.long)
    sorted_ids, order = torch.sort(valid_ids, stable=True)
    sorted_positions = valid_positions.index_select(0, order)
    unique_ids, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
    group_index = torch.repeat_interleave(
        torch.arange(unique_ids.numel(), device=flat.device, dtype=torch.long), counts
    )
    return unique_ids, sorted_positions, group_index, valid


def _row_bytes(ggml_type: int, in_features: int) -> int:
    block = 32 if ggml_type == GGML_IQ4_NL else 256
    type_size = {GGML_IQ2_XS: 74, GGML_IQ3_XXS: 98, GGML_IQ4_NL: 18}[ggml_type]
    if in_features <= 0 or in_features % block:
        raise ValueError(
            f"Qwen4 GGUF quant width must be positive and divisible by {block}, "
            f"got {in_features} for type {ggml_type}"
        )
    return in_features // block * type_size


def _selected_dense(
    packed: torch.Tensor,
    ids: torch.Tensor,
    ggml_type: int,
    out_features: int,
    in_features: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Decode selected packed rows to ``[groups, out_features, in_features]``."""
    from freetoken.models.gguf.dequant import dequantize

    expected = _row_bytes(ggml_type, in_features)
    if packed.ndim < 2 or int(packed.shape[-1]) != expected:
        raise ValueError(
            f"invalid Qwen4 GGUF bank row width: type={ggml_type} "
            f"in_features={in_features} got={packed.shape[-1] if packed.ndim else None} "
            f"expected={expected}"
        )
    rows = packed.index_select(0, ids.to(device=packed.device, dtype=torch.long))
    rows = rows[..., :expected].contiguous()
    # Bank rows are [expert, output-feature, packed-bytes]. Decode each row;
    # complete bank remains quantized.
    dense = dequantize(rows.reshape(-1, expected), int(ggml_type), dtype)
    return dense.reshape(rows.shape[0], out_features, in_features)


def _selected_dense_gate_up(
    gate: torch.Tensor,
    up: torch.Tensor,
    ids: torch.Tensor,
    ggml_type: int,
    out_features: int,
    in_features: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode gate/up rows together, sharing one IQ launch and table lookup.

    Qwen gate and up banks have identical geometry.  Concatenating selected
    packed rows avoids two separate IQ dequant dispatches per expert chunk; no
    dense bank or route-sized weight replication is introduced.
    """
    from freetoken.models.gguf.dequant import dequantize

    expected = _row_bytes(ggml_type, in_features)
    if gate.ndim < 2 or up.ndim < 2 or int(gate.shape[-1]) != expected or int(up.shape[-1]) != expected:
        raise ValueError(
            f"invalid Qwen4 GGUF gate/up row width: expected {expected}, "
            f"got gate={gate.shape[-1] if gate.ndim else None} "
            f"up={up.shape[-1] if up.ndim else None}"
        )
    selected_gate = gate.index_select(0, ids.to(device=gate.device, dtype=torch.long))
    selected_up = up.index_select(0, ids.to(device=up.device, dtype=torch.long))
    count = int(selected_gate.shape[0])
    packed = torch.cat(
        (selected_gate[..., :expected], selected_up[..., :expected]), dim=0
    ).contiguous()
    dense = dequantize(packed.reshape(-1, expected), int(ggml_type), dtype)
    dense = dense.reshape(2, count, out_features, in_features)
    return dense[0], dense[1]


def _groups_per_chunk(
    scratch_mib: int, hidden: int, intermediate: int, element_size: int
) -> int:
    try:
        budget = int(scratch_mib) * (1 << 20)
    except (TypeError, ValueError) as exc:
        raise ValueError("qwen38_moe_scratch_mib must be a positive integer") from exc
    if budget <= 0:
        raise ValueError("qwen38_moe_scratch_mib must be a positive integer")
    # Gate and up slabs coexist until SwiGLU completes. Include conservative
    # factor for dequant's float32 codebook temporaries.
    bytes_per_group = max(1, 4 * intermediate * hidden * element_size)
    return max(1, budget // bytes_per_group)


def _batched_group_gemm(
    hidden_states: torch.Tensor,
    route_out: torch.Tensor,
    group_positions: torch.Tensor,
    local_groups: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    topk_weights: torch.Tensor,
    top_k: int,
    activation: str,
    scratch_bytes: int,
) -> bool:
    """Run several selected experts with one padded batched GEMM triplet.

    ``group_positions`` is already sorted by expert.  Padding only the routed
    activations (never weights per route) removes Python/GPU launch overhead while
    keeping a hard activation bound.  Return ``False`` for highly skewed groups;
    caller then uses the memory-safer one-group loop.
    """
    groups = int(gate_w.shape[0])
    if groups <= 1 or group_positions.numel() == 0:
        return False
    counts = torch.bincount(local_groups, minlength=groups)
    max_routes = int(counts.max().item())
    # A single expert can own every route in a chunk.  Avoid padding that outlier
    # into a large [groups, routes, hidden] slab; the scalar scratch budget is a
    # conservative activation ceiling, independent of selected weight storage.
    padded_bytes = groups * max_routes * hidden_states.shape[1] * hidden_states.element_size()
    if padded_bytes > max(1, scratch_bytes // 2):
        return False

    offsets = torch.cumsum(counts, dim=0) - counts
    ordinal = torch.arange(group_positions.numel(), device=group_positions.device)
    group_slots = ordinal - torch.repeat_interleave(offsets, counts)
    token_positions = torch.div(group_positions, top_k, rounding_mode="floor")
    x_group = hidden_states.index_select(0, token_positions)
    padded = hidden_states.new_zeros((groups, max_routes, hidden_states.shape[1]))
    padded[local_groups, group_slots] = x_group
    del x_group, token_positions, ordinal, offsets, counts

    gate = torch.bmm(padded, gate_w.transpose(1, 2))
    up = torch.bmm(padded, up_w.transpose(1, 2))
    inter = _apply_activation(activation, torch.cat((gate, up), dim=-1))
    del gate, up, padded
    down = torch.bmm(inter, down_w.transpose(1, 2))
    selected = down[local_groups, group_slots]
    route_weights = topk_weights.reshape(-1).index_select(0, group_positions).to(down.dtype)
    route_out.index_copy_(0, group_positions, selected * route_weights.unsqueeze(-1))
    return True


def _batched_torch_qwen4_gguf(
    hidden_states: torch.Tensor,
    gate_q: torch.Tensor,
    up_q: torch.Tensor,
    down_q: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    layer_id: int,
    scratch_mib: int = 128,
) -> torch.Tensor:
    """Vectorized grouped fallback used when fused HIP is unavailable.

    Python iterates over bounded chunks and their device-side route groups only
    to release selected dense slabs between chunks. Each group's shared weight
    is used once per projection; no route-sized weight replication or CPU route
    metadata is materialized.
    """
    act = _ACT.get(activation)
    if act is None:
        raise ValueError(f"unsupported Qwen4Exp activation {activation!r}")
    if hidden_states.ndim != 2 or topk_ids.ndim != 2:
        raise ValueError("Qwen4 GGUF grouped MoE expects [tokens, hidden] and [tokens, top_k]")

    tokens, hidden = hidden_states.shape
    top_k = topk_ids.shape[1]
    intermediate = int(gate_q.shape[1])
    if int(up_q.shape[1]) != intermediate:
        raise ValueError("Qwen4 GGUF gate/up banks have different intermediate widths")
    if int(down_q.shape[1]) != hidden:
        raise ValueError("Qwen4 GGUF down bank has an unexpected hidden width")

    gate_type = GGML_IQ3_XXS if layer_id == 2 else GGML_IQ2_XS
    unique_ids, sorted_positions, group_index, valid = stable_group_routes(topk_ids)
    route_count = tokens * top_k
    route_out = torch.zeros(
        (route_count, hidden), dtype=hidden_states.dtype, device=hidden_states.device
    )
    if unique_ids.numel() == 0:
        return route_out.view(tokens, top_k, hidden).sum(dim=1)

    weights_flat = topk_weights.reshape(-1)
    groups_per_chunk = _groups_per_chunk(
        scratch_mib, hidden, intermediate, hidden_states.element_size()
    )
    # Prefill has many routes per expert, so keep two groups per chunk to bound
    # padded activations. Decode has one route per expert; splitting ten routed
    # experts into five dequant passes multiplies IQ codebook/kernel overhead.
    # In that case the same scratch-derived selected-weight bound can batch all
    # groups without route-sized weight replication (padded activation slab stays
    # tiny). Small decode batches retain this fast path; larger batches use the
    # conservative prefill cap.
    if tokens > 8:
        groups_per_chunk = min(groups_per_chunk, 2)
    scratch_bytes = int(scratch_mib) * (1 << 20)
    _count("batched_fallback_calls")
    _count("grouped_routes", int(sorted_positions.numel()))
    _count("grouped_experts", int(unique_ids.numel()))
    _count("max_chunk_experts", min(groups_per_chunk, int(unique_ids.numel())))

    # Bound is based on selected dense rows, not bank size. No host tensor or
    # route list is materialized.
    num_groups = int(unique_ids.numel())
    for group_start in range(0, num_groups, groups_per_chunk):
        group_end = min(group_start + groups_per_chunk, num_groups)
        route_mask = (group_index >= group_start) & (group_index < group_end)
        group_positions = sorted_positions[route_mask]
        local_groups = group_index[route_mask] - group_start
        selected_ids = unique_ids[group_start:group_end]

        gate_w, up_w = _selected_dense_gate_up(
            gate_q, up_q, selected_ids, gate_type, intermediate, hidden, hidden_states.dtype
        )
        down_w = _selected_dense(
            down_q, selected_ids, GGML_IQ4_NL, hidden, intermediate, hidden_states.dtype
        )
        # ``gate_w.index_select(0, local_groups)`` duplicates a full [I,H] expert
        # weight for every routed token.  At Qwen's 1K prefill size that transient
        # reached ~3.2 GiB despite the selected-expert scratch bound.  Keep one
        # shared weight view per group.  For balanced groups, process two at a time
        # with padded batched GEMMs; skewed groups use the one-group path to avoid
        # padding an outlier activation slab.
        batched = _batched_group_gemm(
            hidden_states, route_out, group_positions, local_groups, gate_w, up_w, down_w,
            topk_weights, top_k, activation, scratch_bytes,
        )
        if not batched:
            for local_group in range(group_end - group_start):
                positions = group_positions[local_groups == local_group]
                if positions.numel() == 0:
                    continue
                token_positions = torch.div(positions, top_k, rounding_mode="floor")
                x_group = hidden_states.index_select(0, token_positions)
                gate = torch.mm(x_group, gate_w[local_group].transpose(0, 1))
                up = torch.mm(x_group, up_w[local_group].transpose(0, 1))
                inter = _apply_activation(activation, torch.cat((gate, up), dim=-1))
                down = torch.mm(inter, down_w[local_group].transpose(0, 1))
                route_weights = weights_flat.index_select(0, positions).to(down.dtype)
                route_out.index_copy_(0, positions, down * route_weights.unsqueeze(-1))
                del token_positions, x_group, gate, up, inter, down, route_weights, positions
        del gate_w, up_w, down_w, group_positions, local_groups, selected_ids

    return route_out.view(tokens, top_k, hidden).sum(dim=1)


def _fused_qwen4_gguf_available(hidden_states: torch.Tensor) -> bool:
    """Capability gate for future Triton/AOT grouped IQ kernel."""
    if hidden_states.device.type != "cuda" or torch.version.hip is None:
        return False
    # Generic IQ kernels are known unsafe on gfx1100. A dedicated module may
    # opt in later; absence is normal batched-Torch fallback.
    try:
        from freetoken.kernel.triton.qwen4exp_quant import qwen4_gguf_grouped_available

        return bool(qwen4_gguf_grouped_available(hidden_states.device))
    except Exception:  # noqa: BLE001 -- capability probe must never break serving
        return False


def fused_experts_qwen4_gguf(
    hidden_states: torch.Tensor,
    gate_q: torch.Tensor,
    up_q: torch.Tensor,
    down_q: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    layer_id: int,
    *,
    grouped: bool = True,
    scratch_mib: int = 128,
    oracle: bool = False,
) -> torch.Tensor:
    """Run grouped Qwen4 GGUF experts.

    ``grouped=False`` is explicit batched-Torch rollback, not permission to use
    old serving loop. ``oracle=True`` is test-only and imports primitive
    reference implementation; production layers never set it.
    """
    if not hidden_states.is_cuda:
        raise RuntimeError("Qwen4Exp GGUF MoE requires HIP/CUDA tensors")
    if oracle:
        _count("oracle_calls")
        from freetoken.moe._qwen4_gguf_oracle import primitive_experts_qwen4_gguf

        return primitive_experts_qwen4_gguf(
            hidden_states, gate_q, up_q, down_q, topk_weights, topk_ids, activation, layer_id
        )

    # Decode is one/few tokens and therefore benefits from a packed GEMV that
    # fuses IQ unpack with dot products.  The Triton kernel is capability-gated
    # and leaves prefill on the numerically stable grouped Torch implementation.
    if grouped and hidden_states.shape[0] <= 8 and _fused_qwen4_gguf_available(hidden_states):
        try:
            _count("triton_decode_calls")
            from freetoken.kernel.triton.qwen4exp_quant import fused_qwen4_gguf_decode

            gate_type = GGML_IQ3_XXS if layer_id == 2 else GGML_IQ2_XS
            gate_out = fused_qwen4_gguf_decode(
                hidden_states, gate_q, topk_ids, ggml_type=gate_type,
                out_features=int(gate_q.shape[1]),
            )
            up_out = fused_qwen4_gguf_decode(
                hidden_states, up_q, topk_ids, ggml_type=gate_type,
                out_features=int(up_q.shape[1]),
            )
            inter = _apply_activation(activation, torch.cat((gate_out, up_out), dim=-1))
            down_out = fused_qwen4_gguf_decode(
                inter, down_q, topk_ids.reshape(-1, 1),
                ggml_type=GGML_IQ4_NL, out_features=int(down_q.shape[1]),
            )
            route_weights = topk_weights.reshape(-1, 1).to(down_out.dtype)
            _count("triton_decode_success")
            global _TRITON_SUCCESS_LOGGED
            if not _TRITON_SUCCESS_LOGGED:
                logging.getLogger(__name__).info("Qwen4 Triton packed GEMV decode active")
                _TRITON_SUCCESS_LOGGED = True
            return (down_out * route_weights).view(hidden_states.shape[0], topk_ids.shape[1], -1).sum(dim=1)
        except Exception as exc:  # noqa: BLE001 -- capability/runtime fallback
            _count("fused_failures")
            _count("triton_decode_failures")
            global _TRITON_FAILURE_LOGGED
            if not _TRITON_FAILURE_LOGGED:
                logging.getLogger(__name__).warning("Qwen4 Triton decode disabled after runtime failure: %s", exc)
                _TRITON_FAILURE_LOGGED = True

    # Dedicated route-fused GEMV handles arbitrary prefill batches without
    # expanding selected expert weights. Failed probe/runtime or non-finite
    # output falls through to finite grouped Torch path.
    if grouped and _fused_qwen4_gguf_available(hidden_states):
        try:
            from freetoken.kernel.triton.qwen4exp_quant import fused_qwen4_gguf_grouped

            out = fused_qwen4_gguf_grouped(
                hidden_states, gate_q, up_q, down_q,
                topk_weights, topk_ids, activation, layer_id,
                scratch_mib=scratch_mib,
            )
            if torch.isfinite(out).all():
                _count("grouped_calls")
                global _GROUPED_SUCCESS_LOGGED
                if not _GROUPED_SUCCESS_LOGGED:
                    logging.getLogger(__name__).info("Qwen4 Triton route-fused GEMV prefill active")
                    _GROUPED_SUCCESS_LOGGED = True
                return out
        except Exception as exc:  # noqa: BLE001 -- process-local fused fallback
            _count("fused_failures")
            global _GROUPED_FAILURE_LOGGED
            if not _GROUPED_FAILURE_LOGGED:
                logging.getLogger(__name__).warning(
                    "Qwen4 Triton grouped prefill disabled after runtime failure: %s", exc
                )
                _GROUPED_FAILURE_LOGGED = True

    return _batched_torch_qwen4_gguf(
        hidden_states, gate_q, up_q, down_q, topk_weights, topk_ids,
        activation, layer_id, scratch_mib=scratch_mib,
    )


__all__ = [
    "fused_experts_qwen4_gguf",
    "stable_group_routes",
    "_batched_torch_qwen4_gguf",
    "reset_qwen4_gguf_stats",
    "qwen4_gguf_stats",
]
