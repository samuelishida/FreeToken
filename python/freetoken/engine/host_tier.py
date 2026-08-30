"""Qwen3.8 host-tier budget resolution.

Keep expert, PLE page/row, and pinned staging reservations explicit.  This
module only reads host metadata; it never allocates checkpoint payloads.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


AUTO_VALUES = frozenset(("auto", "adaptive"))


def parse_budget(value, *, name: str, allow_zero: bool = True) -> int | str:
    if isinstance(value, str):
        value = value.strip().lower()
        if value in AUTO_VALUES:
            return value
        try:
            value = int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be auto, adaptive, or a non-negative integer") from exc
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be auto, adaptive, or a non-negative integer") from exc
    if value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be {'positive' if not allow_zero else 'non-negative'}")
    return value


def mem_available_bytes() -> int | None:
    try:
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def cgroup_memory_limit_bytes() -> int | None:
    """Return effective cgroup memory limit, ignoring the unlimited sentinel."""
    paths = (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    )
    for path in paths:
        try:
            raw = open(path, encoding="ascii").read().strip()
            if raw and raw != "max":
                value = int(raw)
                if 0 < value < (1 << 60):
                    return value
        except (OSError, ValueError):
            continue
    return None


def swap_snapshot() -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                key, _, value = line.partition(":")
                if key in ("SwapFree", "SwapTotal"):
                    result[key] = int(value.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        with open("/proc/self/status", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("VmSwap:"):
                    result["VmSwap"] = int(line.split()[1]) * 1024
                    break
    except (OSError, ValueError, IndexError):
        pass
    return result


@dataclass(frozen=True)
class HostTierBudget:
    available_bytes: int
    cgroup_limit_bytes: int | None
    shared_bytes: int
    expert_bytes: int
    ple_page_bytes: int
    ple_row_bytes: int
    staging_bytes: int
    runtime_reserve_bytes: int
    safety_reserve_bytes: int
    requested_shared: int | str
    requested_expert: int | str
    requested_page: int | str
    requested_row: int | str
    degraded: bool = False

    @property
    def cache_bytes(self) -> int:
        return self.expert_bytes + self.ple_page_bytes + self.ple_row_bytes

    def report(self) -> dict[str, int | bool | None | str]:
        return {
            "available_bytes": self.available_bytes,
            "cgroup_limit_bytes": self.cgroup_limit_bytes,
            "shared_bytes": self.shared_bytes,
            "expert_bytes": self.expert_bytes,
            "ple_page_bytes": self.ple_page_bytes,
            "ple_row_bytes": self.ple_row_bytes,
            "staging_bytes": self.staging_bytes,
            "runtime_reserve_bytes": self.runtime_reserve_bytes,
            "safety_reserve_bytes": self.safety_reserve_bytes,
            "requested_shared": self.requested_shared,
            "requested_expert": self.requested_expert,
            "requested_page": self.requested_page,
            "requested_row": self.requested_row,
            "degraded": self.degraded,
            "swap": swap_snapshot(),
        }


def resolve_host_tier(
    *,
    expert_total_bytes: int,
    available_bytes: int | None = None,
    cgroup_limit_bytes: int | None = None,
    shared_mib: int | str = "auto",
    expert_mib: int | str = "auto",
    page_mib: int | str = "auto",
    row_mib: int | str = "auto",
    staging_bytes: int = 0,
    runtime_reserve_bytes: int = 2 << 30,
    safety_reserve_bytes: int = 1 << 30,
) -> HostTierBudget:
    available = mem_available_bytes() if available_bytes is None else int(available_bytes)
    if available is None:
        raise RuntimeError("cannot resolve Qwen3.8 host tier: MemAvailable is unavailable")
    if cgroup_limit_bytes is None:
        cgroup_limit_bytes = cgroup_memory_limit_bytes()
    if cgroup_limit_bytes is not None:
        # MemAvailable can include pages outside this process' cgroup. Never budget them.
        available = min(available, max(0, int(cgroup_limit_bytes)))
    shared_req = parse_budget(shared_mib, name="qwen38_host_cache_mib")
    expert_req = parse_budget(expert_mib, name="qwen38_expert_host_cache_mib")
    page_req = parse_budget(page_mib, name="ple_ram_cache_mib")
    row_req = parse_budget(row_mib, name="ple_row_cache_mib")
    headroom = max(0, int(runtime_reserve_bytes)) + max(0, int(safety_reserve_bytes))
    usable = max(0, available - headroom - max(0, int(staging_bytes)))
    # Auto remains deliberately conservative: host cache is hot working set, not a
    # second copy of either expert banks or PLE. Launcher may raise explicit caps.
    shared = min(4 << 30, usable) if shared_req in AUTO_VALUES else int(shared_req) << 20
    shared = max(0, shared)
    if shared > usable:
        raise MemoryError(
            f"Qwen3.8 host cache requires {shared / 2**30:.2f} GiB, "
            f"but only {usable / 2**30:.2f} GiB is available after reserves"
        )
    if expert_req in AUTO_VALUES:
        expert = min(max(0, int(expert_total_bytes)), shared * 3 // 4)
    else:
        expert = int(expert_req) << 20
    if page_req in AUTO_VALUES:
        page = shared * 3 // 16  # 75% of non-expert PLE slice below
    else:
        page = int(page_req) << 20
    if row_req in AUTO_VALUES:
        row = shared * 1 // 16
    else:
        row = int(row_req) << 20
    if page + row + expert > shared:
        raise MemoryError(
            "Qwen3.8 host cache sub-budgets exceed shared ceiling: "
            f"expert={expert}, ple_page={page}, ple_row={row}, shared={shared}"
        )
    return HostTierBudget(
        available_bytes=available,
        cgroup_limit_bytes=cgroup_limit_bytes,
        shared_bytes=shared,
        expert_bytes=expert,
        ple_page_bytes=page,
        ple_row_bytes=row,
        staging_bytes=max(0, int(staging_bytes)),
        runtime_reserve_bytes=max(0, int(runtime_reserve_bytes)),
        safety_reserve_bytes=max(0, int(safety_reserve_bytes)),
        requested_shared=shared_req,
        requested_expert=expert_req,
        requested_page=page_req,
        requested_row=row_req,
        degraded=shared < (4 << 30) and shared_req in AUTO_VALUES,
    )


__all__ = [
    "AUTO_VALUES", "HostTierBudget", "cgroup_memory_limit_bytes", "mem_available_bytes",
    "parse_budget", "resolve_host_tier", "swap_snapshot",
]
