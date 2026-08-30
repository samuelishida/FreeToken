from __future__ import annotations

import pytest

from freetoken.engine.host_tier import parse_budget, resolve_host_tier


def test_auto_tier_uses_bounded_split_and_reports_swap(monkeypatch):
    monkeypatch.setattr(
        "freetoken.engine.host_tier.swap_snapshot",
        lambda: {"SwapTotal": 8, "SwapFree": 8, "VmSwap": 0},
    )
    tier = resolve_host_tier(
        expert_total_bytes=40 << 30,
        available_bytes=46 << 30,
        cgroup_limit_bytes=None,
        shared_mib="auto",
        expert_mib="auto",
        page_mib="auto",
        row_mib="auto",
        staging_bytes=32 << 20,
    )
    assert tier.shared_bytes == 4 << 30
    assert tier.expert_bytes == 3 << 30
    assert tier.ple_page_bytes == 768 << 20
    assert tier.ple_row_bytes == 256 << 20
    assert tier.cache_bytes <= tier.shared_bytes
    assert tier.report()["swap"]["VmSwap"] == 0


def test_explicit_host_subbudgets_fail_before_allocation():
    with pytest.raises(MemoryError, match="sub-budgets exceed shared"):
        resolve_host_tier(
            expert_total_bytes=1 << 30,
            available_bytes=16 << 30,
            shared_mib=1024,
            expert_mib=800,
            page_mib=200,
            row_mib=100,
            staging_bytes=0,
            runtime_reserve_bytes=0,
            safety_reserve_bytes=0,
        )


def test_cgroup_limit_caps_available_memory():
    with pytest.raises(MemoryError, match="only"):
        resolve_host_tier(
            expert_total_bytes=1 << 30,
            available_bytes=8 << 30,
            cgroup_limit_bytes=512 << 20,
            shared_mib=1024,
            expert_mib="auto",
            page_mib="auto",
            row_mib="auto",
            staging_bytes=64 << 20,
            runtime_reserve_bytes=0,
            safety_reserve_bytes=0,
        )


def test_budget_parser_preserves_legacy_zero():
    assert parse_budget("adaptive", name="x") == "adaptive"
    assert parse_budget("0", name="x") == 0
    with pytest.raises(ValueError):
        parse_budget("-1", name="x")
