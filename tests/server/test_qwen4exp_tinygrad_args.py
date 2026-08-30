import pytest

from freetoken.server.args import parse_args


def base(*extra):
    return ["--model", "dummy", "--device", "tinygrad", "--use-dummy-weight", *extra]


def test_ssd_is_deprecated_alias_for_paged_and_store_is_optional():
    args, _ = parse_args(base("--ple-mode", "ssd"))
    assert args.ple_mode == "paged" and args.ple_store is None


def test_ram_budget_leaves_headroom():
    with pytest.raises(SystemExit, match="process headroom"):
        parse_args(base("--ple-ram-gib", "48"))


def test_accepted_ple_values():
    args, _ = parse_args(base("--ple-mode", "ssd", "--ple-store", "/nvme/ple", "--ple-ram-gib", "8", "--ple-workers", "3", "--no-ple-prefetch"))
    assert args.ple_mode == "paged" and args.ple_ram_gib == 8 and args.ple_workers == 3 and not args.ple_prefetch


def test_new_paged_knobs_and_zero_gpu_cache_parse():
    args, _ = parse_args(base("--ple-mode", "auto", "--ple-store-build", "never",
                              "--ple-ram-cache-mib", "64", "--ple-gpu-cache-mib", "0",
                              "--ple-staging-mib", "8", "--ple-io", "buffered", "--ple-io-depth", "4"))
    assert (args.ple_mode, args.ple_store_build, args.ple_ram_cache_mib, args.ple_gpu_cache_mib,
            args.ple_staging_mib, args.ple_io, args.ple_io_depth) == ("auto", "never", 64, 0, 8, "buffered", 4)


def test_host_tier_and_auto_prefetch_options_parse():
    args, _ = parse_args(base(
        "--qwen38-expert-residency", "auto-tier",
        "--qwen38-host-cache-mib", "auto",
        "--qwen38-expert-host-cache-mib", "4096",
        "--ple-ram-cache-mib", "adaptive",
        "--ple-row-cache-mib", "256",
        "--ple-prefetch-depth", "auto",
    ))
    assert args.qwen38_expert_residency == "auto-tier"
    assert args.qwen38_host_cache_mib == "auto"
    assert args.qwen38_expert_host_cache_mib == 4096
    assert args.ple_ram_cache_mib == "adaptive"
    assert args.ple_row_cache_mib == 256
    assert args.ple_prefetch_depth == "auto"


def test_rocm_prefill_rollout_flags_are_explicit_and_default_off():
    args, _ = parse_args(base())
    assert not args.ple_batched_cache
    assert not args.ple_fused_dequant
    assert not args.qwen38_qsa_prefill_live_width
    assert args.ple_cache_policy == "lru"

    args, _ = parse_args(base(
        "--ple-cache-policy", "2q", "--ple-batched-cache", "--ple-fused-dequant",
        "--qwen38-qsa-prefill-live-width",
    ))
    assert args.ple_cache_policy == "2q"
    assert args.ple_batched_cache and args.ple_fused_dequant
    assert args.qwen38_qsa_prefill_live_width


def test_q8_kv_reserve_and_fallback_parse():
    args, _ = parse_args(base(
        "--kv-cache-dtype", "q8",
        "--kv-reserve-tokens", "131072",
        "--kv-reserve-fallback-tokens", "98304",
    ))
    assert args.kv_cache_dtype == "q8"
    assert args.kv_reserve_tokens == 131072
    assert args.kv_reserve_fallback_tokens == 98304
