from freetoken.attention.qsa_sparse import _make_prefill_plan


def test_prefill_plan_keeps_nonempty_tail_geometry_before_first_complete_group():
    plan = _make_prefill_plan([1, 3], ratio=4, token_budget=2048, cmp_page_size=16)
    assert (plan.page_columns, plan.score_columns, plan.token_topk, plan.select_width) == (1, 16, 4, 7)


def test_prefill_plan_aligns_score_pages_and_caps_selected_tokens():
    plan = _make_prefill_plan([1024, 65], ratio=4, token_budget=2048, cmp_page_size=16)
    assert (plan.page_columns, plan.score_columns) == (16, 256)
    assert (plan.token_topk, plan.select_width) == (1024, 1027)
    capped = _make_prefill_plan([32768], ratio=4, token_budget=2048, cmp_page_size=16)
    assert (capped.score_columns, capped.token_topk, capped.select_width) == (8192, 2048, 2051)
