"""`eval/golden_chat`'s scenario construction - pure, no stack or LLM call needed.
Covers the mechanical bits that make the ~220-scenario scale trustworthy: every
grounded scenario really does carry a non-empty, positively-judged SKU set
derived from `eval/golden/`, the message-templating rule behaves as documented,
and ids stay unique once the derived and hand-picked scenarios are combined.
"""

from __future__ import annotations

import pytest

from eval.golden import load_golden_set
from eval.golden_chat import _as_chat_message, load_chat_scenarios

TENANTS = ("demo-fashion-in", "demo-electronics-in", "demo-home-goods")


def test_bare_numeric_query_gets_the_identifier_wrapper() -> None:
    assert _as_chat_message("10015819") == "Do you have product 10015819 in stock?"


def test_ordinary_query_gets_the_looking_for_wrapper() -> None:
    assert _as_chat_message("Raymond formal shirt") == "I'm looking for Raymond formal shirt."


def test_grounded_scenario_count_matches_positively_judged_golden_queries() -> None:
    for tenant in TENANTS:
        golden = load_golden_set(tenant)
        expected_grounded = sum(1 for q in golden if any(r > 0 for r in q.judgments.values()))
        scenarios = load_chat_scenarios(tenant)
        grounded = [s for s in scenarios if s.kind == "grounded"]
        assert len(grounded) == expected_grounded, tenant


def test_every_grounded_scenario_has_a_nonempty_expected_sku_set() -> None:
    for tenant in TENANTS:
        for scenario in load_chat_scenarios(tenant):
            if scenario.kind == "grounded":
                assert scenario.expected_skus, scenario.id


def test_every_refusal_scenario_has_no_expected_skus() -> None:
    for tenant in TENANTS:
        for scenario in load_chat_scenarios(tenant):
            if scenario.kind == "refusal":
                assert scenario.expected_skus == frozenset()


def test_scenario_ids_are_unique_within_a_tenant() -> None:
    for tenant in TENANTS:
        ids = [s.id for s in load_chat_scenarios(tenant)]
        assert len(ids) == len(set(ids)), tenant


def test_unknown_tenant_raises() -> None:
    with pytest.raises(ValueError, match="no chat scenarios registered"):
        load_chat_scenarios("not-a-real-tenant")
