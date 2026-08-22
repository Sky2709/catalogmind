"""`eval/generation_metrics.py`'s pure heuristics - free to run, no stack or LLM
call needed. Mirrors `test_citations.py`'s style for the shared
`find_hallucinated_citations` path; adds coverage for the refusal detector (now
`[[NO_MATCH]]`-first, `refuses()` as backstop - see `app/llm/markers.py`) and the
price-mismatch check.
"""

from __future__ import annotations

from eval.generation_metrics import (
    aggregate_generation,
    find_price_mismatch,
    refuses,
    score_scenario,
)

BOOT = {"sku": "BOOT-WP-10", "title": "Waterproof Hiking Boot", "price": "1299.00"}


def test_refusal_cue_is_detected_in_a_plain_no() -> None:
    """`refuses()` itself is unchanged - still the backstop for a compliance
    miss (Claude phrased a refusal without the [[NO_MATCH]] marker)."""
    assert refuses("Sorry, we don't carry smartphones in this catalog.")


def test_refusal_cue_absent_from_an_ordinary_recommendation() -> None:
    assert not refuses("I'd recommend the BOOT-WP-10 - it's waterproof and well reviewed.")


def test_price_mismatch_false_when_the_mentioned_price_matches() -> None:
    answer = "The BOOT-WP-10 is available for ₹1,299."
    assert find_price_mismatch(answer, [BOOT]) is False


def test_price_mismatch_true_when_the_mentioned_price_is_wrong() -> None:
    answer = "The BOOT-WP-10 is available for ₹499."
    assert find_price_mismatch(answer, [BOOT]) is True


def test_price_mismatch_tolerates_minor_rounding() -> None:
    answer = "The BOOT-WP-10 costs about ₹1,300."
    assert find_price_mismatch(answer, [BOOT]) is False


def test_price_mismatch_unscored_when_no_price_is_mentioned() -> None:
    answer = "The BOOT-WP-10 is a great waterproof boot."
    assert find_price_mismatch(answer, [BOOT]) is None


def test_price_mismatch_unscored_when_multiple_products_are_priced() -> None:
    other = {"sku": "SHIRT-CTN-M", "title": "Cotton Shirt", "price": "899.00"}
    answer = "The BOOT-WP-10 is ₹1,299 and the SHIRT-CTN-M is ₹899."
    assert find_price_mismatch(answer, [BOOT, other]) is None


def test_price_mismatch_unscored_when_no_product_has_a_price() -> None:
    unpriced = {"sku": "BOOT-WP-10", "title": "Waterproof Hiking Boot", "price": None}
    answer = "The BOOT-WP-10 costs ₹1,299."
    assert find_price_mismatch(answer, [unpriced]) is None


def test_score_scenario_grounded_hit() -> None:
    result = score_scenario(
        scenario_id="s1",
        kind="grounded",
        expected_skus=frozenset({"BOOT-WP-10"}),
        answer="I'd recommend the [[SKU:BOOT-WP-10]] for ₹1,299 - waterproof and well reviewed.",
        cited_products=[BOOT],
    )
    assert result.grounded is True
    assert result.answer_correct is True
    assert result.price_mismatch is False
    assert result.refused is None


def test_score_scenario_grounded_miss_wrong_sku_cited() -> None:
    """Citing a real, retrieved SKU that isn't the expected one is grounded
    (no fabrication) but not a correct answer - the two questions are independent."""
    other = {"sku": "SHIRT-CTN-M", "title": "Cotton Shirt", "price": "899.00"}
    result = score_scenario(
        scenario_id="s2",
        kind="grounded",
        expected_skus=frozenset({"BOOT-WP-10"}),
        answer="The [[SKU:SHIRT-CTN-M]] looks like a good fit.",
        cited_products=[other],
    )
    assert result.grounded is True
    assert result.answer_correct is False


def test_score_scenario_flags_a_hallucinated_citation() -> None:
    result = score_scenario(
        scenario_id="s3",
        kind="grounded",
        expected_skus=frozenset({"BOOT-WP-10"}),
        answer="Try the [[SKU:JACKET-RAIN-99]] instead.",
        cited_products=[BOOT],
    )
    assert result.grounded is False
    assert result.hallucinated_citations == ("JACKET-RAIN-99",)
    assert result.answer_correct is False


def test_score_scenario_refusal_correct_via_marker() -> None:
    result = score_scenario(
        scenario_id="s4",
        kind="refusal",
        expected_skus=frozenset(),
        answer="[[NO_MATCH]] Sorry, we don't carry smartphones in this catalog.",
        cited_products=[],
    )
    assert result.grounded is True
    assert result.refused is True
    assert result.answer_correct is None


def test_score_scenario_refusal_correct_via_backstop_without_the_marker() -> None:
    """A compliance miss (no [[NO_MATCH]] marker) still scores correctly via the
    `refuses()` backstop - scoring with the same two-signal logic production uses."""
    result = score_scenario(
        scenario_id="s4b",
        kind="refusal",
        expected_skus=frozenset(),
        answer="Sorry, we don't carry smartphones in this catalog.",
        cited_products=[],
    )
    assert result.refused is True


def test_score_scenario_refusal_incorrect_when_not_phrased_as_a_refusal() -> None:
    result = score_scenario(
        scenario_id="s5",
        kind="refusal",
        expected_skus=frozenset(),
        answer="The [[SKU:BOOT-WP-10]] might work for that.",
        cited_products=[BOOT],
    )
    assert result.refused is False


def test_aggregate_generation_reports_undefined_as_none_not_zero() -> None:
    agg = aggregate_generation([])
    assert agg["n_scenarios"] == 0
    assert agg["groundedness_rate"] is None
    assert agg["answer_hit_rate"] is None
    assert agg["refusal_correctness"] is None
    assert agg["hallucinated_attribute_rate"] is None


def test_aggregate_generation_computes_rates_across_mixed_scenarios() -> None:
    grounded_hit = score_scenario(
        scenario_id="g1",
        kind="grounded",
        expected_skus=frozenset({"BOOT-WP-10"}),
        answer="The [[SKU:BOOT-WP-10]] fits.",
        cited_products=[BOOT],
    )
    refusal_correct = score_scenario(
        scenario_id="r1",
        kind="refusal",
        expected_skus=frozenset(),
        answer="[[NO_MATCH]] Sorry, we don't sell that here.",
        cited_products=[],
    )
    refusal_incorrect = score_scenario(
        scenario_id="r2",
        kind="refusal",
        expected_skus=frozenset(),
        answer="The [[SKU:BOOT-WP-10]] might work.",
        cited_products=[BOOT],
    )

    agg = aggregate_generation([grounded_hit, refusal_correct, refusal_incorrect])
    assert agg["n_scenarios"] == 3
    assert agg["groundedness_rate"] == 1.0  # none of these three fabricated a citation
    assert agg["answer_hit_rate"] == 1.0
    assert agg["answer_hit_rate_n"] == 1
    assert agg["refusal_correctness"] == 0.5  # one of two refusal scenarios refused
    assert agg["refusal_correctness_n"] == 2
