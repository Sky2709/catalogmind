"""`app/llm/claims.py` - verifying a chat answer's numeric claims against real
`get_catalog_stats` tool output this turn.

Retired this session (2026-08-22): guessing whether a sentence *meant* "nothing
meets this threshold" from its surface phrasing (`_QUANTIFIABLE_NEGATION_CUE`).
Claude now wraps any stats-backed figure in a `[[STAT:N]]` marker
(`app/llm/markers.py`) regardless of how the sentence around it is phrased, so
these tests check the marker directly rather than a growing list of negation
phrasings.
"""

from __future__ import annotations

from app.llm.claims import (
    find_stat_claim_mismatch,
    find_unverified_quantitative_refusal,
    has_superlative_language,
)

PRICE_STATS = {"kind": "stats", "metric": "price", "count": 12, "minimum": "499", "maximum": "58854", "mean": "5200"}
RATING_STATS = {"kind": "stats", "metric": "rating", "count": 12, "minimum": "2.0", "maximum": "4.8", "mean": "3.9"}


def test_stat_claim_matches_the_real_maximum() -> None:
    answer = "The highest priced item is [[STAT:58854]]."
    assert find_stat_claim_mismatch(answer, [PRICE_STATS]) is False


def test_stat_claim_mismatch_when_the_number_is_wrong() -> None:
    answer = "Nothing here goes above [[STAT:2499]]."
    assert find_stat_claim_mismatch(answer, [PRICE_STATS]) is True


def test_stat_claim_tolerates_minor_rounding() -> None:
    answer = "The priciest item is about [[STAT:58900]]."
    assert find_stat_claim_mismatch(answer, [PRICE_STATS]) is False


def test_stat_claim_unscored_with_no_marker_emitted() -> None:
    answer = "I found several options for you."
    assert find_stat_claim_mismatch(answer, [PRICE_STATS]) is None


def test_stat_claim_unscored_with_multiple_stats_calls() -> None:
    answer = "The highest priced item is [[STAT:58854]]."
    assert find_stat_claim_mismatch(answer, [PRICE_STATS, PRICE_STATS]) is None


def test_stat_claim_unscored_for_non_price_metric() -> None:
    """Rating/review_count claims aren't verified yet - a documented MVP
    boundary, not silently assumed correct."""
    answer = "The highest rated item is [[STAT:4.8]] stars."
    assert find_stat_claim_mismatch(answer, [RATING_STATS]) is None


def test_stat_claim_unscored_with_no_stats_evidence() -> None:
    assert find_stat_claim_mismatch("The highest priced item is [[STAT:58854]].", []) is None


def test_stat_claim_ignores_a_malformed_marker_rather_than_crashing() -> None:
    """A marker Claude fails to fill in with a real number - not this function's
    job to fix, just not to crash on."""
    answer = "The highest priced item is [[STAT:not-a-number]]."
    assert find_stat_claim_mismatch(answer, [PRICE_STATS]) is None


def test_unverified_quantitative_refusal_true_without_a_confirmed_stats_call() -> None:
    answer = "Nothing in our catalog goes above [[STAT:10000]]."
    assert find_unverified_quantitative_refusal(answer, []) is True


def test_unverified_quantitative_refusal_false_when_backed_by_a_real_stats_call() -> None:
    answer = "Nothing in our catalog goes above [[STAT:58854]]."
    assert find_unverified_quantitative_refusal(answer, [PRICE_STATS]) is False


def test_unverified_quantitative_refusal_false_with_no_marker_at_all() -> None:
    """A refusal that never states a checkable figure at all (e.g. a missing
    brand) shouldn't be flagged - there's nothing to verify."""
    answer = "Sorry, we don't carry that brand."
    assert find_unverified_quantitative_refusal(answer, []) is False


def test_superlative_language_detected_in_a_shopper_message() -> None:
    assert has_superlative_language("What's the highest priced item you have?")
    assert has_superlative_language("Do you have anything above ₹10,000?")
    assert has_superlative_language("How many kurtas do you have?")


def test_superlative_language_synonyms_confirmed_by_a_real_llm_judge() -> None:
    """`eval/measure_superlative_heuristic.py` measured these three as real,
    confirmed misses (2026-08-22) - not a guess, an LLM judge agreed all three
    genuinely need a `get_catalog_stats` call the original word list gave no
    signal for at all."""
    assert has_superlative_language("What is the priciest thing you sell?")
    assert has_superlative_language("What is the average price of your products?")
    assert has_superlative_language("What is the total number of products you have?")


def test_mean_alone_is_not_flagged_to_avoid_a_new_false_positive() -> None:
    """"average" was added as a confirmed synonym; "mean" deliberately was not -
    a bare `mean` would false-positive on ordinary usage that has nothing to do
    with a statistic."""
    assert not has_superlative_language("What do you mean by slim fit?")


def test_spelled_out_number_thresholds_are_a_named_accepted_gap() -> None:
    """Confirmed real gap, deliberately left unfixed: catching this generally
    needs a number-word parser, not one more keyword - pins the current,
    honest behaviour rather than silently expecting it to work."""
    assert not has_superlative_language("Do you carry anything over ten thousand rupees?")


def test_superlative_language_absent_from_an_ordinary_query() -> None:
    assert not has_superlative_language("I need waterproof hiking boots")
