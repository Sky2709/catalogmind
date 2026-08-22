"""Retrieval metric maths.

Expected values here are hand-computed and shown in the docstrings, not produced by
running the code and pasting the output. A metric test that just records current
behaviour cannot catch a wrong formula.
"""

from __future__ import annotations

import math

import pytest

from eval.metrics import (
    QueryScores,
    aggregate,
    dcg_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_query,
)

GRADED = {"A": 3, "B": 2, "C": 1}
BINARY = {"A": 1, "B": 1, "C": 1, "D": 1}


# --- DCG / nDCG ------------------------------------------------------------------


def test_dcg_exponential_hand_computed() -> None:
    """ranking [A,B,C] with rel 3,2,1 and exponential gain:

      (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4)
    = 7/1 + 3/1.5849625 + 1/2
    = 7 + 1.8927893 + 0.5
    = 9.3927893
    """
    expected = 7 + 3 / math.log2(3) + 0.5
    assert dcg_at_k(["A", "B", "C"], GRADED, 3) == pytest.approx(expected)
    assert dcg_at_k(["A", "B", "C"], GRADED, 3) == pytest.approx(9.3927893, abs=1e-6)


def test_dcg_linear_hand_computed() -> None:
    """Same ranking, linear gain: 3/1 + 2/log2(3) + 1/2 = 4.7618595."""
    expected = 3 + 2 / math.log2(3) + 0.5
    got = dcg_at_k(["A", "B", "C"], GRADED, 3, exponential=False)
    assert got == pytest.approx(expected)
    assert got == pytest.approx(4.7618595, abs=1e-6)


def test_ndcg_is_one_for_ideal_ranking() -> None:
    assert ndcg_at_k(["A", "B", "C"], GRADED, 3) == pytest.approx(1.0)


def test_ndcg_reversed_ranking_hand_computed() -> None:
    """ranking [C,B,A] reverses the grades:

    DCG  = 1/1 + 3/log2(3) + 7/2      = 6.3927893
    IDCG = 7/1 + 3/log2(3) + 1/2      = 9.3927893
    nDCG = 6.3927893 / 9.3927893      = 0.680606
    """
    assert ndcg_at_k(["C", "B", "A"], GRADED, 3) == pytest.approx(0.680606, abs=1e-6)


def test_ndcg_single_relevant_at_rank_three() -> None:
    """One relevant doc at position 3: DCG = 1/log2(4) = 0.5, IDCG = 1, so nDCG = 0.5."""
    assert ndcg_at_k(["X", "Y", "Z2"], {"Z2": 1}, 10) == pytest.approx(0.5)


def test_gain_functions_agree_on_binary_relevance() -> None:
    """2**1 - 1 == 1, so the two conventions coincide when all grades are 0/1."""
    ranking = ["A", "X", "B"]
    assert ndcg_at_k(ranking, BINARY, 10) == pytest.approx(
        ndcg_at_k(ranking, BINARY, 10, exponential=False)
    )


def test_gain_functions_diverge_on_graded_relevance() -> None:
    """Exponential gain punishes burying the best item harder than linear does."""
    ranking = ["C", "B", "A"]
    assert ndcg_at_k(ranking, GRADED, 3) < ndcg_at_k(ranking, GRADED, 3, exponential=False)


def test_ndcg_is_bounded() -> None:
    for ranking in (["A", "B", "C"], ["C", "A", "B"], ["X", "Y"], []):
        value = ndcg_at_k(ranking, GRADED, 10)
        assert value is None or 0.0 <= value <= 1.0


def test_k_truncates() -> None:
    """Only the top k contributes; the relevant item at rank 3 is outside k=2."""
    assert ndcg_at_k(["X", "Y", "A"], {"A": 1}, 2) == pytest.approx(0.0)


# --- recall / precision ----------------------------------------------------------


def test_recall_at_k_hand_computed() -> None:
    """2 of 4 relevant items land in the top 5 -> 0.5."""
    assert recall_at_k(["A", "B", "X", "Y", "Z"], BINARY, 5) == pytest.approx(0.5)


def test_recall_ignores_grade_magnitude() -> None:
    """Recall is a set operation - a grade of 3 counts once, same as a grade of 1."""
    assert recall_at_k(["A"], GRADED, 10) == pytest.approx(1 / 3)


def test_precision_denominator_is_k_not_result_count() -> None:
    """2 relevant in a 3-item result list, precision@10 = 2/10, not 2/3.

    A retriever that under-fills its result list is penalised for the slots it left
    empty, which is the honest reading when more relevant items exist.
    """
    assert precision_at_k(["A", "B", "X"], BINARY, 10) == pytest.approx(0.2)


# --- MRR / hit rate --------------------------------------------------------------


def test_reciprocal_rank_uses_first_hit() -> None:
    assert reciprocal_rank(["X", "Y", "A"], BINARY) == pytest.approx(1 / 3)
    assert reciprocal_rank(["A", "B"], BINARY) == pytest.approx(1.0)


def test_reciprocal_rank_is_zero_when_relevant_exists_but_is_missed() -> None:
    """A genuine failure, distinct from 'nothing was relevant' (which is None)."""
    assert reciprocal_rank(["X", "Y", "Z"], BINARY) == 0.0


def test_hit_rate() -> None:
    assert hit_rate_at_k(["X", "A"], BINARY, 10) == 1.0
    assert hit_rate_at_k(["X", "Y"], BINARY, 10) == 0.0


# --- undefined vs zero -----------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        lambda r, j: ndcg_at_k(r, j, 10),
        lambda r, j: recall_at_k(r, j, 10),
        lambda r, j: precision_at_k(r, j, 10),
        lambda r, j: reciprocal_rank(r, j),
        lambda r, j: hit_rate_at_k(r, j, 10),
    ],
)
def test_no_relevant_items_is_undefined_not_zero(fn) -> None:
    """Scoring an unlabelled query as 0.0 would drag averages down in proportion to
    how sloppy the labelling was - exactly backwards."""
    assert fn(["A", "B"], {}) is None
    assert fn(["A", "B"], {"A": 0, "B": 0}) is None


def test_empty_ranking_scores_zero_not_none() -> None:
    """There *was* something to find; returning nothing is a real failure."""
    assert ndcg_at_k([], BINARY, 10) == pytest.approx(0.0)
    assert recall_at_k([], BINARY, 10) == pytest.approx(0.0)
    assert reciprocal_rank([], BINARY) == 0.0


# --- gaming defences -------------------------------------------------------------


def test_duplicate_results_cannot_inflate_recall() -> None:
    """Padding the list with repeats must not help.

    Without dedupe, [A,A] at k=2 would occupy both slots with one document and score
    the same recall as genuinely finding two.
    """
    assert recall_at_k(["A", "A"], BINARY, 2) == pytest.approx(0.25)
    assert recall_at_k(["A", "A", "B"], BINARY, 2) == pytest.approx(0.5)


def test_duplicates_do_not_inflate_dcg() -> None:
    assert dcg_at_k(["A", "A"], {"A": 1}, 10) == pytest.approx(1.0)


# --- aggregation -----------------------------------------------------------------


def _scores(query_id: str, ndcg: float | None) -> QueryScores:
    return QueryScores(
        query_id=query_id,
        ndcg_at_10=ndcg,
        recall_at_10=None,
        recall_at_50=None,
        precision_at_10=None,
        reciprocal_rank=None,
        hit_rate_at_10=None,
    )


def test_aggregate_skips_undefined_and_reports_support() -> None:
    """Mean over the two defined values, and n makes the thin support visible."""
    result = aggregate([_scores("q1", 1.0), _scores("q2", 0.0), _scores("q3", None)])
    assert result["n_queries"] == 3
    assert result["ndcg@10"] == pytest.approx(0.5)
    assert result["ndcg@10_n"] == 2
    assert result["recall@10"] is None
    assert result["recall@10_n"] == 0


def test_aggregate_handles_empty_input() -> None:
    assert aggregate([]) == {"n_queries": 0}


def test_score_query_end_to_end() -> None:
    scores = score_query("q1", ["X", "A", "B"], BINARY)
    assert scores.query_id == "q1"
    assert scores.reciprocal_rank == pytest.approx(0.5)
    assert scores.recall_at_10 == pytest.approx(0.5)
    assert scores.hit_rate_at_10 == 1.0
    assert set(scores.as_dict()) == {
        "ndcg@10",
        "recall@10",
        "recall@50",
        "precision@10",
        "mrr",
        "hit_rate@10",
    }
