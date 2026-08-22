"""Intrinsic embedding-quality checks, tested with small hand-constructed vectors -
no model, no stack. See `app/ingestion/embedding_quality.py`'s module docstring for
what these are checking and why they don't need a search endpoint to be useful.
"""

from __future__ import annotations

import math

import pytest

from app.ingestion.embedding_quality import (
    average_pairwise_similarity,
    cosine_similarity,
    group_contrast,
    nearest_neighbors,
)

# --- cosine_similarity ---------------------------------------------------------------


def test_identical_vectors_have_similarity_one() -> None:
    v = [0.6, 0.8]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_opposite_vectors_have_similarity_negative_one() -> None:
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


# --- nearest_neighbors -----------------------------------------------------------------


def test_nearest_neighbors_orders_by_similarity_descending() -> None:
    corpus = [
        ("close", [0.99, 0.14]),
        ("far", [0.0, 1.0]),
        ("exact", [1.0, 0.0]),
    ]
    result = nearest_neighbors([1.0, 0.0], corpus, k=3)
    assert [label for label, _ in result] == ["exact", "close", "far"]


def test_nearest_neighbors_excludes_the_anchor_itself() -> None:
    corpus = [("self", [1.0, 0.0]), ("other", [0.9, 0.1])]
    result = nearest_neighbors([1.0, 0.0], corpus, k=2, exclude="self")
    assert [label for label, _ in result] == ["other"]


def test_nearest_neighbors_respects_k() -> None:
    corpus = [(str(i), [1.0, float(i)]) for i in range(10)]
    result = nearest_neighbors([1.0, 0.0], corpus, k=3)
    assert len(result) == 3


# --- average_pairwise_similarity -------------------------------------------------------


def test_average_similarity_of_identical_vectors_is_one() -> None:
    vectors = [[0.6, 0.8]] * 10
    assert average_pairwise_similarity(vectors) == pytest.approx(1.0)


def test_average_similarity_of_orthogonal_pairs_is_near_zero() -> None:
    vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    # Every cross pair between the two directions is 0; same-direction pairs are 1.
    # With 2 of each, mean over all 6 pairs = (0*4 + 1*2) / 6 = 1/3.
    assert average_pairwise_similarity(vectors) == pytest.approx(1 / 3)


def test_average_similarity_with_fewer_than_two_vectors_is_zero() -> None:
    assert average_pairwise_similarity([]) == 0.0
    assert average_pairwise_similarity([[1.0, 0.0]]) == 0.0


def test_average_similarity_sampling_is_deterministic() -> None:
    vectors = [[math.cos(i), math.sin(i)] for i in range(200)]
    first = average_pairwise_similarity(vectors, n_pairs=50, seed=1)
    second = average_pairwise_similarity(vectors, n_pairs=50, seed=1)
    assert first == second


# --- group_contrast --------------------------------------------------------------------


def test_group_contrast_detects_a_real_signal() -> None:
    """Two groups, each internally near-identical and orthogonal to the other group -
    the embedding "captures" the grouping perfectly, so contrast should be large."""
    items = ["a1", "a2", "a3", "b1", "b2", "b3"]
    vectors = [
        [1.0, 0.01],
        [0.99, 0.02],
        [0.98, 0.01],
        [0.01, 1.0],
        [0.02, 0.99],
        [0.01, 0.98],
    ]

    def key(item: str) -> str:
        return item[0]

    result = group_contrast(items, vectors, key, grouping_field="prefix")
    assert result is not None
    assert result.within_group_mean > 0.9
    assert result.across_group_mean < 0.1
    assert result.contrast > 0.8


def test_group_contrast_is_none_when_fewer_than_two_groups_have_pairs() -> None:
    items = ["a1", "a2", "b1"]  # only group "a" has >= 2 members
    vectors = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
    result = group_contrast(items, vectors, lambda s: s[0], grouping_field="prefix")
    assert result is None


def test_group_contrast_ignores_items_with_no_group_key() -> None:
    items = ["a1", "a2", "a3", None]
    vectors = [[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.0, 1.0]]
    result = group_contrast(items, vectors, lambda s: s[0] if s else None, grouping_field="prefix")
    # Only one real group ("a") has pairs; the None-keyed item contributes nothing.
    assert result is None


def test_group_contrast_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        group_contrast(["a", "b"], [[1.0, 0.0]], lambda s: s, grouping_field="x")
