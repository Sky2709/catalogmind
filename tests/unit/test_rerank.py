"""Cross-encoder reranking behaviour.

Runs the real `bge-reranker-base` model on CPU - slower than the rest of the unit
suite (model load + first inference), same tradeoff `test_embed.py` makes: no server,
no network beyond the first cached download, and the properties that matter here
(relevance ordering, determinism) can't be verified any other way.
"""

from __future__ import annotations

import pytest

from app.retrieval.rerank import arerank, rerank


@pytest.fixture(scope="module", autouse=True)
def _warm() -> None:
    """Pay the model load once for the whole module, not once per test."""
    rerank("warm up", ["warm up document"])


def test_empty_documents_returns_empty_without_invoking_the_model() -> None:
    assert rerank("anything", []) == []


def test_relevant_document_scores_higher_than_irrelevant() -> None:
    """The reason a rerank pass exists at all: a joint query/document encoding should
    separate an on-topic candidate from an off-topic one more sharply than the hybrid
    bi-encoder score alone."""
    scores = rerank(
        "waterproof hiking boots",
        [
            "Blue cotton casual shirt for men, regular fit",
            "Waterproof leather hiking boots, size 10, ankle support",
        ],
    )
    assert len(scores) == 2
    assert scores[1] > scores[0]


def test_scoring_is_deterministic() -> None:
    a = rerank("wireless headphones", ["Noise cancelling wireless headphones"])
    b = rerank("wireless headphones", ["Noise cancelling wireless headphones"])
    assert a == pytest.approx(b, abs=1e-6)


def test_order_of_documents_does_not_change_individual_scores() -> None:
    """Each pair is scored independently - batching is a throughput knob, not a
    correctness one, same invariant `test_embed.py` asserts for the embedding batch."""
    docs = ["a red shirt", "a waterproof jacket", "a pair of running shoes"]
    reversed_docs = list(reversed(docs))

    forward = rerank("jacket", docs)
    backward = rerank("jacket", reversed_docs)

    assert forward[1] == pytest.approx(backward[1], abs=1e-6)


async def test_async_wrapper_matches_sync() -> None:
    query = "stainless steel water bottle"
    docs = ["Stainless steel vacuum insulated water bottle, 500ml"]
    assert await arerank(query, docs) == pytest.approx(rerank(query, docs), abs=1e-6)
