"""Local embedding behaviour.

These run the real model on CPU. They are slower than the rest of the unit suite
(~15s for the one-time load) but they are not integration tests - no server, no
network beyond the first cached download. The properties asserted here are the ones
that silently wreck retrieval when they break: dimension, normalisation, determinism,
and asymmetric query/document encoding.
"""

from __future__ import annotations

import math

import pytest

from app.config import get_settings
from app.ingestion.embed import (
    QUERY_INSTRUCTION,
    embed_documents,
    embed_queries,
    embed_query,
    embedding_dimension,
)


@pytest.fixture(scope="module")
def dim() -> int:
    return embedding_dimension()


def test_dimension_matches_configured_value(dim: int) -> None:
    """A mismatch here becomes a Weaviate insert failure much later and less clearly."""
    assert dim == get_settings().embedding_dim


def test_vectors_are_l2_normalised(dim: int) -> None:
    """Cosine reduces to a dot product only if vectors are unit length.

    Weaviate's COSINE distance and the model's training objective both assume this.
    """
    for vector in embed_documents(["a cotton shirt", "wireless headphones"]):
        assert math.isclose(sum(x * x for x in vector) ** 0.5, 1.0, abs_tol=1e-5)


def test_document_embedding_shape(dim: int) -> None:
    vectors = embed_documents(["one", "two", "three"])
    assert len(vectors) == 3
    assert all(len(v) == dim for v in vectors)


def test_embedding_is_deterministic() -> None:
    """Non-determinism would make the alpha sweep unreproducible."""
    assert (
        embed_documents(["stainless steel bottle"])[0]
        == (embed_documents(["stainless steel bottle"])[0])
    )


def test_empty_input_returns_empty_without_invoking_the_model() -> None:
    assert embed_documents([]) == []
    assert embed_queries([]) == []


def test_query_instruction_changes_the_vector() -> None:
    """Proves the asymmetry is actually applied, not silently dropped.

    If the prefix were ignored, both arms of the Day-4 instruction experiment would
    produce identical numbers and the comparison would be meaningless.
    """
    with_instr = embed_query("warm jacket", use_instruction=True)
    without = embed_query("warm jacket", use_instruction=False)
    assert with_instr != without


def test_query_instruction_is_equivalent_to_manual_prefixing() -> None:
    """Pins exactly what the flag does, so the eval arm is unambiguous."""
    auto = embed_query("warm jacket", use_instruction=True)
    manual = embed_documents([QUERY_INSTRUCTION + "warm jacket"])[0]
    assert auto == pytest.approx(manual, abs=1e-6)


def test_documents_are_never_instruction_prefixed() -> None:
    """The passage side must stay bare - that is what 'asymmetric' means here."""
    bare = embed_documents(["warm jacket"])[0]
    prefixed = embed_documents([QUERY_INSTRUCTION + "warm jacket"])[0]
    assert bare != prefixed


def test_semantic_similarity_beats_lexical_overlap() -> None:
    """The reason a vector half exists at all.

    'keep drinks cold on a hike' shares no content words with the bottle listing, yet
    must rank above a shirt that shares none either. If this fails, the embedding
    model or its normalisation is broken, and no amount of alpha tuning will save the
    exploratory query class.
    """
    catalog = [
        "Blue cotton casual shirt for men, regular fit",
        "Stainless steel vacuum insulated water bottle, 500ml",
    ]
    doc_vectors = embed_documents(catalog)
    query_vector = embed_query("keep my drink cold on a hike")

    scores = [sum(q * d for q, d in zip(query_vector, doc, strict=True)) for doc in doc_vectors]
    assert scores[1] > scores[0]


@pytest.mark.parametrize("batch_size", [1, 2, 100])
def test_batch_size_does_not_change_results(batch_size: int) -> None:
    """Batching is a throughput knob, never a correctness one."""
    texts = ["alpha", "beta", "gamma"]
    reference = embed_documents(texts, batch_size=1)
    actual = embed_documents(texts, batch_size=batch_size)

    assert len(actual) == len(reference)
    # pytest.approx does not descend into nested sequences - compare row by row.
    for got, want in zip(actual, reference, strict=True):
        assert got == pytest.approx(want, abs=1e-6)


async def test_async_wrapper_matches_sync() -> None:
    from app.ingestion.embed import aembed_query

    assert await aembed_query("test", use_instruction=False) == pytest.approx(
        embed_query("test", use_instruction=False), abs=1e-6
    )
