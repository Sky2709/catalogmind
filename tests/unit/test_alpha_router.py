"""Query classification and alpha routing.

These cases are the contract for the dynamic-alpha feature. If the classifier
regresses, the alpha sweep's headline result stops reproducing - so these run on
every PR, before the (much slower) retrieval eval.
"""

from __future__ import annotations

import pytest

from app.retrieval.alpha_router import (
    PRIOR_ALPHA,
    AlphaRouter,
    classify,
    has_identifier_shaped_token,
)
from app.retrieval.base import QueryClass

# --- identifier: exact tokens, BM25 should dominate ---------------------------
IDENTIFIER_QUERIES = [
    "DW-4402B",
    "A1502",
    "55X900H",
    "iPhone 15",
    "WH-1000XM5",
    "sony wh1000xm5",
    "LG 55UP7500",
    '"Air Max 90"',
]

# --- exploratory: intent without catalog vocabulary, vector should dominate ---
EXPLORATORY_QUERIES = [
    "something for a beach wedding",
    "gift ideas for my mother",
    "what should I wear to a summer party",
    "looking for something casual and comfortable",
    "recommend a present for a colleague who likes cooking",
    "I need something suitable for a formal occasion",
    "outfit that goes with white sneakers",
]

# --- attribute: constrained but natural, both signals contribute --------------
ATTRIBUTE_QUERIES = [
    "waterproof hiking boots size 10",
    "cotton shirt size XL",
    "500ml stainless steel water bottle",
    "32 inch 4k smart tv",
    "wireless noise cancelling headphones",
    "black leather wallet for men",
]


@pytest.mark.parametrize("query", IDENTIFIER_QUERIES)
def test_identifier_queries_classified(query: str) -> None:
    assert classify(query).query_class is QueryClass.IDENTIFIER


@pytest.mark.parametrize("query", EXPLORATORY_QUERIES)
def test_exploratory_queries_classified(query: str) -> None:
    assert classify(query).query_class is QueryClass.EXPLORATORY


@pytest.mark.parametrize("query", ATTRIBUTE_QUERIES)
def test_attribute_queries_classified(query: str) -> None:
    assert classify(query).query_class is QueryClass.ATTRIBUTE


# --- casual/formal: garment-style attribute words, not occasion cues ----------
# A real bug caught building `eval/golden/demo_fashion_in.py`: both words used to sit
# in `_EXPLORATORY_CUE`, and "Raymond formal shirt" landed in EXPLORATORY (alpha 0.75)
# purely off "formal" - not cosmetic, that query's nDCG@10 went from ~1.0 at the alpha
# ATTRIBUTE actually needed down to 0.0 at the alpha the wrong class got instead.
GARMENT_STYLE_QUERIES = [
    "Raymond formal shirt",
    "Indian Terrain men slim fit casual shirt",
    "casual sneakers for men",
    "formal shoes size 9",
]


@pytest.mark.parametrize("query", GARMENT_STYLE_QUERIES)
def test_casual_and_formal_classify_as_attribute_not_exploratory(query: str) -> None:
    assert classify(query).query_class is QueryClass.ATTRIBUTE


@pytest.mark.parametrize(
    "query",
    [
        "looking for something casual and comfortable",
        "I need something suitable for a formal occasion",
        "something casual for a beach vacation",
    ],
)
def test_casual_and_formal_still_classify_as_exploratory_with_real_intent_language(
    query: str,
) -> None:
    """The fix must not swallow genuinely occasion-shaped queries that happen to use
    the same words - only the bare garment-style usage (no competing intent cue) was
    the bug."""
    assert classify(query).query_class is QueryClass.EXPLORATORY


# --- need/purpose framing and usage-context: exploratory without an occasion word ---
# Real bug caught by `eval/retrieval_eval.py`'s *live* classification (not the golden
# label, which decides the alpha actually measured): 23 of 170 golden EXPLORATORY
# queries across all three catalogs had no word in `_EXPLORATORY_CUE` and lost the
# unconditional "mid-length query" ATTRIBUTE bonus to the weaker "no numerics"
# EXPLORATORY one - fashion's worst-scoring cell (nDCG@10=0.7379) traced directly to
# this. See PROGRESS.md for the full investigation.
NEED_FRAME_QUERIES = [
    "way to back up my photos",
    "device to track my daily fitness",
    "gear for taking calls hands-free while driving",
    "solution for slow wifi at home",
    "protect my new smartphone screen from scratches",
]

CONTEXT_CUE_QUERIES = [
    "comfortable footwear for daily wear",
    "workout clothes for the gym",
    "sports shoes for running",
    "bag for daily college use",
    "baby clothing for a newborn",
]


@pytest.mark.parametrize("query", NEED_FRAME_QUERIES)
def test_need_frame_queries_classify_as_exploratory(query: str) -> None:
    assert classify(query).query_class is QueryClass.EXPLORATORY


@pytest.mark.parametrize("query", CONTEXT_CUE_QUERIES)
def test_usage_context_queries_classify_as_exploratory(query: str) -> None:
    assert classify(query).query_class is QueryClass.EXPLORATORY


def test_running_shoes_stays_attribute_not_exploratory() -> None:
    """A real false positive caught while adding the usage-context cue above: a bare
    'running' cue misfired on this product-category phrase (the shoe *type*, not a
    purpose/activity description) - narrowed to the 'for running' phrase instead."""
    assert classify("men's running shoes").query_class is QueryClass.ATTRIBUTE


# --- signal-less input: must not default to an extreme -------------------------


@pytest.mark.parametrize("query", ["!!!", "...", "\U0001f45f\U0001f45f\U0001f45f"])
def test_signal_less_query_defaults_to_attribute_not_identifier(query: str) -> None:
    """Pure punctuation or emoji tokenizes to zero real tokens - a real bug had this
    satisfying the 'very short query' (<=3 tokens) check anyway and landing in
    IDENTIFIER (alpha=0.1, the worst possible choice for content with zero chance of
    BM25 term overlap), instead of the documented 'safe middle' default."""
    result = classify(query)
    assert result.query_class is QueryClass.ATTRIBUTE
    assert result.confidence == 0.0


def test_very_long_single_token_does_not_classify_as_short_query() -> None:
    """A pasted URL or paste-error with no spaces is one 'token' by count but not
    short by any reasonable definition - must not route keyword-heavy just because it
    has no whitespace."""
    assert classify("a" * 10_000).query_class is not QueryClass.IDENTIFIER


def test_empty_query_does_not_crash() -> None:
    result = classify("   ")
    assert result.confidence == 0.0
    assert result.query_class in set(QueryClass)


def test_classification_always_reports_reasons() -> None:
    """Every decision must be explainable - this is what makes the router debuggable."""
    for query in IDENTIFIER_QUERIES + EXPLORATORY_QUERIES + ATTRIBUTE_QUERIES:
        assert classify(query).reasons, f"no reasons recorded for {query!r}"


def test_confidence_is_a_proportion() -> None:
    for query in IDENTIFIER_QUERIES + EXPLORATORY_QUERIES + ATTRIBUTE_QUERIES:
        assert 0.0 <= classify(query).confidence <= 1.0


# --- has_identifier_shaped_token: narrower than the IDENTIFIER class -----------


@pytest.mark.parametrize(
    "query",
    [
        "DW-4402B",
        "A1502",
        "55X900H",
        "WH-1000XM5",
        "sony wh1000xm5",
        "get me a WH-1000XM5 please",
        "10015819",
        "find me 10029129 please",
    ],
)
def test_has_identifier_shaped_token_true_for_real_product_codes(query: str) -> None:
    assert has_identifier_shaped_token(query)


def test_has_identifier_shaped_token_true_for_bare_numeric_skus() -> None:
    """A real bug caught building `eval/golden/demo_fashion_in.py`: Myntra-style SKUs
    are all-digit ('10015819'), so the original digit+letter-mix pattern never
    matched them and reranking (no SKU in `rerank_text()`) was destroying an
    otherwise-perfect top-1 match. Requires 6+ digits specifically - see the next
    test for why fewer must not match."""
    assert has_identifier_shaped_token("10015819")
    assert has_identifier_shaped_token("100158")


@pytest.mark.parametrize(
    "query",
    [
        "hiking boots",
        "iPhone 15",
        '"Air Max 90"',
        "cotton shirt",
        "acme trailhead",
        "",
        "size 10 shoes",
        "999",
        "12345",
    ],
)
def test_has_identifier_shaped_token_false_without_a_real_code_token(query: str) -> None:
    """These can still land in `QueryClass.IDENTIFIER` via the class's other, weaker
    triggers (short query with no competing cue, ALLCAPS token, quoted phrase) - the
    exact gap this function exists not to fall into, since a short natural-language
    query like 'hiking boots' can genuinely benefit from reranking."""
    assert not has_identifier_shaped_token(query)


def test_short_query_misclassified_as_identifier_still_has_no_token_signal() -> None:
    """Pins the specific false positive that motivated this function: `classify()`
    puts this in IDENTIFIER (nothing else fires for a short query), but there is no
    actual product-code token in it - `hybrid.py` must not skip reranking here."""
    assert classify("hiking boots").query_class is QueryClass.IDENTIFIER
    assert not has_identifier_shaped_token("hiking boots")


# --- routing ------------------------------------------------------------------


def test_router_maps_class_to_alpha() -> None:
    router = AlphaRouter(enabled=True)
    alpha, classification = router.resolve("something for a beach wedding")
    assert classification.query_class is QueryClass.EXPLORATORY
    assert alpha == router.alpha_by_class[QueryClass.EXPLORATORY]


def test_identifier_gets_lower_alpha_than_exploratory() -> None:
    """The core claim of the feature: keyword-heavy for IDs, vector-heavy for intent."""
    router = AlphaRouter(enabled=True)
    id_alpha, _ = router.resolve("DW-4402B")
    exp_alpha, _ = router.resolve("something for a beach wedding")
    assert id_alpha < exp_alpha


def test_explicit_override_wins() -> None:
    router = AlphaRouter(enabled=True)
    alpha, classification = router.resolve("DW-4402B", override=0.9)
    assert alpha == 0.9
    # ...but the router still reports what it *would* have chosen, for A/B comparison.
    assert classification.query_class is QueryClass.IDENTIFIER


def test_disabled_router_returns_static_default() -> None:
    router = AlphaRouter(enabled=False, default_alpha=0.42)
    alpha, classification = router.resolve("something for a beach wedding")
    assert alpha == 0.42
    assert classification.query_class is QueryClass.EXPLORATORY


def test_falls_back_to_priors_without_tuned_file(tmp_path, monkeypatch) -> None:
    """A fresh clone (no sweep run yet) must still start and serve sane alphas."""
    monkeypatch.setattr("app.retrieval.alpha_router.TUNED_ALPHA_PATH", tmp_path / "absent.json")
    router = AlphaRouter(enabled=True)
    assert router.is_tuned is False
    assert router.alpha_by_class == PRIOR_ALPHA


def test_loads_tuned_alpha_when_present(tmp_path, monkeypatch) -> None:
    path = tmp_path / "tuned_alpha.json"
    path.write_text('{"alpha_by_class": {"identifier": 0.05, "exploratory": 0.88}}')
    monkeypatch.setattr("app.retrieval.alpha_router.TUNED_ALPHA_PATH", path)

    router = AlphaRouter(enabled=True)
    assert router.is_tuned is True
    assert router.alpha_by_class[QueryClass.IDENTIFIER] == 0.05
    assert router.alpha_by_class[QueryClass.EXPLORATORY] == 0.88
    # A class missing from the file keeps its prior rather than vanishing.
    assert router.alpha_by_class[QueryClass.ATTRIBUTE] == PRIOR_ALPHA[QueryClass.ATTRIBUTE]


def test_corrupt_tuned_file_does_not_break_startup(tmp_path, monkeypatch) -> None:
    path = tmp_path / "tuned_alpha.json"
    path.write_text("{ this is not json")
    monkeypatch.setattr("app.retrieval.alpha_router.TUNED_ALPHA_PATH", path)

    router = AlphaRouter(enabled=True)
    assert router.is_tuned is False
    assert router.alpha_by_class == PRIOR_ALPHA
