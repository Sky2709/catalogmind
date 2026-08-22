"""`app/llm/graph.py::validate_and_store`'s citation-display filtering - fully
mocked (`get_stream_writer`/`aembed_query`/`cache_store` all patched, no real
Bedrock/Weaviate/Redis call), free and fast, same rationale as `test_tool_node.py`.

Real production bug this guards against (2026-08-22, live-observed, not
hypothetical): asked "red underwear" against the electronics-only demo catalog,
Claude correctly refused in prose - but the product-card grid still showed the
phone cases and fountain pen `search_catalog` had turned up, contradicting the
words right above them. First fix used a lexical refusal-cue regex
(`refuses()`); that regex itself then missed a second real refusal
("black tshirt for men" - "didn't find"/"don't match" phrasing) within one turn
of shipping, which is why the primary signal is now the `[[NO_MATCH]]` marker
Claude is instructed to emit (`app/llm/markers.py`) - a structural declaration,
not a guess at phrasing. `refuses()` stays wired in only as a backstop for a
compliance miss. Either signal gates what gets *displayed and cached*, never what
gets checked for hallucination - the checks below still run against the full,
unfiltered evidence.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.llm import graph as graph_module
from app.llm.graph import validate_and_store

PRODUCT = {
    "kind": "product",
    "sku": "CASE-001",
    "title": "Phone Case",
    "brand": None,
    "price": "329.0",
    "currency": "INR",
    "in_stock": True,
    "category_path": [],
    "rating": None,
    "image_url": None,
}


def _state(*, answer: str, citations: list[dict[str, Any]], **overrides: Any) -> Any:
    base = {
        "tenant": "demo-electronics-in",
        "messages": [{"role": "user", "content": "red underwear"}],
        "tool_call_rounds": 1,
        "model_used": "anthropic.claude-haiku-4-5",
        "cache_hit": False,
        "final_answer": answer,
        "citations": citations,
        "force_no_tools": False,
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def _mocked_dependencies(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    claim_mismatches: list[dict[str, Any]] = []
    monkeypatch.setattr(graph_module, "get_stream_writer", lambda: events.append)
    monkeypatch.setattr(graph_module, "aembed_query", AsyncMock(return_value=[0.0]))
    monkeypatch.setattr(graph_module, "get_redis_client", lambda: None)
    cache_store = AsyncMock()
    monkeypatch.setattr(graph_module, "cache_store", cache_store)
    monkeypatch.setattr(
        graph_module,
        "observe_claim_mismatch",
        lambda **kwargs: claim_mismatches.append(kwargs),
    )
    return {"events": events, "cache_store": cache_store, "claim_mismatches": claim_mismatches}


async def test_no_match_marker_hides_products_from_display_and_cache(
    _mocked_dependencies: dict[str, Any],
) -> None:
    state = _state(
        answer="[[NO_MATCH]] This catalog doesn't carry underwear or apparel products.",
        citations=[PRODUCT],
    )
    await validate_and_store(state)

    citation_events = [e for e in _mocked_dependencies["events"] if e["type"] == "citations"]
    assert citation_events == [{"type": "citations", "products": [], "cached": False}]

    _mocked_dependencies["cache_store"].assert_awaited_once()
    cached_products = _mocked_dependencies["cache_store"].await_args.args[4]
    assert cached_products == []


async def test_refuses_backstop_still_hides_products_without_the_marker(
    _mocked_dependencies: dict[str, Any],
) -> None:
    """A compliance miss - Claude didn't emit [[NO_MATCH]] but still phrased a
    plain refusal - must still hide the cards via the lexical backstop."""
    state = _state(
        answer="Sorry, this catalog doesn't carry underwear or apparel products.",
        citations=[PRODUCT],
    )
    await validate_and_store(state)

    citation_events = [e for e in _mocked_dependencies["events"] if e["type"] == "citations"]
    assert citation_events == [{"type": "citations", "products": [], "cached": False}]


async def test_non_refusal_answer_still_displays_and_caches_products(
    _mocked_dependencies: dict[str, Any],
) -> None:
    state = _state(
        answer="I'd recommend the [[SKU:CASE-001]] - it's a solid protective case.",
        citations=[PRODUCT],
    )
    await validate_and_store(state)

    citation_events = [e for e in _mocked_dependencies["events"] if e["type"] == "citations"]
    assert citation_events == [{"type": "citations", "products": [PRODUCT], "cached": False}]

    cached_products = _mocked_dependencies["cache_store"].await_args.args[4]
    assert cached_products == [PRODUCT]


async def test_no_match_filtering_does_not_affect_hallucination_detection(
    _mocked_dependencies: dict[str, Any],
) -> None:
    """The filter only changes what's *displayed* - `find_hallucinated_citations`
    must still see the full, unfiltered evidence, so a real fabricated-SKU
    citation inside a [[NO_MATCH]] answer is still caught and metered."""
    state = _state(
        answer="[[NO_MATCH]] We don't have that, but you might like [[SKU:FAKE-999]] instead.",
        citations=[PRODUCT],
    )
    await validate_and_store(state)

    mismatches = _mocked_dependencies["claim_mismatches"]
    assert {"claim_type": "hallucinated_citation", "count": 1} in mismatches


async def test_stats_backed_answer_is_never_cached_even_when_refused(
    _mocked_dependencies: dict[str, Any],
) -> None:
    """Existing "never cache a stats-dependent turn" rule takes precedence -
    cache_store must not be called at all when stats evidence is present,
    refusal or not."""
    state = _state(
        answer="[[NO_MATCH]] Nothing above [[STAT:10000]] in this catalog.",
        citations=[{"kind": "stats", "filters": {}, "metric": "price", "count": 0}],
    )
    await validate_and_store(state)
    _mocked_dependencies["cache_store"].assert_not_awaited()


async def test_semantic_cache_disabled_skips_the_write_too_not_just_the_read(
    _mocked_dependencies: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real bug found live (2026-08-22) while measuring `model_router.py`: this
    write had no `semantic_cache_enabled` guard at all, unlike
    `maybe_serve_from_cache`'s read side - a script that disabled the setting for
    its *lookups* only (`eval/measure_model_router.py`) still silently wrote every
    scored turn, including a real one-off empty-answer response, into the
    *shared* semantic cache, where a real shopper's similar query could have been
    served that broken answer later."""
    monkeypatch.setattr(graph_module.get_settings(), "semantic_cache_enabled", False)
    state = _state(
        answer="I'd recommend the [[SKU:CASE-001]] - it's a solid protective case.",
        citations=[PRODUCT],
    )
    await validate_and_store(state)
    _mocked_dependencies["cache_store"].assert_not_awaited()
