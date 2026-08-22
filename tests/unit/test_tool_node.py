"""`app/llm/graph.py::tool_node`'s dispatch logic - fully mocked (`get_retriever()`
patched, no real Weaviate/Bedrock call), free, and fast. Going from one tool to
three sharing one dispatch function is exactly the kind of change this session's
own history shows breaking only at real-call scale (five separate real bugs found
tonight, several only visible on turn 2+ or under a full paid eval run) - this is
the free, millisecond layer that catches a dispatch regression in CI instead.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from anthropic.types import ToolUseBlock

from app.llm import graph as graph_module
from app.llm.graph import MAX_TOOL_CALL_ROUNDS, tool_node
from app.retrieval.base import CatalogStats, SearchHit, SearchResponse

BOOT = SearchHit(sku="BOOT-WP-10", title="Waterproof Hiking Boot", score=1.0, price=None)
SHIRT = SearchHit(sku="SHIRT-CTN-M", title="Cotton Shirt", score=0.9, price=None)


def _tool_use(name: str, tool_input: dict[str, Any]) -> ToolUseBlock:
    return ToolUseBlock(type="tool_use", id="toolu_test", name=name, input=tool_input)


def _state(*, tool_use: ToolUseBlock, tenant: str = "demo-fashion-in", **overrides: Any) -> Any:
    base = {
        "tenant": tenant,
        "messages": [{"role": "assistant", "content": [tool_use]}],
        "tool_call_rounds": 0,
        "model_used": "anthropic.claude-haiku-4-5",
        "cache_hit": False,
        "final_answer": "",
        "citations": [],
        "force_no_tools": False,
    }
    base.update(overrides)
    return base


class FakeRetriever:
    def __init__(self) -> None:
        self.search = AsyncMock(return_value=SearchResponse(hits=[BOOT]))
        self.stats = AsyncMock(
            return_value=CatalogStats(
                metric="price", count=3, minimum="100", maximum="500", mean="250"
            )
        )
        self.get_by_skus = AsyncMock(return_value=[BOOT])
        self.search_sorted = AsyncMock(return_value=[BOOT])


@pytest.fixture
def fake_retriever(monkeypatch: pytest.MonkeyPatch) -> FakeRetriever:
    retriever = FakeRetriever()
    monkeypatch.setattr(graph_module, "get_retriever", lambda: retriever)
    return retriever


async def test_dispatches_search_catalog(fake_retriever: FakeRetriever) -> None:
    state = _state(tool_use=_tool_use("search_catalog", {"query": "waterproof boots"}))
    result = await tool_node(state)
    fake_retriever.search.assert_awaited_once()
    kinds = {c["kind"] for c in result["citations"]}
    assert kinds == {"product"}
    assert result["citations"][0]["sku"] == "BOOT-WP-10"


async def test_dispatches_get_catalog_stats(fake_retriever: FakeRetriever) -> None:
    state = _state(tool_use=_tool_use("get_catalog_stats", {"min_price": 100}))
    result = await tool_node(state)
    fake_retriever.stats.assert_awaited_once()
    assert result["citations"] == [
        {
            "kind": "stats",
            "filters": {"min_price": 100},
            "metric": "price",
            "count": 3,
            "minimum": "100",
            "maximum": "500",
            "mean": "250",
        }
    ]


async def test_dispatches_get_product_detail(fake_retriever: FakeRetriever) -> None:
    state = _state(tool_use=_tool_use("get_product_detail", {"skus": ["BOOT-WP-10", "NOT-FOUND"]}))
    result = await tool_node(state)
    fake_retriever.get_by_skus.assert_awaited_once_with(
        "demo-fashion-in", ["BOOT-WP-10", "NOT-FOUND"]
    )
    assert result["citations"] == [
        {
            "kind": "product",
            "sku": "BOOT-WP-10",
            "title": "Waterproof Hiking Boot",
            "brand": None,
            "price": None,
            "currency": None,
            "in_stock": True,
            "category_path": [],
            "rating": None,
            "image_url": None,
        }
    ]


async def test_citations_carry_image_url_but_tool_result_does_not(
    fake_retriever: FakeRetriever,
) -> None:
    """`image_url` must reach the browser via `citations` (so product cards can
    render an image) but never reach Claude via the tool_result text - it's pure
    token cost the model has no use for. Regression guard for the split between
    `_citable` (model-facing) and `hits_to_evidence` (browser-facing) in
    `app/llm/prompting.py`."""
    pictured = SearchHit(
        sku="BOOT-WP-10",
        title="Waterproof Hiking Boot",
        score=1.0,
        price=None,
        image_url="https://example.com/boot.jpg",
    )
    fake_retriever.search.return_value = SearchResponse(hits=[pictured])
    state = _state(tool_use=_tool_use("search_catalog", {"query": "boots"}))
    result = await tool_node(state)

    assert result["citations"][0]["image_url"] == "https://example.com/boot.jpg"

    tool_result_message = result["messages"][0]
    tool_result_text = tool_result_message["content"][0]["content"][0]["text"]
    assert "image_url" not in tool_result_text
    assert "boot.jpg" not in tool_result_text


async def test_search_catalog_with_sort_by_uses_search_sorted(
    fake_retriever: FakeRetriever,
) -> None:
    state = _state(
        tool_use=_tool_use("search_catalog", {"query": "watch", "sort_by": "price_desc"})
    )
    await tool_node(state)
    fake_retriever.search_sorted.assert_awaited_once()
    fake_retriever.search.assert_not_awaited()


@pytest.mark.parametrize("tool_name", ["get_catalog_stats", "get_product_detail"])
async def test_tenant_always_comes_from_state_never_from_tool_input(
    fake_retriever: FakeRetriever, tool_name: str
) -> None:
    """A model can emit extra JSON keys nothing in the schema declares - if it
    ever included a `tenant` key (no schema declares one, but nothing stops it),
    the handler must still use `state["tenant"]`, never that value. Guards
    against a future "generic dispatch" refactor quietly reading
    `tool_input.get("tenant", state["tenant"])`."""
    tool_input: dict[str, Any] = {"tenant": "some-other-tenant"}
    if tool_name == "get_product_detail":
        tool_input["skus"] = ["BOOT-WP-10"]
    state = _state(tenant="demo-fashion-in", tool_use=_tool_use(tool_name, tool_input))
    await tool_node(state)

    mock = fake_retriever.stats if tool_name == "get_catalog_stats" else fake_retriever.get_by_skus
    assert mock.await_args is not None
    called_tenant = mock.await_args.args[0]
    assert called_tenant == "demo-fashion-in"


async def test_citations_merge_dedup_across_rounds(fake_retriever: FakeRetriever) -> None:
    existing = [{"kind": "product", "sku": "BOOT-WP-10", "title": "old"}]
    fake_retriever.search.return_value = SearchResponse(hits=[BOOT, SHIRT])
    state = _state(
        tool_use=_tool_use("search_catalog", {"query": "more items"}),
        citations=existing,
    )
    result = await tool_node(state)
    skus = [c["sku"] for c in result["citations"]]
    assert skus == ["BOOT-WP-10", "SHIRT-CTN-M"]  # deduped, first-seen order kept


async def test_stats_evidence_is_never_deduped_by_sku(fake_retriever: FakeRetriever) -> None:
    """A stats entry has no SKU - a second get_catalog_stats call with different
    filters is genuinely different evidence, not a duplicate to drop."""
    existing = [{"kind": "stats", "filters": {}, "metric": "price", "count": 1}]
    state = _state(tool_use=_tool_use("get_catalog_stats", {"max_price": 500}), citations=existing)
    result = await tool_node(state)
    assert len(result["citations"]) == 2


async def test_tool_call_rounds_increments_and_force_no_tools_fires_at_cap(
    fake_retriever: FakeRetriever,
) -> None:
    state = _state(
        tool_use=_tool_use("search_catalog", {"query": "x"}),
        tool_call_rounds=MAX_TOOL_CALL_ROUNDS - 1,
    )
    result = await tool_node(state)
    assert result["tool_call_rounds"] == MAX_TOOL_CALL_ROUNDS
    assert result["force_no_tools"] is True


async def test_unknown_tool_name_raises_instead_of_silently_swallowing(
    fake_retriever: FakeRetriever,
) -> None:
    """A KeyError here means agent()'s declared tools and this dispatch table
    have drifted - a real bug that must surface loudly, not disappear."""
    state = _state(tool_use=_tool_use("not_a_real_tool", {}))
    with pytest.raises(KeyError):
        await tool_node(state)
