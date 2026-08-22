"""`POST /v1/merchants/{tenant}/chat`, end to end against the real stack *and* the
real Bedrock API - the only test file in this suite that costs real money per run,
so every case here is deliberately minimal: forced onto `anthropic.claude-haiku-4-5`
(`app/config.py`'s `model_fast`, never `model_reasoning`) and designed to spend as
few real LLM calls as the assertion actually needs.

Skipped entirely (not failed) when `AWS_BEARER_TOKEN_BEDROCK` isn't a real key -
`make test-all` without one should still exercise every *other* integration test,
not hard-fail here just because a real key hasn't been added yet.
"""

from __future__ import annotations

import json
import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.config import get_settings
from app.database import get_sessionmaker
from app.llm.graph import get_chat_graph
from app.models.db import LlmUsage, Merchant
from app.redis_client import get_redis_client
from app.routers.chat import CHAT_RATE_LIMIT
from tests.integration.conftest import BOOT_IMAGE_URL

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not get_settings().bedrock_api_key
        or get_settings().bedrock_api_key.startswith("bedrock-api-key-xxxx"),
        reason="AWS_BEARER_TOKEN_BEDROCK not set to a real key - see PROGRESS.md's Day 5 notes",
    ),
]


def _parse_sse(raw_text: str) -> list[tuple[str, str]]:
    """`(event, data)` pairs from a raw `text/event-stream` body - the test client
    reads the whole streamed response into `.text` since nothing here needs to
    assert on *timing* between chunks, only on what was eventually sent."""
    events: list[tuple[str, str]] = []
    event_name = None
    for line in raw_text.splitlines():
        if line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
        elif line.startswith("data:") and event_name is not None:
            events.append((event_name, line.removeprefix("data:").strip()))
    return events


async def test_chat_grounds_its_answer_in_a_real_retrieved_sku(
    client: AsyncClient, catalog_merchant
) -> None:
    """The one real-cost golden-path case: ask a question this catalog can
    genuinely answer, and confirm the citations event names a real, seeded
    product - not the text of the streamed answer, which an LLM is free to phrase
    many ways."""
    response = await client.post(
        f"/v1/merchants/{catalog_merchant.tenant}/chat",
        json={"message": "I need waterproof hiking boots"},
        headers=catalog_merchant.headers,
    )
    assert response.status_code == 200, response.text
    events = _parse_sse(response.text)

    assert any(name == "conversation" for name, _ in events)
    citation_events = [data for name, data in events if name == "citations"]
    assert citation_events, "expected at least one citations event"

    cited_products = {
        product["sku"]: product
        for data in citation_events
        for product in json.loads(data)["products"]
    }
    assert "BOOT-WP-10" in cited_products
    # Real regression target: image_url must reach the browser via the citations
    # event even though `_citable` (what Claude itself sees) never includes it -
    # see `app/llm/prompting.py::hits_to_evidence` and the unit-level split test
    # in `tests/unit/test_tool_node.py`.
    assert cited_products["BOOT-WP-10"]["image_url"] == BOOT_IMAGE_URL

    # Day 6 cost tracking: reuses this test's real call(s) rather than spending
    # more - `app/llm/cost_tracking.py::record_llm_usage` writes one ledger row per
    # Bedrock invocation, and a grounded question like this one takes at least two
    # rounds (a `search_catalog` tool-call round, then a final-answer round), so
    # this asserts "every round got recorded correctly," not "exactly one row."
    async with get_sessionmaker()() as session:
        merchant_id = (
            await session.execute(
                select(Merchant.id).where(Merchant.tenant == catalog_merchant.tenant)
            )
        ).scalar_one()
        rows = (
            (await session.execute(select(LlmUsage).where(LlmUsage.merchant_id == merchant_id)))
            .scalars()
            .all()
        )
    assert len(rows) >= 1
    for row in rows:
        assert row.model == get_settings().model_fast
        assert row.input_tokens > 0
        assert row.output_tokens > 0
        assert row.cost_usd > 0


async def test_conversation_id_is_isolated_by_tenant(
    client: AsyncClient, catalog_merchant, make_merchant
) -> None:
    """`ChatRequest.conversation_id` (`app/schemas.py`) is a client-supplied string
    with no format constraint, and LangGraph's `InMemorySaver` keys checkpoints by
    `thread_id` alone - without tenant-namespacing that key
    (`app/routers/chat.py::_stream`), a second tenant reusing the same bare
    `conversation_id` would resume the first tenant's checkpointed message history
    (a real cross-tenant leak found while building Day 6's cost tracking, not a
    hypothetical - see PROGRESS.md's dated entry). Proven directly against
    LangGraph's own state API rather than inferred from LLM prose, and costs only
    the one real call this conversation already needed."""
    shared_conversation_id = f"shared-{uuid.uuid4().hex[:8]}"

    response = await client.post(
        f"/v1/merchants/{catalog_merchant.tenant}/chat",
        json={
            "message": "I need waterproof hiking boots",
            "conversation_id": shared_conversation_id,
        },
        headers=catalog_merchant.headers,
    )
    assert response.status_code == 200, response.text

    other = await make_merchant("chat-iso")
    graph = get_chat_graph()

    # Sanity check first: the catalog merchant's own checkpoint really is populated -
    # a namespacing bug that broke *both* sides identically would otherwise still
    # pass the headline assertion below by leaving both empty.
    own_state = await graph.aget_state(
        {"configurable": {"thread_id": f"{catalog_merchant.tenant}:{shared_conversation_id}"}}
    )
    assert own_state.values.get("messages"), "sanity: the real conversation should be checkpointed"

    # Headline assertion: a different tenant reusing the same bare conversation_id
    # must start from a fresh, empty checkpoint, not the other tenant's history.
    other_state = await graph.aget_state(
        {"configurable": {"thread_id": f"{other.tenant}:{shared_conversation_id}"}}
    )
    assert not other_state.values.get("messages")


async def test_chat_rate_limit_rejects_without_spending_an_llm_call(
    client: AsyncClient, catalog_merchant
) -> None:
    """Pre-fills the rate-limit counter directly in Redis rather than sending
    `CHAT_RATE_LIMIT` real chat requests - the dependency rejects the request
    *before* the graph (and any Bedrock call) ever runs, so this costs zero LLM
    calls to verify."""
    redis = get_redis_client()
    key = f"ratelimit:chat:{catalog_merchant.tenant}"
    for _ in range(CHAT_RATE_LIMIT):
        await redis.incr(key)
    await redis.expire(key, 60)

    response = await client.post(
        f"/v1/merchants/{catalog_merchant.tenant}/chat",
        json={"message": "irrelevant, should never reach the model"},
        headers=catalog_merchant.headers,
    )
    assert response.status_code == 429, response.text


async def test_repeated_question_is_served_from_the_semantic_cache(
    client: AsyncClient, catalog_merchant
) -> None:
    """One real LLM call, then one cache hit - not two real calls. The second
    response's citations event is tagged `cached: true` (`app/llm/graph.py`'s
    `maybe_serve_from_cache` node)."""
    message = {"message": "do you have any waterproof boots"}

    first = await client.post(
        f"/v1/merchants/{catalog_merchant.tenant}/chat",
        json=message,
        headers=catalog_merchant.headers,
    )
    assert first.status_code == 200, first.text

    second = await client.post(
        f"/v1/merchants/{catalog_merchant.tenant}/chat",
        json=message,
        headers=catalog_merchant.headers,
    )
    assert second.status_code == 200, second.text

    citation_events = [data for name, data in _parse_sse(second.text) if name == "citations"]
    assert citation_events
    assert json.loads(citation_events[0])["cached"] is True
