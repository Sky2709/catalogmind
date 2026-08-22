"""`POST /v1/merchants/{tenant}/chat` - the grounded, streaming shopping assistant.

Delegates everything to the LangGraph agent in `app/llm/graph.py`; this module's
only job is the HTTP/SSE boundary: rate limiting, resolving/minting a
`conversation_id`, translating the graph's custom stream events
(`app/llm/graph.py`'s `writer({"type": ..., ...})` calls) into real Server-Sent
Events, and turning a mid-stream failure into a terminal SSE error event instead of
a raw 500 (the response has already started by the time that could happen - there
is no clean way to retry or replace it, only to tell the client honestly that it
failed).
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.runnables import RunnableConfig
from sse_starlette.sse import EventSourceResponse

from app.deps import ScopedMerchant
from app.llm.graph import get_chat_graph
from app.rate_limit import check_rate_limit
from app.redis_client import get_redis_client
from app.schemas import ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/merchants", tags=["chat"])

# An LLM call costs seconds and real money per turn, unlike search's near-instant
# hybrid query - this limit exists to bound cost/latency exposure per tenant, not
# just to catch a runaway script the way search's much higher 120/60s does.
CHAT_RATE_LIMIT = 20
CHAT_RATE_WINDOW_SECONDS = 60


async def enforce_chat_rate_limit(merchant: ScopedMerchant) -> None:
    redis = get_redis_client()
    result = await check_rate_limit(
        redis,
        f"ratelimit:chat:{merchant.tenant}",
        limit=CHAT_RATE_LIMIT,
        window_seconds=CHAT_RATE_WINDOW_SECONDS,
    )
    if not result.allowed:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded: {CHAT_RATE_LIMIT} chat turns per "
                f"{CHAT_RATE_WINDOW_SECONDS}s. Retry after {result.retry_after_seconds}s."
            ),
            headers={"Retry-After": str(result.retry_after_seconds)},
        )


def _sse(event: dict) -> dict:
    """A graph custom-stream event (`{"type": ..., ...}`) into the shape
    `sse-starlette` actually serialises (`ServerSentEvent(**dict)` - `event`/`data`
    kwargs, not our internal `type` key)."""
    payload = {k: v for k, v in event.items() if k != "type"}
    return {"event": event["type"], "data": json.dumps(payload)}


async def _stream(
    merchant: ScopedMerchant, body: ChatRequest, conversation_id: str
) -> AsyncIterator[dict]:
    yield {"event": "conversation", "data": json.dumps({"conversation_id": conversation_id})}

    graph = get_chat_graph()
    # Namespaced by tenant, not the bare client-supplied id. `ChatRequest.conversation_id`
    # is accepted directly from the request body with no format constraint, and
    # LangGraph's `InMemorySaver` keys checkpoints by `thread_id` alone - without this
    # prefix, a client that reuses (or guesses) another tenant's conversation_id would
    # resume that tenant's checkpointed message history via `ChatState.messages`'s
    # `operator.add` accumulation, a real cross-tenant leak (see PROGRESS.md's dated
    # entry). Tenant slugs are regex-validated at merchant-creation time to exclude
    # `:` (`app/models/db.py`'s `Merchant.tenant`), so two tenants can never collide
    # on the composed key regardless of what a client puts in `conversation_id`. The
    # client only ever sees the bare `conversation_id` above - this composed form
    # exists solely as the checkpointer key.
    thread_id = f"{merchant.tenant}:{conversation_id}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # Per-turn scratch state is reset explicitly on every call, including a resumed
    # conversation - only `messages` benefits from the checkpointer's accumulation
    # (via its `operator.add` reducer). Leaving these out on a resumed turn would
    # silently carry the *previous* turn's tool-call-round count etc. forward,
    # which would corrupt the model router's escalation logic - a real, easy-to-miss
    # LangGraph state-design trap, not a hypothetical one.
    turn_input = {
        "tenant": merchant.tenant,
        "merchant_id": merchant.id,
        "conversation_id": conversation_id,
        "messages": [{"role": "user", "content": body.message}],
        "tool_call_rounds": 0,
        "model_used": None,
        "cache_hit": False,
        "final_answer": "",
        "citations": [],
        "force_no_tools": False,
    }

    try:
        async for event in graph.astream(turn_input, config=config, stream_mode="custom"):
            yield _sse(event)
    except Exception as exc:  # noqa: BLE001 - the stream is already open; a client needs a
        # terminal event, not a dropped connection or a raw 500 that arrives too late to matter.
        logger.exception("chat turn failed mid-stream, tenant=%s", merchant.tenant)
        yield {"event": "error", "data": json.dumps({"message": str(exc)})}


@router.post(
    "/{tenant}/chat",
    summary="Grounded conversational commerce (SSE streaming)",
    dependencies=[Depends(enforce_chat_rate_limit)],
    responses={
        429: {"description": "Too many chat turns for this merchant in the current window"},
        200: {
            "description": (
                "`text/event-stream`. Event types: `conversation` (once, carries "
                "`conversation_id` - send it back on the next turn), `tool_call` "
                "(the agent is searching the catalog), `token` (a chunk of the "
                "streamed answer), `citations` (once, the products the answer is "
                "grounded in), `error` (a terminal failure mid-stream)."
            )
        },
    },
)
async def chat(merchant: ScopedMerchant, body: ChatRequest) -> EventSourceResponse:
    conversation_id = body.conversation_id or str(uuid.uuid4())
    return EventSourceResponse(_stream(merchant, body, conversation_id))
