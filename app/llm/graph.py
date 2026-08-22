"""The Day 5 chat agent, as a LangGraph state graph. LangGraph supplies the state
machine (nodes, edges, the checkpointer); every Claude call inside a node goes
through the **raw `anthropic` SDK** (`app/llm/client.py`, via Bedrock), never a
LangChain model wrapper - see `CLAUDE.md`'s "LangGraph scope" section for why that
split matters (prompt-cache control, the model router, the thinking-effort dial, and
cost metering off `response.usage` would all be lost behind a wrapper).

Flow, one turn:

    maybe_serve_from_cache -> (hit) -> END
                            -> (miss) -> agent -> [has tool call?] -> tool_node -> agent (loop)
                                                -> [final answer]  -> validate_and_store -> END

`agent` calls `search_catalog` as a real Claude tool-use call bound directly to the
existing `get_retriever().search()` (`app/routers/search.py`'s own docstring
anticipates exactly this reuse - no new retrieval code). The loop is bounded at
`MAX_TOOL_CALL_ROUNDS` so an indecisive model can't run away on cost/latency, the
same "bounded and explicit" discipline as `INGESTION_CONCURRENCY_LIMIT` /
`RERANK_CONCURRENCY_LIMIT`.

Streaming is a separate concern from the state machine: nodes push custom events
(`tool_call` while searching, `token` for the model's streamed text, `citations`
once at the end) via LangGraph's `get_stream_writer()`, consumed by
`app/routers/chat.py` under `stream_mode="custom"` and forwarded as SSE. Multi-turn
state uses LangGraph's own checkpointer (`InMemorySaver` - correct for this
single-process demo deployment; a Redis/Postgres-backed checkpointer is the
documented production upgrade for multiple worker processes), keyed by
`conversation_id`, so this module does not hand-roll a second conversation store.
"""

from __future__ import annotations

import asyncio
import operator
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from functools import lru_cache
from typing import Annotated, Any, NotRequired, TypedDict, cast

from anthropic.types import ContentBlock, MessageParam, ToolUseBlock, Usage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.config import get_settings
from app.ingestion.embed import aembed_query
from app.llm.citations import find_hallucinated_citations
from app.llm.claims import (
    find_stat_claim_mismatch,
    find_unverified_quantitative_refusal,
    has_superlative_language,
)
from app.llm.client import (
    ANTHROPIC_TRANSIENT_ERRORS,
    BEDROCK_CALL_TIMEOUT_SECONDS,
    get_bedrock_client,
)
from app.llm.cost_tracking import record_llm_usage
from app.llm.markers import is_no_match
from app.llm.model_router import ModelTier, RoutingDecision, classify_complexity
from app.llm.prompting import (
    FORCE_ANSWER_NUDGE,
    GET_CATALOG_STATS_TOOL,
    GET_PRODUCT_DETAIL_TOOL,
    SEARCH_CATALOG_TOOL,
    SYSTEM_PROMPT,
    hits_to_evidence,
    hits_to_tool_result,
    product_detail_to_tool_result,
    stats_to_tool_result,
)
from app.llm.refusal_text import refuses
from app.llm.semantic_cache import lookup as cache_lookup
from app.llm.semantic_cache import store as cache_store
from app.obs.metrics import (
    observe_chat_request,
    observe_chat_tokens,
    observe_chat_tool_call,
    observe_claim_mismatch,
)
from app.redis_client import get_redis_client
from app.retrieval.base import SearchFilters, SearchRequest
from app.retrieval.hybrid import get_retriever
from app.retry import with_retry

MAX_TOOL_CALL_ROUNDS = 2

# Confirmed live (2026-08-21, real Bedrock calls), not left as the guide's
# unconfirmed guess: Haiku 4.5 rejects both `thinking: {"type": "adaptive"}`
# ("adaptive thinking is not supported on this model") and `output_config.effort`
# ("This model does not support the effort parameter") outright - it only accepts
# the older `thinking: {"type": "enabled", "budget_tokens": N}` shape, or no
# `thinking`/`output_config` at all. Sonnet 5 accepts adaptive + effort exactly as
# the reference guide said. Rather than force Haiku into the older explicit-budget
# mode (paying real thinking-token cost for what's supposed to be the cheap,
# fast-lookup tier - defeating the point of having a fast tier), it gets neither
# param: real, tested behaviour, not a guess either way.
_REASONING_TIER_EXTRAS: dict[str, Any] = {
    "thinking": {"type": "adaptive"},
}

# A real, measured lesson carried over from the Gemini build, applied here from the
# start rather than rediscovered: a small max-token budget can be silently consumed
# entirely by thinking tokens, starving the actual visible answer down to nothing.
# 4096 is generous enough that a normal shopping-assistant answer plus real
# thinking overhead should never hit it, without paying for the full 128K ceiling
# Sonnet 5 supports.
MAX_ANSWER_TOKENS = 4096


@dataclass
class _AgentTurn:
    """What one streamed Claude call actually produced. `content_blocks` is the
    *real* response content (`final_message.content` from the SDK's
    `get_final_message()`, not hand-parsed from streaming deltas) - Claude can
    stream a tool call's `input` as fragmented `input_json_delta` chunks that need
    proper JSON reassembly, which the SDK's own streaming helper already does
    correctly. Re-implementing that accumulation by hand would just be inviting the
    same class of bug the old Gemini `thought_signature` mistake was: silently
    dropping something a hand-rolled reconstruction didn't know to carry."""

    text: str = ""
    content_blocks: list[ContentBlock] = field(default_factory=list)
    usage: Usage | None = None


class ChatState(TypedDict):
    tenant: str
    merchant_id: int
    conversation_id: str
    messages: Annotated[list[MessageParam], operator.add]
    tool_call_rounds: int
    model_used: str | None
    cache_hit: bool
    final_answer: str
    citations: list[dict[str, Any]]
    force_no_tools: bool
    """Set by `tool_node` once `tool_call_rounds` hits `MAX_TOOL_CALL_ROUNDS` - see
    the Gemini-era finding this carries forward (`PROGRESS.md`'s Day 5 notes): a
    bounded loop that only stops *routing* to the tool node isn't enough, because
    the *agent* call right at the cap can still request another tool call if the
    tool is still declared - which then has nowhere to go, leaving no final answer
    at all. Omitting `tools` from the request entirely once this is True makes the
    model physically unable to ask again, guaranteeing a real text answer within
    one more turn."""
    force_tier: NotRequired[ModelTier | None]
    """Eval-only override for `classify_complexity`'s decision - lets
    `eval/measure_model_router.py` run the *same* message through both tiers for
    a fair before/after comparison. The exact "eval override, production never
    passes it" shape `WeaviateHybridRetriever`'s `retrieve_top_k` constructor
    override already uses (Day 3 notes, `PROGRESS.md`) - the real `/chat` handler
    (`app/routers/chat.py`) never sets this key, so `.get("force_tier")` defaults
    to `None` and `classify_complexity` runs exactly as it does today."""


def _block_type(block: Any) -> str | None:
    return block.get("type") if isinstance(block, dict) else getattr(block, "type", None)


def _sanitize_messages_for_request(messages: list[MessageParam]) -> list[MessageParam]:
    """`messages` as the API will accept them on *this* outgoing call -
    regardless of whether a `tool_use` block came fresh from a response this
    turn or was round-tripped through LangGraph's checkpoint serializer.

    A real, live bug (2026-08-22, caught by hand-testing a *second* chat turn
    in the browser, not by any automated test): the installed `anthropic`
    SDK's `ToolUseBlock` *response* type carries two newer fields
    (`toolset_name`/`caller`, tied to server-side toolset/tool-search
    features) that Bedrock's current backend rejects as *request* input on a
    replayed conversation - `messages.create` on turn 2 failed with a real
    400 ("Extra inputs are not permitted: toolset_name"). Confirmed why with a
    direct `model_dump(exclude_unset=True)` check: a block parsed from a real
    API response has those fields marked "set" (even at `None`, because the
    server's JSON explicitly included them), so echoing that exact object
    resends them - but a *freshly constructed* `ToolUseBlock` with only
    `id`/`name`/`input`/`type` never touches them, so `exclude_unset`
    correctly drops them.

    **The first fix attempt sanitised only the message being stored this
    turn, and that wasn't enough** - confirmed live, not assumed: fetching the
    checkpointed state back out with `graph.aget_state()` showed
    `model_fields_set` covering *all six* fields, including the two that were
    never touched when the object was built. LangGraph's checkpoint serde
    doesn't preserve pydantic's unset-vs-set distinction across its
    serialize/deserialize round trip for an unregistered type (the
    accompanying "Deserializing unregistered type ... from checkpoint"
    warning is the same mechanism) - so a clean object saved this turn comes
    back dirty next turn regardless of how it was built. The only boundary
    that's actually safe to fix at is the one the failure happens on: right
    before `messages` goes out over the wire, on *every* call, not once at
    storage time. This is the mirror image of why `_AgentTurn.content_blocks`
    keeps the *rest* of the response verbatim in the first place (Gemini's
    `thought_signature` needed exactly that, on that provider) - the same
    "response and request schemas quietly diverged" tension, opposite failure
    direction. Rebuilding only `tool_use` blocks (never text/thinking, which
    have shown no such issue) keeps this fix as narrow as the evidence
    supports.
    """
    sanitized: list[MessageParam] = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            sanitized.append(message)
            continue
        new_content: list[Any] = []
        for block in content:
            if _block_type(block) != "tool_use":
                new_content.append(block)
                continue
            if isinstance(block, dict):
                block_dict = cast(dict[str, Any], block)
                block_id, name, tool_input = (
                    block_dict["id"],
                    block_dict["name"],
                    block_dict["input"],
                )
            else:
                block_obj = cast(Any, block)
                block_id, name, tool_input = block_obj.id, block_obj.name, block_obj.input
            new_content.append(
                ToolUseBlock(type="tool_use", id=block_id, name=name, input=tool_input)
            )
        sanitized.append(cast(MessageParam, {**message, "content": new_content}))
    return sanitized


def _model_for(tier: str) -> str:
    settings = get_settings()
    return settings.model_reasoning if tier == "reasoning" else settings.model_fast


def _last_user_text(messages: list[MessageParam]) -> str:
    # A tool-result turn is also `role="user"` per Anthropic's convention, but its
    # `content` is always a list of blocks, never a bare string - only the
    # shopper's own typed message is ever stored as a plain string here, so that
    # shape alone disambiguates the two without needing a separate marker.
    for message in reversed(messages):
        if message["role"] == "user" and isinstance(message["content"], str):
            return message["content"]
    return ""


async def maybe_serve_from_cache(state: ChatState) -> dict[str, Any]:
    if not get_settings().semantic_cache_enabled:
        return {"cache_hit": False}

    query = _last_user_text(state["messages"])
    embedding = await aembed_query(query)
    cached = await cache_lookup(
        get_redis_client(),
        state["tenant"],
        embedding,
        threshold=get_settings().semantic_cache_threshold,
    )
    if cached is None:
        return {"cache_hit": False}

    writer = get_stream_writer()
    writer({"type": "citations", "products": cached.citations, "cached": True})
    for chunk in cached.answer.split(" "):
        writer({"type": "token", "text": chunk + " "})
    return {
        "cache_hit": True,
        "final_answer": cached.answer,
        "citations": cached.citations,
    }


async def agent(state: ChatState) -> dict[str, Any]:
    forced_tier = state.get("force_tier")
    decision = (
        RoutingDecision(forced_tier, ("forced by eval override",))
        if forced_tier is not None
        else classify_complexity(
            _last_user_text(state["messages"]), tool_call_rounds=state["tool_call_rounds"]
        )
    )
    model = _model_for(decision.tier)
    client = get_bedrock_client()
    writer = get_stream_writer()

    # Omitted entirely (not just discouraged) once the tool-call budget is spent -
    # see `ChatState.force_no_tools`'s docstring for why this has to be an actual
    # capability removal, not a prompt instruction the model could still ignore.
    # Built as a plain kwargs dict (typed `Any`) rather than passing the SDK's
    # `NOT_GIVEN` sentinel directly - the streaming `tools` overload's static type
    # wants its own `Omit` sentinel, not `NotGiven`, even though both mean the same
    # thing to the SDK at runtime.
    stream_kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": MAX_ANSWER_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": _sanitize_messages_for_request(state["messages"]),
    }
    if decision.tier == "reasoning":
        # `_REASONING_TIER_EXTRAS`/`output_config.effort` are Sonnet-5-only, real
        # tested behaviour, not a guess - see that constant's docstring.
        stream_kwargs.update(_REASONING_TIER_EXTRAS)
        stream_kwargs["output_config"] = {"effort": "high"}
    if state.get("force_no_tools"):
        stream_kwargs["system"] = [*SYSTEM_PROMPT, FORCE_ANSWER_NUDGE]
    else:
        stream_kwargs["tools"] = [
            SEARCH_CATALOG_TOOL,
            GET_CATALOG_STATS_TOOL,
            GET_PRODUCT_DETAIL_TOOL,
        ]
        # Real bug, not a hypothetical: Claude will happily emit two `tool_use`
        # blocks in one turn for a compound ask ("black shirt for men" + "red for
        # women"), but `tool_node` below only ever answers `tool_use_blocks[0]` -
        # every downstream piece of state (`tool_call_rounds`, `citations` as a
        # single result set) assumes one call per round. A second, unanswered
        # `tool_use` id then poisons the persisted message history for every later
        # turn on the same conversation (Anthropic rejects the whole request once a
        # `tool_use` has no matching `tool_result`). Disabling parallel tool use
        # keeps the model to one call per round, matching what the rest of the
        # graph actually handles.
        stream_kwargs["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}

    async def _call() -> _AgentTurn:
        turn = _AgentTurn()
        text_parts: list[str] = []
        wrote_any_token = False
        try:
            async with client.messages.stream(**stream_kwargs) as stream:
                stream_iter = stream.__aiter__()
                # A per-*chunk* timeout, not a whole-call one: a real answer can
                # legitimately take longer than `BEDROCK_CALL_TIMEOUT_SECONDS` to
                # finish streaming in full, and that must not be treated as a
                # failure. What must never happen is the gap *between* chunks going
                # unbounded - the real hang this guards against, learned on the
                # Gemini build and applied here from the start (see
                # `app/llm/client.py`'s module docstring).
                while True:
                    try:
                        event = await asyncio.wait_for(
                            stream_iter.__anext__(), timeout=BEDROCK_CALL_TIMEOUT_SECONDS
                        )
                    except StopAsyncIteration:
                        break
                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        wrote_any_token = True
                        text_parts.append(event.delta.text)
                        writer({"type": "token", "text": event.delta.text})
                final_message = await stream.get_final_message()
        except ANTHROPIC_TRANSIENT_ERRORS:
            if wrote_any_token:
                # Some of this turn's answer has already reached the client as
                # real SSE `token` events - retrying from scratch would re-run the
                # whole call and duplicate/garble what's already been sent.
                # Turning this into a non-retryable error instead surfaces it to
                # `app/routers/chat.py` as a terminal `error` event - the
                # documented tradeoff: once streaming has started, a failure is
                # reported, not silently retried.
                raise RuntimeError("Claude stream failed after partial output") from None
            raise
        turn.content_blocks = list(final_message.content)
        turn.usage = final_message.usage
        turn.text = "".join(text_parts)
        return turn

    # `attempts=2`, not the default 3: `BEDROCK_CALL_TIMEOUT_SECONDS` (`app/llm/
    # client.py`) is already generous (60s, per chunk) - a real Gemini-era hang
    # sat for over two minutes with no visible progress before a timeout like this
    # existed anywhere in this codebase. Three attempts at up to 60s each would let
    # one bad call cost a caller three minutes before finally failing; two is
    # still a real retry, not just giving up instantly, without compounding a slow
    # failure into a very slow one.
    turn = await with_retry(_call, retryable=ANTHROPIC_TRANSIENT_ERRORS, attempts=2)
    if turn.usage is not None:
        observe_chat_tokens(
            model=model,
            prompt=turn.usage.input_tokens,
            output=turn.usage.output_tokens,
            cached=turn.usage.cache_read_input_tokens or 0,
        )
        await record_llm_usage(
            merchant_id=state["merchant_id"],
            conversation_id=state["conversation_id"],
            model=model,
            usage=turn.usage,
        )

    tool_use_blocks = [b for b in turn.content_blocks if b.type == "tool_use"]
    # Stored as the *real* response content, unsanitised - see
    # `_sanitize_messages_for_request`'s docstring for why cleaning it here
    # instead wouldn't survive a checkpoint round-trip, and why the actual fix
    # has to live at the point every outgoing call builds `stream_kwargs`.
    assistant_message: MessageParam = {"role": "assistant", "content": turn.content_blocks}

    if tool_use_blocks:
        call = tool_use_blocks[0]
        # `input`, not `query` - two of the three tools (get_catalog_stats,
        # get_product_detail) have no `query` field at all. Wire contract change
        # from the single-tool era: `static/index.html`'s SSE handler must read
        # `payload.input`/`payload.tool` now, not assume `payload.query` exists.
        tool_input = call.input if isinstance(call.input, dict) else {}
        writer({"type": "tool_call", "tool": call.name, "input": tool_input})
        return {"messages": [assistant_message], "model_used": model}

    return {
        "messages": [assistant_message],
        "model_used": model,
        "final_answer": turn.text,
    }


def _filters_from_tool_input(tool_input: Mapping[str, Any]) -> SearchFilters:
    """The four structured filter fields every catalog tool schema shares
    (`app/llm/prompting.py::_PRICE_BRAND_FILTER_PROPERTIES`), translated from the
    model's JSON args into a real `SearchFilters`. Exposed as singular `brand`/
    `category` strings in the tool schema (simpler for the model to emit for the
    common case) - wrapped into the one-item lists `SearchFilters` expects;
    multi-value questions ("Nike or Adidas") are out of scope for now."""
    min_price = tool_input.get("min_price")
    max_price = tool_input.get("max_price")
    brand = tool_input.get("brand")
    category = tool_input.get("category")
    return SearchFilters(
        min_price=Decimal(str(min_price)) if min_price is not None else None,
        max_price=Decimal(str(max_price)) if max_price is not None else None,
        brands=[brand] if brand else None,
        categories=[category] if category else None,
        in_stock_only=bool(tool_input.get("in_stock_only", False)),
    )


# Each handler takes `tenant` as an explicit argument from `state["tenant"]`,
# *never* from `tool_input` - none of the three tool schemas
# (`app/llm/prompting.py`) declare a `tenant`/`merchant`/`store` property, so the
# model has no field to fill even under a prompt-injection attempt. This is the
# existing, correct pattern `search_catalog` already used; preserved exactly
# here rather than let a "generic dispatch" refactor introduce a
# `tool_input.get("tenant", state["tenant"])` convenience path, which would
# quietly reintroduce the caller-controlled-tenant-filter shortcut CLAUDE.md's
# tenant-isolation invariant exists to forbid - except LLM-controlled, which is
# strictly worse. Each returns `(tool_result_for_the_model, evidence_for_state)`.


async def _run_search_catalog(
    tenant: str, tool_input: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query = str(tool_input.get("query", ""))
    filters = _filters_from_tool_input(tool_input)
    sort_by = tool_input.get("sort_by", "relevance")

    if sort_by == "relevance":
        request = SearchRequest(query=query, tenant=tenant, limit=5, filters=filters)
        result = await get_retriever().search(request)
        hits = result.hits
    else:
        hits = await get_retriever().search_sorted(tenant, query, filters, sort_by, limit=5)

    tool_result = hits_to_tool_result(hits)
    return tool_result, hits_to_evidence(hits)


async def _run_get_catalog_stats(
    tenant: str, tool_input: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    filters = _filters_from_tool_input(tool_input)
    metric = tool_input.get("metric", "price")
    stats = await get_retriever().stats(tenant, filters, metric=metric)
    tool_result = stats_to_tool_result(stats)
    return tool_result, [{"kind": "stats", "filters": dict(tool_input), **tool_result}]


async def _run_get_product_detail(
    tenant: str, tool_input: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    requested_skus = [str(s) for s in tool_input.get("skus", [])]
    hits = await get_retriever().get_by_skus(tenant, requested_skus)
    tool_result = product_detail_to_tool_result(hits, requested_skus)
    return tool_result, hits_to_evidence(hits)


_TOOL_HANDLERS = {
    "search_catalog": _run_search_catalog,
    "get_catalog_stats": _run_get_catalog_stats,
    "get_product_detail": _run_get_product_detail,
}


async def tool_node(state: ChatState) -> dict[str, Any]:
    last = state["messages"][-1]
    assert last["role"] == "assistant"
    # `last["content"]` is always `_AgentTurn.content_blocks` (real response
    # objects from `get_final_message()`, see `agent()`) at this point in the
    # graph, never a bare string or a request-side Param dict - the broad
    # `MessageParam.content` union is only this wide because the same field also
    # has to describe outgoing user/tool-result turns.
    content = cast("list[ContentBlock]", last["content"])
    call = cast(ToolUseBlock, next(b for b in content if b.type == "tool_use"))
    tool_input = call.input if isinstance(call.input, dict) else {}

    observe_chat_tool_call(tool=call.name)
    # A KeyError here means agent()'s tool list and this dispatch table have
    # drifted - a real bug to surface loudly, not swallow.
    handler = _TOOL_HANDLERS[call.name]
    tool_result, new_evidence = await handler(state["tenant"], tool_input)

    new_rounds = state["tool_call_rounds"] + 1
    tool_result_message: MessageParam = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": [{"type": "text", "text": str(tool_result)}],
            }
        ],
    }
    # Accumulate across rounds, never overwrite - a real bug caught by running
    # `eval/generation_eval.py` at full scale, not by inspection: a second search
    # round used to *replace* `citations` outright, so a final answer that
    # legitimately cited a product found in round 1 (still visible to the model
    # in its own message history) got flagged as a fabricated citation by
    # `validate_and_store` below - a false positive baked into a live metric, not
    # just eval. Deduped by SKU (product evidence only - a `stats` entry has no
    # SKU to dedupe on, and a repeated stats call with different filters is
    # legitimately different evidence, not a duplicate), first-seen order kept.
    seen_skus = {e["sku"] for e in state["citations"] if e.get("kind") == "product"}
    merged_citations = state["citations"] + [
        e for e in new_evidence if e.get("kind") != "product" or e["sku"] not in seen_skus
    ]
    return {
        "messages": [tool_result_message],
        "tool_call_rounds": new_rounds,
        "citations": merged_citations,
        # This call's pending tool_use is answered above regardless (Anthropic
        # requires every tool_use to get a matching tool_result before the next
        # turn - we can't just stop mid-protocol) - what changes is whether the
        # *next* agent call is still allowed to ask for another one. See
        # `ChatState.force_no_tools`.
        "force_no_tools": new_rounds >= MAX_TOOL_CALL_ROUNDS,
    }


async def validate_and_store(state: ChatState) -> dict[str, Any]:
    query = _last_user_text(state["messages"])
    answer = state["final_answer"]
    # The ledger mixes two evidence kinds now (`"kind": "product"` from
    # search_catalog/get_product_detail, `"kind": "stats"` from
    # get_catalog_stats) - each verification check below only makes sense
    # against its own kind, so split once up front rather than re-filter inside
    # every check.
    product_evidence = [c for c in state["citations"] if c.get("kind") == "product"]
    stats_evidence = [c for c in state["citations"] if c.get("kind") == "stats"]

    hallucinated = find_hallucinated_citations(answer, product_evidence)
    if hallucinated:
        observe_claim_mismatch(claim_type="hallucinated_citation", count=len(hallucinated))

    if find_stat_claim_mismatch(answer, stats_evidence):
        observe_claim_mismatch(claim_type="stat_mismatch", count=1)

    if find_unverified_quantitative_refusal(answer, stats_evidence):
        observe_claim_mismatch(claim_type="unverified_quantitative_refusal", count=1)

    # Signal only, not a gate (confirmed with the user) - matches
    # `find_hallucinated_citations`'s "detection, not prevention" philosophy
    # rather than forcing a retry that could misfire on a lexical false
    # positive. Lets production track "did a superlative-shaped question
    # actually get a get_catalog_stats call" as a leading indicator, the same
    # signal that would have caught the flagship bug before a user hit it.
    if has_superlative_language(query) and not stats_evidence:
        observe_claim_mismatch(claim_type="superlative_without_stats", count=1)

    # A refusal-shaped answer ("this catalog doesn't carry X") means the model
    # itself judged every retrieved product this turn irrelevant - showing them
    # as product cards would visually contradict the words right above them (a
    # real live case: "red underwear" against the electronics catalog correctly
    # refused in prose, while the cards still showed phone cases and a fountain
    # pen as if they were the answer). Only affects what's *displayed and
    # cached* - `hallucinated`/the checks above already ran against the full,
    # unfiltered `product_evidence`, and still should.
    #
    # `is_no_match` (the `[[NO_MATCH]]` marker Claude is instructed to emit,
    # `app/llm/markers.py`) is the primary signal now - a structural declaration,
    # not a guess at refusal phrasing. Measured live (2026-08-22): a raw hybrid
    # retrieval score cannot separate a genuine match from a keyword-coincidence
    # false positive ("Samsung Galaxy S23" top-scoring a "Galaxy Print Gift Bag"),
    # so the fix is Claude's own semantic judgment declared explicitly, not a
    # score threshold - see PROGRESS.md's dated entry for the full measurement.
    # `refuses()` stays wired in as a real backstop, not deleted: if Claude ever
    # fails to emit the marker but its prose still reads as a refusal (a
    # compliance miss), the lexical detector still catches the display-layer
    # symptom.
    displayed_products = [] if is_no_match(answer) or refuses(answer) else product_evidence

    writer = get_stream_writer()
    writer({"type": "citations", "products": displayed_products, "cached": False})

    # Real bug found and fixed the same day this was written (2026-08-22): this
    # write had no `semantic_cache_enabled` guard, unlike `maybe_serve_from_cache`'s
    # read side - `eval/measure_model_router.py` disabled the setting for its
    # *lookups* only, expecting that to make its runs side-effect-free, but every
    # scored turn (including a real one-off empty-answer response) still got
    # written to the *shared* semantic cache regardless, where a real shopper's
    # semantically similar query could have been served that broken answer later.
    # Also skips a wasted embedding call when caching is off entirely.
    if get_settings().semantic_cache_enabled and not stats_evidence:
        embedding = await aembed_query(query)
        await cache_store(
            get_redis_client(), state["tenant"], embedding, answer, displayed_products
        )
    observe_chat_request(model=state["model_used"] or "unknown", cache_hit=False)
    return {}


def _route_after_cache(state: ChatState) -> str:
    return END if state["cache_hit"] else "agent"


def _route_after_agent(state: ChatState) -> str:
    last = state["messages"][-1]
    has_call = any(getattr(b, "type", None) == "tool_use" for b in last["content"])
    if has_call and state["tool_call_rounds"] < MAX_TOOL_CALL_ROUNDS:
        return "tool_node"
    return "validate_and_store"


@lru_cache(maxsize=1)
def get_chat_graph() -> CompiledStateGraph[ChatState, Any, Any, Any]:
    """Compiled once per process, with a process-wide `InMemorySaver` checkpointer -
    see the module docstring for why in-memory is the right call for now and what
    the production upgrade path is."""
    graph = StateGraph(ChatState)
    graph.add_node("maybe_serve_from_cache", maybe_serve_from_cache)
    graph.add_node("agent", agent)
    graph.add_node("tool_node", tool_node)
    graph.add_node("validate_and_store", validate_and_store)

    graph.set_entry_point("maybe_serve_from_cache")
    graph.add_conditional_edges(
        "maybe_serve_from_cache", _route_after_cache, {"agent": "agent", END: END}
    )
    graph.add_conditional_edges(
        "agent",
        _route_after_agent,
        {"tool_node": "tool_node", "validate_and_store": "validate_and_store"},
    )
    graph.add_edge("tool_node", "agent")
    graph.add_edge("validate_and_store", END)

    return graph.compile(checkpointer=InMemorySaver())
