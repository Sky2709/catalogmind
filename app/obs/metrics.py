"""Prometheus metrics for the search hot path.

`SearchResponse.stage_timings_ms` already tells one caller how long *their* request
took at each stage - useful for debugging one slow response, useless for answering
"is p99 search latency healthy across every tenant this hour". That second question
needs samples aggregated across every request in a process (and, once there is more
than one replica, across replicas too), which is exactly what a scrape-based
histogram is for: `histogram_quantile()` computes p50/p95/p99 server-side from the
bucket counts, correctly, across as many processes as are being scraped - client-side
percentile math over one process's own samples does not aggregate that way.

Deliberately **not** labelled by tenant or query. Prometheus's data model is one time
series per unique label combination - a `tenant` label turns "a few histograms" into
"one histogram per merchant, forever", the textbook cardinality mistake. `stage` is
safe because it is a small, fixed set (`embed_ms`, `hybrid_search_ms`, `rerank_ms`,
`total_ms`); per-tenant cost/latency is Day 6's "per-merchant cost tracking" and
belongs in a store built for that shape of query (Postgres), not a Prometheus label.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

SEARCH_STAGE_LATENCY_SECONDS = Histogram(
    "catalogmind_search_stage_latency_seconds",
    "Latency of one stage of the search pipeline (embed, hybrid_search, rerank, total).",
    ["stage"],
    # Tight buckets below 100ms, where embed/hybrid_search mostly land, opening up for
    # rerank and total - a 2.5s+ search is already a bad experience worth its own bucket.
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

SEARCH_REQUESTS_TOTAL = Counter(
    "catalogmind_search_requests_total",
    "Completed search requests, by whether the rerank stage ran and by query class.",
    ["reranked", "query_class"],
)


def observe_search(stage_timings_ms: dict[str, float], *, reranked: bool, query_class: str) -> None:
    for stage, ms in stage_timings_ms.items():
        SEARCH_STAGE_LATENCY_SECONDS.labels(stage=stage).observe(ms / 1000)
    SEARCH_REQUESTS_TOTAL.labels(reranked=str(reranked), query_class=query_class).inc()


# --- chat (Day 5) -----------------------------------------------------------------
#
# Same cardinality discipline as the search metrics above: `stage` and `model` are
# small, fixed sets (two models, a handful of graph stages) - never tenant, never
# conversation/session id. Buckets extend further than search's (15s/30s) because an
# LLM call routinely runs multi-second, unlike a local embed/hybrid-search stage.

CHAT_STAGE_LATENCY_SECONDS = Histogram(
    "catalogmind_chat_stage_latency_seconds",
    "Latency of one stage of the chat agent graph (agent, tool_call, total).",
    ["stage"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0),
)

CHAT_REQUESTS_TOTAL = Counter(
    "catalogmind_chat_requests_total",
    "Completed chat turns, by which model answered and whether it was a cache hit.",
    ["model", "cache_hit"],
)

CHAT_TOKENS_TOTAL = Counter(
    "catalogmind_chat_tokens_total",
    "LLM tokens consumed, by model and token type.",
    ["model", "token_type"],
)

# Not tenant/query-labelled for the same cardinality reason as everything else here -
# this is a count of *events*, not a per-response detail. `claim_type` is a small,
# fixed set (see `app/llm/claims.py`/`app/llm/citations.py` for what each one
# checks) - generalised from a citations-only counter (2026-08-21) once the chat
# agent gained tools whose claims aren't SKU citations at all (a catalog-wide
# stat, a "nothing matches" refusal) - the eval harness's groundedness/refusal-
# correctness rates are what turn these raw counts into a rate; this is the raw
# detection signal each of those checks feeds.
CHAT_CLAIM_MISMATCHES_TOTAL = Counter(
    "catalogmind_chat_claim_mismatches_total",
    "Claims in a chat answer that couldn't be verified against this turn's tool results, by claim type.",
    ["claim_type"],
)

# `tool` is a small, fixed set (the three catalog tools) - the leading indicator
# for whether get_catalog_stats/get_product_detail are actually being invoked for
# the question shapes they exist to cover, not just declared and ignored. Without
# this, a regression back to "the model tried to answer a superlative question
# with search_catalog alone" would only surface as a claim-mismatch after the
# fact, not as a visible shift in which tools get called at all.
CHAT_TOOL_CALLS_TOTAL = Counter(
    "catalogmind_chat_tool_calls_total",
    "Chat tool calls, by tool name.",
    ["tool"],
)


def observe_chat_stage(stage_timings_ms: dict[str, float]) -> None:
    for stage, ms in stage_timings_ms.items():
        CHAT_STAGE_LATENCY_SECONDS.labels(stage=stage).observe(ms / 1000)


def observe_chat_request(*, model: str, cache_hit: bool) -> None:
    CHAT_REQUESTS_TOTAL.labels(model=model, cache_hit=str(cache_hit)).inc()


def observe_chat_tokens(*, model: str, prompt: int, output: int, cached: int) -> None:
    CHAT_TOKENS_TOTAL.labels(model=model, token_type="input").inc(prompt)
    CHAT_TOKENS_TOTAL.labels(model=model, token_type="output").inc(output)
    CHAT_TOKENS_TOTAL.labels(model=model, token_type="cached").inc(cached)


def observe_claim_mismatch(*, claim_type: str, count: int) -> None:
    CHAT_CLAIM_MISMATCHES_TOTAL.labels(claim_type=claim_type).inc(count)


def observe_chat_tool_call(*, tool: str) -> None:
    CHAT_TOOL_CALLS_TOTAL.labels(tool=tool).inc()


def render_latest() -> tuple[bytes, str]:
    """Body and content-type for a `/metrics` scrape endpoint."""
    return generate_latest(), CONTENT_TYPE_LATEST
