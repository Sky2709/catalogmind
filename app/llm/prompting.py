"""What the agent sees: the stable system instruction, the one bound tool, and how a
retrieval result gets shown back to the model.

**Prompt caching, and an honest limit on it**: Bedrock's Claude prompt caching
(confirmed against the reference Bedrock user guide) needs a minimum of 4,096
tokens in a single cache checkpoint to actually cache anything - below that, a
`cache_control` marker is accepted but does nothing. `SYSTEM_INSTRUCTION` below is
a few hundred tokens at most (deliberately generic - no per-merchant customisation,
a scope decision named in the Day 5 plan this was built from), nowhere near that
floor on its own. The `cache_control` marker is still attached, because it's free
and correctly positions this prefix to start actually caching once real multi-turn
conversation history (which shares the same system-prefix + tool-list checkpoint)
grows past 4,096 tokens - but claiming a single-turn system-only prefix gets cached
today would be asserting something not measured, exactly what this project's own
rules say not to do. Padding the prompt artificially just to clear the threshold
was considered and rejected: that spends real tokens on every call to manufacture a
saving that wouldn't actually reflect real usage.
"""

from __future__ import annotations

from anthropic.types import CacheControlEphemeralParam, TextBlockParam, ToolParam

from app.retrieval.base import CatalogStats, SearchHit

# Real production bug this system prompt is written against (2026-08-22, live-
# confirmed, not hypothetical): asked "what's the highest-priced item for men"
# with only `search_catalog` available, the model confidently claimed nothing
# exceeded ₹2,499 - the real answer was a ₹58,854 MOVADO watch, and 94 products
# in the catalog exceeded ₹10,000. `search_catalog` only ever returns a handful of
# relevance-ranked results; it structurally cannot support a catalog-wide claim,
# and the fix is not "search harder" but "use a tool built for the question."
_SYSTEM_TEXT = """\
You are a shopping assistant for one merchant's product catalog. You have three
tools:

- `search_catalog`: relevance-ranked search, returns a handful of results. Use
  it to find products matching what the shopper described. It accepts optional
  price/brand/category/in-stock filters, and an optional `sort_by` for "cheapest/
  priciest/best-rated **matching this description**" (e.g. "cheapest waterproof
  jacket").
- `get_catalog_stats`: exact count/min/max/average price (or rating, or review
  count) over the WHOLE matching catalog, not just a handful of results. This is
  the ONLY tool that can correctly answer a superlative ("highest/lowest priced
  item"), a threshold ("anything above/below a price?"), or a count ("how many
  X do you have?") when the question is scoped by price, brand, category, or
  stock status. A `search_catalog` miss NEVER proves nothing matches - only
  `get_catalog_stats` reporting count=0 does.
- `get_product_detail`: exact lookup by one or more SKUs already mentioned
  earlier in this conversation - use for a follow-up about a specific product,
  or to compare two known products in one call, instead of re-running
  search_catalog and hoping it resurfaces the same item.

Two claims need different levels of confidence, and your wording must reflect
that difference:
1. A claim backed by `get_catalog_stats` (a price/brand/category/stock-scoped
   superlative, threshold, or count) is a real, exact fact about the whole
   catalog - state it plainly.
2. A claim backed by `search_catalog`'s `sort_by` (a superlative scoped by a
   free-text description, e.g. "cheapest waterproof jacket", where the category
   isn't one of the structured filters) is only the best match among the
   results that search found - it is NOT a guaranteed catalog-wide answer.
   Say so explicitly ("among what I found...", "I can't guarantee this is the
   single cheapest such item in the whole catalog") rather than stating it with
   the same certainty as a `get_catalog_stats` fact.

Follow this exact citation and relevance protocol on every answer - it lets what
you say be checked automatically against real catalog data, so match the format
precisely, with nothing else inside the brackets:
- Every time you mention or recommend a specific product, wrap its exact SKU
  like this: [[SKU:THE-EXACT-SKU]].
- Every time you state a figure backed by `get_catalog_stats` (a count, min,
  max, or average), wrap the bare number like this: [[STAT:58854]] (digits and
  at most one decimal point only - no currency symbol, no commas).
- If none of the products `search_catalog` or `get_product_detail` returned are
  a genuine match for what the shopper asked - a different kind of product
  entirely, not just a weaker match - your entire response must begin with the
  exact literal text [[NO_MATCH]] as its very first characters, followed by a
  brief, honest explanation of why nothing matches. Never use [[NO_MATCH]] if
  you go on to recommend or mention any product.

Never invent a SKU, price, count, or product not returned by a tool. A "nothing
matches" answer about a price/brand/category/stock condition is only correct if
`get_catalog_stats` actually reported count=0 for that condition - never
conclude that from `search_catalog` alone returning few or no results. If
`get_product_detail` doesn't find a SKU, say it's "not available in this
catalog" - never speculate about whether it exists somewhere else.

When `search_catalog` returns several matching products, the shopper already
sees each one's full details (title, price, stock, photo) in product cards
shown right alongside your answer - do NOT re-list every result's name, price,
and stock status in prose, that just repeats what's already on screen. Instead
write one short sentence: how many you found, plus one useful highlight (e.g.
the cheapest, the best match, or a standout feature), citing that one product's
[[SKU:...]] so the shopper has an anchor for a follow-up question. Give a fuller
per-item description only when the shopper is asking about, comparing, or being
shown a single specific product.
"""

SYSTEM_PROMPT: list[TextBlockParam] = [
    {
        "type": "text",
        "text": _SYSTEM_TEXT,
        "cache_control": CacheControlEphemeralParam(type="ephemeral"),
    }
]

# Appended only on the turn where `ChatState.force_no_tools` is true (see
# `app/llm/graph.py::agent`) - a real bug caught by `eval/generation_eval.py`, not
# a hypothetical: removing `tools` from that call stops the model from *requesting*
# another call, but says nothing about what to do instead, so it would sometimes
# narrate the call it can no longer make ("Let me try one more search for
# furniture.") as its entire final answer, or return nothing at all. Kept as a
# separate, uncached block appended after `SYSTEM_PROMPT`'s cached one rather than
# folded into `_SYSTEM_TEXT` - it only applies on a small minority of turns (the
# tool-call loop hitting its cap), so baking it into every call's cached prefix
# would be paying for it on turns that never need it.
FORCE_ANSWER_NUDGE: TextBlockParam = {
    "type": "text",
    "text": (
        "You have used your tool-call budget for this turn and no further "
        "catalog tools are available. Give your final, complete answer to the "
        "shopper now, based only on what earlier calls already returned. If "
        "that isn't enough to answer confidently, say so plainly instead of "
        "proposing another call you can no longer make."
    ),
}

_PRICE_BRAND_FILTER_PROPERTIES: dict[str, object] = {
    "min_price": {"type": "number", "description": "Inclusive lower bound."},
    "max_price": {"type": "number", "description": "Inclusive upper bound."},
    "brand": {"type": "string"},
    "category": {"type": "string"},
    "in_stock_only": {"type": "boolean", "default": False},
}

SEARCH_CATALOG_TOOL: ToolParam = {
    "name": "search_catalog",
    "description": (
        "Search this merchant's product catalog by free text, with optional "
        "structured filters. Call again with a refined query if the first "
        "results don't answer the shopper's question. With `sort_by` set to "
        "anything but the default `relevance`, returns the cheapest/priciest/"
        "best-rated items among a wider pool of matches to the query text - use "
        "this for a superlative scoped by a description ('cheapest waterproof "
        "jacket'), not by price/brand/category/stock (use get_catalog_stats for "
        "those instead, since this only searches a bounded pool of candidates "
        "and can't guarantee it saw every match)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for, in the shopper's own words.",
            },
            **_PRICE_BRAND_FILTER_PROPERTIES,
            "sort_by": {
                "type": "string",
                "enum": ["relevance", "price_asc", "price_desc", "rating_desc"],
                "default": "relevance",
            },
        },
        "required": ["query"],
    },
}

GET_CATALOG_STATS_TOOL: ToolParam = {
    "name": "get_catalog_stats",
    "description": (
        "Get exact count/min/max/average for a metric (price by default, or "
        "rating/review count) over the WHOLE matching catalog, not just a "
        "handful of results - filtered by price/brand/category/stock, never by "
        "free text. This is the only correct way to answer a superlative "
        "('highest/lowest priced item'), a threshold ('anything above/under a "
        "price?'), or a count ('how many X do you have?') when the question is "
        "scoped by one of those structured fields. If this reports count=0, "
        "that is a confirmed 'nothing matches' - a search_catalog miss alone "
        "never is."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "metric": {
                "type": "string",
                "enum": ["price", "rating", "review_count"],
                "default": "price",
            },
            **_PRICE_BRAND_FILTER_PROPERTIES,
        },
        "required": [],
    },
}

GET_PRODUCT_DETAIL_TOOL: ToolParam = {
    "name": "get_product_detail",
    "description": (
        "Look up one or more exact products by SKU - use for a follow-up "
        "question about a specific product already cited earlier in this "
        "conversation, or to compare up to 5 known products in a single call, "
        "instead of re-running search_catalog and hoping it resurfaces the "
        "same item(s)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "skus": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 5,
                "description": "Exact SKUs, as previously cited.",
            }
        },
        "required": ["skus"],
    },
}

# Only what's needed to answer and cite - not the full `SearchHit` (vector/keyword/
# rerank scores etc. are provenance for eval and debugging, not something a shopper
# ever needs relayed back to them, and every extra field is tokens paid on every call).
# Deliberately excludes `image_url` too: Claude never reasons about or cites an image,
# so a URL here would be pure token cost with no benefit - see `_evidence` below for
# the richer, browser-facing shape that does need it.


def _citable(hit: SearchHit) -> dict[str, object]:
    return {
        "sku": hit.sku,
        "title": hit.title,
        "brand": hit.brand,
        "price": str(hit.price) if hit.price is not None else None,
        "currency": hit.currency,
        "in_stock": hit.in_stock,
        "category_path": hit.category_path,
        "rating": str(hit.rating) if hit.rating is not None else None,
    }


# What the shopper's browser needs to render a product card in the `citations` SSE
# event - a superset of `_citable`'s model-facing shape, built straight from the same
# `SearchHit`s so the two can never drift apart on sku/title/etc. Never sent to
# Claude: `tool_node` (`app/llm/graph.py`) passes `_citable`'s output to the model and
# this function's output to the frontend as two separate values from the same call.
def _evidence(hit: SearchHit) -> dict[str, object]:
    return {**_citable(hit), "image_url": hit.image_url}


def hits_to_evidence(hits: list[SearchHit]) -> list[dict[str, object]]:
    """The `citations` event payload for a batch of hits - tagged `"kind": "product"`
    to match the shape `tool_node` accumulates in `ChatState.citations`."""
    return [{"kind": "product", **_evidence(hit)} for hit in hits]


def hits_to_tool_result(hits: list[SearchHit]) -> dict:
    """The JSON payload handed back to Claude as a `search_catalog` result."""
    return {"products": [_citable(hit) for hit in hits]}


def stats_to_tool_result(stats: CatalogStats) -> dict:
    """The JSON payload handed back to Claude as a `get_catalog_stats` result."""
    return {
        "metric": stats.metric,
        "count": stats.count,
        "minimum": str(stats.minimum) if stats.minimum is not None else None,
        "maximum": str(stats.maximum) if stats.maximum is not None else None,
        "mean": str(stats.mean) if stats.mean is not None else None,
    }


def product_detail_to_tool_result(hits: list[SearchHit], requested_skus: list[str]) -> dict:
    """The JSON payload handed back to Claude as a `get_product_detail` result.
    `not_found` is explicit so a partial batch miss (2 of 3 SKUs found) doesn't
    read as either a full miss or get silently dropped."""
    found_skus = {hit.sku for hit in hits}
    return {
        "products": [_citable(hit) for hit in hits],
        "not_found": [sku for sku in requested_skus if sku not in found_skus],
    }
