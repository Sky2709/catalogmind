"""Raw hybrid search: keyword + vector, tunable alpha, optional cross-encoder rerank.

Deliberately separate from the future `chat` endpoint - this is the primitive a
shopping assistant's tool calls will sit on top of, but it is also useful on its own
(a merchant's own site search, or `eval/` scoring it directly against golden query
sets) and should not require going through an LLM turn to reach.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.deps import ScopedMerchant
from app.rate_limit import check_rate_limit
from app.redis_client import get_redis_client
from app.retrieval.base import SearchRequest, SearchResponse
from app.retrieval.hybrid import get_retriever
from app.schemas import SearchQuery

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/merchants", tags=["search"])

# Search is a normal per-shopper-query hot path, not an occasional upload - so this
# limit exists only to catch a runaway client or script hammering one tenant, not to
# constrain legitimate traffic. Set far higher than the ingest endpoint's 10/60s for
# exactly that reason: 2 req/sec sustained is nothing for a real storefront, but it
# stops one misbehaving caller from turning every request into an embed + hybrid
# search + N cross-encoder passes indefinitely.
SEARCH_RATE_LIMIT = 120
SEARCH_RATE_WINDOW_SECONDS = 60


async def enforce_search_rate_limit(merchant: ScopedMerchant) -> None:
    redis = get_redis_client()
    result = await check_rate_limit(
        redis,
        f"ratelimit:search:{merchant.tenant}",
        limit=SEARCH_RATE_LIMIT,
        window_seconds=SEARCH_RATE_WINDOW_SECONDS,
    )
    if not result.allowed:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded: {SEARCH_RATE_LIMIT} searches per "
                f"{SEARCH_RATE_WINDOW_SECONDS}s. Retry after {result.retry_after_seconds}s."
            ),
            headers={"Retry-After": str(result.retry_after_seconds)},
        )


@router.post(
    "/{tenant}/search",
    response_model=SearchResponse,
    summary="Hybrid keyword + vector search over one merchant's catalog",
    dependencies=[Depends(enforce_search_rate_limit)],
    responses={429: {"description": "Too many searches for this merchant in the current window"}},
)
async def search_catalog(merchant: ScopedMerchant, body: SearchQuery) -> SearchResponse:
    """Attach per-merchant retrieval defaults, then delegate to the `Retriever`.

    Precedence for `alpha`: an explicit value in this request body wins; otherwise
    the merchant's own stored `alpha_override` applies; only then does the retriever
    fall back to dynamic alpha routing. Resolved here, not inside the retriever, so
    `Retriever` stays generic - it knows about tenants and queries, not about a
    specific merchant's saved preferences. Safe to fully resolve alpha at this layer
    because the retriever has no alpha heuristic of its own to apply in between.

    `rerank` is deliberately **not** collapsed the same way: `merchant.rerank_enabled`
    is passed through as `rerank_default`, a distinct field from the explicit
    `rerank` override, so the retriever's own query-shape heuristic (skip reranking a
    real product-code query - see `hybrid.py`) can still apply between "the caller
    said nothing" and "fall back to the merchant's stored preference." Collapsing it
    here the way `alpha` is would erase that middle state before the retriever ever
    saw it.
    """
    alpha = body.alpha if body.alpha is not None else merchant.alpha_override

    request = SearchRequest(
        query=body.query,
        tenant=merchant.tenant,
        limit=body.limit,
        alpha=alpha,
        rerank=body.rerank,
        rerank_default=merchant.rerank_enabled,
        filters=body.filters,
    )
    return await get_retriever().search(request)
