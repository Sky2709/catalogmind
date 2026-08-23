"""The concrete `Retriever`: Weaviate hybrid search, optionally reranked.

One search request runs, in order:

1. **Classify + choose alpha** (`alpha_router.py`) - microseconds, no I/O.
2. **Embed the query** (`app.ingestion.embed`) - the same BGE model and asymmetric
   instruction ingestion used to embed the catalog.
3. **Hybrid search** against the tenant's Weaviate shard, blending BM25 and vector
   scores by `alpha`, fetching `retrieve_top_k` candidates.
4. **Optional cross-encoder rerank** (`rerank.py`) down to `request.limit` - skipped
   automatically for a query with a real SKU/model-number-shaped token
   (`has_identifier_shaped_token`), since `rerank_text()` never includes the SKU and
   `scripts/bench_search.py` measured reranking at ~8.8s per call at
   `retrieve_top_k=50`: paying that cost to reshuffle results the reranker has no real
   signal for is pure waste, not a quality/latency tradeoff.

Every stage is timed and returned in `SearchResponse.stage_timings_ms` - the whole
point of measuring retrieval quality (Day 4) is knowing what each knob costs, not just
what it's worth.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from decimal import Decimal
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from weaviate.classes.aggregate import Metrics
from weaviate.classes.query import Filter, HybridFusion, MetadataQuery
from weaviate.collections.classes.filters import FilterReturn

from app.config import get_settings
from app.ingestion.embed import aembed_query
from app.obs.metrics import observe_search
from app.retrieval import rerank as rerank_module
from app.retrieval.alpha_router import AlphaRouter, has_identifier_shaped_token
from app.retrieval.base import (
    CatalogStats,
    SearchFilters,
    SearchHit,
    SearchRequest,
    SearchResponse,
    StatsMetric,
)
from app.retrieval.weaviate_client import (
    WEAVIATE_TRANSIENT_ERRORS,
    get_shared_client,
    product_collection,
)
from app.retry import with_retry

if TYPE_CHECKING:
    from weaviate.collections.classes.internal import Object

logger = logging.getLogger(__name__)

# Only what a `SearchHit` needs. `attributes_json` and `attributes_text` are payload
# and rerank input respectively - never indexed for anything else - so there is no
# reason to pull the rest of the schema (content_hash, updated_at, ...) over the wire.
RETURN_PROPERTIES = [
    "sku",
    "title",
    "description",
    "brand",
    "category_path",
    "gender",
    "price",
    "currency",
    "in_stock",
    "attributes_text",
    "attributes_json",
    "image_url",
    "rating",
]


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 2)


def _build_filters(filters: SearchFilters) -> FilterReturn | None:
    """Structured pre-filters, ANDed together. `None` means "no filter" - Weaviate's
    `hybrid()` takes that to mean unfiltered, rather than an empty-result filter."""
    clauses: list[FilterReturn] = []

    if filters.min_price is not None:
        clauses.append(Filter.by_property("price").greater_or_equal(float(filters.min_price)))
    if filters.max_price is not None:
        clauses.append(Filter.by_property("price").less_or_equal(float(filters.max_price)))
    if filters.brands:
        # `equal`, not `contains_any`: brand is a scalar TEXT field and we want exact
        # membership in the given list, not a tokenised keyword match against it.
        brand_filter = Filter.by_property("brand").equal(filters.brands[0])
        for brand in filters.brands[1:]:
            brand_filter = brand_filter | Filter.by_property("brand").equal(brand)
        clauses.append(brand_filter)
    if filters.categories:
        # `category_path` is a TEXT_ARRAY; `contains_any` checks array membership, not
        # a tokenised text match, so "Shoes" matches a path containing "Shoes" exactly.
        clauses.append(Filter.by_property("category_path").contains_any(filters.categories))
    if filters.in_stock_only:
        clauses.append(Filter.by_property("in_stock").equal(True))
    if filters.genders:
        gender_filter = Filter.by_property("gender").equal(filters.genders[0])
        for gender in filters.genders[1:]:
            gender_filter = gender_filter | Filter.by_property("gender").equal(gender)
        clauses.append(gender_filter)
    if filters.min_rating is not None:
        clauses.append(Filter.by_property("rating").greater_or_equal(float(filters.min_rating)))

    if not clauses:
        return None
    combined = clauses[0]
    for clause in clauses[1:]:
        combined = combined & clause
    return combined


def rerank_text(properties: Mapping[str, Any]) -> str:
    """Text handed to the cross-encoder for one candidate.

    Mirrors `Product.embedding_text()`'s field order so the reranker sees the same
    signal the vector index was built from - built from Weaviate's returned
    properties rather than a `Product`, because that is all a search hit has at this
    point in the pipeline. `category_path` deliberately excluded here too, for the
    same measured reason `embedding_text()`'s docstring gives (a real, broad nDCG
    regression, not a hypothesis) - keeping this mirror accurate matters more than
    the extra signal would have.
    """
    parts = [str(properties.get("title") or "")]
    if properties.get("brand"):
        parts.append(str(properties["brand"]))
    if properties.get("description"):
        parts.append(str(properties["description"]))
    if properties.get("attributes_text"):
        parts.append(str(properties["attributes_text"]))
    return "\n".join(parts)


def _to_decimal(value: Any) -> Decimal | None:
    return Decimal(str(value)) if value is not None else None


def hit_from_properties(
    properties: Mapping[str, Any], *, score: float, rank_before_rerank: int
) -> SearchHit:
    """A `SearchHit` from Weaviate's returned properties for one object."""
    attributes: dict[str, Any] = {}
    raw_attributes = properties.get("attributes_json")
    if raw_attributes:
        try:
            attributes = json.loads(raw_attributes)
        except (TypeError, ValueError):
            attributes = {}

    return SearchHit(
        sku=str(properties["sku"]),
        title=str(properties["title"]),
        score=score,
        brand=properties.get("brand"),
        price=_to_decimal(properties.get("price")),
        currency=properties.get("currency"),
        in_stock=bool(properties.get("in_stock", True)),
        image_url=properties.get("image_url"),
        category_path=list(properties.get("category_path") or []),
        gender=properties.get("gender"),
        attributes=attributes,
        rating=_to_decimal(properties.get("rating")),
        rank_before_rerank=rank_before_rerank,
    )


class WeaviateHybridRetriever:
    """Hybrid search over one tenant's Weaviate shard, with optional reranking."""

    name = "weaviate-hybrid"

    def __init__(
        self, alpha_router: AlphaRouter | None = None, retrieve_top_k: int | None = None
    ) -> None:
        """`retrieve_top_k` overrides `Settings.retrieve_top_k` for this instance only -
        `get_retriever()`'s process-wide singleton never passes it, so production is
        unaffected. Exists for `eval/retrieval_eval.py` to measure the effect of the
        candidate-pool depth directly, without touching global settings or needing a
        second env to compare against."""
        settings = get_settings()
        self._retrieve_top_k = (
            retrieve_top_k if retrieve_top_k is not None else settings.retrieve_top_k
        )
        self._rerank_enabled_default = settings.rerank_enabled
        self._alpha_router = alpha_router or AlphaRouter(
            enabled=settings.dynamic_alpha_enabled, default_alpha=settings.default_alpha
        )

    async def health(self) -> bool:
        try:
            client = await get_shared_client()
            return bool(await client.is_ready())
        except Exception:  # noqa: BLE001 - health checks report, never raise
            return False

    async def search(self, request: SearchRequest) -> SearchResponse:
        t_start = time.perf_counter()
        stage_timings: dict[str, float] = {}

        alpha, classification = self._alpha_router.resolve(request.query, override=request.alpha)

        t = time.perf_counter()
        vector = await aembed_query(request.query)
        stage_timings["embed_ms"] = _ms(t)

        client = await get_shared_client()
        collection = product_collection(client, request.tenant)
        weaviate_filter = _build_filters(request.filters)

        # `retrieve_top_k` sizes the reranking candidate pool, not a cap on how many
        # results a caller can ask for (`request.limit` goes up to 100,
        # `SearchQuery`/`SearchRequest` in `schemas.py`/`base.py`) - fetching fewer
        # than `request.limit` would silently truncate a caller's page below what
        # they asked for.
        pool_size = max(self._retrieve_top_k, request.limit)

        async def _hybrid() -> Any:
            return await collection.query.hybrid(
                query=request.query,
                vector=vector,
                alpha=alpha,
                limit=pool_size,
                filters=weaviate_filter,
                fusion_type=HybridFusion.RELATIVE_SCORE,
                return_properties=RETURN_PROPERTIES,
                return_metadata=MetadataQuery(score=True),
            )

        t = time.perf_counter()
        # A dropped connection or a timeout here is retried the same way ingestion
        # retries its Weaviate calls (`with_retry`, `WEAVIATE_TRANSIENT_ERRORS`) -
        # this is the one I/O call on the search hot path that talks to a service
        # this process doesn't control, and a blip should not be a 500 to a shopper.
        result = await with_retry(_hybrid, retryable=WEAVIATE_TRANSIENT_ERRORS)
        stage_timings["hybrid_search_ms"] = _ms(t)

        objects: Sequence[Object[Any, Any]] = result.objects
        retrieved_count = len(objects)
        pre_rerank_rank = {id(obj): idx for idx, obj in enumerate(objects)}

        if request.rerank is not None:
            # An explicit per-call value wins outright, same precedence as alpha - a
            # caller that asked for reranking gets it even on a query the heuristic
            # below would otherwise skip.
            should_rerank = request.rerank
        else:
            # No explicit per-call value: fall back to the caller's stored preference
            # (typically a merchant's `rerank_enabled`), then apply the query-shape
            # heuristic on top of *that* - never above it. `rerank_text()` never
            # includes the SKU (see its docstring), so a real product-code query gets
            # no benefit from reranking regardless of preference: it costs
            # `retrieve_top_k` cross-encoder passes to reshuffle results based on text
            # that was never meant to answer that kind of query. Deliberately checked
            # against the token-level signal, not `classification.query_class ==
            # IDENTIFIER`: that class also catches short natural-language queries like
            # "hiking boots" via its no-competing-signal fallback, and those genuinely
            # can benefit from reranking.
            default = (
                request.rerank_default
                if request.rerank_default is not None
                else self._rerank_enabled_default
            )
            should_rerank = default and not has_identifier_shaped_token(request.query)

        reranked = False
        top: list[tuple[Object[Any, Any], float | None]]
        if should_rerank and objects:
            t = time.perf_counter()
            documents = [rerank_text(obj.properties) for obj in objects]
            scores = await rerank_module.arerank(request.query, documents)
            ranked = sorted(
                zip(objects, scores, strict=True), key=lambda pair: pair[1], reverse=True
            )
            top = list(ranked[: request.limit])
            stage_timings["rerank_ms"] = _ms(t)
            reranked = True
        else:
            top = [(obj, None) for obj in objects[: request.limit]]

        hits = []
        for obj, rerank_score in top:
            base_score = obj.metadata.score if obj.metadata is not None else None
            final_score = rerank_score if rerank_score is not None else (base_score or 0.0)
            hit = hit_from_properties(
                obj.properties,
                score=float(final_score),
                rank_before_rerank=pre_rerank_rank[id(obj)],
            )
            if rerank_score is not None:
                hit = hit.model_copy(update={"rerank_score": float(rerank_score)})
            hits.append(hit)

        stage_timings["total_ms"] = _ms(t_start)
        observe_search(
            stage_timings, reranked=reranked, query_class=classification.query_class.value
        )
        return SearchResponse(
            hits=hits,
            query_class=classification.query_class,
            alpha_used=alpha,
            reranked=reranked,
            retrieved_count=retrieved_count,
            took_ms=stage_timings["total_ms"],
            stage_timings_ms=stage_timings,
            engine=self.name,
        )

    async def stats(
        self, tenant: str, filters: SearchFilters, metric: StatsMetric = "price"
    ) -> CatalogStats:
        """Count/min/max/mean over a whole matching set, via Weaviate's aggregate
        API - no embed, no BM25/vector scoring, no rerank. This is the only
        structurally correct way to answer a superlative ("highest priced item"),
        threshold ("anything above ₹10,000?"), or count ("how many X?") question:
        `search()`'s top-N relevance ranking can never honestly answer these - a
        real production bug (confirmed live: the model claimed nothing exceeded
        ₹2,499 when a real, in-stock item was ₹58,854) traced directly to a chat
        agent trying to infer a catalog-wide fact from 5 relevance-ranked results.

        `metric` picks which numeric property to aggregate - `price` by default,
        but `rating`/`review_count` use the identical mechanism, just a different
        property name, so "your highest rated product" costs nothing extra to
        support once this exists at all.
        """
        client = await get_shared_client()
        collection = product_collection(client, tenant)  # tenant-scoped automatically,
        weaviate_filter = _build_filters(filters)  # same as .query.hybrid()'s own filter

        async def _aggregate() -> Any:
            return await collection.aggregate.over_all(
                filters=weaviate_filter,
                total_count=True,
                return_metrics=Metrics(metric).number(
                    count=True, minimum=True, maximum=True, mean=True
                ),
            )

        result = await with_retry(_aggregate, retryable=WEAVIATE_TRANSIENT_ERRORS)
        agg = result.properties.get(metric)
        return CatalogStats(
            metric=metric,
            # Post-filter count, NOT the metric's own count - the metric's count
            # would silently undercount if any matching product has a null value
            # for that property, which is a real, not hypothetical, shape (not
            # every product has a rating).
            count=result.total_count or 0,
            minimum=_to_decimal(agg.minimum if agg else None),
            maximum=_to_decimal(agg.maximum if agg else None),
            mean=_to_decimal(agg.mean if agg else None),
        )

    async def get_by_skus(self, tenant: str, skus: list[str]) -> list[SearchHit]:
        """Exact lookup by one or more SKUs, bypassing embedding/ranking entirely
        - for a follow-up referencing SKU(s) already known from conversation
        history (a single-item follow-up, or a multi-item comparison resolved in
        one round instead of one round per item). `search()`'s relevance ranking
        has no obligation to resurface a specific known SKU a second time; this
        does an exact, deterministic property match instead. `skus` is expected
        to already be bounded to a small max by the caller (the chat tool schema)
        before this is ever called - not re-validated here."""
        if not skus:
            return []
        client = await get_shared_client()
        collection = product_collection(client, tenant)

        async def _fetch() -> Any:
            return await collection.query.fetch_objects(
                filters=Filter.by_property("sku").contains_any(skus),
                limit=len(skus),
                return_properties=RETURN_PROPERTIES,
            )

        result = await with_retry(_fetch, retryable=WEAVIATE_TRANSIENT_ERRORS)
        return [
            hit_from_properties(obj.properties, score=1.0, rank_before_rerank=0)
            for obj in result.objects
        ]

    async def search_sorted(
        self,
        tenant: str,
        query: str,
        filters: SearchFilters,
        sort_by: str,
        limit: int = 5,
    ) -> list[SearchHit]:
        """For a superlative/count question scoped by FREE TEXT, not a structured
        field ("cheapest waterproof jacket") - `stats()` can't reach this, it only
        aggregates `SearchFilters`' structured fields, and "waterproof jacket"
        isn't one of them. Fetches a wide relevance-matched candidate pool
        (`_SORT_POOL_SIZE`, not the usual handful) via the same `hybrid()`
        mechanism `search()` already uses, then sorts that real, larger,
        text-relevant pool by the requested field before slicing - not a guess
        from a handful of top-ranked items. Reranking is explicitly skipped:
        the pool's order is about to be discarded in favour of a price/rating
        sort, so paying the cross-encoder cost here would be pure waste.

        **Honest limitation, confirmed live, not theoretical**: this is a much
        better answer than the 5-item version that caused a real production bug,
        but it is still a sort over a *candidate pool*, not an exhaustive
        catalog scan - a live check against `demo-fashion-in` found real men's
        watches up to ₹23,395 in a 30-item pool for "men's watch", while the
        catalog's actual highest-priced men's item (a ₹58,854 MOVADO) has "Men"
        only in unindexed `attributes_json`, not in a filterable property, and
        didn't rank in that pool at all. Widening the pool (`_SORT_POOL_SIZE`)
        lowers the miss probability but can never guarantee completeness the
        way `stats()` does for an actual structured filter - callers (the chat
        prompt) must phrase a `search_sorted`-backed superlative claim as "best
        among strongly-matching candidates," never with `stats()`-level
        certainty. See `app/llm/prompting.py`'s tool description for where this
        distinction is actually enforced on the model."""
        key_field, reverse = {
            "price_asc": ("price", False),
            "price_desc": ("price", True),
            "rating_desc": ("rating", True),
        }[sort_by]

        pool = await self.search(
            SearchRequest(
                query=query, tenant=tenant, limit=_SORT_POOL_SIZE, filters=filters, rerank=False
            )
        )
        ranked = sorted(
            (h for h in pool.hits if getattr(h, key_field) is not None),
            key=lambda h: getattr(h, key_field),
            reverse=reverse,
        )
        return ranked[:limit]


_SORT_POOL_SIZE = 50


@lru_cache(maxsize=1)
def get_retriever() -> WeaviateHybridRetriever:
    """Process-wide retriever. Cheap to construct (just loads `tuned_alpha.json`
    once), cached so every request shares one `AlphaRouter` instance."""
    return WeaviateHybridRetriever()
