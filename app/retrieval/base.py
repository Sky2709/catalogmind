"""The retrieval seam.

Every retrieval backend implements `Retriever`. Today there is exactly one
implementation (Weaviate hybrid). The interface exists anyway, for a specific reason:

    A commerce platform that grows by acquisition ends up running several product
    discovery engines at once — a site-search product here, a merchandising engine
    there, a native vector index alongside both. The question stops being "which
    engine is best" and becomes "how do I route per merchant, compare engines on the
    same golden set, and migrate a tenant without downtime".

Keeping retrieval behind a narrow protocol is what makes that tractable: an engine
becomes a swappable component, `eval/` can score any two implementations against the
same labelled queries, and a per-merchant config decides which one serves traffic.

The protocol is deliberately small. Anything an individual engine needs beyond this
(alpha, distance metric, index parameters) belongs in that engine's own config, not
in the shared surface.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class QueryClass(StrEnum):
    """How a shopper's query behaves under lexical vs semantic matching.

    This drives dynamic alpha selection. The classes are not about topic — they are
    about which retrieval signal is trustworthy for that shape of query.
    """

    EXPLORATORY = "exploratory"
    """Intent without vocabulary overlap: 'something for a beach wedding'.
    Vector search wins; BM25 has nothing to latch onto."""

    ATTRIBUTE = "attribute"
    """Constrained but natural: 'waterproof hiking boots size 10'.
    Both signals contribute."""

    IDENTIFIER = "identifier"
    """Exact tokens: 'DW-4402B', 'Air Max 90'.
    BM25 wins; embeddings blur model numbers into their nearest neighbours."""


class SearchFilters(BaseModel):
    """Structured pre-filters applied inside the engine, before scoring."""

    min_price: Decimal | None = None
    max_price: Decimal | None = None
    # Bounded (was unbounded until the chat agent's tool schema started exposing
    # these fields to LLM-constructed calls): an overgenerated list from a
    # prompt-injected shopper message would otherwise build an arbitrarily large
    # OR-filter server-side. 20 is generous for any real merchant question.
    brands: list[str] | None = Field(None, max_length=20)
    categories: list[str] | None = Field(None, max_length=20)
    in_stock_only: bool = False


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    tenant: str = Field(..., description="Merchant identifier; enforced by the engine")
    limit: int = Field(10, ge=1, le=100)

    alpha: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Hybrid blend. 0.0 = pure keyword/BM25, 1.0 = pure vector. "
            "When omitted, the dynamic alpha router picks a value from the query class."
        ),
    )
    rerank: bool | None = Field(
        None,
        description=(
            "Explicit per-call override. When set, wins outright - even over a query "
            "the engine's own heuristics would otherwise skip reranking for."
        ),
    )
    rerank_default: bool | None = Field(
        None,
        description=(
            "The caller's fallback preference when `rerank` is unset - typically a "
            "merchant's stored `rerank_enabled`. Sits *below* the engine's own "
            "query-shape heuristics (e.g. skipping a real product-code query, which "
            "a cross-encoder has no useful signal for regardless of preference), and "
            "above the engine's own global default when unset."
        ),
    )
    filters: SearchFilters = Field(default_factory=SearchFilters)


class SearchHit(BaseModel):
    sku: str
    title: str
    score: float

    brand: str | None = None
    price: Decimal | None = None
    currency: str | None = None
    in_stock: bool = True
    image_url: str | None = None
    category_path: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    rating: Decimal | None = None
    """`None` means no rating recorded for this product, not "not fetched" -
    added to `RETURN_PROPERTIES` unconditionally (cheap, one scalar field) so
    every hit carries it, not just the ones `search_sorted`'s `rating_desc`
    option cares about."""

    # --- provenance: how this hit was produced, for eval and debugging ---
    vector_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None
    rank_before_rerank: int | None = None


class SearchResponse(BaseModel):
    hits: list[SearchHit]

    # --- observability: everything needed to explain or reproduce this result ---
    query_class: QueryClass | None = None
    alpha_used: float | None = None
    reranked: bool = False
    retrieved_count: int = Field(0, description="Candidates fetched before reranking")
    took_ms: float | None = None
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
    engine: str | None = Field(None, description="Which Retriever served this")


StatsMetric = Literal["price", "rating", "review_count"]


class CatalogStats(BaseModel):
    """Count/min/max/mean over a whole matching set - the answer to a
    superlative/threshold/count question, which `SearchHit`-ranked top-N results
    can never honestly give (see `WeaviateHybridRetriever.stats()`'s docstring for
    the real production bug this exists to close). Deliberately not part of the
    `Retriever` Protocol below - aggregation is a Weaviate-specific capability, not
    a swappable-engine concern."""

    metric: StatsMetric
    count: int
    minimum: Decimal | None = None
    maximum: Decimal | None = None
    mean: Decimal | None = None


@runtime_checkable
class Retriever(Protocol):
    """A tenant-scoped product retrieval engine."""

    name: str

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Return ranked hits for one tenant.

        Implementations MUST scope results to `request.tenant`. Isolation is the
        engine's responsibility, not the caller's — a caller that forgets to filter
        must not be able to leak another merchant's catalog.
        """
        ...

    async def health(self) -> bool:
        """True when the engine can serve queries."""
        ...
