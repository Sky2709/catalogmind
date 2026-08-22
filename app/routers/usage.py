"""`GET /v1/merchants/{tenant}/usage` - a merchant's own LLM spend.

Self-service, like `/search` and `/chat`: a merchant reads its own cost ledger
through its own API key, the same `ScopedMerchant` dependency every other
merchant-scoped route uses. Backed by `LlmUsage` (`app/models/db.py`), written by
`app/llm/cost_tracking.py` from inside the chat graph.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Query
from sqlalchemy import func, select

from app.deps import DbSession, ScopedMerchant
from app.models.db import LlmUsage
from app.schemas import ModelUsageBreakdown, UsageSummary

router = APIRouter(prefix="/v1/merchants", tags=["usage"])


@router.get(
    "/{tenant}/usage",
    response_model=UsageSummary,
    summary="This merchant's LLM token usage and estimated cost",
)
async def get_usage(
    merchant: ScopedMerchant,
    session: DbSession,
    since: Annotated[
        datetime | None, Query(description="Only calls at or after this timestamp.")
    ] = None,
    until: Annotated[
        datetime | None, Query(description="Only calls strictly before this timestamp.")
    ] = None,
) -> UsageSummary:
    """Isolation here is only as strong as the `WHERE merchant_id = ...` filter
    below - unlike Weaviate's structurally per-tenant shards, Postgres has no
    row-level security in this codebase (`LlmUsage`'s own docstring). `merchant.id`
    comes from the authenticated key via `ScopedMerchant`, never from the `tenant`
    path segment, so this can't be tricked by editing the URL."""
    query = select(
        LlmUsage.model,
        func.count().label("calls"),
        func.sum(LlmUsage.input_tokens).label("input_tokens"),
        func.sum(LlmUsage.output_tokens).label("output_tokens"),
        func.sum(LlmUsage.cache_read_tokens).label("cache_read_tokens"),
        func.sum(LlmUsage.cache_creation_tokens).label("cache_creation_tokens"),
        func.sum(LlmUsage.cost_usd).label("cost_usd"),
    ).where(LlmUsage.merchant_id == merchant.id)
    if since is not None:
        query = query.where(LlmUsage.created_at >= since)
    if until is not None:
        query = query.where(LlmUsage.created_at < until)
    query = query.group_by(LlmUsage.model)

    rows = (await session.execute(query)).all()

    by_model = [
        ModelUsageBreakdown(
            model=row.model,
            calls=row.calls,
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            cache_read_tokens=row.cache_read_tokens,
            cache_creation_tokens=row.cache_creation_tokens,
            cost_usd=row.cost_usd,
        )
        for row in rows
    ]

    return UsageSummary(
        total_cost_usd=sum((m.cost_usd for m in by_model), Decimal(0)),
        total_calls=sum(m.calls for m in by_model),
        by_model=by_model,
    )
