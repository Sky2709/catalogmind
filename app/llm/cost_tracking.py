"""Per-merchant LLM cost ledger writes.

Called from inside `app/llm/graph.py`'s `agent` node, right after `response.usage`
becomes available - the same spot `observe_chat_tokens` (`app/obs/metrics.py`)
already reads it for the cross-tenant Prometheus counters. This module is the
per-tenant counterpart that counter's own docstring says belongs in Postgres, not
a metrics label.

Opens its own `AsyncSession` via `get_sessionmaker()` rather than
`Depends(session_scope)` - the exact precedent `app/ingestion/pipeline.py`'s
`run_ingestion_job` already sets for DB writes that happen outside a request's own
session lifetime. A LangGraph node runs inside the request's async context, but not
inside the FastAPI dependency that owns the request's session, so this is the same
situation, not a new pattern.

Awaited inline by the caller, not fire-and-forget: a billing ledger has a stronger
durability requirement than a cache warm or a metrics counter (both tolerate losing
one write; this shouldn't). The one indexed INSERT this adds costs single-digit
milliseconds next to a Bedrock call that already takes seconds - not a latency
concern worth trading away durability for.
"""

from __future__ import annotations

import logging

from anthropic.types import Usage

from app.database import get_sessionmaker
from app.llm.pricing import estimate_cost_usd
from app.models.db import LlmUsage

logger = logging.getLogger(__name__)


async def record_llm_usage(
    merchant_id: int, conversation_id: str, model: str, usage: Usage
) -> None:
    """Best-effort: logs and swallows any failure rather than raising.

    A ledger-write failure must not turn an otherwise-successful streamed chat
    answer into a 500 - by the time this runs, the answer has already reached the
    client (see `validate_and_store`'s docstring in `app/llm/graph.py` for the same
    reasoning applied to the citation checker)."""
    try:
        cost = estimate_cost_usd(
            model,
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_input_tokens or 0,
            usage.cache_creation_input_tokens or 0,
        )
        async with get_sessionmaker()() as session:
            session.add(
                LlmUsage(
                    merchant_id=merchant_id,
                    conversation_id=conversation_id,
                    model=model,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_read_tokens=usage.cache_read_input_tokens or 0,
                    cache_creation_tokens=usage.cache_creation_input_tokens or 0,
                    cost_usd=cost,
                )
            )
            await session.commit()
    except Exception:  # noqa: BLE001 - auxiliary ledger write, must never break the chat turn.
        logger.warning(
            "failed to record llm usage merchant_id=%s model=%s", merchant_id, model, exc_info=True
        )
