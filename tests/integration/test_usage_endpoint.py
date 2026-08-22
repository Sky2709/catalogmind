"""`GET /v1/merchants/{tenant}/usage` - Day 6's per-merchant cost ledger endpoint.

No real Bedrock call needed here: the ledger row is written directly via
SQLAlchemy, the same way `LlmUsage` rows are produced in production
(`app/llm/cost_tracking.py`) minus the actual LLM call - this file only needs to
prove the read path aggregates and isolates correctly, not that a real chat turn
produces a row (that's `test_chat.py`'s job, since it needs a real Bedrock call).
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.database import get_sessionmaker
from app.models.db import LlmUsage, Merchant

pytestmark = pytest.mark.integration


async def _insert_usage_row(tenant: str, model: str, cost_usd: str) -> None:
    async with get_sessionmaker()() as session:
        merchant_id = (
            await session.execute(select(Merchant.id).where(Merchant.tenant == tenant))
        ).scalar_one()
        session.add(
            LlmUsage(
                merchant_id=merchant_id,
                conversation_id="test-conversation",
                model=model,
                input_tokens=100,
                output_tokens=50,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                cost_usd=Decimal(cost_usd),
            )
        )
        await session.commit()


async def test_usage_cannot_cross_tenants(client: AsyncClient, make_merchant) -> None:
    """The headline invariant, at the usage endpoint specifically: merchant A must
    see zero cost/usage for a ledger row that only exists for merchant B."""
    a = await make_merchant("usage-iso-a")
    b = await make_merchant("usage-iso-b")
    await _insert_usage_row(b.tenant, "anthropic.claude-haiku-4-5", "0.00075000")

    response = await client.get(f"/v1/merchants/{a.tenant}/usage", headers=a.headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_calls"] == 0
    assert Decimal(body["total_cost_usd"]) == Decimal(0)
    assert body["by_model"] == []

    # Sanity: B really can see its own usage, so the empty result above is
    # isolation, not a broken endpoint.
    sanity = await client.get(f"/v1/merchants/{b.tenant}/usage", headers=b.headers)
    assert sanity.status_code == 200, sanity.text
    sanity_body = sanity.json()
    assert sanity_body["total_calls"] == 1
    assert Decimal(sanity_body["total_cost_usd"]) == Decimal("0.00075000")
    assert sanity_body["by_model"][0]["model"] == "anthropic.claude-haiku-4-5"


async def test_usage_aggregates_multiple_calls_by_model(client: AsyncClient, make_merchant) -> None:
    merchant = await make_merchant("usage-agg")
    await _insert_usage_row(merchant.tenant, "anthropic.claude-haiku-4-5", "0.00050000")
    await _insert_usage_row(merchant.tenant, "anthropic.claude-haiku-4-5", "0.00025000")
    await _insert_usage_row(merchant.tenant, "anthropic.claude-sonnet-5", "0.01000000")

    response = await client.get(f"/v1/merchants/{merchant.tenant}/usage", headers=merchant.headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_calls"] == 3
    assert Decimal(body["total_cost_usd"]) == Decimal("0.01075000")

    by_model = {row["model"]: row for row in body["by_model"]}
    assert by_model["anthropic.claude-haiku-4-5"]["calls"] == 2
    assert Decimal(by_model["anthropic.claude-haiku-4-5"]["cost_usd"]) == Decimal("0.00075000")
    assert by_model["anthropic.claude-sonnet-5"]["calls"] == 1
