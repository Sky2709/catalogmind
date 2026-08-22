"""Rate limiting on `/search`, against the real Redis instance - mirrors
`test_rate_limit.py`'s coverage of the ingest endpoint's limiter, applied to the
newer, much higher-traffic search limiter (`SEARCH_RATE_LIMIT`).

Uses `rerank=False` throughout so the loop is bounded by Redis round trips and a
cheap BM25-only query against an empty tenant, not N cross-encoder passes per call -
the limiter itself, not retrieval quality, is what these tests are about.
"""

from __future__ import annotations

import pytest

from app.routers.search import SEARCH_RATE_LIMIT

pytestmark = pytest.mark.integration


async def test_searches_within_the_limit_all_succeed(client, make_merchant) -> None:
    merchant = await make_merchant("searchlimit-ok")
    for _ in range(SEARCH_RATE_LIMIT):
        resp = await client.post(
            f"/v1/merchants/{merchant.tenant}/search",
            json={"query": "anything", "rerank": False},
            headers=merchant.headers,
        )
        assert resp.status_code == 200, resp.text


async def test_search_beyond_the_limit_is_rejected_with_retry_after(client, make_merchant) -> None:
    merchant = await make_merchant("searchlimit-exceeded")
    for _ in range(SEARCH_RATE_LIMIT):
        resp = await client.post(
            f"/v1/merchants/{merchant.tenant}/search",
            json={"query": "anything", "rerank": False},
            headers=merchant.headers,
        )
        assert resp.status_code == 200, resp.text

    over_limit = await client.post(
        f"/v1/merchants/{merchant.tenant}/search",
        json={"query": "one more", "rerank": False},
        headers=merchant.headers,
    )
    assert over_limit.status_code == 429
    retry_after = int(over_limit.headers["Retry-After"])
    assert 0 < retry_after <= 60


async def test_search_rate_limit_is_scoped_per_tenant(client, make_merchant) -> None:
    """One merchant exhausting its search quota must never throttle another's."""
    merchant_a = await make_merchant("searchlimit-a")
    merchant_b = await make_merchant("searchlimit-b")

    for _ in range(SEARCH_RATE_LIMIT):
        resp = await client.post(
            f"/v1/merchants/{merchant_a.tenant}/search",
            json={"query": "anything", "rerank": False},
            headers=merchant_a.headers,
        )
        assert resp.status_code == 200, resp.text

    exhausted = await client.post(
        f"/v1/merchants/{merchant_a.tenant}/search",
        json={"query": "one more", "rerank": False},
        headers=merchant_a.headers,
    )
    assert exhausted.status_code == 429

    still_fresh = await client.post(
        f"/v1/merchants/{merchant_b.tenant}/search",
        json={"query": "anything", "rerank": False},
        headers=merchant_b.headers,
    )
    assert still_fresh.status_code == 200
