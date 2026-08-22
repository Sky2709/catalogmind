"""Rate limiting on the ingest endpoint, against the real Redis instance - not just
the pure fixed-window logic (see `tests/unit/test_rate_limit.py` for that), but the
actual dependency wired onto the actual route.
"""

from __future__ import annotations

import pytest

from app.routers.ingestion import INGEST_RATE_LIMIT

pytestmark = pytest.mark.integration

CSV_HEADER = "sku,title,description,price,brand"


def _files(sku: str) -> dict[str, tuple[str, bytes, str]]:
    body = f"{CSV_HEADER}\n{sku},Widget,desc,1,Acme\n".encode()
    return {"file": ("feed.csv", body, "text/csv")}


async def test_uploads_within_the_limit_all_succeed(client, make_merchant) -> None:
    merchant = await make_merchant("ratelimit-ok")
    for i in range(INGEST_RATE_LIMIT):
        resp = await client.post(
            f"/v1/merchants/{merchant.tenant}/catalog:ingest",
            files=_files(f"SKU-{i}"),
            headers=merchant.headers,
        )
        assert resp.status_code == 202, resp.text


async def test_upload_beyond_the_limit_is_rejected_with_retry_after(client, make_merchant) -> None:
    merchant = await make_merchant("ratelimit-exceeded")
    for i in range(INGEST_RATE_LIMIT):
        resp = await client.post(
            f"/v1/merchants/{merchant.tenant}/catalog:ingest",
            files=_files(f"SKU-{i}"),
            headers=merchant.headers,
        )
        assert resp.status_code == 202, resp.text

    over_limit = await client.post(
        f"/v1/merchants/{merchant.tenant}/catalog:ingest",
        files=_files("SKU-over-limit"),
        headers=merchant.headers,
    )
    assert over_limit.status_code == 429
    retry_after = int(over_limit.headers["Retry-After"])
    assert 0 < retry_after <= 60


async def test_rate_limit_is_scoped_per_tenant(client, make_merchant) -> None:
    """One merchant hitting its limit must never affect another merchant's quota -
    the whole point of keying the counter by tenant, not globally."""
    merchant_a = await make_merchant("ratelimit-a")
    merchant_b = await make_merchant("ratelimit-b")

    for i in range(INGEST_RATE_LIMIT):
        resp = await client.post(
            f"/v1/merchants/{merchant_a.tenant}/catalog:ingest",
            files=_files(f"SKU-{i}"),
            headers=merchant_a.headers,
        )
        assert resp.status_code == 202, resp.text

    exhausted = await client.post(
        f"/v1/merchants/{merchant_a.tenant}/catalog:ingest",
        files=_files("SKU-over-limit"),
        headers=merchant_a.headers,
    )
    assert exhausted.status_code == 429

    still_fresh = await client.post(
        f"/v1/merchants/{merchant_b.tenant}/catalog:ingest",
        files=_files("SKU-1"),
        headers=merchant_b.headers,
    )
    assert still_fresh.status_code == 202
