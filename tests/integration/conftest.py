"""Fixtures for tests that need the real stack.

Everything here talks to live Postgres and live Weaviate. Run `make up` first;
`make test` excludes these, `make test-all` includes them.

**Everything is async, on one session-scoped event loop.** Starlette's `TestClient` is
deliberately not used: it drives the app on its own private loop, while the async
SQLAlchemy engine caches a connection pool bound to whichever loop touched it first.
Mixing the two produced "Event loop is closed" on every test after the first. An
`httpx.AsyncClient` over `ASGITransport` keeps the app, the fixtures and the tests on a
single loop, which removes the whole class of problem.

Each test provisions uniquely-named merchants and removes them from both systems
afterwards, so the suite is rerunnable without wiping the database.
"""

from __future__ import annotations

import asyncio
import csv
import io
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete

from app.config import get_settings
from app.database import get_sessionmaker
from app.main import app
from app.models.db import Merchant
from app.mongo import raw_products_collection
from app.retrieval import weaviate_client as wv

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class ProvisionedMerchant:
    """A merchant created for one test, with its plaintext key."""

    tenant: str
    name: str
    api_key: str

    @property
    def headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key}


@pytest.fixture(scope="session", autouse=True)
def _require_stack() -> None:
    """Fail loudly and immediately if the stack is not up.

    Otherwise every test fails with a different connection error and it takes a minute
    to realise the answer was `make up`.
    """
    settings = get_settings()
    try:
        response = httpx.get(f"{settings.weaviate_http_url}/v1/.well-known/ready", timeout=5)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        pytest.exit(
            f"Weaviate unreachable at {settings.weaviate_http_url} - run `make up`. ({exc})"
        )


@pytest_asyncio.fixture(scope="session")
async def client() -> AsyncIterator[AsyncClient]:
    """The app, driven in-process on the shared loop. No network, no separate server."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest.fixture(scope="session")
def admin_headers() -> dict[str, str]:
    return {"X-Admin-Token": get_settings().admin_token}


async def _purge(tenant: str) -> None:
    """Remove a tenant from both systems. Safe when it does not exist."""
    async with get_sessionmaker()() as session:
        await session.execute(delete(Merchant).where(Merchant.tenant == tenant))
        await session.commit()

    async with wv.weaviate_client() as wclient:
        collection_exists = await wclient.collections.exists(wv.PRODUCT_COLLECTION)
        if collection_exists and await wv.tenant_exists(wclient, tenant):
            await wv.delete_tenant(wclient, tenant)

    await raw_products_collection(tenant).drop()


MerchantFactory = Callable[..., Awaitable[ProvisionedMerchant]]


@pytest_asyncio.fixture
async def make_merchant(
    client: AsyncClient, admin_headers: dict[str, str]
) -> AsyncIterator[MerchantFactory]:
    """Provision merchants through the real API, and clean them up afterwards.

    Names are randomised per test so a crashed earlier run cannot collide with this one,
    and so tests may run in any order.
    """
    created: list[str] = []

    async def _make(prefix: str = "test", **overrides: object) -> ProvisionedMerchant:
        tenant = f"{prefix}-{uuid.uuid4().hex[:10]}"
        payload = {"tenant": tenant, "name": f"Test {prefix}", **overrides}
        response = await client.post("/v1/merchants", json=payload, headers=admin_headers)
        assert response.status_code == 201, response.text

        created.append(tenant)
        return ProvisionedMerchant(
            tenant=tenant, name=str(payload["name"]), api_key=response.json()["api_key"]
        )

    yield _make

    for tenant in created:
        await _purge(tenant)


# --- shared seeded catalog, for any test that just needs a ready-to-query tenant --
#
# Moved here from `test_search.py` (which still uses it) so `test_chat.py` can share
# the exact same fixture rather than re-ingesting a second copy of the same catalog -
# the ingestion round trip is the slow part of test setup, not worth paying twice.

CSV_HEADER = ("sku", "title", "description", "price", "brand", "in_stock", "image")

# Only `BOOT-WP-10` carries an image - deliberately, so tests can tell "this
# product's image_url survived the pipeline" apart from "every row happens to have
# one," and so the no-image rows exercise the same empty-string-> None path real
# messy feeds hit constantly.
BOOT_IMAGE_URL = "https://example.com/boot.jpg"

CATALOG = [
    (
        "BOOT-WP-10",
        "Trailhead Waterproof Hiking Boots",
        "Grippy sole, ankle support",
        "129.00",
        "Trailhead",
        "true",
        BOOT_IMAGE_URL,
    ),
    (
        "SHIRT-CTN-M",
        "Acme Cotton Casual Shirt",
        "Soft cotton, regular fit",
        "24.00",
        "Acme",
        "true",
        "",
    ),
    (
        "BOTTLE-SS-500",
        "Acme Stainless Steel Water Bottle 500ml",
        "Vacuum insulated",
        "18.00",
        "Acme",
        "true",
        "",
    ),
    (
        "HEADPH-WL-BT",
        "Acme Wireless Noise Cancelling Headphones",
        "Bluetooth, 30h battery",
        "89.00",
        "Acme",
        "false",
        "",
    ),
    (
        "JACKET-WP-L",
        "Trailhead Waterproof Rain Jacket",
        "Packable, sealed seams",
        "149.00",
        "Trailhead",
        "true",
        "",
    ),
]


def _csv(rows: list[tuple[str, ...]]) -> bytes:
    """Proper CSV quoting - several descriptions below contain literal commas, and a
    naive `",".join()` would silently misalign every column after the first one."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(CSV_HEADER)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


async def _ingest_catalog(
    client: AsyncClient, tenant: str, headers: dict[str, str], rows: list[tuple[str, ...]]
) -> None:
    resp = await client.post(
        f"/v1/merchants/{tenant}/catalog:ingest",
        files={"file": ("feed.csv", _csv(rows), "text/csv")},
        headers=headers,
    )
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]

    loop = asyncio.get_event_loop()
    deadline = loop.time() + 20.0
    while True:
        job = (
            await client.get(f"/v1/merchants/{tenant}/ingestion/{job_id}", headers=headers)
        ).json()
        if job["status"] in ("succeeded", "failed", "partial"):
            assert job["status"] == "succeeded", job
            return
        if loop.time() > deadline:
            raise AssertionError(f"ingestion job {job_id} did not finish in time: {job}")
        await asyncio.sleep(0.2)


@pytest.fixture
async def catalog_merchant(client: AsyncClient, make_merchant: MerchantFactory):
    # The default `ColumnMapping` doesn't read an `in_stock` column at all (feeds that
    # omit it are assumed all-sellable) - map it explicitly so the stock filter test
    # below actually has an out-of-stock row to exclude. `image_url` is mapped too so
    # `test_chat.py` can assert a real image survives ingestion -> search -> citation.
    merchant = await make_merchant(
        "search", column_mapping={"in_stock": "in_stock", "image_url": "image"}
    )
    await _ingest_catalog(client, merchant.tenant, merchant.headers, CATALOG)
    return merchant
