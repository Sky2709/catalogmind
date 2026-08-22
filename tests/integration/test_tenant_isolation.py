"""Tenant isolation — the most important test in this project.

If this ever fails, one merchant can see another merchant's catalog, and nothing else
the system does matters. It is deliberately blunt: put a product in merchant B, then
try every way we have of reaching it as merchant A.

The isolation being proved here is *structural*. Weaviate native multi-tenancy gives
each merchant its own shard, so a query cannot cross tenants even if the calling code
forgets to filter. That is the whole argument for not using a `merchant_id` where-clause
instead: a forgotten filter would leak, whereas a forgotten tenant simply errors.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.retrieval import weaviate_client as wv
from app.retrieval.base import SearchFilters
from app.retrieval.hybrid import WeaviateHybridRetriever

pytestmark = pytest.mark.integration

# A SKU no other test would generate by accident.
SECRET_SKU = "B-ONLY-CONFIDENTIAL-9931"
SECRET_TITLE = "Confidential Merchant B Widget"

DIM = 384


def _vector(seed: float) -> list[float]:
    """A deterministic unit-ish vector. Content is irrelevant to isolation."""
    return [seed] * DIM


async def _insert(
    tenant: str, sku: str, title: str, *, price: Decimal | None = None
) -> None:
    async with wv.weaviate_client() as client:
        await wv.ensure_schema(client)
        collection = wv.product_collection(client, tenant)
        properties = {
            "sku": sku,
            "title": title,
            "description": title,
            "in_stock": True,
            **({"price": float(price)} if price is not None else {}),
        }
        await collection.data.insert(properties=properties, vector=_vector(0.1))


async def _all_skus(tenant: str) -> list[str]:
    async with wv.weaviate_client() as client:
        collection = wv.product_collection(client, tenant)
        result = await collection.query.fetch_objects(limit=100)
        return [o.properties["sku"] for o in result.objects]


async def _keyword_search(tenant: str, query: str) -> list[str]:
    async with wv.weaviate_client() as client:
        collection = wv.product_collection(client, tenant)
        result = await collection.query.bm25(query=query, limit=100)
        return [o.properties["sku"] for o in result.objects]


# --- the core proof ---------------------------------------------------------------


async def test_merchant_a_cannot_see_merchant_b_product(make_merchant) -> None:
    """The headline assertion: A's tenant returns ZERO results for B's SKU."""
    a = await make_merchant("iso-a")
    b = await make_merchant("iso-b")

    await _insert(b.tenant, SECRET_SKU, SECRET_TITLE)

    # Sanity: B really can see its own product. Without this, a test that isolates by
    # accidentally storing nothing at all would pass and prove nothing.
    assert SECRET_SKU in await _all_skus(b.tenant)

    assert await _all_skus(a.tenant) == [], "merchant A's tenant should be empty"
    assert SECRET_SKU not in await _all_skus(a.tenant)


async def test_keyword_search_cannot_cross_tenants(make_merchant) -> None:
    """Search, not just storage. BM25 for B's exact title finds nothing in A."""
    a = await make_merchant("iso-a")
    b = await make_merchant("iso-b")

    await _insert(b.tenant, SECRET_SKU, SECRET_TITLE)

    assert SECRET_SKU in await _keyword_search(b.tenant, "Confidential Widget")
    assert await _keyword_search(a.tenant, "Confidential Widget") == []


async def test_both_tenants_hold_their_own_data(make_merchant) -> None:
    """Isolation must not be achieved by simply losing data."""
    a = await make_merchant("iso-a")
    b = await make_merchant("iso-b")

    await _insert(a.tenant, "A-1", "Merchant A Alpha Product")
    await _insert(b.tenant, "B-1", "Merchant B Beta Product")

    assert await _all_skus(a.tenant) == ["A-1"]
    assert await _all_skus(b.tenant) == ["B-1"]


async def test_same_sku_in_two_tenants_stays_separate(make_merchant) -> None:
    """SKUs are unique *per merchant*, not globally.

    Two shops both selling "SHIRT-01" is completely normal, and must not collide.
    """
    a = await make_merchant("iso-a")
    b = await make_merchant("iso-b")

    await _insert(a.tenant, "SHIRT-01", "A's cotton shirt")
    await _insert(b.tenant, "SHIRT-01", "B's linen shirt")

    async with wv.weaviate_client() as client:
        a_objs = await wv.product_collection(client, a.tenant).query.fetch_objects(limit=10)
        b_objs = await wv.product_collection(client, b.tenant).query.fetch_objects(limit=10)

    assert [o.properties["title"] for o in a_objs.objects] == ["A's cotton shirt"]
    assert [o.properties["title"] for o in b_objs.objects] == ["B's linen shirt"]


async def test_deleting_a_merchant_removes_only_its_own_data(make_merchant) -> None:
    """Offboarding one merchant must not touch another's shard."""
    a = await make_merchant("iso-a")
    b = await make_merchant("iso-b")

    await _insert(a.tenant, "A-1", "A product")
    await _insert(b.tenant, "B-1", "B product")

    async with wv.weaviate_client() as client:
        await wv.delete_tenant(client, a.tenant)
        assert not await wv.tenant_exists(client, a.tenant)
        assert await wv.tenant_exists(client, b.tenant)

    assert await _all_skus(b.tenant) == ["B-1"]


# --- the new aggregate/exact-lookup read paths (2026-08-22) ------------------------
#
# `stats()` (Weaviate's aggregate API) and `get_by_skus()` (a plain filtered
# fetch) are genuinely new access patterns added for the chat agent's
# get_catalog_stats/get_product_detail tools - neither goes through
# `.query.hybrid()`/`.query.bm25()`, so the isolation proved above for search
# doesn't automatically cover them. Both route through `product_collection()`
# exactly like search does, so isolation is expected to hold for the same
# structural reason - proved directly rather than assumed.


async def test_get_catalog_stats_cannot_cross_tenants(make_merchant) -> None:
    """A price range that only matches merchant B's product must report
    count=0 for merchant A, not a leaked count - the aggregate-API equivalent
    of `test_keyword_search_cannot_cross_tenants` above."""
    a = await make_merchant("iso-a")
    b = await make_merchant("iso-b")
    await _insert(b.tenant, SECRET_SKU, SECRET_TITLE, price=Decimal("99999"))

    retriever = WeaviateHybridRetriever()
    filters = SearchFilters(min_price=Decimal("99998"), max_price=Decimal("100000"))

    b_stats = await retriever.stats(b.tenant, filters)
    assert b_stats.count == 1, "sanity: B really can see its own priced product"

    a_stats = await retriever.stats(a.tenant, filters)
    assert a_stats.count == 0, "merchant A must not see merchant B's product in an aggregate"
    assert a_stats.maximum is None


async def test_get_by_skus_cannot_cross_tenants(make_merchant) -> None:
    """An exact-SKU lookup for a SKU that only exists in merchant B's tenant
    must return empty for merchant A, not merchant B's product."""
    a = await make_merchant("iso-a")
    b = await make_merchant("iso-b")
    await _insert(b.tenant, SECRET_SKU, SECRET_TITLE)

    retriever = WeaviateHybridRetriever()

    b_hits = await retriever.get_by_skus(b.tenant, [SECRET_SKU])
    assert [h.sku for h in b_hits] == [SECRET_SKU], "sanity: B really can see its own product"

    a_hits = await retriever.get_by_skus(a.tenant, [SECRET_SKU])
    assert a_hits == [], "merchant A must not see merchant B's product by exact SKU lookup"
