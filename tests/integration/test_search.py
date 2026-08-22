"""`POST /v1/merchants/{tenant}/search`, end to end against the real stack.

Ingests a small, deliberately varied catalog through the real upload endpoint (not
`weaviate_client` directly - a break in the search router or its wiring into the
retriever should show up here even if `hybrid.py`'s pure helpers stay green in the
unit suite), then exercises hybrid search, alpha override, rerank toggling, structured
filters, and cross-tenant isolation through the HTTP boundary.

The seeded catalog and the `catalog_merchant` fixture that ingests it now live in
`conftest.py` - `test_chat.py` needs the exact same ready-to-query tenant and there
is no reason to pay for a second ingestion round trip of the same rows.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from tests.integration.conftest import CATALOG, _ingest_catalog

pytestmark = pytest.mark.integration


def _skus(response_json: dict) -> list[str]:
    return [hit["sku"] for hit in response_json["hits"]]


async def test_identifier_query_finds_the_exact_sku(client: AsyncClient, catalog_merchant) -> None:
    """A raw SKU should surface top-1 via BM25 on the `sku` field.

    Reranking disabled deliberately: `rerank_text()` mirrors `Product.embedding_text()`
    and never includes the SKU, so a cross-encoder scoring "BOOT-WP-10" against title
    text alone has no reason to rank it first - this test is about the hybrid/BM25
    stage the identifier query class exists for, not the rerank stage.
    """
    response = await client.post(
        f"/v1/merchants/{catalog_merchant.tenant}/search",
        json={"query": "BOOT-WP-10", "rerank": False},
        headers=catalog_merchant.headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["query_class"] == "identifier"
    assert _skus(body)[0] == "BOOT-WP-10"


async def test_sku_shaped_query_skips_reranking_by_default(
    client: AsyncClient, catalog_merchant
) -> None:
    """The production-hardening fix: a real product-code query skips the (~seconds
    of cross-encoder) rerank stage automatically when the caller didn't explicitly
    ask for it either way - `rerank_text()` has no SKU to score against, so paying
    for reranking here would be pure cost with no chance of helping."""
    response = await client.post(
        f"/v1/merchants/{catalog_merchant.tenant}/search",
        json={"query": "BOOT-WP-10"},
        headers=catalog_merchant.headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["reranked"] is False
    assert all(hit["rerank_score"] is None for hit in body["hits"])
    assert "rerank_ms" not in body["stage_timings_ms"]


async def test_short_query_misclassified_as_identifier_still_reranks(
    client: AsyncClient, catalog_merchant
) -> None:
    """`classify("hiking boots")` lands in the IDENTIFIER class (nothing else fires
    for a short query), but there is no real product-code token in it, so it must
    still get reranked by default - pins the false positive `has_identifier_shaped_token`
    exists to avoid."""
    response = await client.post(
        f"/v1/merchants/{catalog_merchant.tenant}/search",
        json={"query": "hiking boots"},
        headers=catalog_merchant.headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["query_class"] == "identifier"
    assert body["reranked"] is True


async def test_exploratory_query_uses_vector_signal(client: AsyncClient, catalog_merchant) -> None:
    """No lexical overlap with the catalog at all - only the vector half of hybrid
    search can find this."""
    response = await client.post(
        f"/v1/merchants/{catalog_merchant.tenant}/search",
        json={"query": "something to keep my drink cold on a hike"},
        headers=catalog_merchant.headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["query_class"] == "exploratory"
    assert "BOTTLE-SS-500" in _skus(body)


async def test_alpha_override_forces_pure_keyword_search(
    client: AsyncClient, catalog_merchant
) -> None:
    """alpha=0.0 is pure BM25 - a query with real lexical overlap should still win
    even when phrased in a way the dynamic router would classify as exploratory."""
    response = await client.post(
        f"/v1/merchants/{catalog_merchant.tenant}/search",
        json={"query": "waterproof", "alpha": 0.0, "rerank": False},
        headers=catalog_merchant.headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["alpha_used"] == 0.0
    assert not body["reranked"]
    assert {"BOOT-WP-10", "JACKET-WP-L"}.issubset(set(_skus(body)))


async def test_in_stock_filter_excludes_out_of_stock_items(
    client: AsyncClient, catalog_merchant
) -> None:
    response = await client.post(
        f"/v1/merchants/{catalog_merchant.tenant}/search",
        json={
            "query": "wireless headphones",
            "filters": {"in_stock_only": True},
        },
        headers=catalog_merchant.headers,
    )
    assert response.status_code == 200, response.text
    assert "HEADPH-WL-BT" not in _skus(response.json())


async def test_price_filter_is_applied(client: AsyncClient, catalog_merchant) -> None:
    response = await client.post(
        f"/v1/merchants/{catalog_merchant.tenant}/search",
        json={"query": "acme", "filters": {"max_price": 20}},
        headers=catalog_merchant.headers,
    )
    assert response.status_code == 200, response.text
    hits = response.json()["hits"]
    assert hits, "expected at least one hit under $20"
    for hit in hits:
        assert float(hit["price"]) <= 20.0


async def test_brand_filter_is_applied(client: AsyncClient, catalog_merchant) -> None:
    response = await client.post(
        f"/v1/merchants/{catalog_merchant.tenant}/search",
        json={"query": "waterproof jacket boots", "filters": {"brands": ["Acme"]}},
        headers=catalog_merchant.headers,
    )
    assert response.status_code == 200, response.text
    hits = response.json()["hits"]
    assert all(hit["brand"] == "Acme" for hit in hits)
    assert "JACKET-WP-L" not in _skus(response.json())


async def test_rerank_can_be_disabled_per_request(client: AsyncClient, catalog_merchant) -> None:
    response = await client.post(
        f"/v1/merchants/{catalog_merchant.tenant}/search",
        json={"query": "cotton shirt", "rerank": False},
        headers=catalog_merchant.headers,
    )
    body = response.json()
    assert not body["reranked"]
    assert all(hit["rerank_score"] is None for hit in body["hits"])


async def test_reranking_is_on_by_default_and_scores_every_hit(
    client: AsyncClient, catalog_merchant
) -> None:
    response = await client.post(
        f"/v1/merchants/{catalog_merchant.tenant}/search",
        json={"query": "cotton shirt"},
        headers=catalog_merchant.headers,
    )
    body = response.json()
    assert body["reranked"]
    assert body["hits"], "expected at least one hit"
    assert all(hit["rerank_score"] is not None for hit in body["hits"])


async def test_limit_caps_the_number_of_hits(client: AsyncClient, catalog_merchant) -> None:
    response = await client.post(
        f"/v1/merchants/{catalog_merchant.tenant}/search",
        json={"query": "acme trailhead", "limit": 2},
        headers=catalog_merchant.headers,
    )
    assert len(response.json()["hits"]) <= 2


async def test_response_reports_stage_timings_and_engine(
    client: AsyncClient, catalog_merchant
) -> None:
    response = await client.post(
        f"/v1/merchants/{catalog_merchant.tenant}/search",
        json={"query": "hiking boots"},
        headers=catalog_merchant.headers,
    )
    body = response.json()
    assert body["engine"] == "weaviate-hybrid"
    assert body["retrieved_count"] >= 1
    for stage in ("embed_ms", "hybrid_search_ms", "total_ms"):
        assert stage in body["stage_timings_ms"]


async def test_search_cannot_cross_tenants(client: AsyncClient, make_merchant) -> None:
    """The headline invariant, at the search endpoint specifically: merchant A's key
    must return zero results for a product that only exists in merchant B."""
    a = await make_merchant("search-iso-a")
    b = await make_merchant("search-iso-b")
    await _ingest_catalog(client, b.tenant, b.headers, CATALOG[:1])

    response = await client.post(
        f"/v1/merchants/{a.tenant}/search",
        json={"query": "Trailhead Waterproof Hiking Boots", "alpha": 0.0, "rerank": False},
        headers=a.headers,
    )
    assert response.status_code == 200, response.text
    assert response.json()["hits"] == []

    # Sanity: B really can find its own product, so the empty result above is
    # isolation, not a broken search.
    sanity = await client.post(
        f"/v1/merchants/{b.tenant}/search",
        json={"query": "Trailhead Waterproof Hiking Boots", "alpha": 0.0, "rerank": False},
        headers=b.headers,
    )
    assert "BOOT-WP-10" in _skus(sanity.json())


async def test_search_requires_a_valid_api_key(client: AsyncClient, catalog_merchant) -> None:
    response = await client.post(
        f"/v1/merchants/{catalog_merchant.tenant}/search",
        json={"query": "boots"},
    )
    assert response.status_code == 401


async def test_search_key_cannot_target_another_tenant_via_the_url(
    client: AsyncClient, catalog_merchant, make_merchant
) -> None:
    other = await make_merchant("search-other")
    response = await client.post(
        f"/v1/merchants/{other.tenant}/search",
        json={"query": "boots"},
        headers=catalog_merchant.headers,
    )
    assert response.status_code == 404
