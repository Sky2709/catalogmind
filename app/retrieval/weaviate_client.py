"""Weaviate schema, connection, and tenant lifecycle.

**Isolation model.** One collection, `Product`, with Weaviate *native* multi-tenancy
enabled. Every merchant is a tenant, and each tenant gets its own shard with its own
HNSW graph and its own inverted index. A query issued through
`collection.with_tenant("acme")` physically cannot see another tenant's objects.

The alternative — one shared collection plus a `merchant_id` where-filter — is the
shortcut this project exists to reject. It is one forgotten filter away from leaking a
competitor's catalog, it makes per-merchant deletion a full-collection scan, and it
gives every tenant's vectors a single shared index whose recall degrades as unrelated
catalogs pile in. Native tenancy costs a little memory per active tenant and buys
isolation that is structural rather than disciplined.

`auto_tenant_creation` is deliberately **off**. Tenants are created explicitly during
merchant provisioning. If a typo'd tenant name could silently mint a new empty tenant,
"zero results" would become indistinguishable from "wrong merchant" — and the isolation
test would pass for the wrong reason.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import weaviate
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    Tokenization,
    VectorDistances,
)
from weaviate.classes.query import Filter
from weaviate.classes.tenants import Tenant, TenantActivityStatus
from weaviate.collections.classes.data import DataObject
from weaviate.exceptions import (
    WeaviateConnectionError,
    WeaviateGRPCUnavailableError,
    WeaviateTimeoutError,
)
from weaviate.util import generate_uuid5

from app.config import get_settings
from app.models.product import Product

if TYPE_CHECKING:
    from weaviate.client import WeaviateAsyncClient
    from weaviate.collections.classes.batch import BatchObjectReturn
    from weaviate.collections.collection.async_ import CollectionAsync

logger = logging.getLogger(__name__)

PRODUCT_COLLECTION = "Product"

# Shared by every caller that wraps a Weaviate call in `app.retry.with_retry`
# (ingestion's upsert/delete, search's hybrid query): only a dropped connection, a
# timeout, or an unavailable gRPC channel is worth retrying - a schema or validation
# error would just be retried into the same failure.
WEAVIATE_TRANSIENT_ERRORS = (
    WeaviateConnectionError,
    WeaviateGRPCUnavailableError,
    WeaviateTimeoutError,
)


# --- schema --------------------------------------------------------------------


def product_properties() -> list[Property]:
    """Schema for the Product collection.

    Tokenisation notes, because they decide whether the IDENTIFIER query class works:

    `sku` and `title` use WORD tokenisation, which lowercases and splits on
    non-alphanumerics. "DW-4402B" indexes as ["dw", "4402b"], and a shopper typing
    "DW-4402B" produces exactly those tokens, so BM25 lands an exact hit. FIELD
    tokenisation would store the whole string as one token — better for equality
    filters, worse for the keyword search we actually rely on at low alpha.

    `content_hash` is filterable but not searchable: it exists for delta detection
    during re-ingestion, and letting BM25 match hex digests would only add noise.
    """
    return [
        Property(
            name="sku",
            data_type=DataType.TEXT,
            tokenization=Tokenization.WORD,
            index_filterable=True,
            index_searchable=True,
            description="Merchant-scoped unique identifier",
        ),
        Property(
            name="title",
            data_type=DataType.TEXT,
            tokenization=Tokenization.WORD,
            index_searchable=True,
        ),
        Property(
            name="description",
            data_type=DataType.TEXT,
            tokenization=Tokenization.WORD,
            index_searchable=True,
        ),
        Property(
            name="brand",
            data_type=DataType.TEXT,
            tokenization=Tokenization.WORD,
            index_filterable=True,
            index_searchable=True,
        ),
        Property(
            name="category_path",
            data_type=DataType.TEXT_ARRAY,
            index_filterable=True,
            index_searchable=True,
        ),
        Property(name="price", data_type=DataType.NUMBER, index_filterable=True),
        Property(name="original_price", data_type=DataType.NUMBER, index_filterable=True),
        Property(
            name="currency",
            data_type=DataType.TEXT,
            tokenization=Tokenization.FIELD,
            index_filterable=True,
            index_searchable=False,
        ),
        Property(name="in_stock", data_type=DataType.BOOL, index_filterable=True),
        Property(name="rating", data_type=DataType.NUMBER, index_filterable=True),
        Property(name="review_count", data_type=DataType.INT, index_filterable=True),
        # Flattened "key: value" lines. Searchable so attribute queries ("waterproof",
        # "cotton") hit BM25 even when the term never appears in the title.
        Property(
            name="attributes_text",
            data_type=DataType.TEXT,
            tokenization=Tokenization.WORD,
            index_searchable=True,
        ),
        # Verbatim JSON, returned to the caller. Not indexed - it is payload, not signal.
        Property(
            name="attributes_json",
            data_type=DataType.TEXT,
            index_filterable=False,
            index_searchable=False,
        ),
        Property(
            name="image_url",
            data_type=DataType.TEXT,
            index_filterable=False,
            index_searchable=False,
        ),
        Property(
            name="product_url",
            data_type=DataType.TEXT,
            index_filterable=False,
            index_searchable=False,
        ),
        Property(
            name="content_hash",
            data_type=DataType.TEXT,
            tokenization=Tokenization.FIELD,
            index_filterable=True,
            index_searchable=False,
            description="Delta detection: unchanged hash means skip re-embedding",
        ),
        Property(name="updated_at", data_type=DataType.DATE, index_filterable=True),
    ]


async def ensure_schema(
    client: WeaviateAsyncClient,
    *,
    distance_metric: VectorDistances = VectorDistances.COSINE,
    bm25_b: float = 0.75,
    bm25_k1: float = 1.2,
) -> bool:
    """Create the Product collection if absent. Returns True if it was created.

    Idempotent, so `make up` and CI can call it unconditionally.

    Defaults, and why they are defaults rather than conclusions:

    * **COSINE** — BGE models are trained with cosine objectives and emit L2-normalised
      vectors, for which cosine and dot product rank identically. Cosine is the safe
      choice; `eval/` compares it against DOT and L2_SQUARED so the README can report a
      measured justification instead of this paragraph.
    * **bm25_b=0.75, k1=1.2** — the standard Robertson/Sparck-Jones defaults. Product
      titles are short and uniform in length, so b (length normalisation) is a real knob
      here; both are swept alongside alpha.
    """
    if await client.collections.exists(PRODUCT_COLLECTION):
        logger.info("collection %s already exists", PRODUCT_COLLECTION)
        return False

    await client.collections.create(
        name=PRODUCT_COLLECTION,
        description="Tenant-scoped merchant catalog items",
        properties=product_properties(),
        # We embed with a local BGE model, so Weaviate must never vectorise for us.
        vector_config=Configure.Vectors.self_provided(
            vector_index_config=Configure.VectorIndex.hnsw(distance_metric=distance_metric)
        ),
        inverted_index_config=Configure.inverted_index(bm25_b=bm25_b, bm25_k1=bm25_k1),
        multi_tenancy_config=Configure.multi_tenancy(
            enabled=True,
            # Explicit provisioning only - see module docstring.
            auto_tenant_creation=False,
            # Reads against an offloaded tenant transparently warm it back up.
            auto_tenant_activation=True,
        ),
    )
    logger.info(
        "created collection %s (distance=%s, bm25_b=%s, bm25_k1=%s)",
        PRODUCT_COLLECTION,
        distance_metric,
        bm25_b,
        bm25_k1,
    )
    return True


# --- connection ----------------------------------------------------------------


@asynccontextmanager
async def weaviate_client() -> AsyncIterator[WeaviateAsyncClient]:
    """Connected async client, closed on exit."""
    settings = get_settings()
    client = weaviate.use_async_with_local(
        host=settings.weaviate_host,
        port=settings.weaviate_port,
        grpc_port=settings.weaviate_grpc_port,
    )
    await client.connect()
    try:
        yield client
    finally:
        await client.close()


_shared_client: WeaviateAsyncClient | None = None
_shared_client_lock = asyncio.Lock()


async def get_shared_client() -> WeaviateAsyncClient:
    """Process-wide client for the search hot path.

    Ingestion and merchant provisioning use the short-lived `weaviate_client()`
    context manager above, because those are one-off operations - paying a fresh gRPC
    handshake per call is invisible next to an embedding batch or a schema change.
    Search is different: it runs on every shopper query, so a per-request connect
    would put connection setup latency on the one path this project is explicitly
    trying to keep fast (see "no LLM call on the retrieval hot path" in CLAUDE.md -
    the same discipline applies to connection overhead). Lazily created and cached for
    the process lifetime, same pattern as `get_redis_client`/`get_mongo_client`; the
    lock only guards the first concurrent request from opening two connections.
    """
    global _shared_client
    if _shared_client is not None:
        return _shared_client
    async with _shared_client_lock:
        if _shared_client is None:
            settings = get_settings()
            client = weaviate.use_async_with_local(
                host=settings.weaviate_host,
                port=settings.weaviate_port,
                grpc_port=settings.weaviate_grpc_port,
            )
            await client.connect()
            _shared_client = client
    return _shared_client


async def dispose_shared_client() -> None:
    """Close the shared client on shutdown. Safe to call even if never opened."""
    global _shared_client
    if _shared_client is not None:
        await _shared_client.close()
        _shared_client = None


def product_collection(client: WeaviateAsyncClient, tenant: str) -> CollectionAsync:
    """A handle scoped to exactly one tenant.

    This is the only sanctioned way to reach product data. Callers never receive an
    unscoped collection handle, so there is no code path in which forgetting a filter
    could return another merchant's catalog.
    """
    return client.collections.use(PRODUCT_COLLECTION).with_tenant(tenant)


# --- tenant lifecycle ------------------------------------------------------------


async def create_tenant(client: WeaviateAsyncClient, tenant: str) -> None:
    collection = client.collections.use(PRODUCT_COLLECTION)
    await collection.tenants.create(tenants=[Tenant(name=tenant)])
    logger.info("created tenant %s", tenant)


async def delete_tenant(client: WeaviateAsyncClient, tenant: str) -> None:
    """Remove a tenant and everything in it.

    Offboarding a merchant is one call against one shard — not a filtered mass-delete
    across a shared index. This is one of the concrete operational wins of native
    tenancy, and worth pointing at in DECISIONS.md.
    """
    collection = client.collections.use(PRODUCT_COLLECTION)
    await collection.tenants.remove([tenant])
    logger.info("deleted tenant %s", tenant)


async def list_tenants(client: WeaviateAsyncClient) -> dict[str, TenantActivityStatus]:
    collection = client.collections.use(PRODUCT_COLLECTION)
    tenants = await collection.tenants.get()
    return {name: t.activity_status for name, t in tenants.items()}


async def tenant_exists(client: WeaviateAsyncClient, tenant: str) -> bool:
    collection = client.collections.use(PRODUCT_COLLECTION)
    return await collection.tenants.exists(tenant)


async def set_tenant_state(
    client: WeaviateAsyncClient, tenant: str, status: TenantActivityStatus
) -> None:
    """Move a tenant between ACTIVE / INACTIVE / OFFLOADED.

    The cost story for a platform with a long tail of small merchants: ACTIVE keeps the
    HNSW graph resident in memory, INACTIVE drops it to local disk, OFFLOADED pushes the
    shard to object storage. A thousand merchants where fifty are busy should not pay to
    keep nine hundred and fifty graphs in RAM.

    OFFLOADED requires an offload module on the server (MinIO, via the compose
    `offload` profile).
    """
    collection = client.collections.use(PRODUCT_COLLECTION)
    await collection.tenants.update(tenants=[Tenant(name=tenant, activity_status=status)])
    logger.info("tenant %s -> %s", tenant, status)


# --- product I/O -----------------------------------------------------------------


# Object IDs are deterministic, derived from the tenant + SKU rather than left to
# Weaviate to assign. That makes re-ingestion naturally idempotent: inserting the same
# SKU twice overwrites the existing object (confirmed against a live instance - Weaviate's
# batch import upserts on a repeated ID) instead of creating a duplicate that would need
# a delete-then-insert dance.
def _object_id(tenant: str, sku: str) -> str:
    return generate_uuid5(sku, tenant)


def properties_from_product(product: Product) -> dict[str, Any]:
    """A `Product` as Weaviate properties, matching `product_properties()`."""
    attributes_text = "\n".join(
        f"{k}: {v}" for k, v in sorted(product.attributes.items()) if v is not None
    )
    return {
        "sku": product.sku,
        "title": product.title,
        "description": product.description,
        "brand": product.brand,
        "category_path": product.category_path,
        "price": float(product.price) if product.price is not None else None,
        "original_price": (
            float(product.original_price) if product.original_price is not None else None
        ),
        "currency": product.currency,
        "in_stock": product.in_stock,
        "rating": product.rating,
        "review_count": product.review_count,
        "attributes_text": attributes_text,
        "attributes_json": json.dumps(product.attributes, separators=(",", ":")),
        "image_url": product.image_url,
        "product_url": product.product_url,
        "content_hash": product.content_hash(),
        "updated_at": product.updated_at or datetime.now(UTC),
    }


async def existing_content_hashes(client: WeaviateAsyncClient, tenant: str) -> dict[str, str]:
    """Current `sku -> content_hash` for every object already indexed for a tenant.

    Fetched as one pass over the tenant rather than one lookup per SKU - the whole
    point of `content_hash` is to let a re-ingested feed skip re-embedding whatever it
    can prove is unchanged, and that only pays off if the comparison itself is cheap.
    """
    collection = product_collection(client, tenant)
    hashes: dict[str, str] = {}
    async for obj in collection.iterator(return_properties=["sku", "content_hash"]):
        sku = str(obj.properties["sku"])
        hashes[sku] = str(obj.properties.get("content_hash") or "")
    return hashes


async def upsert_products(
    client: WeaviateAsyncClient,
    tenant: str,
    products: Sequence[Product],
    vectors: Sequence[list[float]],
) -> BatchObjectReturn:
    """Batch-write products with their already-computed vectors. See `_object_id`."""
    collection = product_collection(client, tenant)
    objects = [
        DataObject(
            properties=properties_from_product(product),
            uuid=_object_id(tenant, product.sku),
            vector=vector,
        )
        for product, vector in zip(products, vectors, strict=True)
    ]
    return await collection.data.insert_many(objects)


async def delete_products_by_sku(
    client: WeaviateAsyncClient, tenant: str, skus: Sequence[str]
) -> int:
    """Bulk-delete by the same deterministic id `upsert_products` writes under - no
    query needed, since a SKU's id is a pure function of (tenant, sku). Used for
    `full_sync` re-ingestion: a SKU that vanished from a fresh feed is removed from
    the index, not just left stale forever."""
    if not skus:
        return 0
    collection = product_collection(client, tenant)
    ids = [_object_id(tenant, sku) for sku in skus]
    result = await collection.data.delete_many(where=Filter.by_id().contains_any(ids))
    return result.successful
