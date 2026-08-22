"""Async Motor client for raw feed storage.

Postgres owns merchant/job control state and Weaviate owns the searchable product
index; MongoDB's job is narrower than either: keep an unmodified copy of every row a
merchant's feed produced, one collection per tenant, so a bad ingestion can be
inspected or replayed without asking the merchant to re-upload their file.

Collection-per-tenant, not a shared collection with a tenant field, for the same
reason Weaviate uses native multi-tenancy and not a `merchant_id` filter: offboarding
a merchant should be "drop one collection", not "filter a shared one and hope nothing
was missed".
"""

from __future__ import annotations

from functools import lru_cache

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from app.config import get_settings


@lru_cache(maxsize=1)
def get_mongo_client() -> AsyncIOMotorClient:
    return AsyncIOMotorClient(get_settings().mongo_uri)


def raw_products_collection(tenant: str) -> AsyncIOMotorCollection:
    """The `raw_products_{tenant}` collection. Tenant is already a validated slug."""
    db = get_mongo_client()[get_settings().mongo_db]
    return db[f"raw_products_{tenant}"]


def dispose_mongo_client() -> None:
    """Close the client on shutdown. Motor is sync-close despite the async driver."""
    if get_mongo_client.cache_info().currsize:
        get_mongo_client().close()
        get_mongo_client.cache_clear()
