"""Merchant provisioning and API key management.

Provisioning spans two systems that cannot share a transaction: a Postgres row and a
Weaviate tenant. The ordering below is chosen so the only possible inconsistency is the
harmless one.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.deps import CurrentMerchant, DbSession, ScopedMerchant
from app.models.db import ApiKey, Merchant, display_prefix, generate_api_key, hash_api_key
from app.retrieval import weaviate_client as wv
from app.schemas import (
    ApiKeyCreate,
    ApiKeyCreated,
    ApiKeyOut,
    MerchantCreate,
    MerchantCreated,
    MerchantOut,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/merchants", tags=["merchants"])


async def require_admin(
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> None:
    """Operator-only guard for provisioning.

    Creating a merchant mints a credential and allocates a Weaviate shard. That is an
    operator action, not something an anonymous caller should be able to do in a loop.
    """
    if x_admin_token != get_settings().admin_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-Admin-Token.",
        )


@router.post(
    "",
    response_model=MerchantCreated,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_admin)],
    summary="Provision a merchant",
    responses={
        401: {"description": "Missing or invalid admin token"},
        409: {"description": "Tenant already exists"},
    },
)
async def create_merchant(payload: MerchantCreate, session: DbSession) -> MerchantCreated:
    """Create a merchant, its Weaviate tenant, and its first API key.

    **Ordering matters.** Postgres is written first but left uncommitted, then the
    Weaviate tenant is created, then the transaction commits:

    * Weaviate fails  -> Postgres rolls back. Nothing exists. Clean.
    * Commit fails    -> an unused Weaviate tenant is left behind. Harmless, and
                         reclaimed on the next attempt because tenant creation is
                         idempotent.

    The reverse order would allow a merchant row with no tenant behind it, which fails
    later at query time with a far more confusing error.
    """
    existing = await session.scalar(select(Merchant).where(Merchant.tenant == payload.tenant))
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Tenant {payload.tenant!r} already exists.",
        )

    merchant = Merchant(
        tenant=payload.tenant,
        name=payload.name,
        default_currency=payload.default_currency,
        rerank_enabled=payload.rerank_enabled,
        alpha_override=payload.alpha_override,
        column_mapping=payload.column_mapping,
    )
    session.add(merchant)

    raw_key = generate_api_key()
    session.add(
        ApiKey(
            merchant=merchant,
            key_hash=hash_api_key(raw_key),
            key_prefix=display_prefix(raw_key),
            label="initial",
        )
    )

    try:
        # Surfaces the unique/CHECK constraints now, while we can still turn them into
        # a clean HTTP error rather than a 500 from the final commit.
        await session.flush()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Tenant {payload.tenant!r} already exists.",
        ) from exc

    async with wv.weaviate_client() as client:
        await wv.ensure_schema(client)
        if not await wv.tenant_exists(client, payload.tenant):
            await wv.create_tenant(client, payload.tenant)

    logger.info("provisioned merchant tenant=%s", payload.tenant)
    return MerchantCreated(
        merchant=MerchantOut.model_validate(merchant),
        api_key=raw_key,
    )


@router.get(
    "/me",
    response_model=MerchantOut,
    summary="Who am I",
    responses={401: {"description": "Missing or invalid API key"}},
)
async def whoami(merchant: CurrentMerchant) -> MerchantOut:
    """Resolve the caller's API key to its merchant.

    Useful on its own, and the simplest possible demonstration that the key - not any
    request parameter - determines identity.
    """
    return MerchantOut.model_validate(merchant)


@router.get(
    "/{tenant}/keys",
    response_model=list[ApiKeyOut],
    summary="List this merchant's API keys",
)
async def list_keys(merchant: ScopedMerchant, session: DbSession) -> list[ApiKeyOut]:
    result = await session.execute(
        select(ApiKey).where(ApiKey.merchant_id == merchant.id).order_by(ApiKey.id)
    )
    return [ApiKeyOut.model_validate(k) for k in result.scalars()]


@router.post(
    "/{tenant}/keys",
    response_model=ApiKeyCreated,
    status_code=status.HTTP_201_CREATED,
    summary="Issue an additional API key (rotation)",
)
async def create_key(
    payload: ApiKeyCreate, merchant: ScopedMerchant, session: DbSession
) -> ApiKeyCreated:
    """Issue a second key so a credential can be rotated without downtime.

    Rotation is: issue new -> move callers across -> revoke old. That is only possible
    if more than one key can be live at once, which is why keys are a table rather than
    a column on the merchant.
    """
    raw_key = generate_api_key()
    key = ApiKey(
        merchant_id=merchant.id,
        key_hash=hash_api_key(raw_key),
        key_prefix=display_prefix(raw_key),
        label=payload.label,
    )
    session.add(key)
    await session.flush()

    logger.info("issued api key merchant=%s prefix=%s", merchant.tenant, key.key_prefix)
    return ApiKeyCreated(api_key=raw_key, key=ApiKeyOut.model_validate(key))


@router.delete(
    "/{tenant}/keys/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke an API key",
    responses={
        404: {"description": "No such key for this merchant"},
        409: {"description": "Cannot revoke the last active key"},
    },
)
async def revoke_key(key_id: int, merchant: ScopedMerchant, session: DbSession) -> Response:
    """Revoke a key.

    Scoped by `merchant_id` as well as `key_id`, so passing another merchant's key id
    returns 404 rather than revoking it. Guessing integer ids is trivial; this is the
    kind of endpoint IDOR bugs live in.
    """
    key = await session.scalar(
        select(ApiKey).where(ApiKey.id == key_id, ApiKey.merchant_id == merchant.id)
    )
    if key is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such key.")

    if key.is_active:
        active = await session.scalars(
            select(ApiKey).where(ApiKey.merchant_id == merchant.id, ApiKey.revoked_at.is_(None))
        )
        if len(list(active)) <= 1:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "Cannot revoke the only active key - it would lock this merchant "
                    "out permanently. Issue a replacement first."
                ),
            )
        key.revoked_at = datetime.now(UTC)
        logger.info("revoked api key merchant=%s prefix=%s", merchant.tenant, key.key_prefix)

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "",
    response_model=list[MerchantOut],
    dependencies=[Depends(require_admin)],
    summary="List all merchants (operator only)",
)
async def list_merchants(session: DbSession) -> list[MerchantOut]:
    result = await session.execute(
        select(Merchant).options(selectinload(Merchant.api_keys)).order_by(Merchant.tenant)
    )
    return [MerchantOut.model_validate(m) for m in result.scalars()]
