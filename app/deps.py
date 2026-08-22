"""Request-scoped dependencies. Authentication lives here and nowhere else.

**The isolation invariant:** the tenant a request operates on is derived from its API
key and from nothing else. No handler reads a merchant identifier out of a path, query
string or body and uses it to scope data.

Routes still *carry* `{tenant}` in the path, because `/v1/merchants/acme/search` is far
more readable in logs and Swagger than `/v1/search`. But that value is only ever
**compared** against the authenticated merchant — never used to look anything up. If
they disagree the request is refused. A caller cannot reach another merchant's catalog
by editing the URL, because the URL is not what selects the data.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import Depends, HTTPException, Path, status
from fastapi.security import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import session_scope
from app.models.db import ApiKey, Merchant, hash_api_key

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"

# auto_error=False so we can return our own message rather than FastAPI's terse 403.
_api_key_scheme = APIKeyHeader(
    name=API_KEY_HEADER,
    auto_error=False,
    description="Merchant API key, issued at `POST /v1/merchants`. Format: `cm_live_...`",
)

DbSession = Annotated[AsyncSession, Depends(session_scope)]

# Writing last_used_at on every request would mean a database write per read request.
# Throttling to once every few minutes keeps the field useful for spotting dormant keys
# without paying for the precision nobody needs.
LAST_USED_WRITE_INTERVAL = timedelta(minutes=5)

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail=f"Missing or invalid API key. Supply it in the {API_KEY_HEADER} header.",
    headers={"WWW-Authenticate": API_KEY_HEADER},
)


async def get_current_merchant(
    session: DbSession,
    api_key: Annotated[str | None, Depends(_api_key_scheme)] = None,
) -> Merchant:
    """Resolve the API key to its merchant, or refuse the request.

    A revoked key and an unknown key return the *same* error. Distinguishing them would
    tell an attacker which of their guesses were once real.
    """
    if not api_key:
        raise _UNAUTHORIZED

    # Look up by hash. The plaintext key is never stored, so there is nothing to compare
    # in Python and no timing side channel - this is an indexed equality lookup.
    result = await session.execute(
        select(ApiKey)
        .options(selectinload(ApiKey.merchant))
        .where(ApiKey.key_hash == hash_api_key(api_key))
    )
    record = result.scalar_one_or_none()

    if record is None or not record.is_active:
        logger.warning("rejected api key prefix=%s", api_key[:16])
        raise _UNAUTHORIZED

    now = datetime.now(UTC)
    if record.last_used_at is None or now - record.last_used_at > LAST_USED_WRITE_INTERVAL:
        record.last_used_at = now

    return record.merchant


CurrentMerchant = Annotated[Merchant, Depends(get_current_merchant)]


async def tenant_from_path(
    merchant: CurrentMerchant,
    tenant: Annotated[
        str,
        Path(description="Merchant tenant slug. Must match the authenticated API key."),
    ],
) -> Merchant:
    """For routes carrying `{tenant}`: confirm it matches the key, then ignore it.

    Returns the merchant resolved from the *key*, deliberately - so even a handler that
    misuses the return value cannot escape its own tenant.

    404, not 403, on mismatch. Confirming "that merchant exists, you just cannot see it"
    would leak the customer list of a multi-tenant platform to anyone holding one key.
    """
    if tenant != merchant.tenant:
        logger.warning("tenant path mismatch: key=%s requested=%s", merchant.tenant, tenant)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No merchant {tenant!r} is accessible with this API key.",
        )
    return merchant


ScopedMerchant = Annotated[Merchant, Depends(tenant_from_path)]
