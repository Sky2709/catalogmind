"""Tenant isolation enforced at the HTTP boundary.

The companion to `test_tenant_isolation.py`, which proves it at the storage layer. This
file proves the API cannot be talked out of it: no key, a made-up key, a revoked key, or
a valid key pointed at somebody else's URL.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.integration


async def test_no_api_key_is_rejected(client: AsyncClient) -> None:
    response = await client.get("/v1/merchants/me")
    assert response.status_code == 401


async def test_unknown_api_key_is_rejected(client: AsyncClient) -> None:
    response = await client.get(
        "/v1/merchants/me", headers={"X-API-Key": "cm_live_totally-made-up"}
    )
    assert response.status_code == 401


async def test_api_key_identifies_its_own_merchant(client: AsyncClient, make_merchant) -> None:
    a = await make_merchant("iso-a")
    response = await client.get("/v1/merchants/me", headers=a.headers)
    assert response.status_code == 200
    assert response.json()["tenant"] == a.tenant


async def test_key_cannot_operate_on_another_tenant_via_the_url(
    client: AsyncClient, make_merchant
) -> None:
    """Editing the URL must not grant access.

    404 rather than 403 is deliberate: confirming "that merchant exists, you just may
    not see it" would let anyone holding one key enumerate the platform's customers.
    """
    a = await make_merchant("iso-a")
    b = await make_merchant("iso-b")

    denied = await client.get(f"/v1/merchants/{b.tenant}/keys", headers=a.headers)
    assert denied.status_code == 404
    assert b.tenant in denied.json()["detail"]

    # A's own tenant still works, so the 404 is about scope and not a broken route.
    allowed = await client.get(f"/v1/merchants/{a.tenant}/keys", headers=a.headers)
    assert allowed.status_code == 200


async def test_revoked_key_stops_working(client: AsyncClient, make_merchant) -> None:
    a = await make_merchant("iso-a")

    # Rotation needs a second key; revoking the only one is refused by design.
    issued = await client.post(
        f"/v1/merchants/{a.tenant}/keys", json={"label": "temp"}, headers=a.headers
    )
    assert issued.status_code == 201
    second = issued.json()

    revoke = await client.delete(
        f"/v1/merchants/{a.tenant}/keys/{second['key']['id']}", headers=a.headers
    )
    assert revoke.status_code == 204

    dead = await client.get("/v1/merchants/me", headers={"X-API-Key": second["api_key"]})
    assert dead.status_code == 401

    # The original key is unaffected - revocation is per key, not per merchant.
    alive = await client.get("/v1/merchants/me", headers=a.headers)
    assert alive.status_code == 200


async def test_cannot_revoke_the_last_active_key(client: AsyncClient, make_merchant) -> None:
    """Locking a merchant out permanently should take more than one careless call."""
    a = await make_merchant("iso-a")

    keys = (await client.get(f"/v1/merchants/{a.tenant}/keys", headers=a.headers)).json()
    assert len(keys) == 1

    refused = await client.delete(
        f"/v1/merchants/{a.tenant}/keys/{keys[0]['id']}", headers=a.headers
    )
    assert refused.status_code == 409

    still_working = await client.get("/v1/merchants/me", headers=a.headers)
    assert still_working.status_code == 200


async def test_cannot_revoke_another_merchants_key(client: AsyncClient, make_merchant) -> None:
    """Key ids are sequential integers, so this is the classic IDOR target."""
    a = await make_merchant("iso-a")
    b = await make_merchant("iso-b")

    b_keys = (await client.get(f"/v1/merchants/{b.tenant}/keys", headers=b.headers)).json()
    b_key_id = b_keys[0]["id"]

    # Attempted under A's own tenant path, so the path check passes and only the
    # merchant_id scoping on the query stands between this and a cross-tenant revoke.
    attempt = await client.delete(f"/v1/merchants/{a.tenant}/keys/{b_key_id}", headers=a.headers)
    assert attempt.status_code == 404

    survived = await client.get("/v1/merchants/me", headers=b.headers)
    assert survived.status_code == 200


async def test_provisioning_requires_the_admin_token(client: AsyncClient) -> None:
    response = await client.post("/v1/merchants", json={"tenant": "sneaky", "name": "Sneaky"})
    assert response.status_code == 401


async def test_provisioning_rejects_a_wrong_admin_token(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/merchants",
        json={"tenant": "sneaky", "name": "Sneaky"},
        headers={"X-Admin-Token": "not-the-token"},
    )
    assert response.status_code == 401


async def test_duplicate_tenant_is_rejected(
    client: AsyncClient, make_merchant, admin_headers
) -> None:
    a = await make_merchant("iso-a")
    duplicate = await client.post(
        "/v1/merchants",
        json={"tenant": a.tenant, "name": "Impostor"},
        headers=admin_headers,
    )
    assert duplicate.status_code == 409


async def test_invalid_tenant_slug_is_rejected(client: AsyncClient, admin_headers) -> None:
    """The Pydantic rule mirrors the Postgres CHECK, turning a 500 into a clean 422."""
    for bad in ("UPPERCASE", "has spaces", "-leading-dash", "x", "sym$bol"):
        response = await client.post(
            "/v1/merchants", json={"tenant": bad, "name": "x"}, headers=admin_headers
        )
        assert response.status_code == 422, f"{bad!r} should have been rejected"


async def test_api_key_is_returned_once_and_never_again(client: AsyncClient, make_merchant) -> None:
    """Only a hash is stored, so no endpoint can ever reveal the key again."""
    a = await make_merchant("iso-a")
    keys = (await client.get(f"/v1/merchants/{a.tenant}/keys", headers=a.headers)).json()

    assert a.api_key not in str(keys)
    # The prefix is enough to tell two keys apart in a UI, and useless as a credential.
    assert a.api_key.startswith(keys[0]["key_prefix"])
