"""Liveness, OpenAPI surface, and readiness behaviour.

Every outcome here is *injected*, never inferred from whether a stack happens to be
running. That distinction cost a failing test once already: the original version simply
asserted 503 and passed only because no datastores existed yet, then broke the moment
they came up. It had been testing the developer's machine, not the code.

Readiness against genuinely live datastores belongs in tests/integration.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_liveness_is_dependency_free(client: TestClient) -> None:
    """/health must answer even with every datastore down - it is a liveness probe.

    If this ever starts touching a dependency, Kubernetes will restart healthy pods
    during a database blip. That is the bug this test exists to prevent.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_openapi_documents_both_health_routes(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    assert "/health" in spec["paths"]
    assert "/health/ready" in spec["paths"]


def test_openapi_has_usable_metadata(client: TestClient) -> None:
    """The JD calls out OpenAPI explicitly; the generated spec is a deliverable."""
    spec = client.get("/openapi.json").json()
    assert spec["info"]["title"] == "CatalogMind"
    assert spec["info"]["description"].strip()
    tag_names = {tag["name"] for tag in spec.get("tags", [])}
    assert {"health", "merchants", "ingestion", "search", "chat"} <= tag_names


def test_readiness_always_reports_all_four_dependencies(client: TestClient) -> None:
    """Whatever the outcome, every dependency is named and individually attributed.

    On-call needs to see *which* one is down, not just that something is.
    """
    body = client.get("/health/ready").json()
    names = {dep["name"] for dep in body["dependencies"]}
    assert names == {"weaviate", "postgres", "mongo", "redis"}
    for dep in body["dependencies"]:
        assert dep["latency_ms"] is not None
        if not dep["ok"]:
            assert dep["detail"], f"{dep['name']} failed without explaining why"


def test_readiness_degrades_when_a_dependency_fails(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing dependency yields 503 and a named culprit - it never raises.

    The failure is injected rather than inferred from the environment. An earlier
    version of this test simply asserted 503 and passed only because no stack was
    running; the moment the stack came up it failed, having been a test of the
    developer's machine rather than of the code.
    """

    async def boom() -> None:
        raise ConnectionRefusedError("simulated outage")

    monkeypatch.setattr("app.routers.health._check_weaviate", boom)

    response = client.get("/health/ready")
    assert response.status_code == 503

    body = response.json()
    assert body["status"] == "degraded"

    weaviate = next(d for d in body["dependencies"] if d["name"] == "weaviate")
    assert weaviate["ok"] is False
    assert "simulated outage" in weaviate["detail"]


def test_readiness_reports_ready_when_all_checks_pass(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The positive path, also injected so it holds with or without a live stack."""

    async def fine() -> None:
        return None

    for name in ("_check_weaviate", "_check_postgres", "_check_mongo", "_check_redis"):
        monkeypatch.setattr(f"app.routers.health.{name}", fine)

    response = client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert all(dep["ok"] for dep in body["dependencies"])


def test_readiness_respects_its_timeout_ceiling(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hanging dependency must be cut off at the ceiling, not left to the driver.

    Injects a check that never returns - the whole point of the ceiling. Weaviate's
    client took 8s to give up on its own before this existed.
    """
    import asyncio

    from app.routers.health import CHECK_TIMEOUT_SECONDS

    async def hang() -> None:
        await asyncio.sleep(CHECK_TIMEOUT_SECONDS * 10)

    monkeypatch.setattr("app.routers.health._check_weaviate", hang)

    body = client.get("/health/ready").json()
    weaviate = next(d for d in body["dependencies"] if d["name"] == "weaviate")

    assert weaviate["ok"] is False
    assert "timed out" in weaviate["detail"]
    ceiling_ms = CHECK_TIMEOUT_SECONDS * 1000 * 1.5  # slack for slow CI runners
    assert weaviate["latency_ms"] <= ceiling_ms
