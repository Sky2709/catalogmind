"""Prometheus metrics for the search hot path (`app.obs.metrics`) and the `/metrics`
scrape endpoint. No live stack needed - the registry is in-process."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.obs.metrics import (
    CHAT_CLAIM_MISMATCHES_TOTAL,
    CHAT_TOOL_CALLS_TOTAL,
    SEARCH_REQUESTS_TOTAL,
    SEARCH_STAGE_LATENCY_SECONDS,
    observe_chat_tool_call,
    observe_claim_mismatch,
    observe_search,
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _counter_value(counter, **labels) -> float:
    return counter.labels(**labels)._value.get()


def _histogram_sample_count(histogram, **labels) -> float:
    return histogram.labels(**labels)._sum.get()  # noqa: SLF001 - no public accessor exists


def test_observe_search_records_every_stage() -> None:
    before = _histogram_sample_count(SEARCH_STAGE_LATENCY_SECONDS, stage="embed_ms")
    observe_search(
        {"embed_ms": 12.0, "hybrid_search_ms": 34.0, "total_ms": 46.0},
        reranked=False,
        query_class="attribute",
    )
    after = _histogram_sample_count(SEARCH_STAGE_LATENCY_SECONDS, stage="embed_ms")
    assert after == pytest.approx(before + 0.012, abs=1e-9)


def test_observe_search_increments_the_request_counter() -> None:
    before = _counter_value(SEARCH_REQUESTS_TOTAL, reranked="True", query_class="identifier")
    observe_search({"total_ms": 5.0}, reranked=True, query_class="identifier")
    after = _counter_value(SEARCH_REQUESTS_TOTAL, reranked="True", query_class="identifier")
    assert after == before + 1


def test_metrics_endpoint_exposes_the_search_histogram(client: TestClient) -> None:
    observe_search({"total_ms": 7.0}, reranked=False, query_class="exploratory")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "catalogmind_search_stage_latency_seconds" in body
    assert "catalogmind_search_requests_total" in body


def test_metrics_endpoint_is_excluded_from_the_public_openapi_schema(
    client: TestClient,
) -> None:
    """An ops/scrape endpoint, not part of the merchant-facing API surface."""
    spec = client.get("/openapi.json").json()
    assert "/metrics" not in spec["paths"]


@pytest.mark.parametrize(
    "claim_type",
    ["hallucinated_citation", "stat_mismatch", "unverified_quantitative_refusal", "superlative_without_stats"],
)
def test_observe_claim_mismatch_increments_by_claim_type(claim_type: str) -> None:
    before = _counter_value(CHAT_CLAIM_MISMATCHES_TOTAL, claim_type=claim_type)
    observe_claim_mismatch(claim_type=claim_type, count=2)
    after = _counter_value(CHAT_CLAIM_MISMATCHES_TOTAL, claim_type=claim_type)
    assert after == before + 2


def test_observe_chat_tool_call_increments_by_tool_name() -> None:
    before = _counter_value(CHAT_TOOL_CALLS_TOTAL, tool="get_catalog_stats")
    observe_chat_tool_call(tool="get_catalog_stats")
    after = _counter_value(CHAT_TOOL_CALLS_TOTAL, tool="get_catalog_stats")
    assert after == before + 1


def test_metrics_endpoint_exposes_the_new_chat_counters(client: TestClient) -> None:
    observe_claim_mismatch(claim_type="stat_mismatch", count=1)
    observe_chat_tool_call(tool="search_catalog")
    body = client.get("/metrics").text
    assert "catalogmind_chat_claim_mismatches_total" in body
    assert "catalogmind_chat_tool_calls_total" in body
