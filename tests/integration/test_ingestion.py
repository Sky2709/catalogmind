"""Ingestion, end to end: upload a feed through the real API, poll the job.

Runs against live Postgres, Weaviate and Mongo. Exercises exactly the path a real
merchant would use - multipart CSV upload, a `job_id`, polling status - rather than
calling `app.ingestion.pipeline` functions directly, so a break in the router or the
background-task wiring shows up here even if the pipeline's own unit tests stay green.
"""

from __future__ import annotations

import asyncio

import pytest
from httpx import AsyncClient

from app.mongo import raw_products_collection
from app.retrieval import weaviate_client as wv

pytestmark = pytest.mark.integration

CSV_HEADER = "sku,title,description,price,brand"


def _csv(rows: list[tuple[str, str, str, str, str]]) -> bytes:
    lines = [CSV_HEADER, *(",".join(r) for r in rows)]
    return "\n".join(lines).encode("utf-8")


def _files(rows: list[tuple[str, str, str, str, str]]) -> dict[str, tuple[str, bytes, str]]:
    return {"file": ("feed.csv", _csv(rows), "text/csv")}


async def _poll_job(
    client: AsyncClient,
    tenant: str,
    job_id: int,
    headers: dict[str, str],
    *,
    max_wait_seconds: float = 15.0,
) -> dict[str, object]:
    """Poll until the job leaves PENDING/RUNNING. With the in-process ASGI transport
    the background task has usually already finished by the time the POST returns, but
    polling keeps this test honest about what a real client actually does."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + max_wait_seconds
    while True:
        resp = await client.get(f"/v1/merchants/{tenant}/ingestion/{job_id}", headers=headers)
        assert resp.status_code == 200, resp.text
        job = resp.json()
        if job["status"] in ("succeeded", "failed", "partial"):
            return job
        if loop.time() > deadline:
            raise AssertionError(f"job {job_id} did not finish in time: {job}")
        await asyncio.sleep(0.2)


async def test_ingest_indexes_products_and_reports_progress(client, make_merchant) -> None:
    merchant = await make_merchant("ingest")
    rows = [
        ("SKU-1", "Cotton Shirt", "Soft cotton shirt", "999", "Acme"),
        ("SKU-2", "Leather Belt", "Genuine leather", "499", "Acme"),
    ]

    resp = await client.post(
        f"/v1/merchants/{merchant.tenant}/catalog:ingest",
        files=_files(rows),
        headers=merchant.headers,
    )
    assert resp.status_code == 202, resp.text
    assert resp.json()["rows_total"] == 2
    job_id = resp.json()["job_id"]

    job = await _poll_job(client, merchant.tenant, job_id, merchant.headers)
    assert job["status"] == "succeeded"
    assert job["rows_indexed"] == 2
    assert job["rows_failed"] == 0
    assert job["rows_skipped"] == 0

    async with wv.weaviate_client() as wclient:
        hashes = await wv.existing_content_hashes(wclient, merchant.tenant)
    assert set(hashes) == {"SKU-1", "SKU-2"}

    raw = await raw_products_collection(merchant.tenant).find_one({"_id": "SKU-1"})
    assert raw is not None
    assert raw["title"] == "Cotton Shirt"


async def test_reingesting_unchanged_feed_skips_reembedding(client, make_merchant) -> None:
    merchant = await make_merchant("ingest-delta")
    rows = [("SKU-1", "Widget", "desc", "100", "Acme")]

    first = await client.post(
        f"/v1/merchants/{merchant.tenant}/catalog:ingest",
        files=_files(rows),
        headers=merchant.headers,
    )
    job1 = await _poll_job(client, merchant.tenant, first.json()["job_id"], merchant.headers)
    assert job1["rows_indexed"] == 1

    second = await client.post(
        f"/v1/merchants/{merchant.tenant}/catalog:ingest",
        files=_files(rows),
        headers=merchant.headers,
    )
    job2 = await _poll_job(client, merchant.tenant, second.json()["job_id"], merchant.headers)
    assert job2["rows_indexed"] == 0
    assert job2["rows_skipped"] == 1


async def test_reingesting_changed_row_reembeds_only_that_row(client, make_merchant) -> None:
    merchant = await make_merchant("ingest-change")
    original = [
        ("SKU-1", "Widget", "desc", "100", "Acme"),
        ("SKU-2", "Gadget", "desc", "200", "Acme"),
    ]
    first = await client.post(
        f"/v1/merchants/{merchant.tenant}/catalog:ingest",
        files=_files(original),
        headers=merchant.headers,
    )
    await _poll_job(client, merchant.tenant, first.json()["job_id"], merchant.headers)

    updated = [
        ("SKU-1", "Widget v2", "desc", "150", "Acme"),  # changed
        ("SKU-2", "Gadget", "desc", "200", "Acme"),  # unchanged
    ]
    second = await client.post(
        f"/v1/merchants/{merchant.tenant}/catalog:ingest",
        files=_files(updated),
        headers=merchant.headers,
    )
    job2 = await _poll_job(client, merchant.tenant, second.json()["job_id"], merchant.headers)
    assert job2["rows_indexed"] == 1
    assert job2["rows_skipped"] == 1


async def test_bad_rows_are_reported_without_blocking_good_ones(client, make_merchant) -> None:
    merchant = await make_merchant("ingest-errors")
    body = f"{CSV_HEADER}\nSKU-1,Widget,desc,100,Acme\n,Missing SKU,desc,100,Acme\n"

    resp = await client.post(
        f"/v1/merchants/{merchant.tenant}/catalog:ingest",
        files={"file": ("feed.csv", body.encode(), "text/csv")},
        headers=merchant.headers,
    )
    job = await _poll_job(client, merchant.tenant, resp.json()["job_id"], merchant.headers)
    assert job["status"] == "partial"
    assert job["rows_indexed"] == 1
    assert job["rows_failed"] == 1
    assert job["errors"]["reasons"]
    assert job["errors"]["sample"]


async def test_ingestion_job_is_scoped_to_its_merchant(client, make_merchant) -> None:
    merchant_a = await make_merchant("ingest-a")
    merchant_b = await make_merchant("ingest-b")

    resp = await client.post(
        f"/v1/merchants/{merchant_a.tenant}/catalog:ingest",
        files=_files([("SKU-1", "Widget", "desc", "1", "Acme")]),
        headers=merchant_a.headers,
    )
    job_id = resp.json()["job_id"]

    leaked = await client.get(
        f"/v1/merchants/{merchant_b.tenant}/ingestion/{job_id}", headers=merchant_b.headers
    )
    assert leaked.status_code == 404


async def test_empty_upload_is_rejected(client, make_merchant) -> None:
    merchant = await make_merchant("ingest-empty")

    resp = await client.post(
        f"/v1/merchants/{merchant.tenant}/catalog:ingest",
        files={"file": ("feed.csv", b"", "text/csv")},
        headers=merchant.headers,
    )
    assert resp.status_code == 400


async def test_full_sync_deletes_products_absent_from_the_new_feed(client, make_merchant) -> None:
    merchant = await make_merchant("ingest-fullsync")
    original = [
        ("SKU-1", "Widget", "desc", "100", "Acme"),
        ("SKU-2", "Gadget", "desc", "200", "Acme"),
    ]
    first = await client.post(
        f"/v1/merchants/{merchant.tenant}/catalog:ingest",
        files=_files(original),
        headers=merchant.headers,
    )
    await _poll_job(client, merchant.tenant, first.json()["job_id"], merchant.headers)

    # The new feed is the merchant's complete catalog and no longer mentions SKU-2.
    updated = [("SKU-1", "Widget", "desc", "100", "Acme")]
    second = await client.post(
        f"/v1/merchants/{merchant.tenant}/catalog:ingest",
        files=_files(updated),
        data={"full_sync": "true"},
        headers=merchant.headers,
    )
    job2 = await _poll_job(client, merchant.tenant, second.json()["job_id"], merchant.headers)
    assert job2["rows_deleted"] == 1

    async with wv.weaviate_client() as wclient:
        hashes = await wv.existing_content_hashes(wclient, merchant.tenant)
    assert set(hashes) == {"SKU-1"}


async def test_without_full_sync_absent_products_are_left_alone(client, make_merchant) -> None:
    """The default (partial/delta upload) behaviour - the whole reason `full_sync`
    must be opt-in, not automatic."""
    merchant = await make_merchant("ingest-partial")
    original = [
        ("SKU-1", "Widget", "desc", "100", "Acme"),
        ("SKU-2", "Gadget", "desc", "200", "Acme"),
    ]
    first = await client.post(
        f"/v1/merchants/{merchant.tenant}/catalog:ingest",
        files=_files(original),
        headers=merchant.headers,
    )
    await _poll_job(client, merchant.tenant, first.json()["job_id"], merchant.headers)

    # A delta upload that only mentions SKU-1 - full_sync defaults to False.
    partial = [("SKU-1", "Widget v2", "desc", "150", "Acme")]
    second = await client.post(
        f"/v1/merchants/{merchant.tenant}/catalog:ingest",
        files=_files(partial),
        headers=merchant.headers,
    )
    job2 = await _poll_job(client, merchant.tenant, second.json()["job_id"], merchant.headers)
    assert job2["rows_deleted"] == 0

    async with wv.weaviate_client() as wclient:
        hashes = await wv.existing_content_hashes(wclient, merchant.tenant)
    assert set(hashes) == {"SKU-1", "SKU-2"}  # SKU-2 untouched despite not being re-sent
