"""CI eval gate: fail a PR if search quality regresses.

`eval/retrieval_eval.py`'s real 170-query golden set is anchored to real Kaggle
catalogs in `data/raw/` (git-ignored, no scripted CI re-download - see
`data/SOURCES.md`), so it cannot run against a fresh GitHub Actions checkout.
This script runs the same production search path (`WeaviateHybridRetriever`,
dynamic alpha routing, the identifier-aware rerank skip - the exact "shipped
config" pass `eval/retrieval_eval.py` also measures) against a small, self-
contained fixture instead: `eval/ci_fixture/catalog.csv` (24 products, committed
to the repo) and `eval/ci_fixture/golden.py` (15 queries, every judgment grounded
in a real search call against the fixture once ingested - verified 2026-08-22,
not guessed from reading the catalog).

**What this can and can't catch.** A 15-query fixture is a smoke test, not a
statistically meaningful sample - it will not detect a few-percent regression the
way the real 170-query set could. What it reliably catches, based on this
project's own history (`PROGRESS.md`'s Day 4 notes): the kind of regression that's
actually happened here before is dramatic, not subtle - a broken identifier-skip
regex collapsed one query's nDCG@10 from 1.0 to 0.0158, a reranking bug collapsed
overall nDCG@10 from ~0.93 to ~0.52. Those blow up on *any* reasonable golden set,
including this one. Treat a pass here as "nothing is badly broken," not as a
substitute for the full `make eval` run before publishing README numbers.

Compares this run's overall (macro-averaged) nDCG@10/recall@10/MRR against
`eval/ci_fixture/baseline.json`, failing if any drops by more than
`REGRESSION_TOLERANCE`. The baseline only changes via an explicit
`.venv/bin/python -m eval.ci_quality_gate --write-baseline` run, reviewed and
committed like any other change - never regenerated silently by the gate itself,
same "measured, not guessed" discipline as `alpha_router.py`'s `PRIOR_ALPHA`/
`eval/results/tuned_alpha.json`.

Run: `.venv/bin/python -m eval.ci_quality_gate`
Requires `make up` (all four datastores - ingestion writes to Postgres, Mongo and
Weaviate) and `make migrate`. Provisions and re-ingests a fixed tenant
(`CI_TENANT`) fresh on every run for determinism, rather than reusing a
previous run's state.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

from sqlalchemy import delete, select

from app.database import dispose_engine, get_sessionmaker
from app.ingestion.pipeline import ingest
from app.models.db import ApiKey, Merchant, display_prefix, generate_api_key, hash_api_key
from app.mongo import dispose_mongo_client
from app.retrieval import weaviate_client as wv
from app.retrieval.base import SearchFilters, SearchRequest
from app.retrieval.hybrid import WeaviateHybridRetriever
from app.retrieval.weaviate_client import dispose_shared_client
from eval.ci_fixture.golden import QUERIES
from eval.metrics import aggregate, score_query

CI_TENANT = "ci-quality-gate"
CATALOG_PATH = Path(__file__).parent / "ci_fixture" / "catalog.csv"
BASELINE_PATH = Path(__file__).parent / "ci_fixture" / "baseline.json"

# Absolute drop allowed before a metric fails the gate. Generous relative to this
# project's own history of what a *real* regression looks like (a collapse of
# tens of percentage points, not a few) - see the module docstring - while still
# tight enough to catch one. This fixture has no LLM call and a deterministic
# reranker, so run-to-run noise on identical code should be at or near zero;
# this tolerance is headroom for legitimate, deliberate changes to
# retrieve_top_k/alpha/etc., not a hedge against flakiness.
REGRESSION_TOLERANCE = 0.03

TRACKED_METRICS = ("ndcg@10", "recall@10", "mrr")


async def _provision_fixture_tenant() -> None:
    """Delete-then-recreate, not idempotent-reuse: a small fixture is cheap to
    re-ingest every run, and starting fresh removes any risk of a previous run's
    partial state (a failed ingest, a stale product) silently changing what this
    run measures."""
    async with get_sessionmaker()() as session:
        await session.execute(delete(Merchant).where(Merchant.tenant == CI_TENANT))
        await session.commit()

        merchant = Merchant(tenant=CI_TENANT, name="CI quality gate", default_currency="USD")
        session.add(merchant)
        raw_key = generate_api_key()
        session.add(
            ApiKey(
                merchant=merchant,
                key_hash=hash_api_key(raw_key),
                key_prefix=display_prefix(raw_key),
                label="ci-quality-gate",
            )
        )
        await session.commit()

    async with wv.weaviate_client() as client:
        await wv.ensure_schema(client)
        if await wv.tenant_exists(client, CI_TENANT):
            await wv.delete_tenant(client, CI_TENANT)
        await wv.create_tenant(client, CI_TENANT)

    with CATALOG_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]

    async with get_sessionmaker()() as session:
        reloaded = await session.scalar(select(Merchant).where(Merchant.tenant == CI_TENANT))
        assert reloaded is not None, f"just-created merchant {CI_TENANT!r} vanished"
        outcome = await ingest(reloaded, rows)
        if outcome.parse.failed:
            raise SystemExit(
                f"fixture ingestion had {outcome.parse.failed} failures: "
                f"{outcome.parse.reasons} - the fixture catalog itself is broken, "
                "not a real quality regression"
            )


async def _run_queries() -> dict[str, float | None]:
    retriever = WeaviateHybridRetriever()
    scores = []
    for query in QUERIES:
        request = SearchRequest(
            query=query.query, tenant=CI_TENANT, limit=10, filters=SearchFilters()
        )
        response = await retriever.search(request)
        ranking = [hit.sku for hit in response.hits]
        scores.append(score_query(query.id, ranking, query.judgments))
    return aggregate(scores)


def _print_report(label: str, report: dict[str, float | None]) -> None:
    print(
        f"{label:<10} n={report['n_queries']:>3}  "
        f"nDCG@10={report['ndcg@10']:.4f}  "
        f"recall@10={report['recall@10']:.4f}  "
        f"MRR={report['mrr']:.4f}  "
        f"hit_rate@10={report['hit_rate@10']:.4f}"
    )


async def main(write_baseline: bool) -> int:
    await _provision_fixture_tenant()
    report = await _run_queries()
    _print_report("CURRENT", report)

    if write_baseline:
        BASELINE_PATH.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nWrote {BASELINE_PATH} - review the diff before committing it.")
        return 0

    if not BASELINE_PATH.exists():
        print(
            f"\nNo baseline at {BASELINE_PATH} yet - run with --write-baseline "
            "once to create it (review the numbers before committing)."
        )
        return 1

    baseline = json.loads(BASELINE_PATH.read_text())
    _print_report("BASELINE", baseline)

    failures = []
    for metric in TRACKED_METRICS:
        current = report.get(metric)
        expected = baseline.get(metric)
        if current is None or expected is None:
            failures.append(f"{metric}: missing value (current={current}, baseline={expected})")
            continue
        drop = expected - current
        if drop > REGRESSION_TOLERANCE:
            failures.append(
                f"{metric}: {current:.4f} vs baseline {expected:.4f} "
                f"(dropped {drop:.4f}, tolerance {REGRESSION_TOLERANCE})"
            )

    if failures:
        print("\nQUALITY GATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nQUALITY GATE PASSED.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Regenerate eval/ci_fixture/baseline.json from a fresh run instead of checking it.",
    )
    args = parser.parse_args()

    async def _run() -> int:
        try:
            return await main(args.write_baseline)
        finally:
            await dispose_shared_client()
            await dispose_engine()
            dispose_mongo_client()

    sys.exit(asyncio.run(_run()))
