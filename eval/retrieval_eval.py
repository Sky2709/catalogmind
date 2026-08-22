"""Retrieval quality: run every golden query through the real retriever, score with
`eval/metrics.py`, report per-merchant/per-class aggregates and the whole-suite
number.

Three passes over the same 170 queries:

1. **Shipped config** - `retrieve_top_k=50` (the current default), `rerank=None`, i.e.
   exactly what a real search request gets today: dynamic alpha routing and the
   identifier-query rerank skip (`app/retrieval/hybrid.py`) both apply. Reported as
   "what production returns," not compared directly against rerank-off - see below.
2. **Rerank-off** - same queries, same dynamic alpha, `rerank=False`. `retrieve_top_k`
   is irrelevant to this pass: hybrid's own top-10 ordering doesn't change whether a
   shallower or deeper pool was fetched behind it, only reranking cares how deep the
   pool goes.
3. **Rerank at judgment depth** - `rerank=None`, but `retrieve_top_k` set **per query**
   to `len(query.judgments)`, not one blanket number. This, not pass 1, is what gets
   compared against rerank-off to state reranking's quality lift.

Why per-query, not a single constant: an earlier version of this script capped every
query at `retrieve_top_k=15` and still measured a large, suspicious negative lift.
Direct inspection (see `PROGRESS.md`'s Day 4 notes) showed why: many golden queries -
especially exploratory and Hinglish ones - were verified to a shallower depth than
15 during construction (often 4-8 items). A 15-candidate pool still let reranking
correctly promote a genuinely relevant item from rank 9-15 that nobody judged,
displacing verified-relevant items out of the scored top-10 - the exact same
judgment-pool-depth problem as the original `retrieve_top_k=50` comparison, just at
smaller scale. Matching the pool to each query's *own* judgment count removes the
confound entirely: reranking then only ever sees candidates this project actually
verified, so any score change reflects reranking's real effect on ordering, not on
finding things outside the pool. (One further wrinkle this fixed in passing: fashion's
identifier queries are bare numeric SKUs with no letters at all, which
`has_identifier_shaped_token()` didn't originally catch - see
`app/retrieval/alpha_router.py`'s `_IDENTIFIER_TOKEN` - so they were being reranked
and destroyed regardless of pool depth. Fixed at the source; this pass would have
kept measuring that failure otherwise.)

One honest limitation carried over from the golden sets themselves
(`eval/golden/__init__.py`): judgments are a verified top-K pool, not an exhaustive
catalog scan, so `recall@k` here is meaningful for comparing configurations against
each other on this fixed judged set - not a claim about absolute recall over the
whole catalog. Pass 3 exists precisely because that limitation interacts badly with a
candidate pool deeper than what was verified.

Run: `.venv/bin/python -m eval.retrieval_eval`
Requires `make up` and the three demo catalogs seeded (`make seed`). Passes 1 and 3
both rerank every non-identifier query, so budget real wall-clock time - the measured
cost is what makes this worth running, not something to work around.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.retrieval.base import SearchFilters, SearchRequest
from app.retrieval.hybrid import WeaviateHybridRetriever, get_retriever
from app.retrieval.weaviate_client import dispose_shared_client
from eval.golden import GoldenQuery, load_golden_set
from eval.metrics import QueryScores, aggregate, score_query

TENANTS = ("demo-fashion-in", "demo-electronics-in", "demo-home-goods")
RESULTS_PATH = Path("eval/results/retrieval_eval.json")

# One retriever per distinct judgment-pool size, built lazily and reused - judgment
# pool sizes cluster around a handful of values (1, 4-8, 15), so this is a handful of
# retrievers, not 170.
_retrievers_by_top_k: dict[int, WeaviateHybridRetriever] = {}


def _retriever_for_top_k(top_k: int) -> WeaviateHybridRetriever:
    if top_k not in _retrievers_by_top_k:
        _retrievers_by_top_k[top_k] = WeaviateHybridRetriever(retrieve_top_k=top_k)
    return _retrievers_by_top_k[top_k]


async def _run_one(
    retriever: WeaviateHybridRetriever, tenant: str, query: GoldenQuery, *, rerank: bool | None
) -> QueryScores:
    request = SearchRequest(
        query=query.query,
        tenant=tenant,
        limit=10,
        rerank=rerank,
        filters=SearchFilters(),
    )
    response = await retriever.search(request)
    ranking = [hit.sku for hit in response.hits]
    return score_query(query.id, ranking, query.judgments)


async def _run_pass(
    *, rerank: bool | None, match_pool_to_judgment_depth: bool
) -> dict[str, list[QueryScores]]:
    """All 170 queries, grouped by `tenant::class` - the report's row granularity.

    `match_pool_to_judgment_depth=True` picks a per-query retriever sized to that
    query's own judgment count (pass 3); `False` uses the process-wide singleton at
    its configured `retrieve_top_k` (passes 1 and 2).
    """
    grouped: dict[str, list[QueryScores]] = defaultdict(list)
    for tenant in TENANTS:
        for query in load_golden_set(tenant):
            retriever = (
                _retriever_for_top_k(max(len(query.judgments), 1))
                if match_pool_to_judgment_depth
                else get_retriever()
            )
            scores = await _run_one(retriever, tenant, query, rerank=rerank)
            grouped[f"{tenant}::{query.query_class.value}"].append(scores)
    return grouped


def _print_report(title: str, grouped: dict[str, list[QueryScores]]) -> dict[str, Any]:
    print(f"\n{title}")
    print(f"{'tenant :: class':<32} {'n':>4} {'nDCG@10':>8} {'recall@10':>10} {'MRR':>6}")
    all_scores: list[QueryScores] = []
    report: dict[str, Any] = {}
    for key in sorted(grouped):
        scores = grouped[key]
        all_scores.extend(scores)
        agg = aggregate(scores)
        report[key] = agg
        print(
            f"{key:<32} {agg['n_queries']:>4} {agg['ndcg@10']:>8.4f} "
            f"{agg['recall@10']:>10.4f} {agg['mrr']:>6.4f}"
        )
    overall = aggregate(all_scores)
    report["overall"] = overall
    print(
        f"{'OVERALL':<32} {overall['n_queries']:>4} {overall['ndcg@10']:>8.4f} "
        f"{overall['recall@10']:>10.4f} {overall['mrr']:>6.4f}"
    )
    return report


def _write_results(
    shipped_report: dict[str, Any],
    rerank_off_report: dict[str, Any],
    rerank_at_judgment_depth_report: dict[str, Any],
) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "shipped_config": shipped_report,
                "rerank_off": rerank_off_report,
                "rerank_at_judgment_depth": rerank_at_judgment_depth_report,
            },
            indent=2,
        )
    )


async def main() -> None:
    t0 = time.perf_counter()
    shipped = await _run_pass(rerank=None, match_pool_to_judgment_depth=False)
    shipped_report = _print_report(
        "SHIPPED CONFIG (retrieve_top_k=50, dynamic alpha, identifier-aware rerank) "
        "- what production returns today; NOT directly comparable to rerank-off, see module docstring",
        shipped,
    )
    print(f"  ({time.perf_counter() - t0:.1f}s)")

    t0 = time.perf_counter()
    rerank_off = await _run_pass(rerank=False, match_pool_to_judgment_depth=False)
    rerank_off_report = _print_report(
        "RERANK OFF (same dynamic alpha, no cross-encoder)", rerank_off
    )
    print(f"  ({time.perf_counter() - t0:.1f}s)")

    t0 = time.perf_counter()
    fair = await _run_pass(rerank=None, match_pool_to_judgment_depth=True)
    fair_report = _print_report(
        "RERANK AT JUDGMENT DEPTH (retrieve_top_k = each query's own judgment count) "
        "- the fair comparison",
        fair,
    )
    print(f"  ({time.perf_counter() - t0:.1f}s)")

    fair_lift = fair_report["overall"]["ndcg@10"] - rerank_off_report["overall"]["ndcg@10"]
    print(f"\nReranking's nDCG@10 lift, measured fairly at judgment depth: {fair_lift:+.4f}")
    print(
        "(cross-reference against scripts/bench_search.py's per-candidate-count "
        "latency - ~2.4s at k=10, ~8.8s at k=50 - to judge whether that lift is "
        "worth the cost at a given retrieve_top_k.)"
    )

    _write_results(shipped_report, rerank_off_report, fair_report)
    print(f"\nWrote {RESULTS_PATH}")

    await dispose_shared_client()


if __name__ == "__main__":
    asyncio.run(main())
