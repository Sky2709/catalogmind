"""Rigorous, rerunnable benchmark for the search hot path: what reranking actually
costs, and how the pipeline behaves under concurrent load.

Closes a gap flagged in review: Day 2's ingestion benchmarking found CPU embedding
does not scale past ~1-2 concurrent jobs on this dev machine (`bench_ingestion.py`),
and nobody had checked whether the cross-encoder reranker - also synchronous, also
CPU-bound, also offloaded to a thread pool - behaves the same way. This measures it
directly instead of assuming either answer.

Three things, in order:

1. **Rerank cost alone** - latency as a function of candidate count (10/25/50), the
   actual knobs `retrieve_top_k`/`rerank_top_k` operate over.
2. **What reranking costs the whole request** - hybrid-search-alone vs
   hybrid+rerank, same query, same tenant, at concurrency 1.
3. **Concurrency** - N simultaneous full search calls (embed + hybrid + rerank), each
   its own independent request against a real demo tenant, not N threads splitting
   one job - the same methodology correction Day 2's embedding concurrency test
   needed after an initial pass gave a falsely optimistic result.

Run: `.venv/bin/python -m scripts.bench_search`
Requires `make up` and the three demo catalogs seeded (`make seed`) - reads real
indexed products from `demo-fashion-in`, `demo-electronics-in`, `demo-home-goods`,
not synthetic stand-ins.
"""

from __future__ import annotations

import asyncio
import time

from app.ingestion.embed import aembed_query
from app.ingestion.embed import warm_up as warm_up_embedder
from app.retrieval import weaviate_client as wv
from app.retrieval.base import SearchFilters, SearchRequest
from app.retrieval.hybrid import RETURN_PROPERTIES, get_retriever, rerank_text
from app.retrieval.rerank import rerank
from app.retrieval.rerank import warm_up as warm_up_reranker
from scripts.bench_utils import percentiles, print_latency_table

RERANK_REPS = 10
CONCURRENCY_REPS_PER_LEVEL = 8

# One representative, catalog-grounded query per demo tenant - see
# `PROGRESS.md`'s embedding-quality findings for why these particular terms (real
# brands/products in each catalog, not generic category words that are known to
# underperform there).
QUERIES = [
    ("demo-fashion-in", "DKNY women's floral casual shirt"),
    ("demo-electronics-in", "Redmi smartphone with a good camera"),
    ("demo-home-goods", "plastic food storage container with lid"),
]


async def _fetch_rerank_candidates(tenant: str, query: str, limit: int) -> list[str]:
    """Real candidate texts for one query, via the real hybrid path - not synthetic
    strings, so the reranker sees realistic title/description/attribute lengths."""
    vector = await aembed_query(query)
    client = await wv.get_shared_client()
    collection = wv.product_collection(client, tenant)
    result = await collection.query.hybrid(
        query=query,
        vector=vector,
        alpha=0.5,
        limit=limit,
        return_properties=RETURN_PROPERTIES,
    )
    return [rerank_text(obj.properties) for obj in result.objects]


async def bench_rerank_cost() -> None:
    print("=" * 70)
    print("RERANK COST vs CANDIDATE COUNT")
    tenant, query = QUERIES[0]
    candidates = await _fetch_rerank_candidates(tenant, query, limit=50)
    print(f"  fetched {len(candidates)} real candidates from {tenant} for {query!r}")

    rows: list[tuple[str, list[float]]] = []
    for k in (10, 25, 50):
        docs = candidates[:k]
        if len(docs) < k:
            print(f"  skipping k={k}: only {len(docs)} candidates available")
            continue
        latencies = []
        for _ in range(RERANK_REPS):
            t0 = time.perf_counter()
            rerank(query, docs)
            latencies.append(time.perf_counter() - t0)
        rows.append((f"k={k}", latencies))
    print_latency_table(rows)


async def bench_rerank_lift_on_total_latency() -> None:
    """What reranking costs one request, holding the query/tenant fixed - the direct
    "quality vs latency" tradeoff number the project plan asks for."""
    print("=" * 70)
    print("HYBRID-ONLY vs HYBRID+RERANK, same query, concurrency=1")
    retriever = get_retriever()
    tenant, query = QUERIES[0]

    rows: list[tuple[str, list[float]]] = []
    for label, do_rerank in (("hybrid only", False), ("hybrid + rerank", True)):
        latencies = []
        for _ in range(RERANK_REPS):
            request = SearchRequest(
                query=query, tenant=tenant, limit=10, rerank=do_rerank, filters=SearchFilters()
            )
            response = await retriever.search(request)
            latencies.append((response.took_ms or 0.0) / 1000)
        rows.append((label, latencies))
    print_latency_table(rows)


async def bench_search_concurrency() -> None:
    """N simultaneous full search calls, each its own independent request against a
    real tenant - never N callers splitting one job (see module docstring).

    Runs in waves of exactly `concurrency` in-flight requests at a time
    (`_run_batched`), rather than handing the whole call list to one `asyncio.gather`
    - the latter would let the event loop run far more than `concurrency` requests
    at once and silently stop measuring what the label claims to measure.
    """
    print("=" * 70)
    print("CONCURRENT FULL SEARCH (embed + hybrid + rerank)")

    for concurrency in (1, 2, 4, 8):
        calls = [QUERIES[i % len(QUERIES)] for i in range(concurrency * CONCURRENCY_REPS_PER_LEVEL)]
        t0 = time.perf_counter()
        latencies = await _run_batched(calls, concurrency)
        elapsed = time.perf_counter() - t0
        p50, p95, p99 = percentiles(latencies)
        print(
            f"  concurrency={concurrency:>2}: {len(calls)} calls in {elapsed:.2f}s, "
            f"{len(calls) / elapsed:.2f} req/sec aggregate, "
            f"per-call p50={p50 * 1000:.0f}ms p95={p95 * 1000:.0f}ms p99={p99 * 1000:.0f}ms"
        )


async def _run_batched(calls: list[tuple[str, str]], concurrency: int) -> list[float]:
    """Runs `calls` in waves of exactly `concurrency` simultaneous requests - not
    `asyncio.gather` over the whole list, which would let the event loop schedule far
    more than `concurrency` requests in flight at once and stop measuring what the
    label claims to measure."""
    retriever = get_retriever()
    latencies: list[float] = []
    for start in range(0, len(calls), concurrency):
        batch = calls[start : start + concurrency]

        async def _one(tenant: str, query: str) -> float:
            request = SearchRequest(query=query, tenant=tenant, limit=10, filters=SearchFilters())
            response = await retriever.search(request)
            return (response.took_ms or 0.0) / 1000

        latencies.extend(await asyncio.gather(*(_one(t, q) for t, q in batch)))
    return latencies


async def main() -> None:
    print("warming embedding + reranker models...")
    await asyncio.gather(asyncio.to_thread(warm_up_embedder), asyncio.to_thread(warm_up_reranker))

    await bench_rerank_cost()
    await bench_rerank_lift_on_total_latency()
    await bench_search_concurrency()

    await wv.dispose_shared_client()


if __name__ == "__main__":
    asyncio.run(main())
