"""Verification helpers used while authoring `eval/golden/*.py`.

Every judgment in a golden query set has to trace back to a real product a live
query actually returned - these are the two lookups used to find and check that,
kept here (committed, rerunnable) rather than thrown away after the session that
first used them, same reasoning as `scripts/bench_ingestion.py`. Not a generator: no
function here invents a query or a judgment, they only answer "what does the real
catalog actually contain for this term/constraint" so a human (or an agent doing the
same job) can decide.

Run directly for a quick look while drafting a query:

    .venv/bin/python -m scripts.build_golden_sets demo-fashion-in bm25 "kurta"
    .venv/bin/python -m scripts.build_golden_sets demo-fashion-in hybrid "shaadi ke liye kurta" 0.5
"""

from __future__ import annotations

import asyncio
import sys

from app.ingestion.embed import embed_query
from app.retrieval import weaviate_client as wv

PREVIEW_PROPERTIES = ["sku", "brand", "title", "price", "attributes_text"]


async def bm25_preview(tenant: str, query: str, limit: int = 15) -> list[dict[str, object]]:
    """What pure keyword search finds for `query` - useful for checking a term
    exists in the catalog at all before building an attribute/identifier query
    around it (some plausible-sounding terms, e.g. "raincoat" in the fashion
    catalog, return nothing)."""
    async with wv.weaviate_client() as client:
        collection = wv.product_collection(client, tenant)
        result = await collection.query.bm25(
            query=query, limit=limit, return_properties=PREVIEW_PROPERTIES
        )
        return [dict(obj.properties) for obj in result.objects]


async def hybrid_preview(
    tenant: str, query: str, alpha: float, limit: int = 15
) -> list[dict[str, object]]:
    """What the real hybrid path returns for `query` at a given alpha - run this at
    the alpha `PRIOR_ALPHA`/`tuned_alpha.json` actually assigns the query's class, so
    the preview matches what `eval/retrieval_eval.py` will later score against."""
    vector = embed_query(query)
    async with wv.weaviate_client() as client:
        collection = wv.product_collection(client, tenant)
        result = await collection.query.hybrid(
            query=query,
            vector=vector,
            alpha=alpha,
            limit=limit,
            return_properties=PREVIEW_PROPERTIES,
        )
        return [dict(obj.properties) for obj in result.objects]


async def filter_preview(tenant: str, limit: int = 30, **equals: str) -> list[dict[str, object]]:
    """What a filter-only lookup finds, e.g. `filter_preview(tenant, brand="Geox")` -
    for building an identifier/attribute query around a real, exact structured value
    rather than a lexical guess."""
    from weaviate.classes.query import Filter

    clauses = [Filter.by_property(k).equal(v) for k, v in equals.items()]
    combined = clauses[0]
    for clause in clauses[1:]:
        combined = combined & clause

    async with wv.weaviate_client() as client:
        collection = wv.product_collection(client, tenant)
        result = await collection.query.fetch_objects(
            filters=combined, limit=limit, return_properties=PREVIEW_PROPERTIES
        )
        return [dict(obj.properties) for obj in result.objects]


def _print(rows: list[dict[str, object]]) -> None:
    if not rows:
        print("  (no results)")
        return
    for row in rows:
        print(
            f"  {row.get('sku')} | {row.get('brand')} | {row.get('title')} | "
            f"{row.get('price')} | {row.get('attributes_text')}"
        )


async def main() -> None:
    if len(sys.argv) < 4:
        print(__doc__)
        raise SystemExit(1)
    tenant, mode, query = sys.argv[1], sys.argv[2], sys.argv[3]

    if mode == "bm25":
        _print(await bm25_preview(tenant, query))
    elif mode == "hybrid":
        alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
        _print(await hybrid_preview(tenant, query, alpha))
    else:
        print(f"unknown mode {mode!r}: use 'bm25' or 'hybrid'")
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
