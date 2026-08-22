"""Settles the one open question left from Day 4: does BGE's asymmetric-encoding
query instruction (`"Represent this sentence for searching relevant passages: "`,
`app/ingestion/embed.py::QUERY_INSTRUCTION`) actually help on this project's own
catalogs, or was `embed_query_instruction: bool = True` just following the model
card?

The only prior signal (`app/config.py`'s docstring) was a 4-document smoke test that
proved nothing - both arms hit MRR 1.0 because the set was too small to discriminate.
This runs the real 170-query golden set instead, both arms otherwise identical:
dynamic alpha (the shipped router), reranking off throughout - same isolation
rationale as `eval/sweep_alpha.py`, so any score difference is attributable to the
query vector alone, not to a cross-encoder re-reading the text and masking it.

`Settings` is a plain mutable pydantic instance behind `get_settings()`'s
`lru_cache(maxsize=1)`, so flipping `embed_query_instruction` on the cached instance
between passes toggles `embed_query`'s default cleanly without touching any
production call site or needing a second process.

Run: `.venv/bin/python -m eval.sweep_query_instruction`
Requires `make up` and the three demo catalogs seeded (`make seed`).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.retrieval.base import SearchFilters, SearchRequest
from app.retrieval.hybrid import get_retriever
from app.retrieval.weaviate_client import dispose_shared_client
from eval.golden import GoldenQuery, load_golden_set
from eval.metrics import QueryScores, aggregate, score_query

TENANTS = ("demo-fashion-in", "demo-electronics-in", "demo-home-goods")
RESULTS_PATH = Path("eval/results/query_instruction_sweep.json")


async def _run_one(tenant: str, query: GoldenQuery) -> QueryScores:
    request = SearchRequest(
        query=query.query, tenant=tenant, limit=10, rerank=False, filters=SearchFilters()
    )
    response = await get_retriever().search(request)
    ranking = [hit.sku for hit in response.hits]
    return score_query(query.id, ranking, query.judgments)


async def _run_pass(*, use_instruction: bool) -> dict[str, Any]:
    get_settings().embed_query_instruction = use_instruction
    all_scores: list[QueryScores] = []
    for tenant in TENANTS:
        for query in load_golden_set(tenant):
            all_scores.append(await _run_one(tenant, query))
    return aggregate(all_scores)


def _write_results(with_instr: dict[str, Any], without: dict[str, Any], lift: float) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {"with_instruction": with_instr, "without_instruction": without, "lift": lift}, indent=2
        )
    )


async def main() -> None:
    with_instr = await _run_pass(use_instruction=True)
    without = await _run_pass(use_instruction=False)

    lift = with_instr["ndcg@10"] - without["ndcg@10"]
    print(
        f"WITH instruction:    nDCG@10={with_instr['ndcg@10']:.4f} "
        f"recall@10={with_instr['recall@10']:.4f} MRR={with_instr['mrr']:.4f}"
    )
    print(
        f"WITHOUT instruction: nDCG@10={without['ndcg@10']:.4f} "
        f"recall@10={without['recall@10']:.4f} MRR={without['mrr']:.4f}"
    )
    print(f"\nInstruction's nDCG@10 lift on the real 170-query golden set: {lift:+.4f}")

    _write_results(with_instr, without, lift)
    print(f"\nWrote {RESULTS_PATH}")

    await dispose_shared_client()


if __name__ == "__main__":
    asyncio.run(main())
