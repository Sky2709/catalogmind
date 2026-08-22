"""Cross-encoder reranking.

Hybrid search scores query and product independently, then blends the two scores
(`app/retrieval/alpha_router.py`). A cross-encoder scores the pair jointly - the query
and the candidate text are concatenated and run through one forward pass - which
catches interactions a blended bi-encoder score cannot, at the cost of being too slow
to run over a whole catalog. That is why it only ever sees the top-K hybrid candidates,
never the full index: `retrieve_top_k` candidates in, `rerank_top_k` survivors out
(`app/config.py`).

Runs `BAAI/bge-reranker-base` locally on CPU, same reasoning as `app/ingestion/embed.py`
for using local BGE models: the alpha sweep and any future rerank sweep re-score
hundreds of query/candidate pairs, and a per-call hosted reranker would make that either
slow or expensive.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

from app.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_model() -> CrossEncoder:
    """Load the cross-encoder once per process - see `embed.get_model` for why."""
    from sentence_transformers import CrossEncoder

    settings = get_settings()
    logger.info("loading reranker %s (cpu)", settings.reranker_model)
    return CrossEncoder(settings.reranker_model, device="cpu")


def rerank(query: str, documents: Sequence[str]) -> list[float]:
    """Relevance score for each (query, document) pair. Higher is more relevant.

    Scores are the cross-encoder's raw logits, not probabilities - meaningful for
    ranking documents against each other within one query, not for comparing across
    queries or against the hybrid score's own scale.
    """
    if not documents:
        return []
    scores = get_model().predict(
        [(query, document) for document in documents], convert_to_numpy=True
    )
    return [float(s) for s in scores]


RERANK_CONCURRENCY_LIMIT = 1
"""How many `rerank()` calls may run at once, in this one process. Measured, not
guessed: `scripts/bench_search.py` found that running the cross-encoder concurrently
on this dev machine is not just unhelpful but actively pathological. Aggregate
throughput fell monotonically as concurrency rose (0.19 -> 0.14 -> 0.09 -> 0.04
req/sec at concurrency 1/2/4/8 for a full embed+hybrid+rerank call), and per-call
latency did not just grow with queueing, it grew *faster* than queueing predicts -
concurrency=4's 35.9s p50 is close to what 4 serialized ~8.7s rerank calls would cost,
but concurrency=8's 193.5s p50 is roughly 3x what 8 serialized calls should cost, not
the ~70s a pure queue would produce. That gap is consistent with CPU cache/thread
contention inside torch (each `CrossEncoder.predict()` call already tries to use
multiple threads) rather than any useful parallel work happening. Serializing here
turns that pathological blowup back into plain, predictable queueing - still not
fast (rerank cost is still ~O(candidates), see `retrieve_top_k`), but no longer
actively worse than running requests one at a time. Same reasoning as
`INGESTION_CONCURRENCY_LIMIT` in `app/ingestion/pipeline.py`: a per-process semaphore
that does not help once this API runs as more than one worker process."""
_rerank_semaphore = asyncio.Semaphore(RERANK_CONCURRENCY_LIMIT)


async def arerank(query: str, documents: Sequence[str]) -> list[float]:
    async with _rerank_semaphore:
        return await asyncio.to_thread(rerank, query, documents)


def warm_up() -> None:
    """Force model load and one predict at startup - see `embed.warm_up` for why."""
    get_model()
    rerank("warm up", ["warm up document"])
