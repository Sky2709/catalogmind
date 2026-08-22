"""Latency benchmark for the chat agent - cold vs semantic-cache-hit turns, and
concurrent-call behaviour, same rigor as `bench_search.py`.

**Not yet run against Bedrock** - a version of this ran once against Gemini before
the provider rollback (see `PROGRESS.md`'s Day 5 notes for those numbers, kept as
history, not carried over as if they still applied - a different provider is not
the same latency profile). Needs a real `AWS_BEARER_TOKEN_BEDROCK` in `.env`.
Committed now, rerunnable once one is added, same as every other benchmark here.

Deliberately smaller repetition counts than `bench_search.py`'s: every call here is
a real, paid Bedrock request, unlike `bench_search.py`'s free local CPU work - cost
discipline (`CLAUDE.md`) applies directly to how many times this script calls out.

The concurrency test uses **distinct queries per call**, never the same query
repeated - a repeated query would just measure the semantic cache's hit path, not
concurrent *cold* calls, and it exists to check a real open question: Days 2/3 found
local CPU-bound work (embedding, reranking) degrades badly under concurrency, but a
*network-bound* external API call might not share that failure mode. Measuring
settles it either way, rather than assuming the CPU-bound finding generalises.

Run: `.venv/bin/python -m scripts.bench_chat`
Requires `make up`, the three demo catalogs seeded (`make seed`), and a real
`AWS_BEARER_TOKEN_BEDROCK` in `.env`.
"""

from __future__ import annotations

import asyncio
import time
import uuid

from langchain_core.runnables import RunnableConfig
from sqlalchemy import select

from app.database import get_sessionmaker
from app.llm.client import dispose_bedrock_client
from app.llm.graph import ChatState, get_chat_graph
from app.models.db import Merchant
from app.redis_client import dispose_redis_client
from app.retrieval.weaviate_client import dispose_shared_client
from scripts.bench_utils import print_latency_table

CHAT_REPS = 5
CONCURRENCY_REPS_PER_LEVEL = 3

TENANT = "demo-fashion-in"

# Distinct enough per call that none of them collide in the semantic cache
# (`app/llm/semantic_cache.py`'s threshold is 0.95 - these are not near-duplicates).
COLD_QUERIES = [
    "do you have any DKNY shirts",
    "what waterproof jackets are available",
    "show me cotton shirts for men",
    "any wireless headphones in stock",
    "recommend a gift for a hiking trip",
]


_merchant_id_cache: int | None = None


async def _merchant_id() -> int:
    """Resolved once and cached - `ChatState.merchant_id` needs the real integer
    `Merchant.id` (`app/llm/cost_tracking.py`'s FK target), not the `TENANT` slug
    this script otherwise runs on."""
    global _merchant_id_cache
    if _merchant_id_cache is None:
        async with get_sessionmaker()() as session:
            _merchant_id_cache = (
                await session.execute(select(Merchant.id).where(Merchant.tenant == TENANT))
            ).scalar_one()
    return _merchant_id_cache


def _initial_state(tenant: str, merchant_id: int, conversation_id: str, message: str) -> ChatState:
    return {
        "tenant": tenant,
        "merchant_id": merchant_id,
        "conversation_id": conversation_id,
        "messages": [{"role": "user", "content": message}],
        "tool_call_rounds": 0,
        "model_used": None,
        "cache_hit": False,
        "final_answer": "",
        "citations": [],
        "force_no_tools": False,
    }


async def _run_turn(message: str) -> float:
    graph = get_chat_graph()
    conversation_id = str(uuid.uuid4())
    config: RunnableConfig = {"configurable": {"thread_id": conversation_id}}
    state = _initial_state(TENANT, await _merchant_id(), conversation_id, message)
    t0 = time.perf_counter()
    await graph.ainvoke(state, config=config)
    return time.perf_counter() - t0


async def bench_cold_vs_cached() -> None:
    print("=" * 70)
    print("COLD TURN vs SEMANTIC-CACHE-HIT TURN")

    cold_latencies = [await _run_turn(q) for q in COLD_QUERIES[:CHAT_REPS]]

    # Warm the cache once, then repeat the exact same question.
    warm_query = "do you have waterproof hiking boots"
    await _run_turn(warm_query)
    cached_latencies = [await _run_turn(warm_query) for _ in range(CHAT_REPS)]

    print_latency_table([("cold (real LLM call)", cold_latencies), ("cache hit", cached_latencies)])


async def bench_chat_concurrency() -> None:
    print("=" * 70)
    print("CONCURRENT FULL CHAT TURNS (embed + cache-check + Bedrock call)")

    for concurrency in (1, 2, 4):
        n_calls = concurrency * CONCURRENCY_REPS_PER_LEVEL
        queries = [COLD_QUERIES[i % len(COLD_QUERIES)] + f" (call {i})" for i in range(n_calls)]

        t0 = time.perf_counter()
        latencies: list[float] = []
        for start in range(0, len(queries), concurrency):
            batch = queries[start : start + concurrency]
            latencies.extend(await asyncio.gather(*(_run_turn(q) for q in batch)))
        elapsed = time.perf_counter() - t0

        print(
            f"  concurrency={concurrency}: {n_calls} calls in {elapsed:.2f}s, "
            f"{n_calls / elapsed:.2f} req/sec aggregate"
        )
        print_latency_table([(f"concurrency={concurrency}", latencies)])


async def main() -> None:
    await bench_cold_vs_cached()
    await bench_chat_concurrency()
    await dispose_shared_client()
    await dispose_redis_client()
    await dispose_bedrock_client()


if __name__ == "__main__":
    asyncio.run(main())
