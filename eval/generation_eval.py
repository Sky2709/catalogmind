"""Generation quality: run every golden chat scenario through the **real** chat
agent (real Bedrock calls, real retrieval underneath - `app/llm/graph.py`'s actual
compiled graph, not a re-implementation) and score with `eval/generation_metrics.py`.

Unlike `eval/retrieval_eval.py`, this costs real money per run - see
`eval/golden_chat/__init__.py`'s module docstring for the cost/scale reasoning.
Scaled to full parity with the 170-query retrieval golden set (2026-08-21): ~170
grounded scenarios (every positively-judged retrieval query, auto-derived) plus
~48 hand-picked cross-tenant refusal probes, ~218 total. Semantic caching is
deliberately disabled for the duration of this run
(`get_settings().semantic_cache_enabled = False`, restored after): a cache hit
would silently skip the real generation call this eval exists to measure, and
would keep replaying whatever answer happened to be cached from a previous run
instead of testing today's model/prompt.

Each scenario gets its own LangGraph thread id (`scenario.id`) so results can't
leak between scenarios via the checkpointer - `InMemorySaver` starts empty every
process anyway, but this keeps the intent explicit rather than relying on that.

Run: `.venv/bin/python -m eval.generation_eval`
Requires `make up`, the three demo catalogs seeded, and a real `AWS_BEARER_TOKEN_BEDROCK`
- skips with a clear message otherwise, the same discipline `test_chat.py` uses.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from sqlalchemy import select

from app.config import get_settings
from app.database import get_sessionmaker
from app.llm.graph import get_chat_graph
from app.models.db import Merchant
from eval.generation_metrics import ScenarioResult, aggregate_generation, score_scenario
from eval.golden_chat import ChatScenario, load_chat_scenarios

TENANTS = ("demo-fashion-in", "demo-electronics-in", "demo-home-goods")
RESULTS_PATH = Path("eval/results/generation_eval.json")

_merchant_ids: dict[str, int] = {}


async def _merchant_id(tenant: str) -> int:
    """Resolved once per tenant and cached - `ChatState.merchant_id` needs the
    real integer `Merchant.id` (`app/llm/cost_tracking.py`'s FK target), which
    this eval otherwise never looks up (it only ever passes the tenant slug)."""
    if tenant not in _merchant_ids:
        async with get_sessionmaker()() as session:
            _merchant_ids[tenant] = (
                await session.execute(select(Merchant.id).where(Merchant.tenant == tenant))
            ).scalar_one()
    return _merchant_ids[tenant]


async def _run_one(
    tenant: str, scenario: ChatScenario
) -> tuple[ScenarioResult, float, dict[str, Any]]:
    graph = get_chat_graph()
    config: RunnableConfig = {"configurable": {"thread_id": scenario.id}}
    turn_input = {
        "tenant": tenant,
        "merchant_id": await _merchant_id(tenant),
        "conversation_id": scenario.id,
        "messages": [{"role": "user", "content": scenario.message}],
        "tool_call_rounds": 0,
        "model_used": None,
        "cache_hit": False,
        "final_answer": "",
        "citations": [],
        "force_no_tools": False,
    }

    t0 = time.perf_counter()
    final_state = await graph.ainvoke(turn_input, config=config)
    latency_s = time.perf_counter() - t0

    result = score_scenario(
        scenario_id=scenario.id,
        kind=scenario.kind,
        expected_skus=scenario.expected_skus,
        answer=final_state["final_answer"],
        cited_products=final_state["citations"],
    )
    return result, latency_s, final_state


def _is_failure(scenario: ChatScenario, result: ScenarioResult) -> bool:
    if not result.grounded:
        return True
    if scenario.kind == "grounded":
        return not result.answer_correct
    return not result.refused


async def _run_all() -> tuple[
    dict[str, list[ScenarioResult]], dict[str, list[float]], list[dict[str, Any]]
]:
    grouped: dict[str, list[ScenarioResult]] = defaultdict(list)
    latencies: dict[str, list[float]] = defaultdict(list)
    # Raw answer text + retrieved products for any scenario that scored a failure -
    # a bare pass/fail table (like the console output below) can't explain *why* a
    # refusal check or a hallucination flag fired; the report needs to carry enough
    # for a human to judge that without re-running the eval at real cost again.
    failures: list[dict[str, Any]] = []
    for tenant in TENANTS:
        for scenario in load_chat_scenarios(tenant):
            result, latency_s, final_state = await _run_one(tenant, scenario)
            grouped[tenant].append(result)
            latencies[tenant].append(latency_s)
            print(
                f"  {scenario.id:<22} grounded={result.grounded!s:<5} "
                f"answer_correct={result.answer_correct!s:<5} "
                f"refused={result.refused!s:<5} ({latency_s:.1f}s)"
            )
            if _is_failure(scenario, result):
                failures.append(
                    {
                        "scenario_id": scenario.id,
                        "tenant": tenant,
                        "message": scenario.message,
                        "kind": scenario.kind,
                        "answer": final_state["final_answer"],
                        "cited_products": final_state["citations"],
                        "hallucinated_citations": list(result.hallucinated_citations),
                    }
                )
    return grouped, latencies, failures


def _print_and_build_report(
    grouped: dict[str, list[ScenarioResult]],
    latencies: dict[str, list[float]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    print(
        f"\n{'tenant':<22} {'n':>3} {'grounded':>9} {'answer_hit':>11} "
        f"{'refusal_ok':>11} {'p50_latency_s':>14}"
    )
    all_results: list[ScenarioResult] = []
    report: dict[str, Any] = {}
    for tenant in TENANTS:
        results = grouped[tenant]
        all_results.extend(results)
        agg = aggregate_generation(results)
        report[tenant] = agg
        sorted_latencies = sorted(latencies[tenant])
        p50 = sorted_latencies[len(sorted_latencies) // 2] if sorted_latencies else 0.0
        print(
            f"{tenant:<22} {agg['n_scenarios']:>3} "
            f"{_fmt_rate(agg['groundedness_rate']):>9} "
            f"{_fmt_rate(agg['answer_hit_rate']):>11} "
            f"{_fmt_rate(agg['refusal_correctness']):>11} "
            f"{p50:>14.1f}"
        )

    overall = aggregate_generation(all_results)
    report["overall"] = overall
    print(
        f"{'OVERALL':<22} {overall['n_scenarios']:>3} "
        f"{_fmt_rate(overall['groundedness_rate']):>9} "
        f"{_fmt_rate(overall['answer_hit_rate']):>11} "
        f"{_fmt_rate(overall['refusal_correctness']):>11}"
    )
    if overall["hallucinated_attribute_rate"] is not None:
        print(
            f"\nHallucinated-attribute (price) rate: "
            f"{overall['hallucinated_attribute_rate']:.4f} "
            f"(scored on {overall['hallucinated_attribute_rate_n']} of "
            f"{overall['n_scenarios']} scenarios - see module docstring for why "
            f"most are left unscored)"
        )
    if failures:
        print(f"\n{len(failures)} failing scenario(s) - full text below and in {RESULTS_PATH}:")
        for f in failures:
            print(f'\n  [{f["scenario_id"]}] ({f["kind"]}) "{f["message"]}"')
            print(f"  -> {f['answer']}")
            if f["hallucinated_citations"]:
                print(f"     hallucinated: {f['hallucinated_citations']}")
    report["failures"] = failures
    return report


def _fmt_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _write_results(report: dict[str, Any]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2))


async def main() -> None:
    settings = get_settings()
    if not settings.bedrock_api_key or settings.bedrock_api_key.startswith("bedrock-api-key-xxxx"):
        print(
            "AWS_BEARER_TOKEN_BEDROCK not set to a real key - skipping "
            "(see PROGRESS.md's Day 5 notes)."
        )
        sys.exit(0)

    original_cache_setting = settings.semantic_cache_enabled
    settings.semantic_cache_enabled = False
    try:
        grouped, latencies, failures = await _run_all()
    finally:
        settings.semantic_cache_enabled = original_cache_setting

    report = _print_and_build_report(grouped, latencies, failures)
    _write_results(report)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
