"""Measures whether `app/llm/model_router.py`'s escalation heuristic actually
earns its cost - the same "measured, not guessed" discipline `eval/sweep_alpha.py`
already applies to `alpha_router.py`, extended to the generation-side router for
the first time (named as a real, lower-priority gap in the marker-protocol
rework's plan file, `/home/akash/.claude/plans/moonlit-riding-hummingbird.md`).

**A real finding that shaped this script's design, not assumed**: checking
`classify_complexity()` against all 218 existing `eval/golden_chat` messages
(free, no LLM call) found **zero** that naturally escalate to the reasoning tier
at round 0 - every one is a short, templated single-item lookup
("I'm looking for X."), never comparison language, rarely 2+ "and/but/also"
joins, never 25+ tokens. The golden-chat set alone cannot test whether escalation
helps, because it never exercises the escalation path. Two groups instead:

- **BASELINE**: a small, deterministic sample of ordinary golden-chat scenarios
  (the router already leaves these on the fast tier) - checks whether the
  reasoning tier does meaningfully *better* on them anyway, which would mean the
  fast tier is under-serving ordinary lookups and the router should escalate
  more broadly than it does today.
- **ESCALATION_PROBES**: hand-authored messages that genuinely exercise each
  lexical signal (`_COMPARISON_CUE`, `_CONSTRAINT_JOIN`, message length),
  grounded in real SKUs from the live demo catalogs (not synthetic placeholders)
  - checks whether escalating for these reasons actually earns a better answer,
  or whether the fast tier would have done just as well for free.

Every scenario runs through the **real, compiled chat graph**
(`app/llm/graph.py::get_chat_graph`) twice, forced to each tier via
`ChatState.force_tier` (an eval-only override - production's `/chat` handler
never sets this key, see that field's docstring) - not a simplified
re-implementation, so tool-calling behaviour and citation evidence are real.

**Real money, deliberately kept small**: `len(BASELINE) + len(ESCALATION_PROBES)`
scenarios x 2 tiers = real Bedrock calls, roughly half of them the pricier
Sonnet ("reasoning") tier - scoped intentionally smaller than
`eval/generation_eval.py`'s 218-scenario single-tier run, per this project's
cost discipline. Confirm the sample size below before running for real money.

Run: `.venv/bin/python -m eval.measure_model_router`
Requires `make up`, the three demo catalogs seeded, and a real `AWS_BEARER_TOKEN_BEDROCK`.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from sqlalchemy import select

from app.config import get_settings
from app.database import get_sessionmaker
from app.llm.graph import get_chat_graph
from app.models.db import Merchant
from eval.generation_metrics import ScenarioResult, score_scenario
from eval.golden_chat import load_chat_scenarios

RESULTS_PATH = Path("eval/results/model_router_measurement.json")

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


Tier = Literal["fast", "reasoning"]


@dataclass(frozen=True)
class RouterProbe:
    id: str
    tenant: str
    message: str
    kind: Literal["grounded", "refusal"]
    expected_skus: frozenset[str]
    escalation_reason: str
    """Which real `model_router.py` signal this probe is meant to exercise -
    recorded for the report, not used by the run itself."""


# --- BASELINE: a small, deterministic slice of existing golden-chat scenarios,
# all of which the router already leaves on the fast tier (confirmed empirically -
# see module docstring). First 3 grounded + first 2 refusal per tenant, in the
# order `load_chat_scenarios` already returns - deterministic, not random, so a
# re-run samples the exact same messages.
def _baseline_probes() -> list[RouterProbe]:
    probes: list[RouterProbe] = []
    for tenant in ("demo-fashion-in", "demo-electronics-in", "demo-home-goods"):
        scenarios = load_chat_scenarios(tenant)
        grounded = [s for s in scenarios if s.kind == "grounded"][:3]
        refusal = [s for s in scenarios if s.kind == "refusal"][:2]
        for s in grounded + refusal:
            probes.append(
                RouterProbe(
                    id=s.id,
                    tenant=tenant,
                    message=s.message,
                    kind=s.kind,
                    expected_skus=s.expected_skus,
                    escalation_reason="none (baseline - router already picks fast)",
                )
            )
    return probes


# --- ESCALATION_PROBES: hand-authored, grounded in real SKUs pulled live from
# the demo catalogs (2026-08-22) via `get_retriever().search()` - not synthetic
# placeholders. `expected_skus` is whichever real product(s) the question is
# actually about, so `score_scenario` can check groundedness/answer-hit the same
# way it does for golden-chat scenarios.
ESCALATION_PROBES: list[RouterProbe] = [
    RouterProbe(
        id="router-probe-compare-01",
        tenant="demo-electronics-in",
        message=(
            "What's the difference between the talotech Neckband Earphones "
            "(SKU B09XHFKW4M) and the Wireless Bluetooth Headphones for vivo "
            "iQOO Z6 Pro (SKU B0BG33V6PD) - which one should I get?"
        ),
        kind="grounded",
        expected_skus=frozenset({"B09XHFKW4M", "B0BG33V6PD"}),
        escalation_reason="_COMPARISON_CUE (difference between / which one)",
    ),
    RouterProbe(
        id="router-probe-compare-02",
        tenant="demo-fashion-in",
        message=(
            "Compare the Geox Men Navy Blue Suede Boat Shoes (10030411) versus "
            "the Red Tape Men Tan Brown Leather Mid-Top Formal Boots (10063521) "
            "for everyday office wear."
        ),
        kind="grounded",
        expected_skus=frozenset({"10030411", "10063521"}),
        escalation_reason="_COMPARISON_CUE (compare / versus)",
    ),
    RouterProbe(
        id="router-probe-multiconstraint-01",
        tenant="demo-fashion-in",
        message=(
            "I need a formal shirt that's blue, and it needs to be a slim fit, "
            "and it also has to be under 600 rupees."
        ),
        kind="grounded",
        expected_skus=frozenset({"10030085", "10030279"}),
        escalation_reason="_CONSTRAINT_JOIN (and/also, 3+ joins)",
    ),
    RouterProbe(
        id="router-probe-multiconstraint-02",
        tenant="demo-home-goods",
        message=(
            "Show me kitchen storage that's plastic, and versatile, and also "
            "reasonably priced for organizing a small pantry."
        ),
        kind="grounded",
        expected_skus=frozenset({"shein-599e714a81e2"}),
        escalation_reason="_CONSTRAINT_JOIN (and/also, 3+ joins)",
    ),
    RouterProbe(
        id="router-probe-long-01",
        tenant="demo-electronics-in",
        message=(
            "I'm trying to set up a small home entertainment corner and I want "
            "something that streams well, works with a smart Android TV stick, "
            "and I'd also like to know roughly what a realme smart TV stick "
            "costs and whether it's a good starting point for someone who has "
            "never set up a streaming device before."
        ),
        kind="grounded",
        expected_skus=frozenset({"B09LQQYNZQ"}),
        escalation_reason="_LONG_MESSAGE_TOKENS (25+ tokens)",
    ),
]


async def _run_one(
    tenant: str, thread_id: str, message: str, tier: Tier
) -> tuple[dict[str, Any], float]:
    graph = get_chat_graph()
    conversation_id = f"{thread_id}-{tier}"
    config: RunnableConfig = {"configurable": {"thread_id": conversation_id}}
    turn_input = {
        "tenant": tenant,
        "merchant_id": await _merchant_id(tenant),
        "conversation_id": conversation_id,
        "messages": [{"role": "user", "content": message}],
        "tool_call_rounds": 0,
        "model_used": None,
        "cache_hit": False,
        "final_answer": "",
        "citations": [],
        "force_no_tools": False,
        "force_tier": tier,
    }
    t0 = time.perf_counter()
    final_state = await graph.ainvoke(turn_input, config=config)
    latency_s = time.perf_counter() - t0
    return final_state, latency_s


async def _run_probe(probe: RouterProbe) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": probe.id,
        "tenant": probe.tenant,
        "message": probe.message,
        "kind": probe.kind,
        "escalation_reason": probe.escalation_reason,
    }
    for tier in ("fast", "reasoning"):
        final_state, latency_s = await _run_one(probe.tenant, probe.id, probe.message, tier)
        result: ScenarioResult = score_scenario(
            scenario_id=probe.id,
            kind=probe.kind,
            expected_skus=probe.expected_skus,
            answer=final_state["final_answer"],
            cited_products=final_state["citations"],
        )
        row[tier] = {
            "grounded": result.grounded,
            "answer_correct": result.answer_correct,
            "refused": result.refused,
            "hallucinated_citations": list(result.hallucinated_citations),
            "latency_s": round(latency_s, 1),
            "answer": final_state["final_answer"],
        }
        print(
            f"  [{tier:>9}] {probe.id:<28} grounded={result.grounded!s:<5} "
            f"answer_correct={result.answer_correct!s:<5} refused={result.refused!s:<5} "
            f"({latency_s:.1f}s)"
        )
    # The actual question this script exists to answer: did forcing the
    # expensive tier change the outcome at all for this message?
    row["reasoning_changed_outcome"] = (
        row["fast"]["answer_correct"] != row["reasoning"]["answer_correct"]
        or row["fast"]["grounded"] != row["reasoning"]["grounded"]
        or row["fast"]["refused"] != row["reasoning"]["refused"]
    )
    return row


async def main() -> None:
    settings = get_settings()
    if not settings.bedrock_api_key or settings.bedrock_api_key.startswith("bedrock-api-key-xxxx"):
        print(
            "AWS_BEARER_TOKEN_BEDROCK not set to a real key - skipping "
            "(see PROGRESS.md's Day 5 notes)."
        )
        sys.exit(0)

    probes = _baseline_probes() + ESCALATION_PROBES
    print(
        f"Running {len(probes)} probes x 2 tiers = {len(probes) * 2} real Bedrock "
        f"calls through the actual chat graph...\n"
    )

    original_cache_setting = settings.semantic_cache_enabled
    settings.semantic_cache_enabled = False
    try:
        rows = [await _run_probe(p) for p in probes]
    finally:
        settings.semantic_cache_enabled = original_cache_setting

    changed = [r for r in rows if r["reasoning_changed_outcome"]]
    baseline_rows = [r for r in rows if r["escalation_reason"].startswith("none")]
    probe_rows = [r for r in rows if not r["escalation_reason"].startswith("none")]

    print(f"\n{'=' * 70}")
    print(f"Outcome changed by tier: {len(changed)}/{len(rows)}")
    if any(r["reasoning_changed_outcome"] for r in baseline_rows):
        print(
            "  -> at least one BASELINE message (router already picks fast) changed "
            "outcome under reasoning - possible under-escalation, worth a closer look."
        )
    else:
        print("  -> no BASELINE message changed outcome - fast tier looks sufficient for these.")
    escalation_helped = [r for r in probe_rows if r["reasoning_changed_outcome"]]
    print(
        f"  -> {len(escalation_helped)}/{len(probe_rows)} ESCALATION_PROBES changed outcome "
        f"under reasoning - {'escalating for these reasons earns something' if escalation_helped else 'no measured benefit from escalating on these signals in this sample'}."
    )
    for r in changed:
        print(f'\n  CHANGED: [{r["id"]}] "{r["message"][:80]}"')
        print(
            f"    fast:      correct={r['fast']['answer_correct']} grounded={r['fast']['grounded']} refused={r['fast']['refused']}"
        )
        print(
            f"    reasoning: correct={r['reasoning']['answer_correct']} grounded={r['reasoning']['grounded']} refused={r['reasoning']['refused']}"
        )

    _write_results(rows)
    print(f"\nWrote {RESULTS_PATH}")


def _write_results(rows: list[dict[str, Any]]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
