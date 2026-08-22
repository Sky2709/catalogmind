"""Golden chat scenarios: shopper messages for measuring **generation** quality
(groundedness, hallucinated-attribute rate, refusal correctness) - a different
question from `eval/golden/`'s retrieval quality (nDCG/recall/MRR).

**Scaled to full parity with the 170-query retrieval golden set** (2026-08-21,
after the first 12-scenario cut proved the harness works but was too thin a
sample to call "well established" - see `PROGRESS.md`'s Day 6 notes). Two kinds
of scenario, built two different ways:

- **`grounded`**: derived automatically from every query in `eval/golden/` that
  has a positive judgment (`_grounded_scenarios` below) - all 170, across all
  three tenants, zero new hand-labelling. This is deliberate reuse, not
  laziness: the *retrieval* correctness of these anchors is already proven live
  (`eval/retrieval_eval.py`); what this eval checks is a different, independent
  question - whether the generated answer actually *names* the right SKU, not
  merely whether it was retrieved. A golden query's text is turned into a chat
  message by a small mechanical template (`_as_chat_message`), not
  hand-phrased per query - accepted as a known simplification (some read a
  little stiff, e.g. "I'm looking for Raymond formal shirt.") in exchange for
  170 scenarios instead of a dozen hand-written ones. It doesn't undermine the
  check: Claude has no trouble understanding an article-less request, and the
  thing being measured (did it cite the right SKU, did it avoid fabricating
  one) doesn't depend on the message's polish.
- **`refusal`**: no product in *this* tenant's catalog answers the message -
  still hand-picked (`REFUSAL_SCENARIOS` in each tenant module), built by
  pointing a real anchor query from *another* tenant's golden set at this one.
  Catalogs here are domain-disjoint (clothing / electronics / home goods), so
  cross-tenant absence is a safe, mechanical ground truth without a human
  judgment call - but unlike the grounded set there's no free 170-query source
  to derive from, so this stays a curated list (~16 per tenant, ~48 total).

**Cost, stated plainly**: unlike a retrieval query (free, local `hybrid()`
call), each scenario here is a real, paid Bedrock call - agent -> tool call ->
agent again, sometimes twice. At Haiku 4.5 / Sonnet 5 Bedrock pricing and the
short messages here, ~220 scenarios is still well under a dollar - the real
cost of this scale is wall-clock (real sequential calls take tens of minutes,
not per-query milliseconds like retrieval eval), not money.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass, field
from typing import Literal

from eval.golden import load_golden_set

ScenarioKind = Literal["grounded", "refusal"]


@dataclass(frozen=True)
class ChatScenario:
    """One chat turn to send, and what a correct answer looks like.

    `expected_skus` is empty for a `refusal` scenario - there is nothing correct to
    cite, by construction.
    """

    id: str
    message: str
    kind: ScenarioKind
    expected_skus: frozenset[str] = field(default_factory=frozenset)
    note: str | None = None


_TENANT_MODULES = {
    "demo-fashion-in": "demo_fashion_in",
    "demo-electronics-in": "demo_electronics_in",
    "demo-home-goods": "demo_home_goods",
}

_BARE_NUMERIC = re.compile(r"^\d+$")


def _as_chat_message(query_text: str) -> str:
    """Templated, not hand-authored - see module docstring for why that
    tradeoff was accepted at this scale. A bare numeric query (a Myntra-style
    SKU with no letters at all) reads oddly as a sentence on its own, so it gets
    its own wrapper; everything else gets a plain carrier phrase."""
    if _BARE_NUMERIC.match(query_text.strip()):
        return f"Do you have product {query_text} in stock?"
    return f"I'm looking for {query_text}."


def _grounded_scenarios(tenant: str) -> list[ChatScenario]:
    scenarios = []
    for query in load_golden_set(tenant):
        expected = frozenset(sku for sku, rel in query.judgments.items() if rel > 0)
        if not expected:
            continue  # no positive judgment - nothing a correct answer could cite
        scenarios.append(
            ChatScenario(
                id=f"{tenant}-{query.id}",
                message=_as_chat_message(query.query),
                kind="grounded",
                expected_skus=expected,
                note=f"Derived from eval/golden's {query.id} ({query.query_class.value}).",
            )
        )
    return scenarios


def load_chat_scenarios(tenant: str) -> list[ChatScenario]:
    """All chat scenarios for one demo tenant: every positively-judged retrieval
    golden query for this tenant (grounded), plus this tenant's hand-picked
    cross-tenant refusal probes."""
    module_name = _TENANT_MODULES.get(tenant)
    if module_name is None:
        raise ValueError(f"no chat scenarios registered for tenant {tenant!r}")
    module = importlib.import_module(f"eval.golden_chat.{module_name}")
    return _grounded_scenarios(tenant) + list(module.REFUSAL_SCENARIOS)
