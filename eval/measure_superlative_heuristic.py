"""Measures `app/llm/claims.py::has_superlative_language`'s accuracy against an
independent LLM judge - the last item in the marker-protocol plan's Phase 2
(`/home/akash/.claude/plans/moonlit-riding-hummingbird.md`), extending the same
"measured, not guessed" discipline `alpha_router.py` already has to the one
remaining lexical heuristic named as a real (if lower-priority) gap.

`has_superlative_language` only feeds an observability counter
(`observe_claim_mismatch(claim_type="superlative_without_stats")`) - "signal
only, not a gate," confirmed with the user previously - so this measurement is
about *knowing* its real accuracy, not gating anything on the result.

**The judge's verdict is itself a forced tool call, not parsed free text** - the
same "structural declaration over regex-guessing" principle this session's
marker-protocol rework established for the chat agent's own answers, applied
here so the judge's own output isn't one more thing to guess at from prose.

**A repeat of the exact same free finding `eval/measure_model_router.py` made**:
checking `has_superlative_language()` against all 218 existing `eval/golden_chat`
messages (free, no LLM call) found it flags **zero** of them - every message is
a short, templated single-item lookup, never a superlative/aggregate question.
Running the judge against that set would only confirm the heuristic agrees with
itself on messages it was never going to flag either way - uninformative by
construction. Uses a small, hand-authored `PROBES` list instead (confirmed
against the real, unmocked heuristic before spending anything - see the module-
level check), deliberately including paraphrases of the same underlying
questions the flagship phrasings already cover, to test the dangerous direction
(a real aggregate need the heuristic *misses*), not just confirm the phrasings
already in its own word list.

Run: `.venv/bin/python -m eval.measure_superlative_heuristic`
Requires a real `AWS_BEARER_TOKEN_BEDROCK` - skips with a clear message otherwise.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, cast

from anthropic.types import ToolChoiceToolParam, ToolParam, ToolUseBlock

from app.config import get_settings
from app.llm.claims import has_superlative_language
from app.llm.client import get_bedrock_client

RESULTS_PATH = Path("eval/results/superlative_heuristic_measurement.json")

# Hand-authored, not sourced from `eval/golden_chat` (see module docstring for
# why that set is uninformative here). Grouped by what they're meant to probe:
# classic phrasings the heuristic's own word list already covers, deliberate
# paraphrases of the identical underlying need, and ordinary non-aggregate
# questions a correct heuristic must leave unflagged.
PROBES: list[str] = [
    # Classic phrasings - already in _SUPERLATIVE_CUE's word list.
    "What is the highest priced item you have?",
    "How many kurtas do you have in stock?",
    "Do you have anything above 10000 rupees?",
    "What is your cheapest option overall?",
    # Paraphrases of the identical need, in wording the word list doesn't cover.
    "What is the priciest thing you sell?",
    "What is the average price of your products?",
    "What is the average rating across your catalog?",
    "Do you carry anything over ten thousand rupees?",
    "What is the total number of products in your electronics catalog?",
    "Roughly what price range do most of your products fall into?",
    # Ordinary, non-aggregate questions - a correct heuristic must leave these
    # unflagged; includes cases with a digit or a quantity word that isn't a
    # catalog-wide threshold, testing for the opposite (false-positive) failure.
    "I need waterproof hiking boots",
    "Show me some blue formal shirts",
    "I need a shirt in size 42",
    "Do you have more than one colour option for this?",
]

_JUDGE_TOOL: ToolParam = {
    "name": "judge_needs_stats",
    "description": (
        "Record whether correctly answering the shopper's message requires an "
        "exact, catalog-wide aggregate fact (a count, minimum, maximum, or "
        "average price/rating across the WHOLE matching catalog) that only a "
        "full catalog scan could answer correctly - as opposed to just finding "
        "a handful of relevant matching products via ordinary relevance search."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "needs_stats": {
                "type": "boolean",
                "description": "True only if an exact catalog-wide count/min/max/average is required.",
            },
            "reason": {"type": "string", "description": "One brief sentence."},
        },
        "required": ["needs_stats", "reason"],
    },
}
_JUDGE_TOOL_CHOICE: ToolChoiceToolParam = {"type": "tool", "name": "judge_needs_stats"}


# Carries the exact epistemic asymmetry this project's own system prompt
# already encodes (`app/llm/prompting.py::_SYSTEM_TEXT`): a relevance search
# finding a match proves presence, but finding none never proves absence - only
# an exact catalog-wide count does. A first, generic judge prompt without this
# framing disagreed with the heuristic on "anything above 10000 rupees?",
# reasoning it "can be answered by finding relevant matching products" -
# technically true only for a *yes*, not for a confident *no*, which is exactly
# the asymmetry the flagship production bug this whole check exists for was
# about. Spelled out explicitly rather than trusting a judge to infer a
# domain-specific distinction on its own.
_JUDGE_CONTEXT = (
    "Important distinction: a relevance search that FINDS a matching product "
    "proves the store has one, but a relevance search that finds NOTHING never "
    "proves the store has none - search only checks a bounded pool of "
    "candidates, not the whole catalog. So any question asking whether "
    "something exists above/below/over/under a threshold, or asking for an "
    "exact count/min/max/average, needs a real catalog-wide aggregate to "
    "answer with confidence - not just a search."
)


async def _judge(message: str) -> dict[str, Any]:
    client = get_bedrock_client()
    response = await client.messages.create(
        model=get_settings().model_fast,
        max_tokens=200,
        tools=[_JUDGE_TOOL],
        tool_choice=_JUDGE_TOOL_CHOICE,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{_JUDGE_CONTEXT}\n\nA shopper sent this message to a "
                    f"store's chat assistant: {message!r}"
                ),
            }
        ],
    )
    call = cast(ToolUseBlock, next(b for b in response.content if b.type == "tool_use"))
    return dict(call.input) if isinstance(call.input, dict) else {}


async def _run_all() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for message in PROBES:
        heuristic_says = has_superlative_language(message)
        judge = await _judge(message)
        judge_says = bool(judge.get("needs_stats", False))
        row = {
            "message": message,
            "heuristic": heuristic_says,
            "judge": judge_says,
            "judge_reason": judge.get("reason", ""),
            "agree": heuristic_says == judge_says,
        }
        rows.append(row)
        flag = "  " if row["agree"] else "**"
        print(f"{flag} heuristic={heuristic_says!s:<5} judge={judge_says!s:<5} {message}")
    return rows


def _print_summary(rows: list[dict[str, Any]]) -> None:
    disagreements = [r for r in rows if not r["agree"]]
    # The dangerous direction: a message that genuinely needed get_catalog_stats
    # but the heuristic never flagged, so `has_superlative_language`'s own
    # counter would have stayed silent about a real risk.
    false_negatives = [r for r in disagreements if r["judge"] and not r["heuristic"]]
    # The safe direction: a wasted counter increment, not a missed correctness risk.
    false_positives = [r for r in disagreements if r["heuristic"] and not r["judge"]]

    print(f"\n{'=' * 70}")
    agree_n = len(rows) - len(disagreements)
    print(f"Total: {len(rows)} | Agreement: {agree_n}/{len(rows)} ({agree_n / len(rows):.1%})")
    print(f"False negatives (heuristic MISSED a real stats need): {len(false_negatives)}")
    for r in false_negatives:
        print(f"    {r['message']!r} -> {r['judge_reason']}")
    print(f"False positives (heuristic flagged unnecessarily, not dangerous): {len(false_positives)}")
    for r in false_positives:
        print(f"    {r['message']!r} -> {r['judge_reason']}")


def _write_results(rows: list[dict[str, Any]]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(rows, indent=2))


async def main() -> None:
    settings = get_settings()
    if not settings.bedrock_api_key or settings.bedrock_api_key.startswith("bedrock-api-key-xxxx"):
        print(
            "AWS_BEARER_TOKEN_BEDROCK not set to a real key - skipping "
            "(see PROGRESS.md's Day 5 notes)."
        )
        sys.exit(0)

    rows = await _run_all()
    _print_summary(rows)
    _write_results(rows)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
