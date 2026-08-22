"""Grounding, honestly scoped to what a **streamed** response can actually do.

Two layers:

1. **Prompt-level constraint** (primary defense, not in this module - see
   `app/llm/prompting.py`): the system prompt instructs Claude to wrap every SKU
   it cites in a `[[SKU:...]]` marker (`app/llm/markers.py`), and to cite only
   SKUs present in the products it was actually given this turn.
2. **Post-hoc detection** (this module): once the full answer text is assembled,
   check every cited marker against what was actually retrieved. This is
   detection, not prevention - by the time this runs, the answer has already
   streamed to the client token by token, so a hallucinated citation cannot be
   un-sent. Blocking would mean buffering the entire response before sending
   anything, which defeats the point of streaming. What this *can* do: feed a
   metric (`app.obs.metrics`) that Day 6's groundedness/hallucination-rate eval
   consumes - turning "we hope it doesn't hallucinate" into "here's how often it
   does, measured."

**Retired the free-text SKU-shaped-token regex this session (2026-08-22)**, kept
below only as history: the original approach scanned generated prose for tokens
*shaped like* a SKU (`\\b[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+\\b|\\b\\d{6,}\\b`) and
tried to tell a real citation apart from an ordinary word, a model number baked
into a title, a barcode, or a paraphrased spec/price range - accumulating, over
several live-caught false-positive rounds, a hyphenated-idiom carve-out, a
title/brand substring check, and a shopper's-own-words exemption. Every one of
those exists only because guessing which substrings in free prose are a citation
is fundamentally ambiguous. With Claude declaring citations explicitly via
`[[SKU:X]]`, none of that machinery is needed: a marker's SKU either is or isn't
in this turn's real retrieved set, checked as exact (case-insensitive) set
membership. See `PROGRESS.md`'s dated entry for the full before/after account,
including the live Bedrock test proving both model tiers comply reliably.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from app.llm.markers import extract_cited_skus


def find_hallucinated_citations(
    answer: str, retrieved_products: Collection[Mapping[str, Any]]
) -> list[str]:
    """SKUs wrapped in a `[[SKU:...]]` marker in `answer` that don't match any
    SKU actually retrieved this turn. Case-insensitive (a model paraphrasing a
    SKU's case shouldn't count as a miss); order-preserving, de-duplicated.
    `retrieved_products` is `ChatState.citations`' product entries
    (`app/llm/prompting.py::hits_to_evidence`'s shape).
    """
    known_skus = {str(p.get("sku", "")).casefold() for p in retrieved_products}

    seen: set[str] = set()
    hallucinated: list[str] = []
    for sku in extract_cited_skus(answer):
        folded = sku.casefold()
        if folded in known_skus or folded in seen:
            continue
        seen.add(folded)
        hallucinated.append(sku)
    return hallucinated
