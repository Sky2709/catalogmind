"""Refusal probes for `demo-fashion-in`. Grounded scenarios are derived
automatically from `eval/golden/demo_fashion_in.py` (see `eval/golden_chat/
__init__.py`) - only the refusal probes need hand-picking, since there's no
free source of "definitely absent from this catalog" queries the way there is
for "definitely present." Each message below reuses a real, already-verified
anchor query from `eval/golden/demo_electronics_in.py` or
`demo_home_goods.py` - electronics and home-goods products a clothing catalog
genuinely doesn't carry.

**One real limitation this actually caught (2026-08-21, full 218-scenario run)**:
"domain-disjoint" isn't perfectly true. This Myntra fashion catalog genuinely
carries a handful of home-textile items (coasters, placemats, duvet covers, bath
mats) alongside clothing - three home-goods-anchored probes found a real match
and were correctly answered, not refused, which is the *model* behaving
correctly against a *wrong* assumption in the scenario, not a model failure.
Those three were swapped for electronics anchors, which showed zero such
leakage across every probe tried against this tenant.
"""

from __future__ import annotations

from eval.golden_chat import ChatScenario

REFUSAL_SCENARIOS = [
    ChatScenario(
        id="fashion-refusal-001",
        message="Do you sell smartphones?",
        kind="refusal",
        note="Electronics category, absent from a clothing catalog.",
    ),
    ChatScenario(
        id="fashion-refusal-002",
        message="Do you have the Redmi 10 in Caribbean Green?",
        kind="refusal",
        note="Same anchor as electronics-id-002 in eval/golden.",
    ),
    ChatScenario(
        id="fashion-refusal-003",
        message="I need a Logitech G502 gaming mouse",
        kind="refusal",
        note="Same anchor as electronics-id-004 in eval/golden.",
    ),
    ChatScenario(
        id="fashion-refusal-004",
        message="Do you sell noise cancelling headphones?",
        kind="refusal",
    ),
    ChatScenario(
        id="fashion-refusal-005",
        message="I'm looking for a portable power bank with fast charging",
        kind="refusal",
    ),
    ChatScenario(
        id="fashion-refusal-006",
        message="Do you have a dual band wifi router?",
        kind="refusal",
    ),
    ChatScenario(
        id="fashion-refusal-007",
        message="I need an external hard drive with 2TB storage",
        kind="refusal",
    ),
    ChatScenario(
        id="fashion-refusal-008",
        message="Do you have a wireless bluetooth mouse?",
        kind="refusal",
    ),
    ChatScenario(
        id="fashion-refusal-009",
        message="I need a 4K smart Android TV",
        kind="refusal",
    ),
    ChatScenario(
        id="fashion-refusal-010",
        message="I'm looking for a queen size bed frame with storage",
        kind="refusal",
        note="Home-goods furniture, absent from a clothing catalog.",
    ),
    ChatScenario(
        id="fashion-refusal-011",
        message="Do you sell the SONGMICS 10 Tier Shoe Rack?",
        kind="refusal",
        note="Same anchor as homegoods-id-005 in eval/golden.",
    ),
    ChatScenario(
        id="fashion-refusal-012",
        message="I'm looking for a coffee table for my living room",
        kind="refusal",
    ),
    ChatScenario(
        id="fashion-refusal-013",
        message="Do you sell a mechanical gaming keyboard?",
        kind="refusal",
        note=(
            "Replaced 2026-08-21: the original 'kitchen table mat or coaster "
            "set' probe turned out not to be a clean absence - this Myntra "
            "fashion catalog genuinely carries some home-textile items "
            "(coasters, placemats), so the model correctly found a real match "
            "and the scenario's ground truth was wrong, not the model. Swapped "
            "for an electronics anchor - zero domain leakage observed against "
            "this tenant across every other electronics-domain probe."
        ),
    ),
    ChatScenario(
        id="fashion-refusal-014",
        message="I need a smartwatch with bluetooth calling",
        kind="refusal",
        note="Replaced 2026-08-21, same reason as fashion-refusal-013 (the "
        "original 'duvet cover set' probe found a real match in this "
        "catalog's home-textile items).",
    ),
    ChatScenario(
        id="fashion-refusal-015",
        message="Do you sell solar powered garden lights?",
        kind="refusal",
    ),
    ChatScenario(
        id="fashion-refusal-016",
        message="Do you have a microSD memory card for my camera?",
        kind="refusal",
        note="Replaced 2026-08-21, same reason (the original 'anti-slip bath "
        "mat' probe found a real match in this catalog's home-textile items).",
    ),
]
