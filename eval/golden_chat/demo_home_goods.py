"""Refusal probes for `demo-home-goods`. Grounded scenarios are derived
automatically from `eval/golden/demo_home_goods.py` (see
`eval/golden_chat/__init__.py`). Refusal messages reuse real anchors from
`eval/golden/demo_electronics_in.py` and `demo_fashion_in.py` - electronics
and apparel a home-goods catalog genuinely doesn't carry.
"""

from __future__ import annotations

from eval.golden_chat import ChatScenario

REFUSAL_SCENARIOS = [
    ChatScenario(
        id="homegoods-refusal-001",
        message="Do you have wireless bluetooth earbuds?",
        kind="refusal",
        note="Electronics, absent from a home-goods catalog.",
    ),
    ChatScenario(
        id="homegoods-refusal-002",
        message="I'm looking for a Raymond formal shirt",
        kind="refusal",
        note="Fashion apparel, absent from a home-goods catalog.",
    ),
    ChatScenario(
        id="homegoods-refusal-003",
        message="Do you sell the Samsung Galaxy S23 5G?",
        kind="refusal",
        note="Same anchor as electronics-id-001 in eval/golden.",
    ),
    ChatScenario(
        id="homegoods-refusal-004",
        message="I need a 43 inch smart TV",
        kind="refusal",
    ),
    ChatScenario(
        id="homegoods-refusal-005",
        message="Do you have a Logitech gaming mouse?",
        kind="refusal",
    ),
    ChatScenario(
        id="homegoods-refusal-006",
        message="I'm looking for noise cancelling headphones",
        kind="refusal",
    ),
    ChatScenario(
        id="homegoods-refusal-007",
        message="Do you sell a 4K smart Android TV?",
        kind="refusal",
    ),
    ChatScenario(
        id="homegoods-refusal-008",
        message="I need an RGB mechanical gaming keyboard",
        kind="refusal",
    ),
    ChatScenario(
        id="homegoods-refusal-009",
        message="Do you have a fast-charging power bank?",
        kind="refusal",
    ),
    ChatScenario(
        id="homegoods-refusal-010",
        message="I'm looking for Indian Terrain men's casual shirts",
        kind="refusal",
    ),
    ChatScenario(
        id="homegoods-refusal-011",
        message="Do you sell Geox women's ballerina flats?",
        kind="refusal",
    ),
    ChatScenario(
        id="homegoods-refusal-012",
        message="I need Puma sneakers for men",
        kind="refusal",
    ),
    ChatScenario(
        id="homegoods-refusal-013",
        message="Do you have a Titan analogue watch?",
        kind="refusal",
    ),
    ChatScenario(
        id="homegoods-refusal-014",
        message="I'm looking for traditional wear for Diwali",
        kind="refusal",
    ),
    ChatScenario(
        id="homegoods-refusal-015",
        message="I'm looking for a pair of women's sneakers",
        kind="refusal",
        note=(
            "Replaced 2026-08-21: the original 'kids party wear for a "
            "birthday' probe was genuinely ambiguous, not a clean absence - "
            "this catalog carries kids' party *accessories* (hats, decorations)"
            " even though it has no actual clothing, and the model's honest, "
            "hedged answer (found accessories, offered to search for clothing "
            "specifically) doesn't cleanly score as either a hit or a refusal. "
            "Swapped for footwear - Day 2's data-quality report found this "
            "catalog to be exclusively furniture/kitchen/storage/wellness, no "
            "clothing or footwear category at all."
        ),
    ),
    ChatScenario(
        id="homegoods-refusal-016",
        message="I need a smartwatch with bluetooth calling",
        kind="refusal",
    ),
]
