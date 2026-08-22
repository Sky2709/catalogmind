"""Refusal probes for `demo-electronics-in`. Grounded scenarios are derived
automatically from `eval/golden/demo_electronics_in.py` (see
`eval/golden_chat/__init__.py`). Refusal messages reuse real anchors from
`eval/golden/demo_fashion_in.py` and `demo_home_goods.py` - apparel and
furniture an electronics catalog genuinely doesn't carry.
"""

from __future__ import annotations

from eval.golden_chat import ChatScenario

REFUSAL_SCENARIOS = [
    ChatScenario(
        id="electronics-refusal-001",
        message="Do you have a men's kurta for a wedding?",
        kind="refusal",
        note="Fashion apparel, absent from an electronics catalog.",
    ),
    ChatScenario(
        id="electronics-refusal-002",
        message="I'm looking for a queen size upholstered platform bed",
        kind="refusal",
        note="Same anchor as homegoods-id-009 in eval/golden.",
    ),
    ChatScenario(
        id="electronics-refusal-003",
        message="Do you sell Indian Terrain men's casual shirts?",
        kind="refusal",
    ),
    ChatScenario(
        id="electronics-refusal-004",
        message="I need Geox women's ballerina flats",
        kind="refusal",
        note="Same anchor as fashion-id-002 in eval/golden.",
    ),
    ChatScenario(
        id="electronics-refusal-005",
        message="Do you have Puma sneakers for men?",
        kind="refusal",
    ),
    ChatScenario(
        id="electronics-refusal-006",
        message="I'm looking for a Titan analogue watch",
        kind="refusal",
    ),
    ChatScenario(
        id="electronics-refusal-007",
        message="Do you sell traditional wear for Diwali?",
        kind="refusal",
    ),
    ChatScenario(
        id="electronics-refusal-008",
        message="I need an ethnic jewellery set for a festive occasion",
        kind="refusal",
    ),
    ChatScenario(
        id="electronics-refusal-009",
        message="Do you have kids party wear for a birthday?",
        kind="refusal",
    ),
    ChatScenario(
        id="electronics-refusal-010",
        message="I'm looking for a wide mouth juicer machine",
        kind="refusal",
    ),
    ChatScenario(
        id="electronics-refusal-011",
        message="Do you sell a wooden full size platform bed?",
        kind="refusal",
    ),
    ChatScenario(
        id="electronics-refusal-012",
        message="I need a waterproof sofa cover",
        kind="refusal",
    ),
    ChatScenario(
        id="electronics-refusal-013",
        message="Do you have a coffee table for the living room?",
        kind="refusal",
    ),
    ChatScenario(
        id="electronics-refusal-014",
        message="I'm looking for cabinet drawer pull handles",
        kind="refusal",
    ),
    ChatScenario(
        id="electronics-refusal-015",
        message="Do you sell an anti-slip bath mat?",
        kind="refusal",
    ),
    ChatScenario(
        id="electronics-refusal-016",
        message="I need a shoe storage bench",
        kind="refusal",
    ),
]
