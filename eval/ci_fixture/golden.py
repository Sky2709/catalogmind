"""Golden queries for the CI-only fixture catalog (`eval/ci_fixture/catalog.csv`).

**Not the real 170-query golden set** (`eval/golden/`) - this is a small (15-query),
fully self-contained proxy used only by `eval/ci_quality_gate.py`, because the real
golden sets are anchored to real Kaggle datasets that are git-ignored (size/licensing)
and have no scripted CI re-download. This fixture exists so a search-quality
regression gate can run on every PR without external data or secrets - see
`eval/ci_quality_gate.py`'s module docstring for what it can and can't catch.

Same discipline as the real golden sets despite the smaller scale: every judgment
below is grounded in an actual `WeaviateHybridRetriever.search()` call against the
fixture catalog once ingested (verified 2026-08-22), not invented from reading the
catalog and guessing what "should" match. Relevance is graded 0-2 the same way
(`eval/golden/__init__.py`): 2 = an exact match for the query's intent, 1 =
genuinely on-topic but broader, 0 = not recorded - a borderline hit was left
unjudged rather than force-labelled.
"""

from __future__ import annotations

from app.retrieval.base import QueryClass
from eval.golden import GoldenQuery

QUERIES = [
    # --- identifier: exact SKU lookups, all verified rank-1 -----------------------
    GoldenQuery(
        id="ci-id-001",
        query="BOOT-WP-10",
        query_class=QueryClass.IDENTIFIER,
        judgments={"BOOT-WP-10": 2},
    ),
    GoldenQuery(
        id="ci-id-002",
        query="JCKT-RN-22",
        query_class=QueryClass.IDENTIFIER,
        judgments={"JCKT-RN-22": 2},
    ),
    GoldenQuery(
        id="ci-id-003",
        query="HDPH-BT-01",
        query_class=QueryClass.IDENTIFIER,
        judgments={"HDPH-BT-01": 2},
    ),
    GoldenQuery(
        id="ci-id-004",
        query="WATC-SM-07",
        query_class=QueryClass.IDENTIFIER,
        judgments={"WATC-SM-07": 2},
    ),
    GoldenQuery(
        id="ci-id-005",
        query="BTTL-SS-500",
        query_class=QueryClass.IDENTIFIER,
        judgments={"BTTL-SS-500": 2},
    ),
    # --- attribute: brand + product-type constraints -------------------------------
    GoldenQuery(
        id="ci-attr-001",
        query="Trailhead waterproof hiking boots",
        query_class=QueryClass.ATTRIBUTE,
        judgments={"BOOT-WP-10": 2, "BOOT-WP-11": 2},
        note="BOOT-TR-05 is Summit-brand trail running boots, deliberately left unjudged.",
    ),
    GoldenQuery(
        id="ci-attr-002",
        query="Acme slim fit formal shirt",
        query_class=QueryClass.ATTRIBUTE,
        judgments={"SHRT-FRM-B2": 2, "SHRT-FRM-W3": 2},
    ),
    GoldenQuery(
        id="ci-attr-003",
        query="PulseFit smartwatch",
        query_class=QueryClass.ATTRIBUTE,
        judgments={"WATC-SM-07": 2, "WATC-SM-08": 2},
    ),
    GoldenQuery(
        id="ci-attr-004",
        query="Trailhead hiking backpack",
        query_class=QueryClass.ATTRIBUTE,
        judgments={"BPCK-HK-45": 2, "BPCK-HK-30": 2},
    ),
    GoldenQuery(
        id="ci-attr-005",
        query="Acme wireless bluetooth headphones",
        query_class=QueryClass.ATTRIBUTE,
        judgments={"HDPH-BT-01": 2, "HDPH-BT-02": 1},
        note="HDPH-BT-02 is earbuds, not headphones - on-topic but broader.",
    ),
    # --- exploratory: intent without catalog vocabulary ----------------------------
    GoldenQuery(
        id="ci-expl-001",
        query="something for a rainy hiking trip",
        query_class=QueryClass.EXPLORATORY,
        judgments={"JCKT-RN-22": 2, "JCKT-RN-23": 2, "BOOT-WP-10": 2, "BOOT-WP-11": 2},
    ),
    GoldenQuery(
        id="ci-expl-002",
        query="gift for someone who loves running",
        query_class=QueryClass.EXPLORATORY,
        judgments={"SHOE-RUN-21": 2, "SHOE-RUN-22": 2},
    ),
    GoldenQuery(
        id="ci-expl-003",
        query="gear for a multi-day backpacking trip",
        query_class=QueryClass.EXPLORATORY,
        judgments={"BPCK-HK-45": 2, "TENT-2P-01": 2, "BPCK-HK-30": 1},
    ),
    GoldenQuery(
        id="ci-expl-004",
        query="keep my drink cold on a hike",
        query_class=QueryClass.EXPLORATORY,
        judgments={"BTTL-SS-500": 2},
    ),
    GoldenQuery(
        id="ci-expl-005",
        query="something to wear to a formal office",
        query_class=QueryClass.EXPLORATORY,
        judgments={"SHRT-FRM-B2": 2, "SHRT-FRM-W3": 2},
    ),
]
