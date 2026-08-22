"""Intrinsic embedding-quality report: does the vector space this pipeline actually
produces carry semantic signal - answerable today, without a search endpoint or a
labelled query set (Day 3/4). See `app/ingestion/embedding_quality.py` for what each
check means and why it doesn't need those.

Per catalog: an average-pairwise-similarity sanity ceiling, a group-contrast check
against whichever real attribute is available (brand for fashion, category for
home-goods, a rough keyword-based product-type guess for electronics - it has neither
a populated brand nor a non-degenerate category, see `data/SOURCES.md`), a
nearest-neighbour spot check, and a handful of hand-written probe queries. The probe
queries are a cheap sanity check, explicitly NOT a substitute for Day 4's golden
query set - five queries per catalog prove nothing statistically; they just catch an
embedding pipeline that is obviously broken before spending a day building the real
eval harness on top of it.

Run: `.venv/bin/python -m scripts.embedding_quality_report [catalog name]`
Needs the demo catalogs downloaded (`data/raw/`, see `data/SOURCES.md`) but not
seeded - it embeds its own sample directly, independent of what's in Weaviate.
"""

from __future__ import annotations

import re
import sys

from app.ingestion.embed import embed_documents, embed_query, warm_up
from app.ingestion.embedding_quality import (
    average_pairwise_similarity,
    group_contrast,
    nearest_neighbors,
)
from app.models.product import Product
from scripts.bench_ingestion import load_real_products

SAMPLE_SIZE = 500
ANCHOR_COUNT = 4
NEIGHBOR_COUNT = 5

# Rough, deliberately simple keyword buckets - only for this diagnostic, not part of
# the real schema. Electronics has neither a populated brand nor a non-degenerate
# category (see the ingestion quality report), so this is the only available proxy
# for "is the embedding grouping similar product types together".
ELECTRONICS_KEYWORDS: dict[str, re.Pattern[str]] = {
    "phone": re.compile(r"\b(phone|smartphone|redmi|iphone|oneplus|realme)\b", re.I),
    "camera": re.compile(r"\b(camera|dslr|webcam)\b", re.I),
    "audio": re.compile(r"\b(earphone|headphone|earbud|speaker|bluetooth audio)\b", re.I),
    "cable_charger": re.compile(r"\b(cable|charger|adapter|power bank)\b", re.I),
    "computer": re.compile(r"\b(laptop|keyboard|mouse|monitor)\b", re.I),
}

PROBE_QUERIES: dict[str, list[str]] = {
    "fashion": [
        "cotton shirt for men",
        "blue jeans",
        "women's dress",
        "leather bag",
        "running shoes",
    ],
    "electronics": [
        "smartphone",
        "bluetooth earphones",
        "laptop",
        "camera",
        "charging cable",
    ],
    "home": [
        "kitchen storage container",
        "bath towel",
        "bedroom rug",
        "wall decoration",
        "outdoor lighting",
    ],
}


def electronics_bucket(title: str) -> str | None:
    for label, pattern in ELECTRONICS_KEYWORDS.items():
        if pattern.search(title):
            return label
    return None


def group_key_for(catalog_name: str, product: Product) -> str | None:
    if catalog_name == "fashion":
        return product.brand
    if catalog_name == "home":
        return product.category_path[0] if product.category_path else None
    if catalog_name == "electronics":
        return electronics_bucket(product.title)
    return None


def run_catalog_report(catalog_name: str, products: list[Product]) -> None:
    sample = products[:SAMPLE_SIZE]
    texts = [p.embedding_text() for p in sample]

    print("=" * 72)
    print(f"{catalog_name}  (n={len(sample)} sampled of {len(products)})")

    vectors = embed_documents(texts)

    avg_sim = average_pairwise_similarity(vectors)
    print(f"\n  average pairwise similarity (random pairs): {avg_sim:.3f}")
    print("    (near 1.0 would mean the embedding space has collapsed; well below")
    print("    1.0 with real, distinct-looking text below is the healthy case)")

    grouping_field = {"fashion": "brand", "home": "category", "electronics": "keyword-bucket"}[
        catalog_name
    ]
    result = group_contrast(
        sample, vectors, lambda p: group_key_for(catalog_name, p), grouping_field=grouping_field
    )
    if result is None:
        print(f"\n  group contrast ({grouping_field}): not enough grouped pairs in this sample")
    else:
        print(
            f"\n  group contrast ({grouping_field}, {result.groups_compared} groups): "
            f"within={result.within_group_mean:.3f}  across={result.across_group_mean:.3f}  "
            f"contrast={result.contrast:+.3f}"
        )
        print(
            "    (positive and not tiny means the embedding independently captures "
            f"{grouping_field}, which it was never told about directly)"
        )

    print(f"\n  nearest-neighbour spot check ({ANCHOR_COUNT} anchors, read these yourself):")
    corpus = [(p.sku, v) for p, v in zip(sample, vectors, strict=True)]
    by_sku = {p.sku: p for p in sample}
    step = max(1, len(sample) // ANCHOR_COUNT)
    for i in range(0, len(sample), step):
        anchor = sample[i]
        anchor_vec = vectors[i]
        print(f"    anchor: {anchor.title[:70]!r}")
        for sku, sim in nearest_neighbors(anchor_vec, corpus, NEIGHBOR_COUNT, exclude=anchor.sku):
            print(f"      {sim:.3f}  {by_sku[sku].title[:70]!r}")
        print()

    print("  probe queries (not the Day 4 golden set - a cheap sanity check only):")
    for query in PROBE_QUERIES.get(catalog_name, []):
        query_vector = embed_query(query)
        top = nearest_neighbors(query_vector, corpus, 3)
        print(f"    {query!r}")
        for sku, sim in top:
            print(f"      {sim:.3f}  {by_sku[sku].title[:70]!r}")


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else None
    warm_up()
    products_by_catalog = load_real_products()
    for name, products in products_by_catalog.items():
        if target and target != name:
            continue
        run_catalog_report(name, products)


if __name__ == "__main__":
    main()
