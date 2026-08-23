"""Locked category/sub-category taxonomies for the three demo catalogs.

None of the three raw feeds carries a usable category signal (`data/SOURCES.md`,
`PROGRESS.md`'s "category/sub-category" investigation): fashion has no category
column at all, electronics' `main_category`/`sub_category` are a single identical
value across all 9,600 rows, and home-goods' `rank-sub` is real but empty for 78.9%
of rows and an ungrouped 939-value long tail where present. `scripts/
enrich_categories.py` classifies every product against ONE of these locked lists via
a Claude tool call, so the taxonomy has to be defined and reviewed up front, not
discovered per-product - a closed, versioned label set is what makes `category_path`
usable as a real Weaviate filter afterward (`app/retrieval/hybrid.py`'s
`_build_filters` already has a `contains_any` clause for it, wired end-to-end through
the chat agent's tool schema - this module is the only missing piece).

Each leaf is stored as a single `"Category > Subcategory"` string, not two separate
enums - deliberately, so a forced tool call can only ever pick a real, pre-approved
combination (no `Electronics > Skincare`-style nonsense possible), at the cost of a
slightly larger enum than two independent ones would need. `Product._split_category`
(`app/models/product.py`) already splits on `>` for free.

Reviewed and locked with the user, 2026-08-23 - see PROGRESS.md.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

SEPARATOR = " > "

ENRICHMENT_DIR = Path("data/processed/category_enrichment")

UNCATEGORIZED = "Other > Uncategorized"

FASHION_TAXONOMY: tuple[str, ...] = (
    "Apparel > Menswear",
    "Apparel > Womenswear",
    "Apparel > Kidswear",
    "Apparel > Innerwear & Loungewear",
    "Footwear > Men's",
    "Footwear > Women's",
    "Footwear > Kids'",
    "Bags & Luggage > Bags",
    "Bags & Luggage > Backpacks",
    "Bags & Luggage > Luggage & Trolleys",
    "Watches > Watches",
    "Jewelry & Eyewear > Jewelry",
    "Jewelry & Eyewear > Sunglasses & Eyewear",
    "Accessories > Belts & Small Accessories",
    "Beauty & Personal Care > Skincare",
    "Beauty & Personal Care > Makeup",
    "Beauty & Personal Care > Haircare",
    "Beauty & Personal Care > Fragrance & Bath",
    "Sportswear & Activewear > Sportswear & Activewear",
    # Added after the pilot run found a real, systematic gap (not noise): the
    # Myntra feed also carries home decor/furnishing products - ceramic bowls,
    # table lamps, placemats, bedsheets - with zero apparel signal at all.
    # 13.1% of the fashion pilot sample landed in Uncategorized before this was
    # added, almost entirely this one missing category - see PROGRESS.md.
    "Home & Living > Decor & Furnishing",
    "Home & Living > Bedding & Bath",
    "Home & Living > Kitchen & Dining",
    UNCATEGORIZED,
)

ELECTRONICS_TAXONOMY: tuple[str, ...] = (
    "Mobiles & Accessories > Smartphones",
    "Mobiles & Accessories > Cases & Screen Guards",
    "Computing > Laptops",
    "Computing > Storage & Peripherals",
    "Audio > Headphones & Earbuds",
    "Audio > Speakers",
    "TV & Home Entertainment > TV & Home Entertainment",
    "Cameras & Photography > Cameras & Photography",
    "Wearables > Wearables",
    "Cables, Chargers & Power > Cables, Chargers & Power",
    "Home Appliances > Home Appliances",
    UNCATEGORIZED,
)

HOME_GOODS_TAXONOMY: tuple[str, ...] = (
    "Kitchen & Dining > Cookware",
    "Kitchen & Dining > Tableware & Drinkware",
    "Kitchen & Dining > Kitchen Storage",
    "Kitchen & Dining > Small Appliances",
    "Bedding & Bath > Bedding & Linens",
    "Bedding & Bath > Bath Textiles",
    "Bedding & Bath > Bath Accessories",
    "Home Decor > Wall Decor",
    "Home Decor > Decorative Accents",
    "Home Decor > Lighting & Candles",
    "Storage & Organization > Closet & Wardrobe",
    "Storage & Organization > General Storage",
    "Furniture > Living Room",
    "Furniture > Bedroom",
    "Furniture > Outdoor Furniture",
    "Outdoor & Garden > Garden Tools & Decor",
    "Outdoor & Garden > Outdoor Living",
    "Tools & Home Improvement > Hand Tools & Hardware",
    "Tools & Home Improvement > Home Repair",
    "Cleaning & Laundry > Cleaning Supplies",
    "Cleaning & Laundry > Laundry Accessories",
    UNCATEGORIZED,
)

TAXONOMY_BY_TENANT: dict[str, tuple[str, ...]] = {
    "demo-fashion-in": FASHION_TAXONOMY,
    "demo-electronics-in": ELECTRONICS_TAXONOMY,
    "demo-home-goods": HOME_GOODS_TAXONOMY,
}


def taxonomy_for(tenant: str) -> tuple[str, ...]:
    """The locked leaf list for one demo tenant. Raises for an unknown tenant -
    silently returning an empty taxonomy would let a typo skip classification
    entirely rather than fail loudly. Used by `scripts/enrich_categories.py`
    only - that script is only ever invoked by hand, against a known tenant, so
    failing loudly on a typo is the right behaviour there.
    """
    return TAXONOMY_BY_TENANT[tenant]


def category_filter_values(tenant: str) -> tuple[str, ...] | None:
    """Every individual category/subcategory *name* valid as the chat agent's
    `category` filter value for one tenant - e.g. `"Skincare"` (a leaf) or
    `"Beauty & Personal Care"` (its parent, matching every subcategory under
    it too, since `_build_filters`'s `contains_any` checks array membership
    against `category_path`, not the combined `"Category > Subcategory"`
    string). Flattened and deduplicated from `taxonomy_for`'s leaf list - a
    category name that's also used as its own subcategory (e.g. `"Watches >
    Watches"`) only appears once.

    Returns `None`, not an empty tuple, for a tenant with no taxonomy on file -
    a real, expected case for any merchant beyond the three demo catalogs
    `scripts/enrich_categories.py` has ever been run against. `None` is the
    signal `app/llm/prompting.py` uses to omit the `enum` constraint entirely
    rather than passing an empty one (which would make `category` an
    unsatisfiable filter for a tenant this system happens to know nothing
    about) - the live chat path must degrade gracefully here, unlike
    `taxonomy_for`'s deliberate raise for the enrichment script's own use.
    """
    taxonomy = TAXONOMY_BY_TENANT.get(tenant)
    if taxonomy is None:
        return None
    values: set[str] = set()
    for leaf in taxonomy:
        category, _, subcategory = leaf.partition(SEPARATOR)
        values.add(category)
        if subcategory:
            values.add(subcategory)
    return tuple(sorted(values))


@lru_cache(maxsize=3)
def load_enrichment_map(tenant: str) -> dict[str, str]:
    """`identity_key -> "Category > Subcategory"` from
    `scripts/enrich_categories.py`'s output for one tenant. The identity key
    matches whatever that script used for this tenant (title text for fashion/
    home-goods, the ASIN for electronics - see that script's module docstring
    for why it differs per catalog).

    Returns an empty dict if the enrichment file doesn't exist yet, rather than
    raising - lets an adapter fall back to `UNCATEGORIZED` for a fresh clone
    that hasn't run the (paid, one-off) enrichment script, instead of crashing
    ingestion entirely. Cached per-process (`lru_cache`, one entry per tenant):
    this is read once per ingestion adapter construction, not per row, and the
    file only ever changes by re-running the enrichment script by hand.
    """
    path = ENRICHMENT_DIR / f"{tenant}.jsonl"
    if not path.exists():
        return {}
    mapping: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        mapping.update(entry["classifications"])
    return mapping
