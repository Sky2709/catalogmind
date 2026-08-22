# Dataset provenance and licences

Per `CLAUDE.md`: verify a licence before ingesting anything, and record it here. All
three catalogs land in `data/raw/` (git-ignored) and are the source for the three demo
merchants (Day 2). None of this data, nor anything derived from it, is sold — this is a
public portfolio project with a free live demo.

## Fashion — `data/raw/fashion-myntra/`

- **Dataset:** [Myntra Fashion Products](https://www.kaggle.com/datasets/nirokey/myntra-fashion-products) (nirokey)
- **Licence:** CC0-1.0 (public domain) — confirmed via `kaggle datasets download` output
- **Rows:** ~12,762
- **Shape:** `name, sku, mpn, price, in_stock, currency, brand, description, images, gender`
- **Downloaded:** 2026-08-20
- **Notes:** Real Indian-market fashion listings (Myntra). `images` packs multiple URLs
  separated by `~` — needs a `preprocess` hook to take the first, not a plain
  `ColumnMapping`. **Fixed 2026-08-22**: this gap meant `image_url` was never mapped at
  all (products indexed with no image), found via manual chat-UI testing — see
  `PROGRESS.md`'s "Image pipeline fix" entry. `MyntraFashionAdapter`
  (`app/ingestion/adapters/demo_catalogs.py`) now splits it and keeps the first URL;
  verified against all 12,491 rows that the first segment is always a non-empty,
  valid `http(s)` URL. `sku`/`mpn` are identical in every row seen so far; worth
  spot-checking before treating `sku` as reliably unique.

## Electronics — `data/raw/electronics-amazon/`

- **Dataset:** [Amazon Electronics Products 10k items - 2023](https://www.kaggle.com/datasets/akeshkumarhp/electronics-products-amazon-10k-items) (akeshkumarhp)
- **Licence:** ODbL-1.0 (Open Database License — attribution + share-alike on the
  database) — confirmed via `kaggle datasets metadata`
- **Rows:** ~9,601
- **Shape:** unnamed index column, `name, main_category, sub_category, image, link,
  ratings, no_of_ratings, discount_price, actual_price`
- **Downloaded:** 2026-08-20
- **Notes:** Amazon India listings; prices are `"₹10,999"`-style strings — exactly what
  `app/ingestion/normalize.py::parse_price` exists to handle. **No SKU column at all** —
  a real adapter will need to derive one, e.g. from the ASIN embedded in `link`
  (`/dp/B09Y64H8VS/`). Originally tried `datafiniti/electronic-products-prices`
  (Datafiniti's electronics pricing set) — **rejected**: licensed CC-BY-NC-SA-4.0, and
  the NonCommercial + ShareAlike terms are too ambiguous for a public repo + live demo
  to risk. That file was downloaded then deleted, never ingested.

## Home goods (deliberately messy) — `data/raw/home-shein/`

- **Dataset:** [Dirty E-Commerce Data \[80,000+ Products\]](https://www.kaggle.com/datasets/oleksiimartusiuk/e-commerce-data-shein) (oleksiimartusiuk) — three of its twenty per-category files
- **Licence:** ODC-By (attribution required) — confirmed via `kaggle datasets download` output
- **Files used:** `us-shein-home_and_kitchen-3719.csv`, `us-shein-home_textile-3883.csv`,
  `us-shein-tools_and_home_improvement-3903.csv` (~11,503 rows combined). The other
  seventeen category files (electronics, kids, pet supplies, ...) were left undownloaded
  — off-topic for the "home goods" vertical, not a licensing concern.
- **Shape:** `goods-title-link, rank-title, rank-sub, price, selling_proposition, discount`
- **Downloaded:** 2026-08-20
- **Notes:** This is the genuinely messy one, by design. No SKU/id column at all;
  `goods-title-link` appears to fuse the title and a URL together. Will need a real
  `FeedAdapter` subclass (`preprocess`), not just a `ColumnMapping` — which is exactly
  the point of this vertical per the project blueprint.

## Attribution

When the README ships, credit:
- Myntra Fashion Products by nirokey (CC0-1.0, no attribution legally required, credited anyway)
- Amazon Electronics Products 10k items by akeshkumarhp (ODbL-1.0)
- Dirty E-Commerce Data by oleksiimartusiuk (ODC-By)
