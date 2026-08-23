"""Golden queries for `demo-home-goods` (the deliberately messy SHEIN home-goods catalog).

Same methodology as the other two golden sets - every judgment below came from a
real, live `hybrid()` call against the tenant this session, not invented. See
`eval/golden/__init__.py`'s module docstring for the verified-top-K-pool limitation
and the 0-2 relevance scale.

This catalog earns its "deliberately messy" name in a way that shaped several
queries below: SKUs are synthetic hashes of the title (`app/ingestion/adapters/
demo_catalogs.py`), there is no brand field at all, `category_path` is almost always
the generic "Home Goods" fallback, and - the one that mattered most while building
this set - Day 2's ~14% duplicate rate (`SOURCES.md`, `app/ingestion/quality.py`)
turned out to be directly visible here: two identifier candidates that should have
had one clear target instead matched 2-3 near-identical listings for what is
obviously the same real product (same brand/model text, different price or minor
wording), not a retrieval bug. Both were kept as ATTRIBUTE queries with every
matching variant judged relevant, precisely because that ambiguity is a real,
observed property of this catalog, not something to paper over.
"""

from __future__ import annotations

from app.retrieval.base import QueryClass
from eval.golden import GoldenQuery

QUERIES = [
    # --- identifier: distinctive phrases pulled from real (long, SEO-stuffed)
    # titles, each verified top-1. Raw SKUs are internal hashes (see module
    # docstring) that no real shopper would ever type, so - like electronics -
    # these use title text instead. ---
    GoldenQuery(
        id="homegoods-id-001",
        query="Iron Door Back Hook",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-f352d14300c4": 2},
    ),
    GoldenQuery(
        id="homegoods-id-002",
        query="Multifunctional Car Steering Wheel Table Tray Table",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-845dc7a8d054": 2},
    ),
    GoldenQuery(
        id="homegoods-id-003",
        query="YITAHOME High Gloss Coffee Table",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-315b59012542": 2},
    ),
    GoldenQuery(
        id="homegoods-id-004",
        query="VEVOR Floor Cutter 13 Inch",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-5ecab9ca2392": 2},
    ),
    GoldenQuery(
        id="homegoods-id-005",
        query="SONGMICS 10 Tier Shoe Rack 11 X 17.7 inches",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-13b74eb4dbcc": 2},
        note=(
            "Needed the exact dimensions to disambiguate - 'SONGMICS 10 Tier Shoe "
            "Rack' alone top-1s a near-identical *different* SONGMICS listing "
            "(same product family, different size/price), the catalog's duplicate "
            "problem showing up directly during construction."
        ),
    ),
    GoldenQuery(
        id="homegoods-id-006",
        query="Tribesigns Farmhouse Kitchen Table Round wooden texture",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-47f81b4f8911": 2},
    ),
    GoldenQuery(
        id="homegoods-id-007",
        query="Mjkone Folding Sofa Bed Couch",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-da892eff7529": 2},
    ),
    GoldenQuery(
        id="homegoods-id-008",
        query="Member's Mark Hotel Premier Collection Washcloth",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-b2a1c17e0cc6": 2},
    ),
    GoldenQuery(
        id="homegoods-id-009",
        query="Nestfair Queen Size Upholstered Platform Bed with support legs",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-8133172b51d7": 2},
    ),
    GoldenQuery(
        id="homegoods-id-010",
        query="Wooden Full Size Platform Bed With Headboard And Footrest",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-5877ccfb3e14": 2},
    ),
    GoldenQuery(
        id="homegoods-id-011",
        query="King Size Bed Frame Charging Station LED Black",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-de5034a805c2": 2},
    ),
    GoldenQuery(
        id="homegoods-id-012",
        query="VEVOR Bar Clamps For Woodworking",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-2437b618c97f": 2},
    ),
    GoldenQuery(
        id="homegoods-id-013",
        query="Juilist Juicer Machines Wide Mouth Juicer",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-420c12b6ebca": 2},
    ),
    GoldenQuery(
        id="homegoods-id-014",
        query="Shoe Bench With Cushion 12 Cubbies White",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-9ee3e6323975": 2},
    ),
    GoldenQuery(
        id="homegoods-id-015",
        query="3 in 1 Evaporative Air Cooler Portable",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-68e732cf8bf9": 2},
    ),
    GoldenQuery(
        id="homegoods-id-016",
        query="White Coffee Table Clear Coffee Table Modern Side Center Tables",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-ff17cc64ef0a": 2},
    ),
    GoldenQuery(
        id="homegoods-id-017",
        query="Unbeatablesale Cordless Cellular Shade Cream 45 X 64",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-e3d543af6349": 2},
        note=(
            "Needed colour+size to disambiguate from a same-brand Gray/50x64 "
            "variant - without them the two genuinely tie at the top."
        ),
    ),
    GoldenQuery(
        id="homegoods-id-018",
        query="Knee Brace With Side Stabilizers Patella Gel Pads",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-d41373c97bd8": 2},
    ),
    GoldenQuery(
        id="homegoods-id-019",
        query="Wooden Pattern Trash Can",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-872b583aa8df": 2},
    ),
    GoldenQuery(
        id="homegoods-id-020",
        query="Personalized Silver Sequin Tablecloth",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-7bdb628bd16b": 2},
    ),
    # --- attribute: product-type + material/use constraints (no brand/category
    # field is usable here - see module docstring) -------------------------------
    GoldenQuery(
        id="homegoods-attr-001",
        query="waterproof sofa cover",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "shein-79d159c37cfd",
                "shein-3d17410271b4",
                "shein-850a2a6e1dc1",
                "shein-64ac44d7a36f",
                "shein-dcfe6f6e7db8",
                "shein-b1810279d1ac",
                "shein-c3bd50a8b270",
                "shein-9dc2264f7e73",
            )
        },
        note="All 8 top hits are waterproof sofa/cushion covers.",
    ),
    GoldenQuery(
        id="homegoods-attr-002",
        query="solar powered garden light",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "shein-a3d971a69bf9",
                "shein-e68f5709ac5b",
                "shein-5e9b2916cd39",
                "shein-6b94c0ace581",
                "shein-2900b6c292e6",
                "shein-84e1b09155e0",
                "shein-563ea9fa37cf",
                "shein-949e61adcd31",
            )
        },
        note="All 8 top hits are solar-powered outdoor/garden lights.",
    ),
    GoldenQuery(
        id="homegoods-attr-003",
        query="platform bed with storage",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "shein-c0545af6533b",
                "shein-00f8f42f33dc",
                "shein-ce407e36075a",
                "shein-8f97182fe4d3",
                "shein-de5034a805c2",
                "shein-26567ccedda4",
                "shein-ee0521f065e4",
                "shein-5877ccfb3e14",
            )
        },
        note="All 8 top hits are platform bed frames with built-in storage.",
    ),
    GoldenQuery(
        id="homegoods-attr-004",
        query="microfiber cooling towel",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            "shein-4aa073ed94ec": 2,
            "shein-c1f543081efb": 2,
            "shein-7b7cd6bc831f": 2,
            "shein-0f4cc0c33f24": 2,
            "shein-96994505dfcc": 1,  # a beach towel - microfiber, not marketed as cooling
            "shein-7fa774b7de9e": 1,  # a beach towel
            "shein-10df3bef0de9": 1,  # a beach towel
            "shein-a7bb548a48ef": 1,  # a face-wash towel, not a cooling towel
        },
        note="4 explicitly 'cooling' towels at 2; 4 other microfiber towels (beach/face) at 1.",
    ),
    GoldenQuery(
        id="homegoods-attr-005",
        query="artificial plant home decor",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "shein-289930e9d8e5",
                "shein-ce7bfb31c6a2",
                "shein-1e692bd95c34",
                "shein-5bb786f2f6e4",
                "shein-481b6986f070",
                "shein-2c44e6d14dfa",
                "shein-01bfa80c9cef",
                "shein-e9ccf6da538e",
            )
        },
        note="All 8 top hits are artificial/faux plants for home decor.",
    ),
    GoldenQuery(
        id="homegoods-attr-006",
        query="duvet cover set",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "shein-40cb994e7d02",
                "shein-9f2e6a8f146f",
                "shein-f243f5a1156c",
                "shein-c4b29112d8c4",
                "shein-16ca1b0ce7cf",
                "shein-a6355d73affc",
                "shein-cf1cdd1e5ad8",
                "shein-e99f86fc8ec8",
            )
        },
        note="All 8 top hits are duvet cover sets.",
    ),
    GoldenQuery(
        id="homegoods-attr-007",
        query="cabinet drawer pull handle",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "shein-523d5dda00ce",
                "shein-7701520b4cf2",
                "shein-13a8d6d830dc",
                "shein-eeba27e9178c",
                "shein-bcaf28b71d65",
                "shein-9c26dd2b0e32",
                "shein-169db6f51dd0",
                "shein-e2379c0c60b3",
            )
        },
        note="All 8 top hits are cabinet/drawer pull handles.",
    ),
    GoldenQuery(
        id="homegoods-attr-008",
        query="bath mat anti slip",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "shein-0ccaabcba7b4",
                "shein-7c836082bdfe",
                "shein-06be7062ec9f",
                "shein-1f3f51aea9e3",
                "shein-4f42516dd142",
                "shein-6dd32fcb829f",
                "shein-23bdb2cc499b",
                "shein-731f48142601",
            )
        },
        note="All 8 top hits are anti-slip bath mats/rugs.",
    ),
    GoldenQuery(
        id="homegoods-attr-009",
        query="wall sticker decal",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "shein-2b9d45c37aa0",
                "shein-c628e665e0d7",
                "shein-93a3aa177d55",
                "shein-aedd96b8cfed",
                "shein-7bc5841e09ff",
                "shein-7255c277f4e7",
                "shein-e37dbdb4619d",
                "shein-b87fdcfb5025",
            )
        },
        note="All 8 top hits are wall stickers/decals.",
    ),
    GoldenQuery(
        id="homegoods-attr-010",
        query="chair slipcover dining room",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "shein-a4ececca98c4",
                "shein-aa16d90ebd8c",
                "shein-67704ed1b532",
                "shein-635579ebd41d",
                "shein-7894f83df9a5",
                "shein-bf2ce35bc19e",
                "shein-a54e68028d86",
                "shein-6f8ac30b5ed8",
            )
        },
        note="All 8 top hits are dining chair slipcovers.",
    ),
    GoldenQuery(
        id="homegoods-attr-011",
        query="coffee table living room",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "shein-74f54438223e",
                "shein-9ce1caf8595d",
                "shein-413ff11b27cb",
                "shein-ff17cc64ef0a",  # also homegoods-id-016's exact target
                "shein-b451358c8d1a",
                "shein-9af4551af181",
                "shein-b0d6d32ae403",
                "shein-6dcbed2d6e17",
            )
        },
        note="All 8 top hits are coffee/centre tables for the living room.",
    ),
    GoldenQuery(
        id="homegoods-attr-012",
        query="pillowcase without filler",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "shein-7818474c5586",
                "shein-298dd1a5ef58",
                "shein-79e96af543a9",
                "shein-7fbc01b3ba98",
                "shein-d11d74ae013c",
                "shein-781c34c9acce",
                "shein-9e05dd506f22",
                "shein-585321488af0",
            )
        },
        note="All 8 top hits are pillowcases sold without a filler/insert.",
    ),
    GoldenQuery(
        id="homegoods-attr-013",
        query="storage bench shoe rack",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "shein-9ee3e6323975",  # also homegoods-id-014's exact target
                "shein-9344bdc30d0c",
                "shein-847e5e8c3af5",
                "shein-4478e35d9dd8",
                "shein-b97128f0462a",
                "shein-bec6fe3cf65b",
                "shein-587fe49e8a21",
                "shein-219fe91ed3b5",
            )
        },
        note="All 8 top hits are shoe storage racks/organizers.",
    ),
    GoldenQuery(
        id="homegoods-attr-014",
        query="kitchen table mat coaster",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "shein-09dd31a1c37b",
                "shein-1b150963c92e",
                "shein-eaab78b67f39",
                "shein-30c1d3570171",
                "shein-d239323fb5a7",
                "shein-1966ebbcc07a",
                "shein-0edae24a1683",
                "shein-5c87d0a74bd0",
            )
        },
        note="All 8 top hits are table mats/coasters.",
    ),
    GoldenQuery(
        id="homegoods-attr-015",
        query="canvas poster wall art no frame",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "shein-209e271c1592",
                "shein-92bef1fbbbc3",
                "shein-30d840f36808",
                "shein-f46a707505cf",
                "shein-73b822e068cd",
                "shein-ac73bb5f4180",
                "shein-575accc15ec3",
                "shein-4fbbbcc8c100",
            )
        },
        note="All 8 top hits are unframed canvas posters/wall art.",
    ),
    GoldenQuery(
        id="homegoods-attr-016",
        query="solar led string lights outdoor",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "shein-3cec56a04eac",
                "shein-21f880dc33a4",
                "shein-1917438cc640",
                "shein-f4ca64f44100",
                "shein-eeb6ffc10790",
                "shein-e68f5709ac5b",
                "shein-4a887cb2ff24",
            )
        },
        note="7 of the top 8 hits are solar/LED outdoor string lights.",
    ),
    GoldenQuery(
        id="homegoods-attr-017",
        query="food storage bag silicone",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            "shein-c21035d8ad7e": 2,
            "shein-eb9f57bf3230": 2,
            "shein-1f42d7f30642": 2,
            "shein-ee2b66e317f3": 1,  # sealing clips, not the storage bag itself
            "shein-d5f7c7f54e5d": 1,  # an insulated lunch bag, not a silicone storage bag
            "shein-d3470e90ae93": 1,
            "shein-92db7d929c37": 1,  # a fridge organizer box, different product
            "shein-127e6e01712e": 1,
        },
        note="3 genuine silicone reusable food storage bags at 2; 5 adjacent food-storage products at 1.",
    ),
    GoldenQuery(
        id="homegoods-attr-018",
        query="carpet rug for living room",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "shein-b63cb8aca2f4",
                "shein-80655df947f0",
                "shein-99f4d23d4555",
                "shein-6e02c354f5d0",
                "shein-5cf4dc3679c1",
                "shein-20f1a6b466d1",
                "shein-741683b7de29",
                "shein-1c834b3b4f81",
            )
        },
        note="All 8 top hits are living-room carpets/rugs.",
    ),
    GoldenQuery(
        id="homegoods-attr-019",
        query="smart body fat scale",
        query_class=QueryClass.ATTRIBUTE,
        judgments={"shein-cc1ebb566047": 2, "shein-c5e9b3884848": 2},
        note=(
            "Two genuinely different real listings for the same kind of product "
            "(digital body-fat/bathroom scale) - unlike the exact-duplicate cases "
            "moved out of the identifier set, these have meaningfully different "
            "titles, so both are judged relevant rather than picked as one "
            "'correct' target."
        ),
    ),
    GoldenQuery(
        id="homegoods-attr-020",
        query="GINRGINR LED nightstand table",
        query_class=QueryClass.ATTRIBUTE,
        judgments={"shein-b81ab802bd9e": 2, "shein-12e25fffbc19": 2},
        note=(
            "The exact-duplicate case flagged in this module's docstring: same "
            "brand and near-identical description, different SKU and price - Day "
            "2's ~14% duplicate-rate finding for this catalog, encountered directly "
            "while building this set rather than just read off a report."
        ),
    ),
    # --- exploratory: occasion/intent, no lexical overlap with the target product --
    GoldenQuery(
        id="homegoods-exp-001",
        query="something to make my bedroom cozy for winter",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            sku: 2
            for sku in (
                "shein-cfbdca00cbbd",
                "shein-bc18106e2bc8",
                "shein-d54bf07f8d37",
                "shein-f2eff42dff9d",
                "shein-5a1f29d89930",
                "shein-1e5f44b39b28",
            )
        },
        note="Every top-6 hit is a cozy blanket/throw explicitly marketed for warmth.",
    ),
    GoldenQuery(
        id="homegoods-exp-002",
        query="gift for someone who loves gardening",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "shein-842491e6953c": 2,
            "shein-0eb5b01f7342": 2,
            "shein-57db88096bb3": 2,
            "shein-9ca4a94ef13e": 2,
        },
        note=(
            "4 genuine garden-decor ornaments at 2; a generic 'grow up' desk plaque "
            "and a floral-print tapestry (not garden-specific) in the same top-6 "
            "were omitted."
        ),
    ),
    GoldenQuery(
        id="homegoods-exp-003",
        query="way to organize a messy closet",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            sku: 2
            for sku in (
                "shein-5ef43fb6b146",
                "shein-132eec41f578",
                "shein-3b9b6ee45077",
                "shein-3da42e20bee3",
                "shein-70782e6f25dc",
                "shein-07fb199e346b",
            )
        },
        note="Every top-6 hit is a genuine closet/wardrobe organizer.",
    ),
    GoldenQuery(
        id="homegoods-exp-004",
        query="decor for a kid birthday party",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            sku: 2
            for sku in (
                "shein-ddd2cd9795bd",
                "shein-2fadc3464ba8",
                "shein-49a2e3505f27",
                "shein-db53911b12ef",
                "shein-3261ccbe5c57",
                "shein-32da07178c41",
            )
        },
        note="Every top-6 hit is a genuine party decoration item.",
    ),
    GoldenQuery(
        id="homegoods-exp-005",
        query="something to keep my kitchen counters clean",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "shein-9f61488d13d8": 2,  # stove/counter gap cover, explicit
            "shein-239c6a56623c": 2,  # stove/counter gap cover, explicit
            "shein-ba3df77cbf72": 1,  # kitchen organizer - tidiness, not counter-cleaning specifically
            "shein-a1ffbfe6bb5a": 1,
            "shein-0c10a001519e": 1,
            "shein-c481e32c7687": 1,
        },
        note="2 counter-gap covers directly on-target at 2; 4 broader kitchen-tidiness tools at 1.",
    ),
    GoldenQuery(
        id="homegoods-exp-006",
        query="way to stay cool during a hot summer",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            sku: 2
            for sku in (
                "shein-7c8dc9cd6aa3",
                "shein-0f4cc0c33f24",
                "shein-07644ea37afa",
                "shein-c6be28d85a5b",
                "shein-611f90bf8fe4",
                "shein-f814cd241102",
            )
        },
        note="Every top-6 hit is a genuine cooling product (neck wrap, fan, cooling comforter/towels, sunshade).",
    ),
    GoldenQuery(
        id="homegoods-exp-007",
        query="something for a relaxing bath",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "shein-50b69f15d051": 2,  # bathrobe
            "shein-421c660d0094": 2,  # bath towel + turban set
            "shein-8f51d6b03b84": 2,  # wearable bath towel/shower wrap
            "shein-d60197f3d9b9": 1,  # a pregnancy/postpartum sitting bath tub - specific use case
            "shein-4e00cb16a15e": 1,  # a foot-soaking basin - bath-adjacent, narrower
        },
        note="3 general bath-relaxation textiles at 2; 2 narrower special-purpose bath products at 1.",
    ),
    GoldenQuery(
        id="homegoods-exp-008",
        query="way to protect furniture from pets",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            sku: 2
            for sku in (
                "shein-ae2c240d0f74",
                "shein-27648d5ea922",
                "shein-4c1957e6fc98",
                "shein-2496fb47489b",
                "shein-2ab895ca7efe",
                "shein-73c6b86b50d4",
            )
        },
        note="Every top-6 hit is a pet-scratch guard or a sofa/chair cover explicitly marketed for pet protection.",
    ),
    GoldenQuery(
        id="homegoods-exp-009",
        query="something to brighten up a dark room",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            sku: 2
            for sku in (
                "shein-98a0e92ef749",
                "shein-a57121ce2312",
                "shein-d18a4c7bd738",
                "shein-b13fd4b87a5b",
                "shein-eb23a2157672",
                "shein-a0b2464a4358",
            )
        },
        note="Every top-6 hit is a decorative light/lamp.",
    ),
    GoldenQuery(
        id="homegoods-exp-010",
        query="decor for a bohemian style bedroom",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            sku: 2
            for sku in (
                "shein-1a1d2c8d1865",
                "shein-5fbc2bf8ec39",
                "shein-d5418712034d",
                "shein-21b806d9ffc5",
                "shein-e7d196e2f795",
                "shein-044eaa6a83b9",
            )
        },
        note="Every top-6 hit is explicitly labelled 'Bohemian style' bedroom decor.",
    ),
    GoldenQuery(
        id="homegoods-exp-011",
        query="way to keep bugs out of the house",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            sku: 2
            for sku in (
                "shein-208ac41c020b",
                "shein-f1520481ecf8",
                "shein-73f215c38e83",
                "shein-fccf3d398980",
                "shein-7e63dd7f3c66",
                "shein-bedcf85bd3a7",
            )
        },
        note="Every top-6 hit is an insect trap or mosquito/insect screen.",
    ),
    GoldenQuery(
        id="homegoods-exp-012",
        query="something for a rainy day outdoors",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "shein-b8c9240a8833": 1,
            "shein-631ef9a1d3be": 1,
            "shein-5101d4a48af7": 1,
            "shein-a3c731d80347": 1,
            "shein-ef48000e26a3": 1,
            "shein-dbd9c2c033c8": 1,
        },
        note=(
            "All 6 top hits are waterproof outdoor gear (cushions, mats, shoe "
            "covers) - genuinely water-resistant, but none is specific rain "
            "protection the way an actual raincoat (present elsewhere in this "
            "catalog) would be, so graded 1 rather than 2 across the board - an "
            "honest, moderate match rather than an inflated one."
        ),
    ),
    GoldenQuery(
        id="homegoods-exp-013",
        query="something to help me sleep better at night",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "shein-d4817cfbe6d6": 2,  # sleep-aid light / white noise machine
            "shein-d1d2c62632f9": 2,  # cooling sleep eye mask
            "shein-599cb59b817c": 1,  # plain eye mask - simpler, less sleep-aid-specific
            "shein-554bc7299a73": 1,  # a bedside night light - only tangentially sleep-aid
        },
        note=(
            "A leg pillow (comfort, not sleep-specific) and a novelty 'game block "
            "lamp' torch (a toy, not a sleep aid) in the same top-6 were omitted."
        ),
    ),
    GoldenQuery(
        id="homegoods-exp-014",
        query="decor for a farmhouse style kitchen",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            sku: 2
            for sku in (
                "shein-e97c648bd0b3",
                "shein-cc64501fcf83",
                "shein-c188f36013b7",
                "shein-bf96b605e2e0",
                "shein-92f49ba51f4a",
                "shein-269ddc900092",
            )
        },
        note="Every top-6 hit is explicitly labelled farmhouse/rustic/country-style kitchen decor.",
    ),
    GoldenQuery(
        id="homegoods-exp-015",
        query="way to add storage to a small bedroom",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "shein-072479e88e2e": 2,  # bedside storage cabinet
            "shein-3c46ba5debba": 2,  # bedroom storage table
            "shein-69a52d9edf20": 2,  # narrow dresser, explicitly space-saving
            "shein-872bc505616f": 1,  # bed-bottom storage - shoe-specific, narrower
            "shein-68697a9a81a5": 1,  # a desk storage rack - desk, not bedroom-general
            "shein-814b662df6d0": 1,  # a shoe rack - narrower than general bedroom storage
        },
        note="3 general bedroom storage furniture pieces at 2; 3 narrower/adjacent storage items at 1.",
    ),
    GoldenQuery(
        id="homegoods-exp-016",
        query="something festive for Christmas decoration",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "shein-ef41c8741058": 2,
            "shein-de904fc68d09": 2,
            "shein-0d16df6cdccc": 2,
            "shein-38f3f847b3dd": 2,
        },
        note=(
            "4 genuine Christmas items at 2; a generic festive vase (wedding/party, "
            "not Christmas-specific) and a Halloween decoration light (wrong "
            "holiday entirely - keyword overlap on 'decoration') in the same top-6 "
            "were omitted."
        ),
    ),
]
