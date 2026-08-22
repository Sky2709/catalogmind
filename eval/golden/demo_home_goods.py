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
        judgments={"shein-ae12db356467": 2},
    ),
    GoldenQuery(
        id="homegoods-id-002",
        query="Multifunctional Car Steering Wheel Table Tray Table",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-5878aaafb1d0": 2},
    ),
    GoldenQuery(
        id="homegoods-id-003",
        query="YITAHOME High Gloss Coffee Table",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-131d32f6359f": 2},
    ),
    GoldenQuery(
        id="homegoods-id-004",
        query="VEVOR Floor Cutter 13 Inch",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-1fd5f14c6545": 2},
    ),
    GoldenQuery(
        id="homegoods-id-005",
        query="SONGMICS 10 Tier Shoe Rack 11 X 17.7 inches",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-8c8054c63e5a": 2},
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
        judgments={"shein-cdfa044edd52": 2},
    ),
    GoldenQuery(
        id="homegoods-id-007",
        query="Mjkone Folding Sofa Bed Couch",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-0186483437b0": 2},
    ),
    GoldenQuery(
        id="homegoods-id-008",
        query="Member's Mark Hotel Premier Collection Washcloth",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-0bd3e25aa06a": 2},
    ),
    GoldenQuery(
        id="homegoods-id-009",
        query="Nestfair Queen Size Upholstered Platform Bed with support legs",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-a2f986ca6d1b": 2},
    ),
    GoldenQuery(
        id="homegoods-id-010",
        query="Wooden Full Size Platform Bed With Headboard And Footrest",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-4488e595dfb1": 2},
    ),
    GoldenQuery(
        id="homegoods-id-011",
        query="King Size Bed Frame Charging Station LED Black",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-f5404711f2f5": 2},
    ),
    GoldenQuery(
        id="homegoods-id-012",
        query="VEVOR Bar Clamps For Woodworking",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-26e11ef8b4c8": 2},
    ),
    GoldenQuery(
        id="homegoods-id-013",
        query="Juilist Juicer Machines Wide Mouth Juicer",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-7c7bb38a563d": 2},
    ),
    GoldenQuery(
        id="homegoods-id-014",
        query="Shoe Bench With Cushion 12 Cubbies White",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-d2248f1bbc11": 2},
    ),
    GoldenQuery(
        id="homegoods-id-015",
        query="3 in 1 Evaporative Air Cooler Portable",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-197344b5cb58": 2},
    ),
    GoldenQuery(
        id="homegoods-id-016",
        query="White Coffee Table Clear Coffee Table Modern Side Center Tables",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-5946e305aff9": 2},
    ),
    GoldenQuery(
        id="homegoods-id-017",
        query="Unbeatablesale Cordless Cellular Shade Cream 45 X 64",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-76737dc39918": 2},
        note=(
            "Needed colour+size to disambiguate from a same-brand Gray/50x64 "
            "variant - without them the two genuinely tie at the top."
        ),
    ),
    GoldenQuery(
        id="homegoods-id-018",
        query="Knee Brace With Side Stabilizers Patella Gel Pads",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-076c69686a3d": 2},
    ),
    GoldenQuery(
        id="homegoods-id-019",
        query="Wooden Pattern Trash Can",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-6959ff590eaa": 2},
    ),
    GoldenQuery(
        id="homegoods-id-020",
        query="Personalized Silver Sequin Tablecloth",
        query_class=QueryClass.IDENTIFIER,
        judgments={"shein-f01c09e91bd1": 2},
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
                "shein-41fafa9bf6ee",
                "shein-c9d61e09383d",
                "shein-49ebfdac77a9",
                "shein-ac55343fb1cd",
                "shein-d6eab134575c",
                "shein-adfd2fd636e1",
                "shein-f1ed0b0226e8",
                "shein-f95c7eb095a5",
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
                "shein-35e4318cb65c",
                "shein-b4c34f6a97bd",
                "shein-94fbb8cc342e",
                "shein-b653ebad3e00",
                "shein-fdeb87f6c50c",
                "shein-cfaa6add870d",
                "shein-9147c41f4098",
                "shein-6d975a155ce8",
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
                "shein-4879635eb74c",
                "shein-3a738249f2db",
                "shein-bdc7134ff599",
                "shein-f13ea658f58d",
                "shein-f5404711f2f5",
                "shein-75b9daef4e73",
                "shein-5c8f7a81babe",
                "shein-4488e595dfb1",
            )
        },
        note="All 8 top hits are platform bed frames with built-in storage.",
    ),
    GoldenQuery(
        id="homegoods-attr-004",
        query="microfiber cooling towel",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            "shein-39907ca63f3b": 2,
            "shein-574644eaaf75": 2,
            "shein-2c8f0a188a1f": 2,
            "shein-df3cf78a6227": 2,
            "shein-1e1adbcee04b": 1,  # a beach towel - microfiber, not marketed as cooling
            "shein-3f69fb7fdf9d": 1,  # a beach towel
            "shein-4373e73ad57d": 1,  # a beach towel
            "shein-f0f5c4f1e6ff": 1,  # a face-wash towel, not a cooling towel
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
                "shein-d3a81f8f86f4",
                "shein-84ebb8ef1e44",
                "shein-167794ea8a31",
                "shein-27e4c0eae2bb",
                "shein-8224460dc6fd",
                "shein-7584c2df056f",
                "shein-f30c3eda50a3",
                "shein-1838c727a942",
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
                "shein-002ce42e8d82",
                "shein-2f6c3bdfc41a",
                "shein-c94f22348ec2",
                "shein-9eb1ae6252c6",
                "shein-76bf4ffca119",
                "shein-44a36b1e5ba5",
                "shein-34610cfcb978",
                "shein-7ff70e0188cd",
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
                "shein-9dacc75e47ba",
                "shein-ef91cdf64475",
                "shein-c824ebf85f8e",
                "shein-f4196ab1e7db",
                "shein-72d1f2988e80",
                "shein-8d4a2663fc7e",
                "shein-c3f38453fd3a",
                "shein-c8bd69dd2e49",
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
                "shein-cdf08c7f2d82",
                "shein-6f37978adbde",
                "shein-b6e9c970eda7",
                "shein-700d77ccab5f",
                "shein-5b64d6457a7c",
                "shein-607abe8e5c59",
                "shein-c3b2a8df1a77",
                "shein-d3d925ef39a5",
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
                "shein-833757d7b309",
                "shein-2d533f225acc",
                "shein-1a84733deb01",
                "shein-696a3e89575a",
                "shein-5135cc4789b0",
                "shein-d9574e4985b4",
                "shein-f374ceb56deb",
                "shein-582c76a91191",
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
                "shein-42f136c9984a",
                "shein-a551b03a0b3e",
                "shein-ea4b0cd587c1",
                "shein-eb7e887f071e",
                "shein-73024e45fd44",
                "shein-f17c783cf18f",
                "shein-d1d0e30daead",
                "shein-f19f26bc6534",
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
                "shein-c99cbf4003b0",
                "shein-bf48ef90c0b0",
                "shein-4623036c0013",
                "shein-5946e305aff9",  # also homegoods-id-016's exact target
                "shein-28eb8dbd894b",
                "shein-f8827e931c7f",
                "shein-b582431e2d18",
                "shein-705c182f6aaa",
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
                "shein-bb1cdab11b14",
                "shein-79a30ad81315",
                "shein-952df8f2bdcb",
                "shein-40bc37094c3f",
                "shein-90b11b7e445f",
                "shein-2ea6cc9b8f6d",
                "shein-1a120b449eeb",
                "shein-12372fa99a96",
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
                "shein-d2248f1bbc11",  # also homegoods-id-014's exact target
                "shein-146e125b350e",
                "shein-fa25643ffe13",
                "shein-792876cab4d1",
                "shein-e3931b0179c4",
                "shein-3c363ec90c45",
                "shein-f546ac546327",
                "shein-6b1882a2c91e",
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
                "shein-2f55713ebd81",
                "shein-e2751dc39882",
                "shein-e84ac0779273",
                "shein-d88a38b5fd42",
                "shein-f54cc4c80a2b",
                "shein-8350995701ae",
                "shein-80f8e0d0fe25",
                "shein-c294ded2d156",
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
                "shein-2e9362fffc84",
                "shein-08177c3ec663",
                "shein-1ed4c3ba32a1",
                "shein-89d0492342fc",
                "shein-53469806764c",
                "shein-6ff3d06f7aa3",
                "shein-50bce09d004a",
                "shein-7c76d9cb46ec",
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
                "shein-0844d0dfb576",
                "shein-3db5f3aca525",
                "shein-ea630cb99ad9",
                "shein-09c1c4876a81",
                "shein-17418d0d396a",
                "shein-b4c34f6a97bd",
                "shein-ef36d9818a04",
            )
        },
        note="7 of the top 8 hits are solar/LED outdoor string lights.",
    ),
    GoldenQuery(
        id="homegoods-attr-017",
        query="food storage bag silicone",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            "shein-1cea35dbc971": 2,
            "shein-0041d5f44416": 2,
            "shein-3824e4e723bc": 2,
            "shein-7b358da6bb77": 1,  # sealing clips, not the storage bag itself
            "shein-9306ced3dc41": 1,  # an insulated lunch bag, not a silicone storage bag
            "shein-a594c6f6b3c1": 1,
            "shein-0285fd0ac501": 1,  # a fridge organizer box, different product
            "shein-14f640d748dc": 1,
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
                "shein-d05b58cbd721",
                "shein-f22e0d95f922",
                "shein-1c53d5d9ec0c",
                "shein-9b33c99d6881",
                "shein-433402766c35",
                "shein-8a899b2bdc5e",
                "shein-27c72a6bde21",
                "shein-92187dc52ae1",
            )
        },
        note="All 8 top hits are living-room carpets/rugs.",
    ),
    GoldenQuery(
        id="homegoods-attr-019",
        query="smart body fat scale",
        query_class=QueryClass.ATTRIBUTE,
        judgments={"shein-d9be7fea6911": 2, "shein-62e15068a938": 2},
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
        judgments={"shein-4c3d3e7a9172": 2, "shein-2b29aae760f6": 2},
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
                "shein-e4de7ed089ad",
                "shein-e936183d781d",
                "shein-a4f2ecf8b0af",
                "shein-ad6707f9128f",
                "shein-74d70ccf7a72",
                "shein-e70c56a68bbb",
            )
        },
        note="Every top-6 hit is a cozy blanket/throw explicitly marketed for warmth.",
    ),
    GoldenQuery(
        id="homegoods-exp-002",
        query="gift for someone who loves gardening",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "shein-8285a9b39c49": 2,
            "shein-36ae81a34b77": 2,
            "shein-bc497d65c037": 2,
            "shein-cc82711c7e5a": 2,
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
                "shein-0ab8fc2b458f",
                "shein-ca9c15a54143",
                "shein-8ae4ef5545be",
                "shein-64f872bfb7a8",
                "shein-3f021ad9452b",
                "shein-0c7e71826bb1",
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
                "shein-c7f2c7213b75",
                "shein-a01c4cc84485",
                "shein-52404b21c7a3",
                "shein-7352fd316353",
                "shein-30cb0726eb8a",
                "shein-b8324217ce13",
            )
        },
        note="Every top-6 hit is a genuine party decoration item.",
    ),
    GoldenQuery(
        id="homegoods-exp-005",
        query="something to keep my kitchen counters clean",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "shein-3dcb1ce3aab3": 2,  # stove/counter gap cover, explicit
            "shein-2b0c1d8c7848": 2,  # stove/counter gap cover, explicit
            "shein-ed6d2ec47bc3": 1,  # kitchen organizer - tidiness, not counter-cleaning specifically
            "shein-ca3884a83539": 1,
            "shein-f771809b179f": 1,
            "shein-85abe50763e3": 1,
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
                "shein-4b645c66faad",
                "shein-df3cf78a6227",
                "shein-509e609dc6ec",
                "shein-832184ea8092",
                "shein-dfc60a1e0da4",
                "shein-a9d9cd3ad58f",
            )
        },
        note="Every top-6 hit is a genuine cooling product (neck wrap, fan, cooling comforter/towels, sunshade).",
    ),
    GoldenQuery(
        id="homegoods-exp-007",
        query="something for a relaxing bath",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "shein-823a67cb20aa": 2,  # bathrobe
            "shein-91065b2edec2": 2,  # bath towel + turban set
            "shein-89c49006a8da": 2,  # wearable bath towel/shower wrap
            "shein-6ede7ecc5309": 1,  # a pregnancy/postpartum sitting bath tub - specific use case
            "shein-7c9fcc5681b3": 1,  # a foot-soaking basin - bath-adjacent, narrower
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
                "shein-0f5e359cc74c",
                "shein-a05013f78989",
                "shein-773c31fadd78",
                "shein-d42b512035da",
                "shein-e05fb205b734",
                "shein-b9f9928afb50",
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
                "shein-e34990d4d384",
                "shein-b534ee981705",
                "shein-2e51e3b2784b",
                "shein-5c55357e3200",
                "shein-1987c1f08651",
                "shein-14d328082877",
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
                "shein-8d5f3ebd75e0",
                "shein-0edc1da685f7",
                "shein-4bac952a2e2c",
                "shein-c773de87427e",
                "shein-a2a96d334978",
                "shein-64c143de64cf",
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
                "shein-23aa8d072d8a",
                "shein-1aa4b0214f7b",
                "shein-836f11ebe4a3",
                "shein-7832ba37e428",
                "shein-e77eccfe1aee",
                "shein-c2b97504fa31",
            )
        },
        note="Every top-6 hit is an insect trap or mosquito/insect screen.",
    ),
    GoldenQuery(
        id="homegoods-exp-012",
        query="something for a rainy day outdoors",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "shein-6fcfba482989": 1,
            "shein-9d1b3c8b48c2": 1,
            "shein-0d25b203d38a": 1,
            "shein-607061e22e5f": 1,
            "shein-f8346a1c3595": 1,
            "shein-a3b137e31721": 1,
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
            "shein-d98d53b28294": 2,  # sleep-aid light / white noise machine
            "shein-dafc664a8da2": 2,  # cooling sleep eye mask
            "shein-d1a3b472b850": 1,  # plain eye mask - simpler, less sleep-aid-specific
            "shein-b1dcab57e642": 1,  # a bedside night light - only tangentially sleep-aid
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
                "shein-79501d26e7ca",
                "shein-158d921f6bbb",
                "shein-b99a54b7ee69",
                "shein-c8bef0e86353",
                "shein-b91cf8ad2d08",
                "shein-0faa3b5b6b2c",
            )
        },
        note="Every top-6 hit is explicitly labelled farmhouse/rustic/country-style kitchen decor.",
    ),
    GoldenQuery(
        id="homegoods-exp-015",
        query="way to add storage to a small bedroom",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "shein-28c4b3eefd6e": 2,  # bedside storage cabinet
            "shein-e049c0a3e1a8": 2,  # bedroom storage table
            "shein-a61147fd9dfa": 2,  # narrow dresser, explicitly space-saving
            "shein-553e89b2b986": 1,  # bed-bottom storage - shoe-specific, narrower
            "shein-cfe393d89e20": 1,  # a desk storage rack - desk, not bedroom-general
            "shein-f9ac64c4f37e": 1,  # a shoe rack - narrower than general bedroom storage
        },
        note="3 general bedroom storage furniture pieces at 2; 3 narrower/adjacent storage items at 1.",
    ),
    GoldenQuery(
        id="homegoods-exp-016",
        query="something festive for Christmas decoration",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "shein-cc86f2bdd3ea": 2,
            "shein-4f199495b259": 2,
            "shein-23981c36b21e": 2,
            "shein-da05e6241d90": 2,
        },
        note=(
            "4 genuine Christmas items at 2; a generic festive vase (wedding/party, "
            "not Christmas-specific) and a Halloween decoration light (wrong "
            "holiday entirely - keyword overlap on 'decoration') in the same top-6 "
            "were omitted."
        ),
    ),
]
