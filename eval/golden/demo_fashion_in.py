"""Golden queries for `demo-fashion-in` (Myntra Indian fashion catalog).

**Phase A sample — 10 of a planned 60.** Every SKU below was pulled from a real
`collection.query.hybrid()` call against the live tenant, at the alpha
`PRIOR_ALPHA` (`app/retrieval/alpha_router.py`) already assigns that query's class
(0.15 identifier / 0.50 attribute / 0.75 exploratory) - the same code path
`eval/retrieval_eval.py` runs, not a guess about what "should" match. See
`eval/golden/__init__.py`'s module docstring for the verified-top-K-pool limitation
and the 0-2 relevance scale.
"""

from __future__ import annotations

from app.retrieval.base import QueryClass
from eval.golden import GoldenQuery

QUERIES = [
    # --- identifier: a shopper who already knows a product number -----------------
    GoldenQuery(
        id="fashion-id-001",
        query="10015819",
        query_class=QueryClass.IDENTIFIER,
        judgments={"10015819": 2},
        note=(
            "Raymond Men Maroon Slim Fit Formal Shirt. Verified top-1 at alpha=0.15 "
            "even though the next 5 hits are unrelated (perfume, a lehenga) - the "
            "rank-1 case MRR/nDCG@10 are built to reward."
        ),
    ),
    GoldenQuery(
        id="fashion-id-002",
        query="10029129",
        query_class=QueryClass.IDENTIFIER,
        judgments={"10029129": 2},
        note="Geox Women Black Solid Ballerinas.",
    ),
    GoldenQuery(
        id="fashion-id-003",
        query="10203335",
        query_class=QueryClass.IDENTIFIER,
        judgments={"10203335": 2},
        note="HERE&NOW Women Black Solid Sweatshirt.",
    ),
    # --- attribute: brand + garment-type constraint, both signals contribute -------
    GoldenQuery(
        id="fashion-attr-001",
        query="Indian Terrain men slim fit casual shirt",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "10155463",
                "10153137",
                "10154867",
                "10154653",
                "10152795",
                "10155391",
                "10152489",
                "10153323",
                "10155661",
                "10153495",
                "10153051",
                "10153445",
                "10152973",
                "10151463",
                "10153901",
            )
        },
        note="All 15 top hits at alpha=0.5 are genuinely Indian Terrain slim-fit casual shirts.",
    ),
    GoldenQuery(
        id="fashion-attr-002",
        query="Raymond formal shirt",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "10018845",
                "10018857",
                "10015839",
                "10237323",
                "10015819",  # also the exact target of fashion-id-001 - legitimate reuse
                "10015831",
                "10015805",
                "10018841",
                "10015829",
                "10018839",
                "10254679",
                "10237167",
                "10254565",
                "10254589",
                "10254587",
            )
        },
        note="All 15 top hits at alpha=0.5 are Raymond formal shirts.",
    ),
    GoldenQuery(
        id="fashion-attr-003",
        query="office ke liye formal shirt mard",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            "10028413": 2,
            "10030219": 2,
            "10029543": 2,
            "10028269": 2,
            "10028121": 2,
            "10028367": 2,
        },
        note=(
            "Hinglish. Park Avenue / Next Look men's formal shirts - English content "
            "words ('formal shirt') carry BM25 signal fine even Hindi-mixed."
        ),
    ),
    GoldenQuery(
        id="fashion-attr-004",
        query="shaadi ke liye kurta women",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            "10016727": 2,
            "10013041": 2,
            "10222897": 2,
            "10076055": 2,
            "10266547": 2,
            "10266059": 2,
        },
        note="Hinglish. Vishudh/Sera/Ahalyaa/ZIYAA/Ishin/Jompers women's kurtas.",
    ),
    # --- exploratory: occasion/intent, no lexical overlap with the target garment --
    GoldenQuery(
        id="fashion-exp-001",
        query="cozy warm outfit for a chilly evening",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "10002455": 2,  # GAS Pullover
            "10244301": 2,  # GAP Girls Puffer Jacket
            "10032801": 2,  # HERE&NOW Sweater
            "10030681": 1,  # HRX "Lifestyle Sweatshirt" - warm, less precisely "evening"
        },
        note=(
            "Two other top-6 hits (a raw dress-material fabric, workout tights) were "
            "judged not relevant and omitted - live verification catching a "
            "plausible-looking-but-wrong hit before it entered the set."
        ),
    ),
    GoldenQuery(
        id="fashion-exp-002",
        query="elegant outfit for a wedding reception",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "10138179": 2,  # Libas ethnic jacket dress
            "10260567": 2,  # MISH maxi dress
            "10197661": 2,  # EthnoVogue gown
            "10194297": 1,  # Madame embellished top - dressy, but "a top" != "an outfit"
        },
        note=(
            "A Swarovski necklace and the same raw dress-material fabric were also in "
            "the top-6 and were omitted - an accessory and a fabric are not outfits."
        ),
    ),
    GoldenQuery(
        id="fashion-exp-003",
        query="comfortable footwear for daily wear",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "10262305": 2,
            "10253197": 2,
            "10014975": 2,
            "10253199": 2,
            "10137177": 2,
        },
        note=(
            "4 Puma sneakers + 1 Crew STREET sneaker. A pair of GAP khakis was omitted. "
            "Labelled EXPLORATORY by design (no lexical overlap with 'sneakers'/'shoes'), "
            "but the live lexical classifier has no cue to catch that and assigns "
            "ATTRIBUTE instead (alpha=0.5) - which drops this specific query's nDCG@10 "
            "to 0.0, since the relevant items only surface at the higher, vector-heavy "
            "alpha exploratory queries actually need. A genuine hard case for a "
            "surface-lexical classifier, not a labelling mistake - kept in the set "
            "rather than dropped, since a sweep that only contains easy cases would "
            "overstate how well any fixed or dynamic alpha choice actually does."
        ),
    ),
    # ================================================================================
    # Phase B: scaled to the full set. Same methodology as Phase A above - every
    # judgment below came from an actual live `hybrid()`/`fetch_objects()` call
    # against `demo-fashion-in` this session (see `scripts/build_golden_sets.py`),
    # not invented. Identifier SKUs were pulled from a diverse, paginated sample of
    # ~150 real rows spanning brand/gender/category, specifically to avoid the first
    # 10 rows' bias toward whatever category happens to sort first.
    # ================================================================================
    # --- identifier: 17 more real, diverse SKUs ------------------------------------
    GoldenQuery(
        id="fashion-id-004",
        query="10052311",
        query_class=QueryClass.IDENTIFIER,
        judgments={"10052311": 2},
        note="Titan Women Maroon Analogue Watch 95062WD01.",
    ),
    GoldenQuery(
        id="fashion-id-005",
        query="10176789",
        query_class=QueryClass.IDENTIFIER,
        judgments={"10176789": 2},
        note="Tommy Hilfiger Men Pack of 3 Trunks.",
    ),
    GoldenQuery(
        id="fashion-id-006",
        query="10233965",
        query_class=QueryClass.IDENTIFIER,
        judgments={"10233965": 2},
        note="ADIDAS Men Blue ENERGYFALCON Running Shoes.",
    ),
    GoldenQuery(
        id="fashion-id-007",
        query="10268221",
        query_class=QueryClass.IDENTIFIER,
        judgments={"10268221": 2},
        note="Calvin Klein Jeans Men Black Solid Round Neck T-shirt.",
    ),
    GoldenQuery(
        id="fashion-id-008",
        query="10071599",
        query_class=QueryClass.IDENTIFIER,
        judgments={"10071599": 2},
        note="Kenneth Cole Men Brown Leather Formal Brogues.",
    ),
    GoldenQuery(
        id="fashion-id-009",
        query="10197757",
        query_class=QueryClass.IDENTIFIER,
        judgments={"10197757": 2},
        note="CALFNERO Men Black Genuine Leather Solid Messenger Bag.",
    ),
    GoldenQuery(
        id="fashion-id-010",
        query="10159509",
        query_class=QueryClass.IDENTIFIER,
        judgments={"10159509": 2},
        note="Eske Black Colourblocked Leather Handheld Bag.",
    ),
    GoldenQuery(
        id="fashion-id-011",
        query="10013491",
        query_class=QueryClass.IDENTIFIER,
        judgments={"10013491": 2},
        note="PARFAIT Plus Size Red Underwired Lightly Padded T-shirt Bra P5391.",
    ),
    GoldenQuery(
        id="fashion-id-012",
        query="10052451",
        query_class=QueryClass.IDENTIFIER,
        judgments={"10052451": 2},
        note="Titan Mechanical Men Blue Analogue watch 90111WL01.",
    ),
    GoldenQuery(
        id="fashion-id-013",
        query="10248671",
        query_class=QueryClass.IDENTIFIER,
        judgments={"10248671": 2},
        note="Puma Women Black & Grey Solid Force IDP Sports Sandals.",
    ),
    GoldenQuery(
        id="fashion-id-014",
        query="10021499",
        query_class=QueryClass.IDENTIFIER,
        judgments={"10021499": 2},
        note="Bitiya by Bhama Girls Red A-Line Dress.",
    ),
    GoldenQuery(
        id="fashion-id-015",
        query="10105877",
        query_class=QueryClass.IDENTIFIER,
        judgments={"10105877": 2},
        note="Genie Women Purple & Coral Pink Geometric Pattern Laptop Backpack with Pouch.",
    ),
    GoldenQuery(
        id="fashion-id-016",
        query="10038141",
        query_class=QueryClass.IDENTIFIER,
        judgments={"10038141": 2},
        note="Infuzze Women Oxidised Silver-Toned Brass-Plated Peacock Shaped Drop Earrings.",
    ),
    GoldenQuery(
        id="fashion-id-017",
        query="10191047",
        query_class=QueryClass.IDENTIFIER,
        judgments={"10191047": 2},
        note="Fastrack Men Navy Blue & Brown Colourblocked Belt.",
    ),
    GoldenQuery(
        id="fashion-id-018",
        query="10080229",
        query_class=QueryClass.IDENTIFIER,
        judgments={"10080229": 2},
        note="Shaily Off-White Printed Cotton Blend Saree.",
    ),
    GoldenQuery(
        id="fashion-id-019",
        query="10245867",
        query_class=QueryClass.IDENTIFIER,
        judgments={"10245867": 2},
        note="Ecko Unltd Men Black Solid Hooded Puffer Jacket.",
    ),
    GoldenQuery(
        id="fashion-id-020",
        query="10196265",
        query_class=QueryClass.IDENTIFIER,
        judgments={"10196265": 2},
        note="CLAY CRAFT White Set of 4 Bowls - Myntra sells homeware too, real row.",
    ),
    # --- attribute: 16 more brand/product-type clusters, 1 more Hinglish ----------
    GoldenQuery(
        id="fashion-attr-005",
        query="UCLA men slim fit checked casual shirt",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "10073485",
                "10073171",
                "10073073",
                "10073415",
                "10073181",
                "10073463",
                "10073119",
                "10073123",
                "10073363",
                "10072995",
            )
        },
        note="All 10 top hits are UCLA slim-fit checked casual shirts.",
    ),
    GoldenQuery(
        id="fashion-attr-006",
        query="GAP girls jeans",
        query_class=QueryClass.ATTRIBUTE,
        judgments={sku: 2 for sku in ("10144961", "10145017", "10144973", "10145033", "10145011")},
        note=(
            "5 of the top 10 hits are genuine GAP girls jeans; a boys-jeans, two tops "
            "and a hoodie sweatshirt in the same top-10 were omitted (same brand, "
            "wrong garment/gender)."
        ),
    ),
    GoldenQuery(
        id="fashion-attr-007",
        query="Park Avenue men formal shirt",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "10028121",
                "10028367",
                "10028157",
                "10027685",
                "10028413",
                "10028215",
                "10028269",
                "10027833",
                "10028217",
                "10028095",
            )
        },
        note="All 10 top hits are Park Avenue formal shirts - a distinct brand from Raymond.",
    ),
    GoldenQuery(
        id="fashion-attr-008",
        query="Indian Terrain men trousers",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "10150015",
                "10148345",
                "10148343",
                "10150641",
                "10150247",
                "10150581",
                "10150241",
                "10148383",
                "10148275",
                "10148931",
            )
        },
        note="All 10 top hits are Indian Terrain trousers - same brand as the shirt queries, different garment.",
    ),
    GoldenQuery(
        id="fashion-attr-009",
        query="Saree mall printed saree",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "10266329",
                "10258127",
                "10173315",
                "10173357",
                "10173367",
                "10173359",
                "10266279",
                "10173325",
                "10258123",
                "10173333",
            )
        },
        note="All 10 top hits are Saree mall printed sarees.",
    ),
    GoldenQuery(
        id="fashion-attr-010",
        query="AURELIA women printed straight kurta",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "10185135",
                "10185033",
                "10185253",
                "10185061",
                "10185055",
                "10185043",
                "10185365",
                "10185311",
                "10185231",
                "10185395",
            )
        },
        note="All 10 top hits are AURELIA printed straight kurtas.",
    ),
    GoldenQuery(
        id="fashion-attr-011",
        query="U.S. Polo Assn Kids boys t-shirt",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            "10213029": 2,
            "10212949": 2,
            "10213093": 2,
            "10212981": 2,
            "10212957": 2,
            "10212987": 2,
            "10212877": 2,
            "10212925": 2,
            "10213001": 2,
            "10213013": 1,  # a sweatshirt, not a t-shirt - same brand, near-miss garment
        },
        note="9 t-shirts at relevance 2; one sweatshirt from the same brand at relevance 1.",
    ),
    GoldenQuery(
        id="fashion-attr-012",
        query="Vishudh women printed kurta with palazzos",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "10073597",
                "10016727",
                "10030591",
                "10073603",
                "10073609",
                "10030593",
                "10016737",
                "10089613",
                "10089243",
                "10016747",
            )
        },
        note="All 10 top hits are Vishudh kurta-with-palazzo sets (one spelled 'Kurti').",
    ),
    GoldenQuery(
        id="fashion-attr-013",
        query="Puma sneakers men",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "10018193",
                "10018075",
                "10018013",
                "10018217",
                "10080309",
                "10075955",
                "10071055",
                "10075931",
                "10018191",
                "10018011",
            )
        },
        note="All 10 top hits are Puma men's sneakers.",
    ),
    GoldenQuery(
        id="fashion-attr-014",
        query="Titan analogue watch",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "10052311",  # also fashion-id-004's exact target - legitimate reuse
                "10052137",
                "10052601",
                "10052169",
                "10052321",
                "10052147",
                "10052015",
                "10052419",
                "10052443",
                "10052201",
            )
        },
        note="All 10 top hits are Titan analogue watches, men's and women's.",
    ),
    GoldenQuery(
        id="fashion-attr-015",
        query="Roadster women stretchable jeans",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "10058275",
                "10058471",
                "10058245",
                "10058315",
                "10058353",
                "10058357",
                "10058181",
                "10058253",
                "10058339",
                "10058205",
            )
        },
        note="All 10 top hits are Roadster women's stretchable jeans.",
    ),
    GoldenQuery(
        id="fashion-attr-016",
        query="Calvin Klein Jeans men shirt",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            "10268147": 2,
            "10267679": 2,
            "10268041": 2,
            "10268467": 2,
            "10267843": 2,
            "10268255": 2,
            "10268715": 1,  # polo T-shirt, not a "shirt" in the queried sense
            "10267767": 1,  # polo T-shirt
            "10267983": 1,  # round-neck T-shirt
            "10268439": 1,  # polo T-shirt
        },
        note="6 casual/corduroy shirts at 2; 4 T-shirts/polos from the same brand at 1.",
    ),
    GoldenQuery(
        id="fashion-attr-017",
        query="garmi ke liye cotton kurta women",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            "10206911": 2,  # explicitly "Pure Cotton"
            "10234851": 2,  # explicitly "Cotton"
            "10172907": 1,
            "10206925": 1,
            "10138175": 1,
            "10172993": 1,
            "10222897": 1,  # silk, not cotton, but still a real kurta match
            "10138055": 1,
            "10197173": 1,
            "10138139": 1,
        },
        note=(
            "Hinglish (garmi=summer). Only 2 of the 10 top hits explicitly say "
            "'cotton' in the title; the rest are real kurtas but the material isn't "
            "confirmed, so graded 1 rather than 2 - the judgment reflects what the "
            "title actually states, not what the query hoped for."
        ),
    ),
    GoldenQuery(
        id="fashion-attr-018",
        query="men's running shoes",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "10029809",
                "10029731",
                "10029797",
                "10029811",
                "10029819",
                "10029745",
                "10029869",
                "10248547",
                "10071803",
                "10233965",
            )
        },
        note=(
            "Not a single-brand cluster (Force 10, Puma, Campus, ADIDAS) - a plain "
            "product-type + gender constraint, deliberately included for diversity "
            "against the mostly brand-anchored attribute queries above."
        ),
    ),
    GoldenQuery(
        id="fashion-attr-019",
        query="Geox women ballerinas",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            "10029195": 2,
            "10029129": 2,  # also fashion-id-002's exact target - legitimate reuse
            "10029205": 2,
            "10029231": 2,
            "10029043": 2,
            "10029293": 2,
            "10029365": 2,
            "10029037": 2,
            "10029091": 1,  # a pump, not a ballerina - same brand, near-miss garment
            "10029077": 1,  # a pump
        },
        note="8 Geox ballerinas at relevance 2; 2 Geox pumps (same brand, different shoe) at 1.",
    ),
    GoldenQuery(
        id="fashion-attr-020",
        query="HRX by Hrithik Roshan sweatshirt",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in ("10031241", "10020899", "10030665", "10030681", "10020911", "10030679")
        },
        note=(
            "6 of the top 10 hits are genuine HRX sweatshirts; 4 T-shirts/cargos from "
            "the same brand in the same top-10 were omitted (wrong garment)."
        ),
    ),
    # --- exploratory: 16 more occasion/intent queries ------------------------------
    GoldenQuery(
        id="fashion-exp-004",
        query="workout clothes for the gym",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            sku: 2
            for sku in (
                "10203229",
                "10231599",
                "10193605",
                "10203233",
                "10244149",
                "10137263",
                "10231589",
                "10193621",
            )
        },
        note="Gym/training shoes + a sports bra + training tights - genuinely gym-appropriate.",
    ),
    GoldenQuery(
        id="fashion-exp-005",
        query="traditional wear for Diwali",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "10238709": 2,
            "10258093": 2,
            "10128407": 2,
            "10140437": 2,
            "10247015": 2,
            "10268825": 2,
            "10258273": 2,
            "10135443": 1,  # earrings - an accessory, not "wear" itself
        },
        note="Sarees, ethnic dresses, kurta-with-dhoti-pants sets - real festive wear.",
    ),
    GoldenQuery(
        id="fashion-exp-006",
        query="gift for dad birthday",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            sku: 1
            for sku in (
                "10115285",
                "10115319",
                "10115245",
                "10115207",
                "10115281",
                "10115197",
                "10115337",
                "10115247",
            )
        },
        note=(
            "Every top-8 hit is an Archies 'Love Gifts' mug - a plausible generic "
            "birthday gift, but branded for Valentine's, not fathers specifically. "
            "Graded 1 (weak match), not 2, being honest about how strong the signal "
            "actually is rather than inflating it."
        ),
    ),
    GoldenQuery(
        id="fashion-exp-007",
        query="warm layers for winter travel",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "10185279": 2,  # titled "Winter Kurta"
            "10185559": 2,  # titled "Winter Kurta"
            "10185227": 2,  # titled "Winter Kurta"
            "10185123": 2,  # titled "Winter Kurta"
            "10002455": 2,  # GAS Pullover
            "10231665": 1,  # Reebok training tights - warm, but workout-specific
        },
        note=(
            "AURELIA's own titles say 'Winter Kurta' explicitly - strong real signal. "
            "A Divine Casa bed comforter in the same top-8 was omitted (bedding, not "
            "wearable)."
        ),
    ),
    GoldenQuery(
        id="fashion-exp-008",
        query="everyday jewelry for work",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            sku: 2
            for sku in (
                "10125213",
                "10106591",
                "10161943",
                "10161929",
                "10161959",
                "10125229",
                "10106609",
                "10125231",
            )
        },
        note="Rings, necklaces, jewellery sets - all genuinely jewelry.",
    ),
    GoldenQuery(
        id="fashion-exp-009",
        query="loungewear for working from home",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            sku: 2
            for sku in (
                "10235307",
                "10235351",
                "10235313",
                "10235323",
                "10235327",
                "10235347",
                "10235343",
                "10235297",
            )
        },
        note="Every top-8 hit is a Soie 'Lounge T-shirt' - the product name matches the intent directly.",
    ),
    GoldenQuery(
        id="fashion-exp-010",
        query="sports shoes for running",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            sku: 2
            for sku in (
                "10029819",
                "10029809",
                "10029745",
                "10233965",
                "10029869",
                "10029731",
                "10029797",
                "10242891",
            )
        },
        note=(
            "Overlaps in real products with fashion-attr-018's brand-agnostic "
            "'men's running shoes' - deliberately kept as a separate, differently "
            "phrased query since the two may still get classified/routed differently."
        ),
    ),
    GoldenQuery(
        id="fashion-exp-011",
        query="bag for daily college use",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "10182275": 2,  # MBOSS laptop bag
            "10143543": 2,  # Calvin Klein Jeans laptop backpack
            "10252709": 1,
            "10252711": 1,
            "10252693": 1,
            "10252723": 1,
            "10128245": 1,
            "10128215": 1,
        },
        note=(
            "Laptop bag/backpack graded 2 (genuinely practical for daily college use); "
            "handheld/shoulder/sling bags graded 1 (real bags, but a weaker fit for "
            "'daily college')."
        ),
    ),
    GoldenQuery(
        id="fashion-exp-012",
        query="makeup for a party look",
        query_class=QueryClass.EXPLORATORY,
        judgments={"10236207": 2, "10236211": 2, "10232357": 2},
        note=(
            "Only 3 of the top-8 hits are actual makeup (2 eyeshadow palettes, a "
            "highlighter); a party shirt, palazzos, dress material, a suit, and "
            "sandals in the same top-8 were omitted as not makeup."
        ),
    ),
    GoldenQuery(
        id="fashion-exp-013",
        query="birthday gift for a teenage girl",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "10212955": 2,
            "10030547": 2,
            "10213455": 2,
            "10212963": 2,
            "10158061": 2,
            "10212883": 2,
            "10225783": 1,  # Swarovski necklace - plausible gift, less teen-specific
            "10225753": 1,  # Swarovski pendant
        },
        note="Girls tops/dresses/sweatshirt at 2; jewelry from the same result set at 1.",
    ),
    GoldenQuery(
        id="fashion-exp-014",
        query="cozy home wear for a lazy Sunday",
        query_class=QueryClass.EXPLORATORY,
        judgments={"10032801": 2, "10002455": 2},
        note=(
            "Only a sweater and a pullover from the top-8 actually fit 'cozy home "
            "wear' - 4 jeggings (regular daywear, not loungewear) and a curtain/"
            "bedsheet (home decor, not wearable) were omitted. A small relevant set "
            "is a legitimate outcome of live verification, not a flaw to pad out. "
            "Also misclassified ATTRIBUTE by the live lexical classifier (no lexical "
            "overlap cue it can catch) - same known hard case as fashion-exp-003, "
            "scores nDCG@10=0.0 at the current alpha for the same reason."
        ),
    ),
    GoldenQuery(
        id="fashion-exp-015",
        query="kids party wear for a birthday",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            sku: 2
            for sku in (
                "10213415",
                "10213251",
                "10213395",
                "10213401",
                "10213409",
                "10213437",
                "10213455",
                "10213377",
            )
        },
        note="U.S. Polo Assn. Kids girls party dresses + a boys blazer - genuinely party-appropriate.",
    ),
    GoldenQuery(
        id="fashion-exp-016",
        query="formal watch for office wear",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "10223359": 2,
            "10044435": 2,
            "10223367": 2,
            "10223363": 2,
            "1022179": 2,
        },
        note=(
            "SEIKO/TIMESMITH/SKAGEN watches at 2; 3 CODE by Lifestyle formal *shirts* "
            "in the same top-8 (keyword overlap on 'formal', wrong product type) "
            "were omitted."
        ),
    ),
    GoldenQuery(
        id="fashion-exp-017",
        query="trendy sunglasses for summer",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            sku: 2
            for sku in (
                "10236609",
                "10236611",
                "10030529",
                "10236615",
                "10180299",
                "10203355",
                "10173199",
                "10185967",
            )
        },
        note="Every top-8 hit is genuine sunglasses - the cleanest exploratory probe of the batch.",
    ),
    GoldenQuery(
        id="fashion-exp-018",
        query="ethnic jewellery set for a festive occasion",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "10135429": 2,
            "10139533": 2,
            "10161985": 2,
            "10135509": 2,
            "10135495": 2,
        },
        note=(
            "5 real jewellery sets at 2; 2 home-decor 'Ethnic Motifs' cushion covers "
            "and a curtain in the same top-8 (keyword overlap on 'ethnic') were "
            "omitted - the catalog uses 'ethnic' for home decor prints too, not just "
            "fashion."
        ),
    ),
    GoldenQuery(
        id="fashion-exp-019",
        query="baby clothing for a newborn",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            sku: 2
            for sku in (
                "10244371",
                "10244671",
                "10213151",
                "10145021",
                "10244377",
                "10244669",
                "10244683",
                "10145065",
            )
        },
        note="GAP Baby / U.S. Polo Assn. Infants items - all genuinely newborn/baby clothing.",
    ),
]
