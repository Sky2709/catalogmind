"""Golden queries for `demo-electronics-in` (Amazon India electronics catalog).

Same methodology as `demo_fashion_in.py` - every judgment below came from a real,
live `hybrid()` call against the tenant this session (see
`scripts/build_golden_sets.py`), not invented. See `eval/golden/__init__.py`'s module
docstring for the verified-top-K-pool limitation and the 0-2 relevance scale.

This catalog's real constraints (discovered by sampling live data, see
`PROGRESS.md`'s Day 4 notes) shape how these queries had to be built differently from
fashion's: `brand` is **null on every row** (the adapter never extracts it) and
`category_path` is the single value `"tv, audio & cameras"` for the whole catalog
(HHI=1.0, a Day 2 finding) - there is no usable brand/category filter here. Every
attribute query below is a title-keyword constraint (spec words, product type), not a
brand+category combination the way fashion's are. Identifier queries use a
distinctive model-name phrase pulled straight from a real title (e.g. "Redmi 10"),
verified to land top-1, rather than the raw ASIN - closer to what a real shopper
would actually type.
"""

from __future__ import annotations

from app.retrieval.base import QueryClass
from eval.golden import GoldenQuery

QUERIES = [
    # --- identifier: distinctive model-name phrases, each verified top-1 ----------
    GoldenQuery(
        id="electronics-id-001",
        query="Samsung Galaxy S23 5G Cream 8GB 256GB",
        query_class=QueryClass.IDENTIFIER,
        judgments={"B0BT9F4DVQ": 2},
    ),
    GoldenQuery(
        id="electronics-id-002",
        query="Redmi 10 Caribbean Green",
        query_class=QueryClass.IDENTIFIER,
        judgments={"B09XR8591Q": 2},
    ),
    GoldenQuery(
        id="electronics-id-003",
        query="OnePlus Y Series 43 inch Smart TV",
        query_class=QueryClass.IDENTIFIER,
        judgments={"B09Q5P2MT3": 2},
    ),
    GoldenQuery(
        id="electronics-id-004",
        query="Logitech G502 Hero gaming mouse",
        query_class=QueryClass.IDENTIFIER,
        judgments={"B07GBZ4Q68": 2},
    ),
    GoldenQuery(
        id="electronics-id-005",
        query="Xiaomi Pad 5 Snapdragon 860",
        query_class=QueryClass.IDENTIFIER,
        judgments={"B09XXZXQC1": 2},
    ),
    GoldenQuery(
        id="electronics-id-006",
        query="Sennheiser HD 458 ANC headphones",
        query_class=QueryClass.IDENTIFIER,
        judgments={"B08D1CD3Q9": 2},
    ),
    GoldenQuery(
        id="electronics-id-007",
        query="Dell WM118 Wireless Mouse",
        query_class=QueryClass.IDENTIFIER,
        judgments={"B07JPX9CR7": 2},
    ),
    GoldenQuery(
        id="electronics-id-008",
        query="Yubico YubiKey 5 NFC",
        query_class=QueryClass.IDENTIFIER,
        judgments={"B07HBD71HL": 2},
    ),
    GoldenQuery(
        id="electronics-id-009",
        query="Seagate Barracuda 2 TB internal hard drive",
        query_class=QueryClass.IDENTIFIER,
        judgments={"B07GWRP5LN": 2},
    ),
    GoldenQuery(
        id="electronics-id-010",
        query="iQOO 9 SE 5G Space Fusion",
        query_class=QueryClass.IDENTIFIER,
        judgments={"B07WHQRCZD": 2},
    ),
    GoldenQuery(
        id="electronics-id-011",
        query="Kingston SSDNow A400 240GB",
        query_class=QueryClass.IDENTIFIER,
        judgments={"B079TH8YZQ": 2},
    ),
    GoldenQuery(
        id="electronics-id-012",
        query="Sansui 55 inches 4K Android TV",
        query_class=QueryClass.IDENTIFIER,
        judgments={"B09NNGHG22": 2},
    ),
    GoldenQuery(
        id="electronics-id-013",
        query="boAt Immortal 121 TWS earbuds",
        query_class=QueryClass.IDENTIFIER,
        judgments={"B0BKZLW5T5": 2},
    ),
    GoldenQuery(
        id="electronics-id-014",
        query="JBL Live Pro 2 earbuds",
        query_class=QueryClass.IDENTIFIER,
        judgments={"B0B1SJ7YSK": 2},
    ),
    GoldenQuery(
        id="electronics-id-015",
        query="Crucial RAM 8GB DDR4 3200MHz",
        query_class=QueryClass.IDENTIFIER,
        judgments={"B08C4Z69LN": 2},
    ),
    GoldenQuery(
        id="electronics-id-016",
        query="TAGG Verve Connect Ultra smartwatch",
        query_class=QueryClass.IDENTIFIER,
        judgments={"B0B56WM84H": 2},
    ),
    GoldenQuery(
        id="electronics-id-017",
        query="SanDisk Extreme Pro microSD 128GB",
        query_class=QueryClass.IDENTIFIER,
        judgments={"B09X7DNF6G": 2},
    ),
    GoldenQuery(
        id="electronics-id-018",
        query="HP GK320 mechanical keyboard",
        query_class=QueryClass.IDENTIFIER,
        judgments={"B08498H13H": 2},
    ),
    GoldenQuery(
        id="electronics-id-019",
        query="SanDisk 256 GB iXpand Flash Drive for iPhone",
        query_class=QueryClass.IDENTIFIER,
        judgments={"B08JD2LXXG": 2},
        note=(
            "Replaces an initial pick, 'Fastrack Reflex Vox Smartwatch' - that model "
            "name maps to 4 near-identical colour-variant SKUs with truncated titles "
            "that don't disambiguate, so it isn't a genuine single-target identifier "
            "query. Moved to electronics-attr-001 instead, where multiple valid "
            "targets is exactly the point."
        ),
    ),
    GoldenQuery(
        id="electronics-id-020",
        query="Wildcraft Turnaround Polyester 14 inch Laptop Bag",
        query_class=QueryClass.IDENTIFIER,
        judgments={"B012D4QWES": 2},
        note=(
            "Replaces an initial pick, 'ZEBRONICS Zeb-Thunder wireless headphone' - "
            "same issue as above (multiple real colour variants share the query "
            "text); moved to electronics-attr-002."
        ),
    ),
    # --- attribute: title-keyword spec/product-type constraints -------------------
    GoldenQuery(
        id="electronics-attr-001",
        query="Fastrack Reflex Vox Smartwatch",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            "B09QKLV4D7": 2,
            "B09QKL7BM6": 2,
            "B09QKLM2VK": 2,
            "B09QKLB8Y2": 2,
            "B0BHTHWJ81": 1,  # "Reflex VOX 2.0" - related but a different model
            "B0BHTZ5V9H": 1,  # "Reflex VOX 2.0"
            "B0BB7H8RL1": 1,  # "Reflex Play +" - different model
            "B09SZBK6BW": 1,  # "Reflex Activity Tracker" - different product line
        },
        note=(
            "The exact model name maps to 4 real colour-variant SKUs (truncated "
            "titles don't show which colour) - this is the multi-target case "
            "electronics-id-019/020 moved away from. 4 name-sibling models (VOX 2.0, "
            "Play+, Activity Tracker) graded 1."
        ),
    ),
    GoldenQuery(
        id="electronics-attr-002",
        query="ZEBRONICS Zeb-Thunder wireless headphone",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            "B07L8KNP5F": 2,
            "B09B9SD1QH": 2,
            "B07L8KV2KC": 2,
            "B07L8LTS3J": 2,
            "B07L8JTZ4H": 2,
            "B09B5F3QBH": 2,
            "B097JN5V18": 1,  # "Zeb-Thunder PRO" - a different, newer model
            "B088FLHXJY": 1,  # "Zeb-Duke" - a different product line entirely
        },
        note="6 real Zeb-Thunder colour variants at 2; a PRO variant and a different product line at 1.",
    ),
    GoldenQuery(
        id="electronics-attr-003",
        query="boAt true wireless earbuds",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "B0B58J5N8M",
                "B0BV5Y6D9P",
                "B08JQM8SMH",
                "B08JQN8DGZ",
                "B0BNPSTPP1",
                "B08CVTT65T",
                "B0BNPTJ7T2",
                "B0B7J5ZTXD",
            )
        },
        note="All 8 top hits are genuine boAt Airdopes TWS earbuds models.",
    ),
    GoldenQuery(
        id="electronics-attr-004",
        query="Redmi back cover case",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "B089B7ZX22",
                "B0B1N42H9G",
                "B08KFSMQ84",
                "B089B242K1",
                "B089B4YMZS",
                "B089B24HVX",
                "B07XLSH9M1",
                "B089B628P8",
            )
        },
        note="All 8 top hits are back cover cases for various Redmi phone models.",
    ),
    GoldenQuery(
        id="electronics-attr-005",
        query="Samsung Galaxy back cover case",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "B0BW67T23Q",
                "B09JFRT3KK",
                "B07FLZJ57M",
                "B09JFSJVTQ",
                "B089B826K6",
                "B08QNNXD84",
                "B0BJS9D29K",
                "B0938NB5PB",
            )
        },
        note="All 8 top hits are back cover cases for various Samsung Galaxy models.",
    ),
    GoldenQuery(
        id="electronics-attr-006",
        query="wireless bluetooth mouse",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "B09J8FX6VZ",
                "B07X4QDCD6",
                "B0BFBDYRTP",
                "B0BFBG1QF9",
                "B0BFBHW6MT",
                "B07X2L5Z8C",
                "B07X3PJLQR",
                "B07PT6Q376",
            )
        },
        note="All 8 top hits are genuine wireless/Bluetooth mice.",
    ),
    GoldenQuery(
        id="electronics-attr-007",
        query="HDMI cable 4K",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "B07KSMBL2H",
                "B08TGG316Z",
                "B09WDHMP94",
                "B0BQ6MCQKD",
                "B0BQ6KSXJW",
                "B0BJK9WCXN",
                "B01M4GGIVU",
                "B0BJK8JXD4",
            )
        },
        note="All 8 top hits are HDMI cables supporting 4K.",
    ),
    GoldenQuery(
        id="electronics-attr-008",
        query="USB type C fast charging cable",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "B08DDRGWTJ",
                "B07GVGTSLN",
                "B01GGKZ2SC",
                "B0BTHLFZQB",
                "B0B86CDHL1",
                "B09C6HWG18",
                "B08PSVSTPH",
                "B093C171PL",
            )
        },
        note="All 8 top hits are USB-C fast-charging cables.",
    ),
    GoldenQuery(
        id="electronics-attr-009",
        query="4K smart Android TV",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "B095JQVC7N",
                "B0B1YZX72F",
                "B08FD3HN12",
                "B09MJ77786",
                "B095JPKPH3",
                "B0B99NL2JM",
                "B09RWQ7YR6",
                "B09RFC46VP",
            )
        },
        note="All 8 top hits are 4K Android smart TVs across multiple brands.",
    ),
    GoldenQuery(
        id="electronics-attr-010",
        query="gaming mechanical keyboard RGB",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "B019O9BLVY",
                "B08498H13H",
                "B09BVCVTBC",
                "B0BSNNRSQV",
                "B09SH9PS1G",
                "B09JSZYKYR",
                "B0B1MRNF93",
                "B0BS9QGDF3",
            )
        },
        note="All 8 top hits are RGB mechanical gaming keyboards.",
    ),
    GoldenQuery(
        id="electronics-attr-011",
        query="noise cancelling headphones",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "B098FKXT8L",
                "B09XSDMT4F",
                "B09XS7JWHH",
                "B0B4PSQHD5",
                "B08S7291WK",
                "B0BD7T5287",
                "B009LJ2BXA",
                "B08J4CS8MX",
            )
        },
        note="All 8 top hits genuinely advertise noise cancelling.",
    ),
    GoldenQuery(
        id="electronics-attr-012",
        query="power bank fast charging",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "B0851WN735",
                "B0851WMSDS",
                "B08JVY8LGG",
                "B08JVY8LGD",
                "B08HF4W2CT",
                "B08XXN8CNS",
                "B08JW1GVS7",
                "B08JVY7QYC",
            )
        },
        note="All 8 top hits are fast-charging power banks.",
    ),
    GoldenQuery(
        id="electronics-attr-013",
        query="microSD memory card 128GB",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            "B07DJGJ18T": 2,
            "B08L5DBMMS": 2,
            "B08TJRVWV1": 2,
            "B0B2DD66GS": 2,
            "B09X7DNF6G": 2,  # also electronics-id-017's exact target
            "B0BX8ZSCM8": 2,
            "B07DJGJ2H1": 1,  # 16GB, not 128GB
            "B07DJGB43S": 1,  # 64GB, not 128GB
        },
        note="6 real 128GB microSD cards at 2; 2 lower-capacity cards from the same brand at 1.",
    ),
    GoldenQuery(
        id="electronics-attr-014",
        query="laptop sleeve case 15.6 inch",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "B09BFW41H4",
                "B0BRJ7F2VD",
                "B09BW574DG",
                "B07WJRY2QK",
                "B09BFTHSBN",
                "B091NYBNVY",
                "B09X6657S8",
                "B0B3TKNF9V",
            )
        },
        note="All 8 top hits are 15.6-inch laptop sleeves/cases.",
    ),
    GoldenQuery(
        id="electronics-attr-015",
        query="smartwatch bluetooth calling",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "B0B53QLB9H",
                "B0B53QFZPY",
                "B0BKLK2RL2",
                "B09RKCM1J3",
                "B09YV3K34W",
                "B09YV4MW2T",
                "B09YV4DC29",
                "B0BP1X1YGG",
            )
        },
        note="All 8 top hits are Bluetooth-calling smartwatches.",
    ),
    GoldenQuery(
        id="electronics-attr-016",
        query="extension board with USB ports",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "B08S7K8LNM",
                "B0B3DRQ246",
                "B096XGFTLX",
                "B08S76PDN2",
                "B09RKGP4NW",
                "B08S6KKXNX",
                "B08FR9JMD7",
                "B096Y48N1P",
            )
        },
        note="All 8 top hits are extension boards/power strips with USB charging ports.",
    ),
    GoldenQuery(
        id="electronics-attr-017",
        query="screen protector tempered glass iPhone",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "B0BRB7XJNT",
                "B08FBJXYJD",
                "B07NZST6WH",
                "B0BCX2GZTR",
                "B0BCX5CCLH",
                "B0BD55NSNZ",
                "B095RTJH1M",
                "B09XR9CYCV",
            )
        },
        note="All 8 top hits are tempered-glass screen protectors for various iPhone models.",
    ),
    GoldenQuery(
        id="electronics-attr-018",
        query="external hard drive 2TB",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            "B08ZJG6TVT": 2,
            "B07B4KXTQK": 2,
            "B07VTFN6HM": 2,
            "B07DNDLHH8": 2,
            "B01L3XLIFI": 1,  # a protective CASE for a drive, not a drive itself
            "B01L3YBQZO": 1,  # same - a case
            "B01KI6FLI6": 1,  # same - a case
            "B01KI42UPK": 1,  # same - a case
        },
        note="4 real 2TB external drives at 2; 4 empty drive-carrying cases (same search terms, different product) at 1.",
    ),
    GoldenQuery(
        id="electronics-attr-019",
        query="wifi router dual band",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "B08FYJKGSX",
                "B07CC7BHXH",
                "B07DGPYKLP",
                "B08GGC84CR",
                "B0BJF1F84W",
                "B0756FP9DH",
                "B00KXULGJQ",
                "B09DQ3RV16",
            )
        },
        note="All 8 top hits are dual-band WiFi routers/extenders.",
    ),
    GoldenQuery(
        id="electronics-attr-020",
        query="selfie stick tripod bluetooth",
        query_class=QueryClass.ATTRIBUTE,
        judgments={
            sku: 2
            for sku in (
                "B08HJ1ST92",
                "B09PYFSKCV",
                "B092R7KJBK",
                "B08YJV9M3Z",
                "B09PTSTZJQ",
                "B07Q64Q5R5",
                "B0BDVV221L",
                "B0B3F9D5TP",
            )
        },
        note="All 8 top hits are Bluetooth selfie sticks with tripod stands.",
    ),
    # --- exploratory: occasion/intent, no lexical overlap with the target product --
    GoldenQuery(
        id="electronics-exp-001",
        query="something to protect my phone from drops",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "B0BTLL44NH": 2,  # an actual phone drop-protection back cover
            "B09K4J1TZD": 1,  # earbuds case, same "protective case" concept, wrong device
            "B09PMKLZJ7": 1,
            "B0BBW3TYHM": 1,
        },
        note=(
            "Only one of the top-6 hits is a phone case specifically; 3 are earbuds "
            "cases (weaker match, graded 1) and a generic waterproof pouch was "
            "omitted as unclear."
        ),
    ),
    GoldenQuery(
        id="electronics-exp-002",
        query="device to track my daily fitness",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            sku: 2
            for sku in (
                "B09KPXBMKT",
                "B09NNMV5GV",
                "B0BSCVHVQW",
                "B0B5TTLRBS",
                "B0BHQQNYTM",
                "B0BG8RJMSZ",
            )
        },
        note="Every top-6 hit is a genuine fitness tracker / activity band.",
    ),
    GoldenQuery(
        id="electronics-exp-003",
        query="gear for taking calls hands-free while driving",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "B07XY2MFZC": 2,  # Bluetooth FM transmitter for hands-free calling
            "B0BKLD6FS8": 2,  # Bluetooth hands-free car kit
            "B08F7PPGBQ": 2,  # motorcycle Bluetooth headset, hands-free
            "B07RVCVSB2": 1,  # a phone mount - relevant to driving, not audio/calling
            "B09QGQ1F2M": 1,  # generic wired earphones, no driving-specific feature
            "B0BM41N38K": 1,
        },
        note="3 genuinely hands-free-calling products at 2; a phone mount and generic earphones at 1.",
    ),
    GoldenQuery(
        id="electronics-exp-004",
        query="solution for slow wifi at home",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            sku: 2
            for sku in (
                "B07CC7BHXH",
                "B0151AM5UG",
                "B07L44RHC2",
                "B07L45LZP5",
                "B07KJ2TDMR",
                "B07P7KM4Y6",
            )
        },
        note="Every top-6 hit is a WiFi router or whole-home mesh system.",
    ),
    GoldenQuery(
        id="electronics-exp-005",
        query="way to back up my photos",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            sku: 2
            for sku in (
                "B07VTWX8MN",
                "B07VTGBWYH",
                "B088P72F93",
                "B07VTW2LQ5",
                "B07VP5X239",
                "B07LGDLPCC",
            )
        },
        note="Every top-6 hit is an external drive with 'Automatic Backup' explicitly in its title.",
    ),
    GoldenQuery(
        id="electronics-exp-006",
        query="something for better sound while gaming",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            sku: 2
            for sku in (
                "B07W7K3BTD",
                "B09374KTPK",
                "B0B14SGH8F",
                "B0B14TSYCH",
                "B07JLFFTY1",
                "B07MZY8GFV",
            )
        },
        note="Every top-6 hit is a gaming headset, gaming earbuds, mic, or sound card.",
    ),
    GoldenQuery(
        id="electronics-exp-007",
        query="device to monitor my home while away",
        query_class=QueryClass.EXPLORATORY,
        judgments={"B09RSVG8NJ": 2, "B092VX42D4": 2, "B0B42TZQ4C": 2},
        note=(
            "Only the 3 security cameras in the top-6 actually fit 'monitor my home' "
            "- 2 vehicle GPS trackers and a fitness smartwatch (keyword overlap on "
            "'monitor'/'track') were omitted as the wrong kind of monitoring."
        ),
    ),
    GoldenQuery(
        id="electronics-exp-008",
        query="way to type comfortably for long hours",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            sku: 2 for sku in ("B0B29G5SSW", "B09ZV676R9", "B0BQJK1YZV", "B08CF4SCNP", "B0B71WT5XT")
        },
        note="5 of the top-6 hits are keyboards emphasizing comfort/ergonomics explicitly.",
    ),
    GoldenQuery(
        id="electronics-exp-009",
        query="something to keep my laptop safe while travelling",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "B07XYTZP4Z": 2,
            "B097JMR4M6": 2,
            "B097JM98M1": 2,
            "B0B34WJK3X": 2,
            "B00EU6TXC6": 1,  # a laptop stand - not travel-specific
        },
        note="4 travel laptop backpacks at 2; a laptop stand (not travel gear) at 1.",
    ),
    GoldenQuery(
        id="electronics-exp-010",
        query="gadget for capturing memories on a trip",
        query_class=QueryClass.EXPLORATORY,
        judgments={"B0BRGCCGP4": 2, "B099ZYBSWB": 2, "B08KY6SQFP": 2, "B09FK2639V": 2},
        note=(
            "A kids camera and 2 DJI gimbal stabilizers + a mini tripod are real "
            "capture gear; a kids LCD writing pad and a sticky-note pack in the same "
            "top-6 (no real connection) were omitted."
        ),
    ),
    GoldenQuery(
        id="electronics-exp-011",
        query="way to listen to music without disturbing others",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            sku: 2
            for sku in (
                "B08SW6MQMD",
                "B08SW5F9GJ",
                "B00KGZZ824",
                "B00HVLUR18",
                "B0BD7T5287",
                "B09RWXBMQJ",
            )
        },
        note="Every top-6 hit is a personal headphone/earphone/neckband - genuinely private listening.",
    ),
    GoldenQuery(
        id="electronics-exp-012",
        query="gift for a gamer",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            sku: 2 for sku in ("B0BJ99Q8JD", "B0BK4WTSHD", "B09BVCY8RF", "B09SN34HZ7", "B09GG3BB59")
        },
        note=(
            "5 of the top-6 hits are real gaming peripherals (keyboards, gloves, "
            "mouse); a kids birthday pencil-gift set (keyword overlap on 'gift') was "
            "omitted."
        ),
    ),
    GoldenQuery(
        id="electronics-exp-013",
        query="something to keep my desk cables organized",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "B09NTTMC73": 2,
            "B084X79CRW": 2,
            "B098KH4TMP": 2,
            "B07RCGXNW5": 2,
            "B07R99NBVB": 2,
            "B0BC9Y2RVX": 1,  # a monitor stand with storage - desk accessory, not cable-specific
        },
        note="5 genuine cable-management products at 2; a monitor stand (desk accessory, not cable-specific) at 1.",
    ),
    GoldenQuery(
        id="electronics-exp-014",
        query="way to reduce eye strain while reading at night",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            "B07QMRHWJD": 2,  # "Eye Protection" desk lamp, explicit
            "B0BCG1FNXW": 2,  # "Eye Protection" desk lamp, explicit
            "B0BCG3WNQ1": 2,  # "Eye Protection" desk lamp, explicit
            "B09CLPB75T": 2,  # privacy screen filter, titled "NO Eye Strain"
            "B07XLMSV5W": 1,  # a magnifier - helps reading, not specifically eye-strain
            "B07XTJMS82": 1,
        },
        note="4 products with 'eye protection'/'eye strain' explicitly in their titles at 2; 2 magnifiers at 1.",
    ),
    GoldenQuery(
        id="electronics-exp-015",
        query="protect my new smartphone screen from scratches",
        query_class=QueryClass.EXPLORATORY,
        judgments={
            sku: 2
            for sku in (
                "B09Q3M3WLJ",
                "B0BTVN781V",
                "B0BNF4LHHV",
                "B09Q3HXLVS",
                "B0B6RMW3VB",
                "B08CTQP51L",
            )
        },
        note="Every top-6 hit is a tempered-glass screen protector or camera-lens protector.",
    ),
]
