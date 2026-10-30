"""Action space of the RL agent.

Each turn every unit (farmer or hired worker) receives a pair (target, operation):
  target = tile 0..99 (x + 10*y), operation = one of OPS.
Executor: if the unit is not on the target, one step towards the target (first along x, then along y); if it is on the target, the operation.
Shed operations (DROP, PICKUP_*) are meaningful only on the 4 tiles next to the shed (SHED_TILES).

Market: per product, a bin for the sell quantity; purchases of seeds/products/animals, hire, land are
separate heads with bins. Orders are assembled in the order given by the template chosen by the ORDER head (ORDER_TEMPLATES), at most 10.
"""  # noqa: E501

# This file is copied verbatim into the submitted agent (rl/export_agent.py), and entry_points/verify_submission.py
# compares its code with the archives sent to Kaggle: change only comments and formatting.
N = 10
NT = N * N
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
SHOPS = (
    "BAKERY",
    "BRUNCH_SPOT",
    "FARMERS_MARKET",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "PIZZA_SHOP",
    "SMOOTHIE_SHOP",
    "YARN_STORE",
)
SHED_TILES = ((4, 4), (5, 4), (4, 5), (5, 5))
SHED_IDX = tuple(x + N * y for x, y in SHED_TILES)
QUADS = ("NW", "NE", "SW", "SE")

PICKUP_WHEAT_N = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
PICKUP_FERT_N = (1, 2, 3, 4, 5, 6, 7, 8)
PICKUP_ANIMAL_N = (1, 2, 3)

# 56 operations. PICKUP_<item>_<k> takes k units from the shed; PLACE_<product>_ALL puts the unit's whole stack of
# that product into the shed; PLACE_<animal> puts a carried animal on an empty coop/pasture.
OPS = (
    ["STAY"]
    + [f"PLANT_{c}" for c in CROPS]
    + ["WATER", "HARVEST", "FERTILIZE", "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE", "DIG"]
    + ["DROP"]
    + [f"PICKUP_WHEAT_{k}" for k in PICKUP_WHEAT_N]
    + [f"PICKUP_FERTILIZER_{k}" for k in PICKUP_FERT_N]
    + [f"PICKUP_{a}_{k}" for a in ANIMALS for k in PICKUP_ANIMAL_N]
    + [f"PLACE_{a}" for a in ANIMALS]
    + [f"PLACE_{p}_ALL" for p in PRODUCTS]
    + ["PLACE_WHEAT_1"]
)
OP_ID = {o: i for i, o in enumerate(OPS)}
NOPS = len(OPS)
SHED_OPS = frozenset(
    i for i, o in enumerate(OPS) if o == "DROP" or o.startswith("PICKUP_") or o.endswith("_ALL") or o == "PLACE_WHEAT_1"
)

# Quantity bins for the market (0 = no order). SELL: the last bin = "everything in the shed".
SELL_BINS = tuple(range(0, 25)) + (28, 32, 36, 40, 48, 64, -1)  # exact 0..24; -1 = the whole shed
BUY_SEED_BINS = tuple(range(0, 13)) + (16, 20, 24, 32)
BUY_PRODUCT_BINS = tuple(range(0, 25)) + (32, 48, 64)  # WHEAT, FERTILIZER
BUY_ANIMAL_BINS = tuple(range(0, 7))
HIRE_BINS = tuple(range(0, 13))
LAND_BINS = (0, 1)
BUY_PRODUCTS = ("WHEAT", "FERTILIZER")
# Order of order groups in the list (orders are executed one by one in lockstep with the opponent, at most 10):
# S sells, P product purchases, D seeds, A animals, H hire, L land. The templates cover the teacher's orderings.
ORDER_TEMPLATES = ("SPHLDA", "PSHLAD", "PDLAHS", "SLPHDA", "SHPLDA", "DSPHLA", "HSPLDA", "ASPHLD")

MARKET_HEADS = (
    [("SELL", p, SELL_BINS) for p in PRODUCTS]
    + [("BUY_SEED", c, BUY_SEED_BINS) for c in CROPS]
    + [("BUY_PRODUCT", p, BUY_PRODUCT_BINS) for p in BUY_PRODUCTS]
    + [("BUY_ANIMAL", a, BUY_ANIMAL_BINS) for a in ANIMALS]
    + [
        ("HIRE", None, HIRE_BINS),
        ("BUY_LAND", None, LAND_BINS),
        ("ORDER", None, tuple(range(len(ORDER_TEMPLATES)))),
        ("SELL_FIRST", None, tuple(range(len(PRODUCTS) + 1))),
        ("SEED_FIRST", None, tuple(range(len(CROPS) + 1))),
    ]
)
# SELL_FIRST / SEED_FIRST: 0 = PRODUCTS / CROPS order; k = the k-th product / seed goes first in its group
GROUP_OF = {"SELL": "S", "BUY_PRODUCT": "P", "BUY_SEED": "D", "BUY_ANIMAL": "A", "HIRE": "H", "BUY_LAND": "L"}
MAX_UNITS = 16  # farmer + up to 15 hired workers (from the 16th on: PASS)
MAX_ORDERS = 10


def tile_xy(t):
    return t % N, t // N


def tile_id(x, y):
    return x + N * y
