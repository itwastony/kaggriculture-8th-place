"""Caretaker v0.9: v0.8 + counter-tape warfare iteration (2026-08-04).
Behavior-identical to v0.8 at default params (certified batch-exact); adds:
- full EGG/goose vocabulary (BUILD_COOP, PLACE:GOOSE, feed chain, egg sales)
  behind GOOSE_TARGET (default 0: every enabled config measured -3..-29k --
  a single coop tile shifts the weed-RNG stream and reshuffles the season's
  shop-unlock schedule, which is systematically pro-tape; see
  research/warfare_report.md);
- lineage-gated adversarial market machinery (WAR_*): per-product floor
  overrides (flood/hold), flush-rate override, drain-tick sell alignment.
  All levers measured neutral-to-negative vs the ladder tape (the v0.7
  fb-outlook already prices its baked schedule), so defaults are neutral.

v0.8 was: v0.7 + production-tempo capital-plan fixes (this layer
was measured change-by-change; see research/tempo_report.md):
- SE cash hoard removed: v0.7 parked $4000 from d10 for the SE quadrant,
  stalling the cow ramp to d12-15 (+2.1k on the 601001 batch from removal
  alone; +1.1k mean over the full 32-game gate for the whole layer);
- optional SE refusal (LAND_MAX_EXTRA=2) -- the ladder tape never buys SE;
- partial feed-wheat buys at cash troughs (all-or-nothing orders bought
  nothing and the herd starved into escapes on trough seeds).
MARGIN LAW learned from the failed rewrites (kept for the record): vs a
fixed-schedule tape, never withdraw volume from a market where the tape
holds >= our units (milk 230, straw 268, wheat 840, fert 216 fixed units) --
withdrawing supply hands its book a price recovery (straw cut measured -13k
margin, wool cut -13k); volume floods are margin-safe because the mutual
crash costs the bigger holder more (milk 451u: their -19k vs our -12k).

v0.7 base: state-driven farm agent with a day-route planner,
production-integrity layers (anti-burn, herd uptime, melon fertilization)
and a market-outlook layer (v0.7): exact town-drain forward sim, baked
ladder-lineage flow schedule verified online by inventory residuals,
crash-flush below the sell floors when the glut is predicted to deepen,
fb-gated liquidation quota, recovery-aware pressure relief, and (vs live
non-lineage opponents only) crash harvest deferral.

Architecture: every turn -> collect service jobs from live tile state -> at day
start build balanced boustrophedon tours (one per planned unit, shed-anchored,
feeders pre-load wheat) -> units walk their tours performing every pending op
at each stop -> free units adopt orphan tours / claim unrouted jobs / steal the
tail of the thickest tour, so nobody idles while work exists -> emergency layer
late in the day interrupts tours for must-water/must-feed tiles -> liquidity
first market queue.  Key economy: ongoing crops only need water every 2nd day
(production is unconditional), so watering is scheduled lazily; harvests are
batched (yield accumulates on the tile); fertilizer goes into strawberries on
production eves.
"""

CROPS = {
    "WHEAT": {"seed": 10, "first": 2, "max_day": 4, "interval": 0, "ongoing": False, "max_yield": 6},
    "CARROT": {"seed": 20, "first": 2, "max_day": 3, "interval": 0, "ongoing": False, "max_yield": 4},
    "TOMATO": {"seed": 50, "first": 8, "max_day": 8, "interval": 1, "ongoing": True, "max_yield": 4},
    "STRAWBERRY": {"seed": 100, "first": 10, "max_day": 10, "interval": 2, "ongoing": True, "max_yield": 4},
    "MELON": {"seed": 80, "first": 10, "max_day": 12, "interval": 0, "ongoing": False, "max_yield": 6},
}
ANIMALS = {
    "COW": {"cost": 400, "first": 8, "interval": 2, "max_held": 6},
    "SHEEP": {"cost": 500, "first": 6, "interval": 3, "max_held": 6},
    "GOOSE": {"cost": 300, "first": 4, "interval": 1, "max_held": 4},
}
SHED_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]
MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
OP_RANK = {"DIG": 0, "BUILD_PASTURE": 1, "BUILD_COOP": 1, "PLACE": 2, "PLANT": 3,
           "FEED": 4, "FERTILIZE": 5, "WATER": 6, "CARE": 7,
           "COLLECT_FERTILIZER": 8, "HARVEST": 9}

import json
import math
import os

# ---- tunable constants (Optuna surface).  Defaults = study caretaker_v05
# trial 659 (2026-08-04, 767 trials): picked from the top-10 by fresh-seed
# arena robustness, not by study value.  A trial overrides via env var
# CARETAKER_PARAMS='{"TARGET_COWS": 9, ...}' at import.
PARAMS = {
    "TARGET_COWS": 10,
    "TARGET_SHEEP": 7,          # leaky herd (uptime is the multiplier on count)
    "ANIMAL_BUY_LAST_DAY": 13,
    "TARGET_STRAWBERRY": 46,   # 1.32.4 re-tune: peak of 42/44/46/48/50 sweep
    "STRAWBERRY_LAST_PLANT_DAY": 16,
    "EARLY_WHEAT_TILES": 7,
    "MID_WHEAT_EXTRA": 4,       # mid-game wheat cap = EARLY + this
    "MELON_WAVE": 10,
    "MELON_LAST_PLANT_DAY": 18,
    "LATE_WHEAT_FROM_DAY": 23,  # earlier endgame wheat flood: the tape sells
                                # 840 wheat units/game, extra supply keeps its
                                # book cheap and ours sells at $44-50 anyway
                                # (23 measured best of 21/23/25: -6.9/-8.4/-7.1k)
    "LATE_WHEAT_LAST_PLANT_DAY": 26,
    "HANDS_RAMP_SHIFT": 1,      # added to every entry of the early-day hand cap
    "DEFAULT_MAX_HANDS": 12,
    "MIN_HANDS": 4,
    "HAND_CAPACITY": 16,        # useful turns one unit contributes to the day plan
    "FLOOR_STRAWBERRY": 0.8442864788316762,
    "FLOOR_MELON": 0.656248697650075,
    "FLOOR_MILK": 0.7726800233906806,
    "FLOOR_WOOL": 0.4975536532867258,
    "FLOOR_TOMATO": 0.50,
    "FLOOR_CARROT": 0.40,
    "LIQ_START": 624,           # endgame: decay floors + force a liquidation pace
    "LIQ_END": 701,             # fully liquid before tapes' 712 terminal dump
    "LIQ_FLOOR_MIN": 0.15,
    "LIQ_SPIKE_FRAC": 0.3,
    "SHED_SOFT": 94,            # shed cap is 100, overflow burns at day end
    "SHED_SOFT_EVE": 96,        # exact day-end target (hours 21+)
    "FEED_BUFFER_EXTRA": 3,     # feed buffer = herd size + this
    "FERT_RESERVE_UNTIL_DAY": 12,
    "LAND_DAY_1": 2,
    "LAND_DAY_2": 6,
    "LAND_DAY_3": 11,
    "LAND_MAX_EXTRA": 3,        # 2 = refuse the $4000 SE quadrant
    "MELON_FIRST_DAY": 99,      # from this day melon replant outranks straw
    "EARLY_SHEEP_TARGET": 7,    # pre-strawberry greedy sheep buys
    "EARLY_SHEEP_LAST_DAY": 9,
    "COW_BASE": 5,             # 1.32.4 re-tune: earlier cows (the (5,5) fix
                               # reopened this lever); saturates >=5. +1.2k holdout
                               # combined w/ TARGET_STRAWBERRY=46 (regresses from
                               # +3.8k tuning-seed -> lottery variance caps gains)
    "COW_RAMP_START_DAY": 7,
    "SHEEP_RAMP_START_DAY": 4,
    "SHEEP_RAMP_RATE": 3,
    "DEFER_HARVEST_THRESHOLD": 72,
    "ANIMAL_RESERVE_EXTRA": 200,
    # --- EGG line (v0.9): mechanically complete (BUILD_COOP / PLACE:GOOSE /
    # feed chain / egg sales) but DISABLED by default -- measured NEGATIVE in
    # every configuration (research/warfare_log.md): a single coop tile
    # perturbs the farm layout, which shifts the shared weed-RNG stream and
    # reshuffles the whole season's shop-unlock order; the reshuffle is
    # systematically pro-tape (their milk/straw books gain +10..28k/game
    # while our wool book eats the crash).  Egg economics themselves are
    # sound (goose $300, first yield d+4, interval 1, ~2 eggs/day with care,
    # log above-curve = dump-safe) -- revisit only with a layout-invariant
    # scheme.
    "GOOSE_TARGET": 0,
    "GOOSE_START_DAY": 2,
    "GOOSE_RATE": 2,            # buy cap ramps RATE/day from START_DAY
    "GOOSE_LAST_DAY": 12,       # payback needs ~5 producing days + liq window
    "GOOSE_PRIO": 0,            # 0: after cows; 1: before the sheep/cow ramp;
                                # 2: before even the early-sheep greedy buys
    "GOOSE_FAR": 0,             # coop distance-from-shed preference: tiles at
                                # |dist-GOOSE_FAR| rank first (0 = closest;
                                # >0 keeps the near ring for the melon wave)
    "GOOSE_RESERVE_EXTRA": 200,  # goose buy guard = reserve + this
    "EGG_RATE_DAY": 1.9,        # own production, units/day/goose (care-banked)
    # --- adversarial market warfare vs DETECTED lineage tapes (v0.9) ---
    # Margin law, weaponized: vs a fixed-schedule book, holding stock through
    # a glut (our floors) gifts the tape's remaining book a price recovery;
    # flooding a market where THEY hold the bigger book costs them more than
    # us (milk crash measured -19k them vs -12k us).  Gated on the one-shot
    # lineage lock (step 240, tapes err<=0.022 vs reactive 0.51) so mirror/
    # solo behavior is untouched.  Floors are frac-of-base; -1 keeps normal.
    # ALL levers measured neutral-to-negative on the 4x8-game gate (see
    # research/warfare_report.md) -- the v0.7 fb-outlook already implements
    # the margin-law-optimal glut policy against a known schedule, so a
    # flood can only sell below fb (worse ex-ante).  Defaults are NEUTRAL
    # (= v0.8 market behavior); the machinery stays for future experiments.
    "WAR_ENABLE": 1,
    "WAR_FLOOR_MILK": -1.0,        # 0.03 flood measured -0.2k/0.0 (no-op)
    "WAR_FLOOR_STRAWBERRY": -1.0,  # 0.03 flood measured -1.4k (harmful)
    "WAR_FLOOR_MELON": -1.0,       # borderline books (them 114-120, us ~100)
    "WAR_FLOOR_WOOL": -1.0,        # 0.70 hold measured net 0 over 32 games
    "WAR_FLUSH_RATE": 2,           # 4 measured no-op (flush rarely binds >2)
    "WAR_END_STEP": 720,
    # Drain-tick alignment: market orders fill BEFORE the town tick of the
    # same step (engine: _process_market then _town_consume), so a unit
    # deferred off a tick step (step%4==0; center ticks are a subset) sells
    # next turn into the drained, higher quote -- deterministic +0.2-1.4%/u.
    # Only small (rate-limited) doses defer, to keep budget timing stable.
    # DISABLED: vs a continuously-selling tape the edge inverts (its sells
    # refill the drain first; measured -2.0k/+0.8k, unstable).
    "WAR_TICK_DEFER": 0,
    "WAR_TICK_MAX": 4,
    # --- market outlook (v0.7 microstructure layer) ---
    "OUT_HORIZON": 288,         # forward-sim horizon, steps
    "OUT_SAMPLE": 8,            # price-eval stride along the predicted path
    "FB_HAIRCUT": 0.9,          # pessimism on the predicted future-best quote
    "FB_OVERHANG_W": 0.25,      # our held stock leaks into the future ladder
    "FB_OWNPROD_W": 1.0,        # fraction of our production sold along the path
                                # (everything we make is liquidated before 701,
                                # so the future price must carry all of it)
    "FB_OVERHANG_GROWTH": 0.0,  # extra production-into-overhang (double-counts
                                # with FB_OWNPROD_W=1.0; keep for tuning)
    "FLUSH_RATE": 2,            # crash-flush units/turn/item below the floor
    "FLUSH_MIN_PRICE": 3,       # never flush into quotes below this
    "HOLD_RECOV_MULT": 1.25,    # defer harvest when fb > price * this
    "RELIEF_RECOV_W": 0.6,      # pressure-relief penalty for recoverable items
    "SCHED_ERR_MAX": 0.35,      # residual error gate for the baked opp schedule
    "SCHED_MIN_STEP": 72,       # trust the lineage prior until here
    "LINEAGE_LOCK_STEP": 240,   # one-shot tape-vs-not classification here
    "LINEAGE_ERR_MAX": 0.15,    # ladder tapes measure <=0.022, caretaker 0.51
    "LIQ_FB_GATE": 0.85,        # liq quota fires when quote >= fb * this
    "LIQ_USE_FB": 1,            # 0: keep the v0.6 EMA spike gate in the ramp
}
PARAMS.update(json.loads(os.environ.get("CARETAKER_PARAMS", "{}")))

TARGET_COWS = PARAMS["TARGET_COWS"]
TARGET_SHEEP = PARAMS["TARGET_SHEEP"]
ANIMAL_BUY_LAST_DAY = PARAMS["ANIMAL_BUY_LAST_DAY"]
TARGET_STRAWBERRY = PARAMS["TARGET_STRAWBERRY"]
STRAWBERRY_LAST_PLANT_DAY = PARAMS["STRAWBERRY_LAST_PLANT_DAY"]
EARLY_WHEAT_TILES = PARAMS["EARLY_WHEAT_TILES"]
MID_WHEAT_EXTRA = PARAMS["MID_WHEAT_EXTRA"]
MELON_WAVE = PARAMS["MELON_WAVE"]
MELON_LAST_PLANT_DAY = PARAMS["MELON_LAST_PLANT_DAY"]
LATE_WHEAT_FROM_DAY = PARAMS["LATE_WHEAT_FROM_DAY"]
LATE_WHEAT_LAST_PLANT_DAY = PARAMS["LATE_WHEAT_LAST_PLANT_DAY"]
DEFAULT_MAX_HANDS = PARAMS["DEFAULT_MAX_HANDS"]
MIN_HANDS = PARAMS["MIN_HANDS"]
MAX_HANDS_BY_DAY = {
    d: max(MIN_HANDS, min(DEFAULT_MAX_HANDS, v + PARAMS["HANDS_RAMP_SHIFT"]))
    for d, v in enumerate([4, 5, 6, 6, 7, 7, 8, 9, 10, 11])}
HAND_CAPACITY = PARAMS["HAND_CAPACITY"]
DEFER_HARVEST_THRESHOLD = PARAMS["DEFER_HARVEST_THRESHOLD"]
FEED_BUFFER_EXTRA = PARAMS["FEED_BUFFER_EXTRA"]
FERT_RESERVE_UNTIL_DAY = PARAMS["FERT_RESERVE_UNTIL_DAY"]
LAND_DAYS = [PARAMS["LAND_DAY_1"], PARAMS["LAND_DAY_2"], PARAMS["LAND_DAY_3"]]
EARLY_SHEEP_TARGET = PARAMS["EARLY_SHEEP_TARGET"]
EARLY_SHEEP_LAST_DAY = PARAMS["EARLY_SHEEP_LAST_DAY"]
COW_BASE = PARAMS["COW_BASE"]
COW_RAMP_START_DAY = PARAMS["COW_RAMP_START_DAY"]
SHEEP_RAMP_START_DAY = PARAMS["SHEEP_RAMP_START_DAY"]
SHEEP_RAMP_RATE = PARAMS["SHEEP_RAMP_RATE"]
ANIMAL_RESERVE_EXTRA = PARAMS["ANIMAL_RESERVE_EXTRA"]
TERMINAL_STEP = 712         # from here: walk to shed, drop, sell out
FIB = [1, 1]
while len(FIB) < 20:
    FIB.append(FIB[-1] + FIB[-2])

# ---- market price model (mirrors engine MARKET_PARAMS exactly) ----
MARKET_I0 = 10000
# item: (base, T, below_func, below_target, above_func, above_target)
MKT = {
    "WHEAT":      (25,  400, "sqrt",   0.80, "log",    0.20),
    "CARROT":     (35,  450, "log",    0.20, "sqrt",   0.70),
    "TOMATO":     (60,  200, "linear", 0.40, "sqrt",   0.60),
    "STRAWBERRY": (120, 100, "sqrt",   0.70, "linear", 1.60),
    "MELON":      (250, 300, "log",    0.20, "sq",     3.60),
    "EGG":        (50,  332, "linear", 0.40, "log",    0.20),
    "MILK":       (160, 122, "sqrt",   0.60, "linear", 1.60),
    "WOOL":       (200, 105, "log",    0.20, "sq",     3.20),
    "FERTILIZER": (100, 200, "linear", 0.40, "linear", 0.40),
}


def _shape(func, x):
    x = max(0.0, x)
    if func == "linear": return x
    if func == "sq":     return x * x
    if func == "sqrt":   return math.sqrt(x)
    if func == "log":    return math.log(1.0 + x)
    return x


def _mprice(item, inv):
    base, T, bf, bt, af, at = MKT[item]
    if inv < MARKET_I0:
        price = base + (bt * base / _shape(bf, T)) * _shape(bf, MARKET_I0 - inv)
    else:
        price = base - (at * base / _shape(af, T)) * _shape(af, inv - MARKET_I0)
    return max(1, int(round(price)))


# ---- town demand model (mirrors engine _town_consume / _end_of_day) ----
# Shops tick every 4 steps (single-product shops eat 2, multi eat 1 each);
# town center ticks every 12 steps (x2 from day 10, x4 from day 20), all
# products except FERTILIZER (zero drain: a fert glut never recovers).
# One random unopened shop unlocks at the start of every 3rd day -> ALL
# shops are open by day 24; P(shop open by day D) = min(1, slots/remaining).
TOWN_SHOPS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}


def _shop_amt(shop, item):
    prods = TOWN_SHOPS.get(shop, ())
    return (2 if len(prods) == 1 else 1) if item in prods else 0


def _center_mult(day):
    return 4 if day >= 20 else (2 if day >= 10 else 1)


# Canonical net market flow (inventory-adding sells minus product buys) of the
# dominant ladder lineage, per step, measured from executed transactions of
# local episodes (seed-invariant except the 710+ terminal dump, which is past
# our liquidation window anyway).  Replaced by generate script; see
# harness/bake_opp_sched.py.
OPP_SCHED = {'auto': {'WHEAT': [(1, -6), (2, -2), (3, -2), (4, -3), (6, -6), (8, -2), (10, 1), (11, -1), (23, 1), (24, -1), (25, 2), (26, -1), (30, 4), (32, -1), (33, -2), (35, -2), (39, -1), (49, -4), (50, -4), (63, 7), (64, -1), (74, -1), (75, -2), (76, 2), (78, 1), (82, -1), (84, -1), (88, -1), (96, 4), (97, -2), (98, -1), (99, -2), (119, 3), (120, 31), (121, -4), (122, -3), (123, -3), (125, -1), (137, -1), (139, -1), (144, 6), (145, -7), (146, -3), (147, -4), (150, -1), (152, -2), (154, -1), (157, -1), (160, -2), (161, 1), (162, 6), (164, -1), (166, -1), (167, 1), (168, -1), (169, -7), (170, 9), (171, -5), (174, -1), (178, -1), (181, 9), (182, 2), (183, -1), (184, 1), (185, -1), (188, -1), (190, 1), (192, -1), (193, -6), (194, -1), (195, -5), (197, -2), (198, -1), (200, -3), (216, 17), (217, -11), (218, 8), (219, -1), (223, -1), (225, -1), (226, -3), (227, -1), (228, -1), (231, -6), (233, -1), (236, -9), (241, -12), (242, 11), (243, -10), (257, -11), (258, -1), (259, -4), (264, -2), (265, -13), (266, -3), (267, -9), (269, -2), (271, -2), (273, -2), (289, -22), (290, -9), (291, -10), (295, -1), (296, 49), (297, 36), (299, 12), (302, 1), (306, 10), (310, -10), (312, -10), (313, -4), (314, -9), (315, -4), (316, -5), (318, -1), (319, -3), (320, -1), (321, -2), (322, -2), (323, -1), (324, -2), (325, -1), (326, -2), (327, -2), (328, -2), (329, -2), (330, -4), (332, -4), (334, -4), (335, -6), (336, -4), (337, -5), (338, -3), (339, -7), (341, -1), (343, 1), (344, 32), (345, -1), (346, 3), (347, -2), (361, -14), (362, -1), (363, -12), (365, -1), (367, -3), (369, -2), (370, -3), (373, -2), (374, 8), (375, -1), (379, 22), (385, -12), (386, -1), (387, -12), (389, -1), (391, 9), (393, -1), (394, -5), (396, -1), (397, -1), (399, -1), (401, 14), (409, -15), (410, -1), (411, -12), (413, -2), (415, -2), (416, -1), (418, -2), (419, -2), (420, -1), (421, 27), (423, 11), (426, -1), (433, -14), (434, -1), (435, -12), (437, -1), (439, -3), (441, -1), (442, -2), (444, -2), (445, 12), (446, 28), (457, -14), (458, -1), (459, -8), (461, -3), (463, -3), (465, -3), (466, -1), (467, -1), (468, -1), (469, 17), (474, -1), (477, 9), (481, -13), (482, -1), (483, -12), (485, -5), (488, -3), (490, -1), (491, -2), (493, 31), (494, 7), (496, 3), (502, 1), (505, -12), (506, -1), (507, -12), (509, -1), (512, -1), (514, -3), (515, -1), (516, -1), (517, 13), (518, 18), (529, -15), (530, -7), (531, -12), (533, -1), (536, -1), (538, -7), (539, -2), (540, -1), (542, 1), (543, -1), (544, -1), (545, -1), (553, -14), (554, -1), (555, -12), (557, -1), (559, -1), (560, 6), (561, -3), (562, 25), (563, -1), (565, 3), (566, 13), (567, -1), (568, 5), (569, -2), (570, -1), (571, -2), (572, -4), (573, -1), (574, -5), (575, 12), (576, -1), (577, -11), (578, -2), (579, -10), (581, -1), (583, -1), (584, 1), (585, -4), (586, -1), (587, -2), (588, 4), (589, -1), (591, -2), (592, -2), (593, 11), (594, -1), (595, -2), (596, -2), (597, -1), (598, 8), (599, 9), (600, -2), (601, -11), (602, -1), (603, -8), (607, -1), (609, -1), (610, -3), (611, -1), (612, -1), (613, -5), (614, 6), (617, -1), (641, 6), (645, 3), (650, 1), (652, -1), (655, -1), (658, -2), (659, -4), (662, -3), (663, -5), (667, 4), (679, -1), (680, -11), (690, 3), (692, 7), (693, 1), (694, 1), (705, 6), (709, 8)], 'FERTILIZER': [(31, 2), (48, 2), (73, 4), (79, 2), (92, 1), (96, 1), (117, 2), (120, 2), (139, 2), (144, 2), (158, 2), (167, 1), (168, 3), (180, 2), (184, 1), (192, 3), (215, 3), (230, 3), (236, 2), (240, 6), (262, 2), (264, 9), (288, 11), (305, 2), (308, 2), (312, 7), (333, 2), (336, 10), (360, 14), (384, 12), (408, 8), (423, 1), (424, 1), (430, 1), (432, 2), (450, 2), (460, 1), (464, 1), (480, 7), (497, 4), (501, 6), (528, 3), (547, 5), (552, 3), (554, 1), (555, 1), (568, 3), (569, 1), (576, 5), (595, 2), (600, 10), (619, 1), (622, 1), (649, 8), (673, 2), (697, 7)], 'WOOL': [(158, 6), (229, 4), (280, 6), (309, 4), (348, 6), (353, 6), (354, 6), (358, 6), (365, 4), (384, 4), (423, 4), (424, 3), (444, 2), (445, 2), (448, 2), (449, 2), (456, 4), (457, 4), (460, 1), (537, 24), (566, 3), (612, 9), (613, 9), (616, 3), (657, 1), (660, 1), (661, 1), (699, 21)], 'MILK': [(195, 6), (198, 12), (293, 9), (302, 3), (308, 2), (324, 3), (325, 3), (342, 3), (343, 1), (366, 9), (386, 9), (408, 9), (437, 3), (438, 8), (439, 1), (470, 9), (477, 6), (484, 15), (520, 2), (521, 2), (524, 2), (525, 2), (536, 15), (562, 4), (565, 4), (566, 1), (569, 1), (572, 2), (607, 6), (608, 15), (633, 15), (656, 9), (673, 12), (674, 3), (698, 9)], 'MELON': [(256, 6), (257, 6), (259, 6), (262, 6), (264, 18), (306, 6), (359, 6), (360, 6), (423, 4), (505, 2), (527, 6), (562, 6), (599, 6), (622, 6), (625, 6), (707, 1), (709, 8)], 'STRAWBERRY': [(416, 5), (417, 3), (436, 8), (450, 1), (464, 1), (465, 1), (466, 10), (467, 3), (474, 6), (481, 12), (492, 2), (497, 2), (499, 4), (501, 2), (504, 14), (527, 2), (528, 14), (547, 6), (548, 6), (551, 2), (561, 13), (566, 2), (569, 2), (573, 6), (576, 29), (589, 2), (595, 3), (598, 4), (600, 20), (617, 6), (619, 6), (632, 20), (657, 2), (661, 2), (663, 8), (664, 11), (681, 3), (682, 12), (697, 13)]}, 'manual': {'WHEAT': [(1, -6), (2, -2), (3, -2), (4, -3), (6, -6), (8, -2), (10, 1), (11, -1), (23, 1), (24, -1), (25, 2), (26, -1), (30, 4), (32, -1), (33, -2), (35, -2), (39, -1), (49, -4), (50, -4), (63, 7), (64, -1), (74, -1), (75, -2), (76, 2), (78, 1), (82, -1), (84, -1), (88, -1), (96, 4), (97, -2), (98, -1), (99, -2), (119, 3), (120, 31), (121, -4), (122, -3), (123, -3), (125, -1), (137, -1), (139, -1), (144, 6), (145, -7), (146, -3), (147, -4), (150, -1), (152, -2), (154, -1), (157, -1), (160, -2), (161, 1), (162, 6), (164, -1), (166, -1), (167, 1), (168, -1), (169, -7), (170, 9), (171, -5), (174, -1), (178, -1), (181, 9), (182, 2), (183, -1), (184, 1), (185, -1), (188, -1), (190, 1), (192, -1), (193, -6), (194, -1), (195, -5), (197, -2), (198, -1), (200, -3), (216, 17), (217, -11), (218, 8), (219, -1), (223, -1), (225, -1), (226, -3), (227, -1), (228, -1), (231, -6), (233, -1), (236, -9), (241, -12), (242, 11), (243, -10), (257, -11), (258, -1), (259, -4), (264, -2), (265, -13), (266, -3), (267, -9), (269, -2), (271, -2), (273, -2), (289, -22), (290, -9), (291, -10), (295, -1), (296, 49), (297, 36), (299, 12), (302, 1), (306, 10), (310, -10), (312, -10), (313, -4), (314, -9), (315, -4), (316, -5), (318, -1), (319, -3), (320, -1), (321, -2), (322, -2), (323, -1), (324, -2), (325, -1), (326, -2), (327, -2), (328, -2), (329, -2), (330, -4), (332, -4), (334, -4), (335, -6), (336, -4), (337, -5), (338, -3), (339, -7), (341, -1), (343, 1), (344, 32), (345, -1), (346, 3), (347, -2), (361, -14), (362, -1), (363, -12), (365, -1), (367, -3), (369, -2), (370, -3), (373, -2), (374, 8), (375, -1), (379, 22), (385, -12), (386, -1), (387, -12), (389, -1), (391, 9), (393, -1), (394, -5), (396, -1), (397, -1), (399, -1), (401, 14), (409, -15), (410, -1), (411, -12), (413, -2), (415, -2), (416, -1), (418, -2), (419, -2), (420, -1), (421, 27), (423, 11), (426, -1), (433, -14), (434, -1), (435, -12), (437, -1), (439, -3), (441, -1), (442, -2), (444, -2), (445, 12), (446, 28), (457, -14), (458, -1), (459, -8), (461, -3), (463, -3), (465, -3), (466, -1), (467, -1), (468, -1), (469, 17), (474, -1), (477, 9), (481, -13), (482, -1), (483, -12), (485, -5), (488, -3), (490, -1), (491, -2), (493, 31), (494, 7), (496, 3), (502, 1), (505, -12), (506, -1), (507, -12), (509, -1), (512, -1), (514, -3), (515, -1), (516, -1), (517, 13), (518, 18), (529, -15), (530, -7), (531, -12), (533, -1), (536, -1), (538, -7), (539, -2), (540, -1), (542, 1), (543, -1), (544, -1), (545, -1), (553, -14), (554, -1), (555, -12), (557, -1), (559, -1), (560, 6), (561, -3), (562, 25), (563, -1), (565, 3), (566, 13), (567, -1), (568, 5), (569, -2), (570, -1), (571, -2), (572, -4), (573, -1), (574, -5), (575, 12), (576, -1), (577, -11), (578, -2), (579, -10), (581, -1), (583, -1), (584, 1), (585, -4), (586, -1), (587, -2), (588, 4), (589, -1), (591, -2), (592, -2), (593, 11), (594, -1), (595, -2), (596, -2), (597, -1), (598, 8), (599, 9), (600, -2), (601, -11), (602, -1), (603, -8), (607, -1), (609, -1), (610, -3), (611, -1), (612, -1), (613, -5), (614, 6), (617, -1), (641, 6), (645, 3), (650, 1), (652, -1), (655, -1), (658, -2), (659, -4), (662, -3), (663, -5), (667, 4), (679, -1), (680, -11), (690, 3), (692, 7), (693, 1), (694, 1), (705, 6), (709, 8)], 'FERTILIZER': [(31, 2), (48, 2), (73, 4), (79, 2), (92, 1), (96, 1), (117, 2), (120, 2), (139, 2), (144, 2), (158, 2), (167, 1), (168, 3), (180, 2), (184, 1), (192, 3), (215, 3), (230, 3), (236, 2), (240, 6), (262, 2), (264, 9), (288, 11), (305, 2), (308, 2), (312, 7), (333, 2), (336, 10), (360, 14), (384, 12), (408, 8), (423, 1), (424, 1), (430, 1), (432, 2), (450, 2), (460, 1), (464, 1), (480, 7), (497, 4), (501, 6), (528, 3), (547, 5), (552, 3), (554, 1), (555, 1), (568, 3), (569, 1), (576, 5), (595, 2), (600, 10), (619, 1), (622, 1), (649, 8), (673, 2), (697, 7)], 'WOOL': [(158, 6), (229, 4), (280, 6), (309, 4), (348, 6), (353, 6), (354, 6), (358, 1), (366, 4), (384, 4), (423, 4), (424, 3), (445, 2), (449, 2), (457, 4), (461, 4), (466, 4), (470, 1), (538, 24), (566, 3), (613, 9), (617, 9), (625, 1), (627, 2), (657, 1), (661, 1), (699, 22)], 'MILK': [(195, 6), (198, 12), (294, 9), (302, 3), (308, 2), (325, 3), (343, 3), (344, 4), (367, 9), (387, 9), (408, 9), (438, 3), (439, 5), (440, 4), (471, 9), (477, 6), (485, 15), (521, 2), (525, 2), (537, 19), (562, 4), (566, 4), (569, 1), (573, 3), (608, 6), (609, 9), (612, 6), (633, 15), (657, 9), (673, 4), (674, 8), (678, 3), (698, 9)], 'MELON': [(256, 6), (257, 6), (259, 6), (262, 6), (264, 18), (306, 6), (359, 5), (360, 7), (423, 4), (506, 2), (527, 6), (563, 6), (576, 6), (599, 6), (622, 6), (625, 6), (649, 6), (707, 1), (709, 8)], 'STRAWBERRY': [(417, 5), (421, 3), (437, 8), (450, 1), (465, 1), (467, 10), (468, 1), (469, 1), (470, 2), (474, 6), (482, 12), (493, 2), (497, 2), (499, 4), (501, 2), (504, 14), (527, 2), (528, 14), (547, 6), (548, 6), (551, 2), (562, 13), (566, 2), (569, 2), (573, 6), (576, 29), (589, 2), (595, 3), (599, 4), (600, 20), (617, 6), (619, 6), (633, 20), (657, 2), (661, 2), (664, 8), (665, 4), (668, 7), (681, 3), (682, 12), (697, 13)]}}

_ANIMAL_PRODUCT = {"COW": "MILK", "SHEEP": "WOOL", "GOOSE": "EGG"}
# expected own production, units/day per source (for future-glut prediction)
_OWN_RATE_DAY = {"MILK": 1.5, "WOOL": 1.35, "EGG": PARAMS["EGG_RATE_DAY"],
                 "STRAWBERRY": 0.55, "MELON": 0.5, "FERTILIZER": 0.35}


class Outlook:
    """Market microstructure model: exact town-drain forward sim + opponent
    flow model (baked lineage schedule, verified online by inventory
    residuals: delta_inv - predicted_drain - our_flows = opponent net flow;
    falls back to an EMA flow extrapolation when the residuals reject the
    schedule).  fb(item) = haircut best future first-unit quote within the
    horizon -- the opportunity cost of selling a held unit now."""

    def __init__(self):
        self.prev = None          # (step, inv snapshot, open shops)
        self.our_net = {}         # our net inventory adds ordered last turn
        self.cum_opp = {}         # item -> cumulative observed opp net flow
        self.rate = {}            # item -> EMA per-step opp net flow
        self.sched_cum = {n: {} for n in OPP_SCHED}
        self.sched_ptr = {n: {} for n in OPP_SCHED}
        self.active = next(iter(OPP_SCHED), None)  # lineage prior
        self.err = None
        self.opp_active = False   # opponent shows any economic activity
        self.opp_money0 = None
        self.lineage = None       # one-shot classification at LINEAGE_LOCK_STEP

    @staticmethod
    def _drain_at(step, shops):
        out = {}
        if step % 4 == 0:
            for shop in shops:
                prods = TOWN_SHOPS.get(shop, ())
                amt = 2 if len(prods) == 1 else 1
                for it in prods:
                    out[it] = out.get(it, 0) + amt
        if step % 12 == 0:
            m = _center_mult(step // 24)
            for it in MKT:
                if it != "FERTILIZER":
                    out[it] = out.get(it, 0) + m
        return out

    def observe(self, step, inv, shops):
        if inv and self.prev is not None and self.prev[0] == step - 1:
            pstep, pinv, pshops = self.prev
            drain = self._drain_at(pstep, pshops)
            for item in MKT:
                if item not in inv or item not in pinv:
                    continue
                opp = (inv[item] - pinv[item]
                       - self.our_net.get(item, 0) + drain.get(item, 0))
                self.cum_opp[item] = self.cum_opp.get(item, 0) + opp
                r = self.rate.get(item, 0.0)
                self.rate[item] = r + 0.05 * (opp - r)
            best, best_err = None, None
            for name, sched in OPP_SCHED.items():
                cums, ptrs = self.sched_cum[name], self.sched_ptr[name]
                num = den = 0.0
                for item, entries in sched.items():
                    i = ptrs.get(item, 0)
                    c = cums.get(item, 0)
                    while i < len(entries) and entries[i][0] <= pstep:
                        c += entries[i][1]
                        i += 1
                    ptrs[item], cums[item] = i, c
                    num += abs(self.cum_opp.get(item, 0) - c)
                    den += abs(c)
                e = num / (30.0 + den)
                if best_err is None or e < best_err:
                    best, best_err = name, e
            self.err = best_err
            if best is not None and (step < PARAMS["SCHED_MIN_STEP"]
                                     or best_err <= PARAMS["SCHED_ERR_MAX"]):
                self.active = best
            else:
                self.active = None
            if self.lineage is None and step >= PARAMS["LINEAGE_LOCK_STEP"]:
                # by now every ladder-lineage tape has near-zero residuals
                # while reactive agents (incl. ourselves) sit far outside
                self.lineage = (best_err is not None
                                and best_err <= PARAMS["LINEAGE_ERR_MAX"])
        if inv:
            self.prev = (step, dict(inv), list(shops))
        self.our_net = {}

    def fb(self, item, inv_now, held, own_rate_day, step):
        if inv_now is None or item not in MKT or item == "WHEAT":
            return None
        K = min(PARAMS["OUT_HORIZON"], LIQ_END - step)
        if K < PARAMS["OUT_SAMPLE"]:
            return None
        shops = self.prev[2] if self.prev else []
        open_amt = sum(_shop_amt(s, item) for s in shops)
        unopened = [s for s in TOWN_SHOPS if s not in shops]
        r = len(unopened)
        un_amt = sum(_shop_amt(s, item) for s in unopened)
        flows = ()
        rate = 0.0
        if self.active is not None:
            entries = OPP_SCHED.get(self.active, {}).get(item, ())
            flows = [(s, n) for (s, n) in entries if step < s <= step + K]
        else:
            rate = max(-1.5, min(1.5, self.rate.get(item, 0.0)))
        own = PARAMS["FB_OWNPROD_W"] * own_rate_day / 24.0
        over = PARAMS["FB_OVERHANG_W"] * held
        # holding is not free: our own production keeps stacking onto the
        # stock that must eventually cross the same ladder
        over_g = (PARAMS["FB_OVERHANG_W"] * PARAMS["FB_OVERHANG_GROWTH"]
                  * own_rate_day / 24.0)
        day0_slots = (step // 24) // 3
        inv_k = float(inv_now)
        fi = 0
        best = None
        sample = PARAMS["OUT_SAMPLE"]
        for k in range(1, K + 1):
            s = step + k
            while fi < len(flows) and flows[fi][0] == s:
                inv_k += flows[fi][1]
                fi += 1
            inv_k += rate + own
            if s % 4 == 0:
                inv_k -= open_amt
                if r and un_amt:
                    slots = (s // 24) // 3 - day0_slots
                    if slots > 0:
                        inv_k -= min(1.0, slots / float(r)) * un_amt
            if s % 12 == 0 and item != "FERTILIZER":
                inv_k -= _center_mult(s // 24)
            if k % sample == 0 or k == K:
                q = _mprice(item, int(round(inv_k + over + over_g * k)))
                if best is None or q > best:
                    best = q
        return PARAMS["FB_HAIRCUT"] * best if best is not None else None


# Sell dosing.  The meta is deficit-dominated: town drain outruns supply for
# most goods until late game, so prices sit ABOVE base and selling fast into
# the shared demand pool (before the opponent fills it) is right.  Floors are
# crash protection only: when the opponent's dump has walked a premium price
# deep below base, hold and let town drain recover it.  FERTILIZER has zero
# town drain (no shop, excluded from town center) -> a fert glut never
# recovers, so sell it first-mover, no floor.
SELL_FLOOR_FRAC = {"STRAWBERRY": PARAMS["FLOOR_STRAWBERRY"],
                   "MELON": PARAMS["FLOOR_MELON"],
                   "MILK": PARAMS["FLOOR_MILK"],
                   "WOOL": PARAMS["FLOOR_WOOL"],
                   "TOMATO": PARAMS["FLOOR_TOMATO"],
                   "CARROT": PARAMS["FLOOR_CARROT"]}
LIQ_START = PARAMS["LIQ_START"]
LIQ_END = PARAMS["LIQ_END"]
LIQ_FLOOR_MIN = PARAMS["LIQ_FLOOR_MIN"]
LIQ_SPIKE_FRAC = PARAMS["LIQ_SPIKE_FRAC"]
SHED_SOFT = PARAMS["SHED_SOFT"]
SHED_SOFT_EVE = PARAMS["SHED_SOFT_EVE"]


WAR_FLOORS = {"MILK": PARAMS["WAR_FLOOR_MILK"],
              "STRAWBERRY": PARAMS["WAR_FLOOR_STRAWBERRY"],
              "MELON": PARAMS["WAR_FLOOR_MELON"],
              "WOOL": PARAMS["WAR_FLOOR_WOOL"]}


def _dose(item, inv, qty, step, ema=None, fb=None, war=False):
    """(units, modeled_revenue): max prefix of `qty` units whose marginal
    quotes (engine walks inventory +1 per unit sold) stay above the floor;
    in the endgame ramp additionally force ceil(qty/turns_left) per turn so
    stock spreads across crash/recovery cycles instead of dumping at 719.
    The forced quota fires at predicted local peaks (quote >= fb gate when
    the outlook is live, price-EMA spike otherwise).  Below the floor a
    rate-limited crash-flush sells while the marginal quote still beats the
    predicted future best (fb) -- a deepening glut only gets worse, so the
    floor must not hold stock through it; when fb says the market recovers
    (town drain / shop unlocks), the floor holds as before."""
    frac = SELL_FLOOR_FRAC.get(item, 0.0)
    flush_cap = PARAMS["FLUSH_RATE"]
    if war:
        # vs a detected fixed-schedule tape: drop the floor on products where
        # its book is the bigger holder (holding just props its quotes; the
        # mutual crash costs the bigger book more), and flush faster out of
        # deepening gluts.  Never sell into the $1 band: at the price floor
        # sales stop raising market inventory, so there is nothing to deny.
        wf = WAR_FLOORS.get(item)
        if wf is not None and wf >= 0:
            frac = wf
        flush_cap = PARAMS["WAR_FLUSH_RATE"]
    forced = 0
    if step >= LIQ_START:
        t = max(0.0, (LIQ_END - step) / float(LIQ_END - LIQ_START))
        frac = max(frac * t, LIQ_FLOOR_MIN if frac > 0 else 0.0)
        forced = -(-qty // max(1, LIQ_END - step + 1))
        if inv is not None and step < LIQ_END - 10:
            if fb is not None and PARAMS["LIQ_USE_FB"]:
                ref = fb * PARAMS["LIQ_FB_GATE"]
            elif ema is not None:
                ref = max(ema, LIQ_SPIKE_FRAC * MKT[item][0])
            else:
                ref = None
            if ref is not None and _mprice(item, inv) < ref:
                forced = 0          # defer the quota to a real spike
    if inv is None:                     # no market inventory in obs: dump
        return qty, _mprice(item, MARKET_I0) * qty
    floor_price = frac * MKT[item][0]
    units, rev = 0, 0
    while units < qty:
        p = _mprice(item, inv + units)
        if p < floor_price and units >= forced:
            break
        rev += p
        units += 1
    if fb is not None and fb < floor_price and units < qty:
        flushed = 0
        while units < qty and flushed < flush_cap:
            p = _mprice(item, inv + units)
            if p < fb or p < PARAMS["FLUSH_MIN_PRICE"]:
                break
            rev += p
            units += 1
            flushed += 1
    return units, rev


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _unlocked(tiles, x, y):
    t = tiles[y][x]
    return not (isinstance(t, str) and t == "LOCKED")


def _nearest_shed(pos, tiles):
    open_tiles = [s for s in SHED_TILES if _unlocked(tiles, s[0], s[1])]
    return min(open_tiles, key=lambda s: _dist(pos, s)) if open_tiles else (4, 4)


def _at_shed(pos, tiles):
    """Standing on an UNLOCKED shed-access tile (hands can spawn on locked
    ones, where every action except movement is a silent no-op)."""
    return pos in SHED_TILES and _unlocked(tiles, pos[0], pos[1])


def _step_toward(pos, target, tiles):
    x, y = pos
    tx, ty = target
    if (x, y) == (tx, ty):
        return None
    prefs = []
    if abs(tx - x) >= abs(ty - y):
        if tx > x: prefs.append(("EAST", x + 1, y))
        if tx < x: prefs.append(("WEST", x - 1, y))
        if ty > y: prefs.append(("SOUTH", x, y + 1))
        if ty < y: prefs.append(("NORTH", x, y - 1))
    else:
        if ty > y: prefs.append(("SOUTH", x, y + 1))
        if ty < y: prefs.append(("NORTH", x, y - 1))
        if tx > x: prefs.append(("EAST", x + 1, y))
        if tx < x: prefs.append(("WEST", x - 1, y))
    for op, nx, ny in prefs:
        if 0 <= nx < 10 and 0 <= ny < 10 and _unlocked(tiles, nx, ny):
            return op
    for op, dx, dy in (("NORTH", 0, -1), ("WEST", -1, 0), ("SOUTH", 0, 1), ("EAST", 1, 0)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < 10 and 0 <= ny < 10 and _unlocked(tiles, nx, ny):
            return op
    return None


def _op_to_action(op):
    if op.startswith("PLANT:"):
        return ["PLANT", op[6:]]
    if op.startswith("PLACE:"):
        return ["PLACE", op[6:]]
    return [op]


def _op_rank(op):
    return OP_RANK.get(op.split(":", 1)[0], 99)


class Brain:
    def __init__(self):
        self.plan_day = -1
        self.routes = []        # routes[slot] = ordered list of (x, y) stops
        self.planned_hands = 2
        self.price_ema = {}     # item -> smoothed market price (spike detector)
        self.outlook = Outlook()

    @staticmethod
    def _mk_hold(item, prices, fbs):
        """Crashed but predicted to recover: keep yield banked on the tile or
        animal (free storage) instead of harvesting into a dead market."""
        frac = SELL_FLOOR_FRAC.get(item)
        if not frac:
            return False
        p, fbv = prices.get(item), fbs.get(item)
        if p is None or fbv is None:
            return False
        return p < frac * MKT[item][0] and fbv > p * PARAMS["HOLD_RECOV_MULT"]

    # ------------------------------------------------------------------ plan
    def _plan_day(self, jobs, day, extra_load):
        stops = {}
        for (x, y, op), prio in jobs.items():
            if prio >= 9:
                continue        # luxury jobs never anchor a route
            stops[(x, y)] = stops.get((x, y), 0) + 1
        order = []
        for y in range(10):
            xs = range(10) if y % 2 == 0 else range(9, -1, -1)
            for x in xs:
                if (x, y) in stops:
                    order.append((x, y))
        inc = []
        prev = (4, 4)
        for t in order:
            inc.append(stops[t] + _dist(prev, t))
            prev = t
        total = sum(inc)
        load = total * 1.15 + extra_load
        cap = MAX_HANDS_BY_DAY.get(day, DEFAULT_MAX_HANDS)
        hands = max(MIN_HANDS, min(cap, int(load / HAND_CAPACITY + 0.999)))
        k = hands + 1
        routes = []
        i = 0
        remaining = float(total)
        for r in range(k):
            if i >= len(order):
                break
            target = remaining / (k - r)
            chunk = [order[i]]
            acc = inc[i]
            i += 1
            while i < len(order) and acc + inc[i] <= target + 1.0:
                chunk.append(order[i])
                acc += inc[i]
                i += 1
            remaining -= acc
            d0 = min(_dist(chunk[0], s) for s in SHED_TILES)
            d1 = min(_dist(chunk[-1], s) for s in SHED_TILES)
            if d1 < d0:
                chunk.reverse()
            routes.append(chunk)
        while i < len(order):        # spillover (shouldn't happen often)
            routes[-1].append(order[i])
            i += 1
        self.routes = routes
        self.planned_hands = hands
        self.plan_day = day

    # ------------------------------------------------------------- unit move
    def _pending(self, tile_xy, jobs_at, claimed, inv, shed, seeds, plant_issued):
        d = jobs_at.get(tile_xy)
        if not d:
            return []
        out = []
        for op in sorted(d, key=_op_rank):
            if (tile_xy[0], tile_xy[1], op) in claimed:
                continue
            base = op.split(":", 1)[0]
            # FEED stays feasible with no wheat anywhere: the market buys it
            # and the feeder waits at the shed (starved animals are the one
            # unrecoverable loss)
            if base == "FERTILIZE" and inv.get("FERTILIZER", 0) <= 0 and shed.get("FERTILIZER", 0) <= 0:
                continue
            if base == "PLACE" and inv.get(op[6:], 0) <= 0 and shed.get(op[6:], 0) <= 0:
                continue
            if base == "PLANT" and seeds.get(op[6:], 0) - plant_issued.get(op[6:], 0) <= 0:
                continue
            out.append(op)
        return out

    def _unit_action(self, i, pos, n_units, tiles, jobs, jobs_at, claimed, heads,
                     inv, shed, seeds, plant_issued, day, hour):
        # spawned on a locked tile (engine quirk): only movement works there
        if not _unlocked(tiles, pos[0], pos[1]):
            mv = _step_toward(pos, (4, 4), tiles)
            return [mv] if mv else ["PASS"]
        route = self.routes[i]

        def pending(t):
            return self._pending(t, jobs_at, claimed, inv, shed, seeds, plant_issued)

        for _ in range(3):
            while route and not pending(route[0]):
                route.pop(0)
            if route:
                break
            # adopt an orphan route (planned for a hire that didn't happen yet)
            adopted = False
            for j in range(n_units, len(self.routes)):
                if self.routes[j]:
                    self.routes[i] = self.routes[j]
                    self.routes[j] = []
                    route = self.routes[i]
                    adopted = True
                    break
            if adopted:
                continue
            routed = {t for r in self.routes for t in r}
            # claim the best unrouted real job
            best = None
            for (x, y, op), prio in jobs.items():
                if prio > 7 or (x, y) in routed or (x, y, op) in claimed:
                    continue
                base = op.split(":", 1)[0]
                if base == "PLANT" and seeds.get(op[6:], 0) - plant_issued.get(op[6:], 0) <= 0:
                    continue
                score = prio * 6 + _dist(pos, (x, y))
                if best is None or score < best[0]:
                    best = (score, (x, y))
            if best is not None:
                # mini-tour: sweep up other unrouted jobs near the claim
                T = best[1]
                near = sorted(
                    {(x, y) for (x, y, op), prio in jobs.items()
                     if prio <= 7 and (x, y) != T and (x, y) not in routed
                     and _dist((x, y), T) <= 2},
                    key=lambda t: _dist(t, T))[:3]
                route.append(T)
                route.extend(near)
                continue
            # steal the tail of the thickest active route
            cand = None
            for j in range(min(n_units, len(self.routes))):
                if j == i:
                    continue
                L = len(self.routes[j])
                if L >= 6 and (cand is None or L > cand[0]):
                    cand = (L, j)
            if cand is not None:
                L, j = cand
                keep = L - L // 2
                self.routes[i] = self.routes[j][keep:]
                self.routes[j] = self.routes[j][:keep]
                route = self.routes[i]
                continue
            # adjacent luxury jobs (prio 9): better than idling, but never
            # worth a hike — the mandatory route tomorrow visits them cheaper
            best = None
            for (x, y, op), prio in jobs.items():
                if prio < 8 or (x, y) in routed or (x, y, op) in claimed:
                    continue
                score = _dist(pos, (x, y))
                if score <= 1 and (best is None or score < best[0]):
                    best = (score, (x, y))
            if best is not None:
                route.append(best[1])
                continue
            break

        # evening depot: haul harvests to the shed while the market can still
        # sell them today -- goods still in hands at day-end burn once the
        # shed cap is hit.  Only when no survival-critical stop is pending.
        if day <= 28 and hour >= 19 and route and sum(inv.values()) >= 4:
            crit = any(pr <= 2 for t in route[:4]
                       for pr in jobs_at.get(t, {}).values())
            if not crit:
                if _at_shed(pos, tiles):
                    return ["DROP"]
                mv = _step_toward(pos, _nearest_shed(pos, tiles), tiles)
                if mv:
                    return [mv]

        if not route:
            thr = 1 if (day >= 29 or hour >= 19) else 4
            if sum(inv.values()) >= thr:
                if _at_shed(pos, tiles):
                    return ["DROP"]
                mv = _step_toward(pos, _nearest_shed(pos, tiles), tiles)
                return [mv] if mv else ["PASS"]
            return ["PASS"]

        tx, ty = route[0]
        ops = pending((tx, ty))
        if not ops:
            # adoption loop exhausted its 3 rounds with a still-unvalidated
            # head: don't crash the whole turn, just walk toward it
            mv = _step_toward(pos, (tx, ty), tiles)
            return [mv] if mv else ["PASS"]
        op = ops[0]
        base = op.split(":", 1)[0]

        # material logistics: batch pickups at the shed
        if _at_shed(pos, tiles):
            feed_need = sum(1 for t in route if "FEED" in jobs_at.get(t, {})) - inv.get("WHEAT", 0)
            if feed_need > 0 and shed.get("WHEAT", 0) > 0:
                take = min(feed_need, shed["WHEAT"], 8)
                shed["WHEAT"] -= take
                return ["PICKUP", "WHEAT", take]
            fert_need = min(5, sum(1 for t in route
                                   if "FERTILIZE" in jobs_at.get(t, {}))) - inv.get("FERTILIZER", 0)
            if fert_need > 0 and shed.get("FERTILIZER", 0) > 0:
                take = min(fert_need, shed["FERTILIZER"])
                shed["FERTILIZER"] -= take
                return ["PICKUP", "FERTILIZER", take]

        need_item = None
        if base == "FEED" and inv.get("WHEAT", 0) <= 0:
            need_item = "WHEAT"
        elif base == "FERTILIZE" and inv.get("FERTILIZER", 0) <= 0:
            need_item = "FERTILIZER"
        elif base == "PLACE" and inv.get(op[6:], 0) <= 0:
            need_item = op[6:]
        if need_item is not None:
            if _at_shed(pos, tiles) and shed.get(need_item, 0) > 0:
                if need_item == "WHEAT":
                    qty = min(8, shed["WHEAT"],
                              max(1, sum(1 for t in route if "FEED" in jobs_at.get(t, {}))))
                elif need_item == "FERTILIZER":
                    qty = min(5, shed["FERTILIZER"],
                              max(1, sum(1 for t in route if "FERTILIZE" in jobs_at.get(t, {}))))
                else:
                    qty = 1
                shed[need_item] -= qty
                return ["PICKUP", need_item, qty]
            # not at the shed, or the shed is empty: walk there and wait
            # (market restocks wheat within a couple of turns)
            mv = _step_toward(pos, _nearest_shed(pos, tiles), tiles)
            return [mv] if mv else ["PASS"]

        if pos == (tx, ty):
            claimed.add((tx, ty, op))
            if base == "PLANT":
                plant_issued[op[6:]] = plant_issued.get(op[6:], 0) + 1
            return _op_to_action(op)

        # opportunistic service of the tile underfoot (no fetches, not
        # somebody's current stop)
        if pos not in heads or heads.get(pos) == i:
            for hop in pending(pos):
                hbase = hop.split(":", 1)[0]
                if hbase == "FEED" and inv.get("WHEAT", 0) <= 0:
                    continue
                if hbase == "FERTILIZE" and inv.get("FERTILIZER", 0) <= 0:
                    continue
                if hbase == "PLACE" and inv.get(hop[6:], 0) <= 0:
                    continue
                claimed.add((pos[0], pos[1], hop))
                if hbase == "PLANT":
                    plant_issued[hop[6:]] = plant_issued.get(hop[6:], 0) + 1
                return _op_to_action(hop)
        mv = _step_toward(pos, (tx, ty), tiles)
        return [mv] if mv else ["PASS"]

    # ------------------------------------------------------------------ act
    def act(self, obs):
        player = int(obs.get("player", 0) or 0)
        step = int(obs.get("step", 0) or 0)
        day = int(obs.get("day", step // 24) or 0)
        hour = int(obs.get("hour", step % 24) or 0)
        farm = obs["farms"][player]
        tiles = farm["tiles"]
        private = obs.get("private", {}) or {}
        shed = dict(private.get("shed", {}) or {})
        seeds = dict(private.get("seeds", {}) or {})
        invs = [dict(i or {}) for i in (private.get("inventories", []) or [{}])]
        market = obs.get("market", {}) or {}
        prices = dict(market.get("prices", {}) or {})
        inv_mkt = market.get("inventory", {}) or {}
        town = obs.get("town", {}) or {}
        self.outlook.observe(step, inv_mkt,
                             town.get("unlocked_shops", []) or [])
        farms_all = obs.get("farms", []) or []
        if not self.outlook.opp_active and len(farms_all) == 2:
            of = farms_all[1 - player] or {}
            om = float(of.get("money", 0) or 0)
            if getattr(self.outlook, "opp_money0", None) is None:
                self.outlook.opp_money0 = om
            if om != self.outlook.opp_money0 or of.get("hands"):
                self.outlook.opp_active = True
        money = float(farm.get("money", 0) or 0)

        units = [tuple(farm["farmer"])] + [tuple(h) for h in farm.get("hands", [])]
        while len(invs) < len(units):
            invs.append({})
        n_units = len(units)
        actions = [None] * n_units
        market_orders = []

        # ---------- terminal: walk in, drop, sell out ----------
        if step >= TERMINAL_STEP:
            drop_now = []
            for i, pos in enumerate(units):
                if sum(invs[i].values()) > 0:
                    if _at_shed(pos, tiles):
                        actions[i] = ["DROP"]
                        drop_now.append(i)
                    else:
                        mv = _step_toward(pos, _nearest_shed(pos, tiles), tiles)
                        actions[i] = [mv] if mv else ["PASS"]
                else:
                    tile = tiles[pos[1]][pos[0]]
                    if (isinstance(tile, dict) and tile.get("yield_units", 0) > 0
                            and step <= 716 and not (tile.get("kind") == "PLANT"
                            and day - tile.get("planted_day", 0) < CROPS[tile["crop"]]["first"])):
                        actions[i] = ["HARVEST"]
                    else:
                        actions[i] = ["PASS"]
            sellable = dict(shed)
            for i in drop_now:
                for k, v in invs[i].items():
                    sellable[k] = sellable.get(k, 0) + v
            order = sorted(sellable, key=lambda k: -prices.get(k, 0) * sellable.get(k, 0))
            market_orders = [["SELL", k, sellable[k]] for k in order
                             if sellable.get(k, 0) > 0 and k in prices][:10]
            self.outlook.our_net = {
                o[1]: o[2] for o in market_orders if prices.get(o[1], 0) > 1}
            return {"farmer": actions[0] or ["PASS"],
                    "hands": [a or ["PASS"] for a in actions[1:]],
                    "market": market_orders}

        # ---------- census ----------
        plants, animals, weeds, empties, empty_pastures = [], [], [], [], []
        empty_coops = []
        counts = {}
        for y in range(10):
            for x in range(10):
                t = tiles[y][x]
                if t is None:
                    empties.append((x, y))
                elif isinstance(t, dict):
                    kind = t.get("kind")
                    if kind == "PLANT":
                        plants.append((x, y, t))
                        counts[t["crop"]] = counts.get(t["crop"], 0) + 1
                    elif kind == "WEED":
                        weeds.append((x, y))
                    elif t.get("animal"):
                        animals.append((x, y, t))
                        counts[t["animal"]] = counts.get(t["animal"], 0) + 1
                    elif kind == "PASTURE":
                        empty_pastures.append((x, y))
                    elif kind == "COOP":
                        empty_coops.append((x, y))

        carried = {}
        for inv in invs:
            for k, v in inv.items():
                carried[k] = carried.get(k, 0) + v
        total_fert = shed.get("FERTILIZER", 0) + carried.get("FERTILIZER", 0)

        # ---- market outlook: predicted best future quote per product ----
        own_rate = {
            "MILK": _OWN_RATE_DAY["MILK"] * counts.get("COW", 0),
            "WOOL": _OWN_RATE_DAY["WOOL"] * counts.get("SHEEP", 0),
            "EGG": _OWN_RATE_DAY["EGG"] * counts.get("GOOSE", 0),
            "STRAWBERRY": _OWN_RATE_DAY["STRAWBERRY"] * counts.get("STRAWBERRY", 0),
            "MELON": _OWN_RATE_DAY["MELON"] * counts.get("MELON", 0),
            "FERTILIZER": _OWN_RATE_DAY["FERTILIZER"]
                          * (counts.get("COW", 0) + counts.get("SHEEP", 0)
                             + counts.get("GOOSE", 0)),
        }
        fbs = {}
        for item in set(own_rate) | {k for k, v in shed.items()
                                     if v > 0 and k in MKT}:
            held = shed.get(item, 0) + carried.get(item, 0)
            fbs[item] = self.outlook.fb(item, inv_mkt.get(item), held,
                                        own_rate.get(item, 0.0), step)
        # harvest-defer only against a live NON-lineage opponent: the baked
        # tape keeps selling through our hold (we would just prop its quotes;
        # measured -1..-2k/batch and -6.4k on solo seed 777004), while vs
        # caretaker-like opponents mutual holding wins the glut exchange
        # (mirror +3.9k with defer on)
        defer_ok = self.outlook.lineage is False and self.outlook.opp_active
        # late-evening harvest deferral: yield parks safely on non-dying tiles
        # overnight, while units' hauls burn at the day-end 100-cap shed drop
        defer_harvest = (hour >= 18 and day <= 27
                         and sum(shed.values()) + sum(carried.values())
                         > DEFER_HARVEST_THRESHOLD)

        # ---------- jobs ----------
        jobs = {}

        def add(prio, x, y, op):
            key = (x, y, op)
            if key not in jobs or jobs[key] > prio:
                jobs[key] = prio

        for (x, y, t) in animals:
            a = ANIMALS[t["animal"]]
            dsf = day + 1 - t.get("placed_day", 0) - a["first"]
            prod_eve = dsf >= 0 and dsf % a["interval"] == 0
            bank = t.get("pending_care_bonus", 0)
            if not t.get("fed_today"):
                if day <= 27:
                    # tonight's payout consumes the bank only if fed: an unfed
                    # production eve wipes the whole banked care bonus.  A
                    # placement-day miss starts an escape countdown + delays
                    # the first care bank, so burst-placed animals feed today.
                    urgent = (t.get("consecutive_unfed", 0) >= 1
                              or (prod_eve and bank > 0)
                              or t.get("placed_day", -1) == day)
                    add(0 if urgent else 2, x, y, "FEED")
                elif day == 28 and prod_eve and bank > 0:
                    add(2, x, y, "FEED")
            if not t.get("cared_today") and day <= 27:
                # bank above max_held-1 can never pay out -- skip the op,
                # except on production eves where care banks into the NEXT cycle
                if bank < a["max_held"] - 1 or prod_eve:
                    add(4, x, y, "CARE")
            if t.get("fertilizer_available"):
                add(6, x, y, "COLLECT_FERTILIZER")
            yu = t.get("yield_units", 0)
            if yu >= 2 or (day >= 28 and yu >= 1):
                # tonight's payout (1+bank) could overflow max_held: harvest
                # anyway when close to the cap, else defer under shed pressure
                bank_room = (yu + 1 + t.get("pending_care_bonus", 0)
                             <= a["max_held"])
                # crashed-but-recovering market: bank yield on the animal
                # (free storage) until the town drain / shop unlock restores
                # the price, instead of harvesting into a dead market
                mk_hold = (defer_ok and bank_room and day <= 27
                           and step < LIQ_START
                           and self._mk_hold(_ANIMAL_PRODUCT.get(t["animal"]),
                                             prices, fbs))
                if not (defer_harvest and bank_room) and not mk_hold:
                    add(3, x, y, "HARVEST")

        fert_targets_melon = 0
        for (x, y, t) in plants:
            crop = CROPS[t["crop"]]
            age = day - t.get("planted_day", 0)
            cu = t.get("consecutive_unwatered", 0)
            yu = t.get("yield_units", 0)
            harvest_here = False
            dying = False
            if yu > 0 and age >= crop["first"]:
                if crop["ongoing"]:
                    # dying = final production reached, tile decays to weed soon
                    dying = t.get("max_lifespan_step", -1) >= 0
                    harvest_here = yu >= 2 or dying or day >= 28
                else:
                    harvest_here = age >= crop["max_day"] or yu >= crop["max_yield"] or day >= 28
            if (defer_ok and harvest_here and crop["ongoing"] and not dying
                    and day <= 26 and step < LIQ_START
                    and yu + 2 <= crop["max_yield"]
                    and self._mk_hold(t["crop"], prices, fbs)):
                harvest_here = False    # bank yield on the tile through the crash
            if harvest_here:
                # decay bleeds 1 unit/2 steps once past max_day (non-ongoing)
                # or past final production (ongoing dying): those must go now
                decaying = dying or (not crop["ongoing"] and age >= crop["max_day"])
                # under evening shed pressure only decaying tiles must be
                # cleared; everything else keeps its yield overnight loss-free
                if not defer_harvest or decaying or (
                        crop["ongoing"] and yu + 2 > crop["max_yield"]):
                    add(2 if decaying else 3, x, y, "HARVEST")
            if not t.get("watered_today"):
                if crop["ongoing"]:
                    dsf = day + 1 - t.get("planted_day", 0) - crop["first"]
                    prod_eve = dsf >= 0 and dsf % crop["interval"] == 0
                    fert_active = t.get("fertilized_until_day", -1) >= day
                    if cu >= 1 and day <= 28:
                        add(0, x, y, "WATER")
                    elif fert_active and prod_eve and day <= 28:
                        add(2, x, y, "WATER")
                    elif day <= 26:
                        add(9, x, y, "WATER")
                else:
                    ws = (crop["max_day"] + 1) // 2
                    if ws <= age <= crop["max_day"]:
                        add(0 if cu >= 1 else 2, x, y, "WATER")
                    elif cu >= 1 and day <= 28 and not harvest_here:
                        add(0, x, y, "WATER")
                    elif age < ws and day <= 26:
                        add(9, x, y, "WATER")
            if (t["crop"] == "STRAWBERRY" and day <= 26 and total_fert > 0
                    and t.get("fertilized_until_day", -1) < day):
                # the 3-day coverage window (day..day+2) must catch a
                # production eve; for interval-2 crops any mature day works
                dsf = day + 1 - t.get("planted_day", 0) - crop["first"]
                if dsf >= -2:
                    add(4, x, y, "FERTILIZE")
            if t["crop"] == "MELON":
                # fert+water on a window day pays +2/day: one fertilizer takes
                # a melon tile from 1 to its 6-cap in ~3 watered days ($230/u)
                ws_m = (crop["max_day"] + 1) // 2
                if ws_m - 1 <= age <= crop["max_day"] - 1 and yu < 5:
                    fert_targets_melon += 1
                    if (total_fert > 0 and day <= 27
                            and t.get("fertilized_until_day", -1) < day):
                        add(4, x, y, "FERTILIZE")

        # animals waiting in the shed (or carried): build pasture + place
        animals_avail = (shed.get("COW", 0) + shed.get("SHEEP", 0)
                         + carried.get("COW", 0) + carried.get("SHEEP", 0))
        for (x, y) in empty_pastures[:animals_avail]:
            animal = "COW" if shed.get("COW", 0) + carried.get("COW", 0) > 0 else "SHEEP"
            add(2, x, y, "PLACE:" + animal)
        need_pastures = animals_avail - len(empty_pastures)
        if need_pastures > 0:
            for (x, y) in sorted(empties, key=lambda p: _dist(p, (4, 4)))[:need_pastures]:
                add(2, x, y, "BUILD_PASTURE")
                empties.remove((x, y))
        # geese waiting in the shed: build coop + place (coops are a separate
        # structure; PLACE checks tile.kind == COOP)
        geese_avail = shed.get("GOOSE", 0) + carried.get("GOOSE", 0)
        for (x, y) in empty_coops[:geese_avail]:
            add(2, x, y, "PLACE:GOOSE")
        need_coops = geese_avail - len(empty_coops)
        if need_coops > 0:
            gf = PARAMS["GOOSE_FAR"]
            for (x, y) in sorted(empties,
                                 key=lambda p: (abs(_dist(p, (4, 4)) - gf),
                                                _dist(p, (4, 4))))[:need_coops]:
                add(2, x, y, "BUILD_COOP")
                empties.remove((x, y))

        # planting plan (high-value crops claim empty tiles first; from
        # MELON_FIRST_DAY the melon replant outranks the strawberry ramp so
        # the d10-13 wave-2 tiles are not stolen while cows finish placing)
        desired = []
        if day >= PARAMS["MELON_FIRST_DAY"] and day <= MELON_LAST_PLANT_DAY:
            desired.append(("MELON", max(MELON_WAVE, counts.get("MELON", 0))))
        if 1 <= day <= STRAWBERRY_LAST_PLANT_DAY:
            desired.append(("STRAWBERRY", TARGET_STRAWBERRY))
        if day < PARAMS["MELON_FIRST_DAY"] and day <= MELON_LAST_PLANT_DAY:
            desired.append(("MELON", max(MELON_WAVE, counts.get("MELON", 0))))
        if LATE_WHEAT_FROM_DAY <= day <= LATE_WHEAT_LAST_PLANT_DAY:
            # endgame filler: ops are cheap now, wheat matures in 4 days
            desired.append(("WHEAT", EARLY_WHEAT_TILES + len(empties)))
        elif STRAWBERRY_LAST_PLANT_DAY < day < LATE_WHEAT_FROM_DAY:
            # modest mid-game wheat only: every extra wheat tile costs water +
            # harvest ops that $230-melons and animal care need more
            desired.append(("WHEAT", EARLY_WHEAT_TILES + MID_WHEAT_EXTRA))
        elif day < LATE_WHEAT_FROM_DAY:
            desired.append(("WHEAT", EARLY_WHEAT_TILES))
        seed_deficit = {}
        for crop, tgt in desired:
            deficit = tgt - counts.get(crop, 0)
            if deficit <= 0:
                continue
            if hour <= 21:
                plantable = min(deficit, seeds.get(crop, 0))
                spots = sorted(empties, key=lambda p: _dist(p, (4, 4)))[:plantable]
                for (x, y) in spots:
                    add(7, x, y, "PLANT:" + crop)
                    empties.remove((x, y))
            missing = deficit - seeds.get(crop, 0)
            if missing > 0:
                seed_deficit[crop] = min(missing, 6)
        if day <= 26:
            for (x, y) in weeds:
                add(4, x, y, "DIG")

        # ---------- day plan ----------
        if day != self.plan_day:
            self._plan_day(jobs, day, extra_load=3 * sum(seed_deficit.values()))
        while len(self.routes) < n_units:
            self.routes.append([])

        # ---------- emergency override (late-day survival) ----------
        if 16 <= hour and day <= 28:
            covered = set()
            for j in range(min(n_units, len(self.routes))):
                covered.update(self.routes[j][:3])
            emg_load = [0] * n_units
            for (x, y, op), prio in sorted(jobs.items(), key=lambda kv: kv[1]):
                if prio != 0:
                    continue
                T = (x, y)
                if T in covered:
                    continue
                best_j, best_s = None, None
                for j in range(n_units):
                    s = _dist(units[j], T) + 5 * emg_load[j]
                    if op == "FEED" and invs[j].get("WHEAT", 0) <= 0:
                        s += 6
                    if best_s is None or s < best_s:
                        best_j, best_s = j, s
                for r in self.routes:
                    if T in r:
                        r.remove(T)
                self.routes[best_j].insert(0, T)
                emg_load[best_j] += 1
                covered.add(T)

        # ---------- mid-day insertion of new unrouted jobs ----------
        jobs_at = {}
        for (x, y, op), prio in jobs.items():
            jobs_at.setdefault((x, y), {})[op] = prio
        routed = {t for r in self.routes for t in r}
        if day <= 28 and all(self.routes[j] for j in range(n_units)):
            cands = sorted(((prio, (x, y)) for (x, y, op), prio in jobs.items()
                            if prio <= 7 and (x, y) not in routed), key=lambda c: c[0])
            for prio, T in cands[:3]:
                if T in routed:
                    continue
                best = None      # (delta, j, position)
                for j in range(n_units):
                    r = self.routes[j]
                    pts = [units[j]] + r
                    for p in range(len(r) + 1):
                        a = pts[p]
                        if p < len(r):
                            delta = _dist(a, T) + _dist(T, r[p]) - _dist(a, r[p])
                        else:
                            delta = _dist(a, T)
                        if best is None or delta < best[0]:
                            best = (delta, j, p)
                if best is not None:
                    _, j, p = best
                    self.routes[j].insert(p, T)
                    routed.add(T)

        # ---------- unit actions ----------
        heads = {}
        for j in range(min(n_units, len(self.routes))):
            if self.routes[j]:
                heads[self.routes[j][0]] = j
        claimed = set()
        plant_issued = {}
        for i, pos in enumerate(units):
            if actions[i] is None:
                actions[i] = self._unit_action(
                    i, pos, n_units, tiles, jobs, jobs_at, claimed, heads,
                    invs[i], shed, seeds, plant_issued, day, hour)

        # ---------- market ----------
        hungry = sum(1 for (_, _, t) in animals if not t.get("fed_today"))
        reserve = hungry * (prices.get("WHEAT", 25) + 5) + 60
        # one day of feeds + slack; a 2x buffer used to hog ~40 shed slots the
        # pressure-relief seller may not touch -> evening harvests burned at
        # the day-end 100-cap drop (measured 100-160 units/game vs v9_auto)
        if day <= 27:
            # include animals still in the shed/hands: burst placements used
            # to outrun the buffer and starve the newcomers on day one
            feed_buffer = (len(animals) + FEED_BUFFER_EXTRA
                           + shed.get("COW", 0) + shed.get("SHEEP", 0)
                           + shed.get("GOOSE", 0)
                           + carried.get("COW", 0) + carried.get("SHEEP", 0)
                           + carried.get("GOOSE", 0))
        elif day == 28:
            feed_buffer = 2
        else:
            feed_buffer = 0
        n_strawberries = counts.get("STRAWBERRY", 0)
        # fertilizer has no town drain: a glutted fert market never recovers,
        # so release the reserve early (day 22) instead of ramp-dumping at $30.
        # Melon window tiles pre-reserve theirs ($230/unit beats a $50 sale).
        fert_reserve = ((n_strawberries // 3 + 2)
                        if (n_strawberries > 0 and day <= FERT_RESERVE_UNTIL_DAY)
                        else 0)
        fert_reserve += fert_targets_melon if day <= 26 else 0

        avail = {}
        for item, qty in shed.items():
            if qty <= 0 or item not in prices:
                continue
            if item == "WHEAT":
                qty = max(0, qty - feed_buffer)
            if item == "FERTILIZER":
                qty = max(0, qty - fert_reserve)
            if qty > 0:
                avail[item] = qty
        for item, p in prices.items():
            e = self.price_ema.get(item)
            self.price_ema[item] = p if e is None else e + 0.1 * (p - e)
        # adversarial mode: only vs a live opponent positively classified as
        # the ladder lineage (one-shot lock at step 240); reactive/unknown
        # opponents and solo keep exact v0.8 market behavior (mirror gate)
        war = (bool(PARAMS["WAR_ENABLE"])
               and self.outlook.lineage is True
               and self.outlook.opp_active
               and step < PARAMS["WAR_END_STEP"])
        plan = {}                      # item -> [units, modeled revenue]
        for item, qty in avail.items():
            n_u, rv = _dose(item, inv_mkt.get(item), qty, step,
                            self.price_ema.get(item), fbs.get(item), war=war)
            if n_u > 0:
                plan[item] = [n_u, rv]
        # drain-tick alignment (see WAR_TICK_DEFER): on tick steps defer
        # small doses one step so they sell into the post-drain quote
        if (war and PARAMS["WAR_TICK_DEFER"] and step % 4 == 0
                and step < LIQ_END - 8):
            for item in list(plan):
                if item == "FERTILIZER":
                    continue        # fert has zero town drain
                if plan[item][0] <= PARAMS["WAR_TICK_MAX"]:
                    del plan[item]
        # shed-pressure relief: overflow burns at day-end hand dissolution,
        # so when shed+carried outgrow the soft cap, sell extra units past
        # the floors -- preferring items with the least predicted upside
        # (venting a recoverable product at its crash price is the one
        # mistake this layer used to make; fb prices that regret in).
        pressure = (sum(shed.values()) + sum(carried.values())
                    - sum(u for u, _ in plan.values()))
        soft = SHED_SOFT_EVE if hour >= 21 else SHED_SOFT
        while pressure > soft:
            best = None
            for item, qty in avail.items():
                sold = plan.get(item, [0, 0])[0]
                if sold >= qty:
                    continue
                p = _mprice(item, inv_mkt[item] + sold) if item in inv_mkt else prices[item]
                fbv = fbs.get(item)
                score = p - PARAMS["RELIEF_RECOV_W"] * max(0, (fbv or p) - p)
                if best is None or score > best[0]:
                    best = (score, p, item)
            if best is None:
                break
            _, p, item = best
            plan.setdefault(item, [0, 0])
            plan[item][0] += 1
            plan[item][1] += p
            pressure -= 1
        sellable = sorted(((rev, item, units) for item, (units, rev) in plan.items()),
                          reverse=True)
        for _, item, qty in sellable:
            market_orders.append(["SELL", item, qty])
        budget = money + sum(v for v, _, _ in sellable)

        # feed wheat BEFORE hires: on cash-crunch games hires drained the
        # budget, the wheat buy failed and animals starved into escapes
        wheat_stock = shed.get("WHEAT", 0) + carried.get("WHEAT", 0)
        if wheat_stock < feed_buffer and day <= 27:
            unit_cost = prices.get("WHEAT", 25) + 3
            # partial buys: at a cash trough an all-or-nothing 6-wheat order
            # bought nothing and the herd starved into escapes
            need = min(feed_buffer - wheat_stock, 6, int(budget // unit_cost))
            if need > 0:
                market_orders.append(["BUY_PRODUCT", "WHEAT", need])
                budget -= need * unit_cost

        # hire right after sells: hands are the cheapest force multiplier.
        # Spawn = least-occupied shed-access tile (NWSE tie-break), computed on
        # post-move positions; (5,5) is a trap while NE/SW/SE are all locked.
        unlocked_quads = set(farm.get("unlocked_quadrants", ["NW"]))
        hires_today = int(farm.get("hires_today", 0) or 0)
        post_pos = []
        for i, pos in enumerate(units):
            a = actions[i]
            p = pos
            if a and a[0] in MOVES:
                dx, dy = MOVES[a[0]]
                nx, ny = pos[0] + dx, pos[1] + dy
                if 0 <= nx < 10 and 0 <= ny < 10 and _unlocked(tiles, nx, ny):
                    p = (nx, ny)
            post_pos.append(p)
        occ = {s: 0 for s in SHED_TILES}
        for p in post_pos:
            if p in occ:
                occ[p] += 1
        trap_live = not ({"NE", "SW", "SE"} & unlocked_quads)
        n_ordered_hires = 0
        while (n_units - 1 + n_ordered_hires < self.planned_hands and hour <= 18
               and len(market_orders) < 9):
            spawn = min(SHED_TILES, key=lambda s: (occ[s], SHED_TILES.index(s)))
            if trap_live and spawn == (5, 5):
                break
            cost = FIB[min(hires_today + n_ordered_hires, len(FIB) - 1)]
            guard = 0 if cost <= 8 else reserve
            if budget - guard < cost:
                break
            market_orders.append(["HIRE"])
            occ[spawn] += 1
            n_ordered_hires += 1
            budget -= cost

        # land
        n_extra = len(unlocked_quads) - 1
        if n_extra < PARAMS["LAND_MAX_EXTRA"]:
            land_price = [1000, 2000, 4000][n_extra]
            land_day = LAND_DAYS[n_extra]
            if day >= land_day and budget - reserve >= land_price:
                market_orders.append(["BUY_LAND"])
                budget -= land_price

        n_cows = counts.get("COW", 0) + shed.get("COW", 0) + carried.get("COW", 0)
        n_sheep = counts.get("SHEEP", 0) + shed.get("SHEEP", 0) + carried.get("SHEEP", 0)
        # geese: the tape's egg book is ZERO, so all egg town drain is
        # uncontested demand and the above-curve is log (dump-safe); a goose
        # pays back its $300 in ~3 producing days (2 eggs + 1 fertilizer/day).
        # Late geese are worthless (first yield is placement+4): buy early.
        n_geese = (counts.get("GOOSE", 0) + shed.get("GOOSE", 0)
                   + carried.get("GOOSE", 0))
        goose_cap = (0 if day < PARAMS["GOOSE_START_DAY"]
                     else min(PARAMS["GOOSE_TARGET"],
                              PARAMS["GOOSE_RATE"]
                              * (day - PARAMS["GOOSE_START_DAY"] + 1)))
        animal_reserve = reserve + ANIMAL_RESERVE_EXTRA
        goose_state = {"n": n_geese,
                       "budget_guard": reserve + PARAMS["GOOSE_RESERVE_EXTRA"]}

        def try_buy_goose(bdg):
            if (goose_state["n"] < goose_cap
                    and bdg - goose_state["budget_guard"] >= 300
                    and day <= PARAMS["GOOSE_LAST_DAY"]
                    and len(market_orders) < 10):
                market_orders.append(["BUY_ANIMAL", "GOOSE", 1])
                goose_state["n"] += 1
                return bdg - 300
            return bdg

        if PARAMS["GOOSE_PRIO"] >= 2:
            budget = try_buy_goose(budget)
        # sheep beat extra strawberry tiles for early cash: wool is the best
        # $/day in the game (4/cycle at $200+ base) and an early sheep repays
        # itself within two cycles; the meta fields 6 by day 8
        if (day <= EARLY_SHEEP_LAST_DAY and n_sheep < EARLY_SHEEP_TARGET
                and budget - reserve >= 500 and len(market_orders) < 10):
            market_orders.append(["BUY_ANIMAL", "SHEEP", 1])
            budget -= 500
            n_sheep += 1
        # seeds then animals
        for crop, need in sorted(seed_deficit.items(), key=lambda kv: CROPS[kv[0]]["seed"]):
            cost = need * CROPS[crop]["seed"]
            if budget - reserve >= cost and len(market_orders) < 10:
                market_orders.append(["BUY_SEED", crop, need])
                budget -= cost
        cow_cap = (COW_BASE if day < COW_RAMP_START_DAY
                   else min(TARGET_COWS, COW_BASE + (day - COW_RAMP_START_DAY + 1)))
        # sheep are the best $/day animal (4 wool per 3-day cycle, wool
        # chronically drained above base): front-load them like the meta does
        # (6 sheep by day 8) instead of trickling 1/day from day 5
        sheep_cap = (0 if day < SHEEP_RAMP_START_DAY
                     else min(TARGET_SHEEP,
                              SHEEP_RAMP_RATE * (day - SHEEP_RAMP_START_DAY + 1)))
        if PARAMS["GOOSE_PRIO"] == 1:
            budget = try_buy_goose(budget)
        if (n_sheep < sheep_cap and budget - animal_reserve >= 500
                and day <= ANIMAL_BUY_LAST_DAY and len(market_orders) < 10):
            market_orders.append(["BUY_ANIMAL", "SHEEP", 1])
            budget -= 500
        if (n_cows < cow_cap and budget - animal_reserve >= 400
                and day <= ANIMAL_BUY_LAST_DAY and len(market_orders) < 10):
            market_orders.append(["BUY_ANIMAL", "COW", 1])
            budget -= 400
        if PARAMS["GOOSE_PRIO"] == 0:
            budget = try_buy_goose(budget)

        our_net = {}
        for o in market_orders[:10]:
            if o[0] == "SELL" and prices.get(o[1], 0) > 1:
                our_net[o[1]] = our_net.get(o[1], 0) + int(o[2])
            elif o[0] == "BUY_PRODUCT":
                our_net[o[1]] = our_net.get(o[1], 0) - int(o[2])
        self.outlook.our_net = our_net

        return {"farmer": actions[0] or ["PASS"],
                "hands": [a or ["PASS"] for a in actions[1:]],
                "market": market_orders[:10]}


_BRAIN = Brain()


def agent(obs, config=None):
    try:
        return _BRAIN.act(obs)
    except Exception:
        return {"farmer": ["PASS"], "hands": [], "market": []}
