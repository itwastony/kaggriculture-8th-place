"""Encoding of an observation into numpy arrays (shared code for data collection, training and the submission).

encode(obs) -> dict:
  tiles  float32 [100, FT]   — own farm, tile = token
  units  float32 [16, FU]    — farmer + hired workers (zero padding), unit_pos int [16] (tile), unit_mask bool [16]
  glob   float32 [FG]        — time, money, shed, seeds, market, shops, summary of the opponent's farm
  tmask  bool [16, 100, NOPS] — validity of (target, operation) for each unit
  mmask  bool [sum bins]      — validity of market bins
"""

# This file is copied verbatim into the submitted agent (rl/export_agent.py), and entry_points/verify_submission.py
# compares its code with the archives sent to Kaggle: change only comments and formatting.
import numpy as np

from . import spec as S

KINDS = ("EMPTY", "LOCKED", "WEED", "PLANT", "COOP", "PASTURE")
ITEMS = S.PRODUCTS + S.ANIMALS  # 12 inventory/shed items
BASE_PRICE = {
    "WHEAT": 25,
    "CARROT": 35,
    "TOMATO": 60,
    "STRAWBERRY": 120,
    "MELON": 250,
    "EGG": 50,
    "MILK": 160,
    "WOOL": 200,
    "FERTILIZER": 100,
}
# tile: kind one-hot, crop one-hot, animal one-hot, 6 plant features, 7 animal features, (shed tile, x, y)
FT = len(KINDS) + len(S.CROPS) + len(S.ANIMALS) + 6 + 7 + 3
# unit: (is farmer, x, y), carried items, unit index
FU = 3 + len(ITEMS) + 1
FG = 0  # computed below (_fg encodes a minimal observation)
MB = sum(len(b) for _, _, b in S.MARKET_HEADS)
SHED_SET = frozenset(S.SHED_IDX)


def _tile_kind(t):
    if t is None:
        return "EMPTY"
    if isinstance(t, str):
        return "LOCKED" if t == "LOCKED" else "EMPTY"
    return t.get("kind", "EMPTY")


def _opp_summary(f):
    c = np.zeros(len(S.CROPS) + len(S.ANIMALS) + 5, np.float32)
    for row in f["tiles"]:
        for t in row:
            k = _tile_kind(t)
            if k == "PLANT":
                cr = t.get("crop")
                if cr in S.CROPS:
                    c[S.CROPS.index(cr)] += 1
                c[-3] += float(t.get("yield_units", 0) or 0)
            elif k in ("COOP", "PASTURE"):
                an = t.get("animal")
                if an in S.ANIMALS:
                    c[len(S.CROPS) + S.ANIMALS.index(an)] += 1
                    c[-2] += float(t.get("yield_units", 0) or 0)
            elif k == "WEED":
                c[-4] += 1
    c[-5] = len(f.get("hands") or [])
    c[-1] = len(f.get("unlocked_quadrants") or [])
    return c / np.array([25.0] * len(S.CROPS) + [10.0] * len(S.ANIMALS) + [16.0, 20.0, 100.0, 60.0, 4.0], np.float32)


def encode(obs):
    me = obs["player"]
    f = obs["farms"][me]
    fo = obs["farms"][1 - me]
    pv = obs["private"]
    day, step = int(obs["day"]), int(obs["step"])
    tiles = np.zeros((S.NT, FT), np.float32)
    kind_of = [None] * S.NT
    animal_of = [None] * S.NT
    for y, row in enumerate(f["tiles"]):
        for x, t in enumerate(row):
            i = x + S.N * y
            k = _tile_kind(t)
            kind_of[i] = k
            v = tiles[i]
            v[KINDS.index(k)] = 1
            o = len(KINDS)
            if k == "PLANT":
                cr = t.get("crop")
                if cr in S.CROPS:
                    v[o + S.CROPS.index(cr)] = 1
                p = o + len(S.CROPS) + len(S.ANIMALS)
                v[p + 0] = (day - int(t.get("planted_day", day))) / 16.0
                v[p + 1] = float(bool(t.get("watered_today")))
                v[p + 2] = float(t.get("consecutive_unwatered", 0) or 0) / 2.0
                v[p + 3] = float(t.get("yield_units", 0) or 0) / 6.0
                v[p + 4] = max(-1.0, min(1.0, (float(t.get("max_lifespan_step", step) or step) - step) / 240.0))
                v[p + 5] = float((t.get("fertilized_until_day") or -1) >= day)
            elif k in ("COOP", "PASTURE"):
                an = t.get("animal")
                if an in S.ANIMALS:
                    v[o + len(S.CROPS) + S.ANIMALS.index(an)] = 1
                    animal_of[i] = an
                p = o + len(S.CROPS) + len(S.ANIMALS) + 6
                v[p + 0] = float(bool(t.get("fed_today")))
                v[p + 1] = float(bool(t.get("cared_today")))
                v[p + 2] = float(t.get("consecutive_unfed", 0) or 0) / 2.0
                v[p + 3] = float(t.get("yield_units", 0) or 0) / 6.0
                v[p + 4] = float(t.get("fertilizer_available", 0) or 0)
                v[p + 5] = float(t.get("pending_care_bonus", 0) or 0) / 6.0
                v[p + 6] = (day - int(t.get("placed_day", day) or day)) / 10.0 if an else 0.0
            q = FT - 3
            v[q + 0] = float(i in SHED_SET)
            v[q + 1] = x / 9.0
            v[q + 2] = y / 9.0
    # units
    pos = [tuple(f["farmer"])] + [tuple(h) for h in (f.get("hands") or [])]
    nu = min(len(pos), S.MAX_UNITS)
    units = np.zeros((S.MAX_UNITS, FU), np.float32)
    unit_pos = np.zeros(S.MAX_UNITS, np.int64)
    unit_mask = np.zeros(S.MAX_UNITS, bool)
    invs = pv.get("inventories") or []
    carry = []
    for i in range(nu):
        x, y = pos[i]
        unit_pos[i] = x + S.N * y
        unit_mask[i] = True
        units[i, 0] = float(i == 0)
        units[i, 1] = x / 9.0
        units[i, 2] = y / 9.0
        inv = invs[i] if i < len(invs) else {}
        carry.append(inv)
        for j, it in enumerate(ITEMS):
            units[i, 3 + j] = float(inv.get(it, 0) or 0) / 10.0
        units[i, -1] = i / 16.0
    # global
    shed = pv.get("shed") or {}
    seeds = pv.get("seeds") or {}
    mk = obs["market"]
    px, minv = mk.get("prices") or {}, mk.get("inventory") or {}
    shops = list((obs.get("town") or {}).get("unlocked_shops") or [])
    hour = int(obs["hour"])
    g = [
        step / 720.0,
        day / 30.0,
        hour / 24.0,
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        float(f["money"]) / 1e5,
        float(fo["money"]) / 1e5,
        (float(f["money"]) - float(fo["money"])) / 1e5,
        np.log1p(max(0.0, float(f["money"]))) / 12.0,
        float(f.get("hires_today", 0) or 0) / 12.0,
        nu / 16.0,
        sum(float(v) for v in shed.values()) / 100.0,
    ]
    g += [float(shed.get(it, 0) or 0) / 50.0 for it in ITEMS]
    g += [float(seeds.get(c, 0) or 0) / 50.0 for c in S.CROPS]
    g += [float(px.get(p, BASE_PRICE[p]) or 0) / BASE_PRICE[p] for p in S.PRODUCTS]
    g += [(float(minv.get(p, 10000) or 10000) - 10000.0) / 200.0 for p in S.PRODUCTS]
    g += [shops.count(s) / 2.0 for s in S.SHOPS] + [len(shops) / 8.0]
    g += [float(q in (f.get("unlocked_quadrants") or [])) for q in S.QUADS]
    g += [float(q in (fo.get("unlocked_quadrants") or [])) for q in S.QUADS]
    glob = np.concatenate([np.asarray(g, np.float32), _opp_summary(fo)])
    # masks (target, operation)
    tmask = np.zeros((S.MAX_UNITS, S.NT, S.NOPS), bool)
    base = np.zeros((S.NT, S.NOPS), bool)
    base[:, S.OP_ID["STAY"]] = True
    for i in range(S.NT):
        k = kind_of[i]
        if k == "EMPTY":
            for c in S.CROPS:  # seeds may arrive via the market while the unit walks to the tile
                base[i, S.OP_ID[f"PLANT_{c}"]] = True
            base[i, S.OP_ID["BUILD_COOP"]] = True
            base[i, S.OP_ID["BUILD_PASTURE"]] = True
        elif k == "PLANT":
            for o in ("WATER", "HARVEST", "FERTILIZE", "DIG"):
                base[i, S.OP_ID[o]] = True
        elif k == "WEED":
            base[i, S.OP_ID["DIG"]] = True
        elif k in ("COOP", "PASTURE"):
            if animal_of[i]:
                for o in ("FEED", "HARVEST", "COLLECT_FERTILIZER", "CARE"):
                    base[i, S.OP_ID[o]] = True
            else:
                base[i, S.OP_ID["DIG"]] = True
    # the shed is refilled by the market while the unit walks: PICKUP is not masked by the current stock
    for s_i in S.SHED_IDX:
        for o, name in enumerate(S.OPS):
            if name.startswith("PICKUP_"):
                base[s_i, o] = True
    for u in range(nu):
        tmask[u] = base
        inv = carry[u]
        if any(int(v or 0) > 0 for v in inv.values()):
            for s_i in S.SHED_IDX:
                tmask[u, s_i, S.OP_ID["DROP"]] = True
                for p in S.PRODUCTS:
                    if int(inv.get(p, 0) or 0) > 0:
                        tmask[u, s_i, S.OP_ID[f"PLACE_{p}_ALL"]] = True
                if int(inv.get("WHEAT", 0) or 0) > 0:
                    tmask[u, s_i, S.OP_ID["PLACE_WHEAT_1"]] = True
        for a in S.ANIMALS:
            if int(inv.get(a, 0) or 0) > 0:
                want = "COOP" if a == "GOOSE" else "PASTURE"
                for i in range(S.NT):
                    if kind_of[i] == want and animal_of[i] is None:
                        tmask[u, i, S.OP_ID[f"PLACE_{a}"]] = True
    # market mask: every bin is allowed (the engine silently skips what cannot be executed)
    mm = []
    for kind, item, bins in S.MARKET_HEADS:  # noqa: B007  names kept as submitted
        m = np.ones(len(bins), bool)
        mm.append(m)
    mmask = np.concatenate(mm)
    return {
        "tiles": tiles,
        "units": units,
        "unit_pos": unit_pos,
        "unit_mask": unit_mask,
        "glob": glob,
        "tmask": tmask,
        "mmask": mmask,
    }


def _fg():
    import copy

    obs = {
        "player": 0,
        "day": 0,
        "hour": 0,
        "step": 0,
        "farms": [
            {
                "money": 3000,
                "tiles": [[None] * 10 for _ in range(10)],
                "farmer": [4, 4],
                "hands": [],
                "unlocked_quadrants": ["NW"],
                "hires_today": 0,
            }
        ]
        * 2,
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
        "market": {"prices": {}, "inventory": {}},
        "town": {"unlocked_shops": []},
    }
    return encode(copy.deepcopy(obs))["glob"].shape[0]


FG = _fg()
