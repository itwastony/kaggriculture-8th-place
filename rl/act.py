"""Action executor and conversion of engine actions into (target, operation) labels + market bins.

execute(obs, unit_choices, market_choices) -> engine action dict.
unit_labels(traj) / market_labels(obs, action) -> per step: (target, operation) for each unit + market bin indices
  (hindsight: movement target = the tile where the unit performs its next non-move operation on the same day).
"""

# This file is copied verbatim into the submitted agent (rl/export_agent.py), and entry_points/verify_submission.py
# compares its code with the archives sent to Kaggle: change only comments and formatting.
from . import spec as S

MOVES = ("NORTH", "SOUTH", "EAST", "WEST")
_BIN_OPS_W = {k: S.OP_ID[f"PICKUP_WHEAT_{k}"] for k in S.PICKUP_WHEAT_N}
_BIN_OPS_F = {k: S.OP_ID[f"PICKUP_FERTILIZER_{k}"] for k in S.PICKUP_FERT_N}


def _floor_bin(bins, q):
    """Index of the largest positive bin <= q (q>0 -> at least the first positive one)."""
    best = 0
    for i, b in enumerate(bins):
        if b > 0 and b <= q:
            best = i
    if q > 0 and best == 0:
        best = next(i for i, b in enumerate(bins) if b > 0)
    return best


def _nearest_key(keys, q):
    return min(keys, key=lambda k: (abs(k - q), k))


def op_to_engine(op_id):
    o = S.OPS[op_id]
    if o == "STAY":
        return ["PASS"]
    if o.startswith("PLANT_"):
        return ["PLANT", o[6:]]
    if o.startswith("PICKUP_WHEAT_"):
        return ["PICKUP", "WHEAT", int(o.rsplit("_", 1)[1])]
    if o.startswith("PICKUP_FERTILIZER_"):
        return ["PICKUP", "FERTILIZER", int(o.rsplit("_", 1)[1])]
    if o.startswith("PICKUP_"):
        item, k = o[7:].rsplit("_", 1)
        return ["PICKUP", item, int(k)]
    if o.endswith("_ALL"):
        return ["PLACE", o[6:-4], 1000]
    if o == "PLACE_WHEAT_1":
        return ["PLACE", "WHEAT", 1]
    if o.startswith("PLACE_"):
        return ["PLACE", o[6:], 1]
    return [o]


def engine_to_op(u):
    """Non-move engine operation -> operation id (None for movement; PASS -> STAY)."""
    if not u:
        return S.OP_ID["STAY"]
    k = u[0]
    if k in MOVES:
        return None
    if k == "PASS":
        return S.OP_ID["STAY"]
    if k == "PLANT":
        return S.OP_ID.get(f"PLANT_{u[1]}", S.OP_ID["STAY"]) if len(u) > 1 else S.OP_ID["STAY"]
    if k == "PICKUP":
        item = u[1] if len(u) > 1 else None
        n = int(u[2]) if len(u) > 2 else 1
        if item == "WHEAT":
            return _BIN_OPS_W[_nearest_key(S.PICKUP_WHEAT_N, n)]
        if item == "FERTILIZER":
            return _BIN_OPS_F[_nearest_key(S.PICKUP_FERT_N, n)]
        if item in S.ANIMALS:
            return S.OP_ID[f"PICKUP_{item}_{_nearest_key(S.PICKUP_ANIMAL_N, n)}"]
        return S.OP_ID["STAY"]
    if k == "PLACE":
        item = u[1] if len(u) > 1 else None
        if item in S.ANIMALS:
            return S.OP_ID[f"PLACE_{item}"]
        n = int(u[2]) if len(u) > 2 else 1
        if item == "WHEAT" and n == 1:
            return S.OP_ID["PLACE_WHEAT_1"]
        if item in S.PRODUCTS:
            return S.OP_ID[f"PLACE_{item}_ALL"]
        return S.OP_ID["DROP"]
    if k in S.OP_ID:
        return S.OP_ID[k]
    return S.OP_ID["STAY"]


def step_toward(pos, target):
    x, y = pos
    tx, ty = S.tile_xy(target)
    if tx > x:
        return "EAST"
    if tx < x:
        return "WEST"
    if ty > y:
        return "SOUTH"
    if ty < y:
        return "NORTH"
    return None


def unit_positions(obs):
    f = obs["farms"][obs["player"]]
    return [tuple(f["farmer"])] + [tuple(h) for h in f["hands"]]


def unit_action(pos, target, op_id):
    mv = step_toward(pos, target)
    if mv is not None:
        return [mv]
    return op_to_engine(op_id)


SHED_XY = frozenset(S.SHED_TILES)


def projected_shed(obs, unit_acts):
    """Shed after this turn's unit actions (units act before the market): DROP/PLACE into the shed, PICKUP."""
    pv = obs["private"]
    shed = {k: int(v) for k, v in pv["shed"].items()}
    invs = pv["inventories"]
    pos = unit_positions(obs)
    cap = 100
    for i, u in enumerate(unit_acts):
        if not u or i >= len(pos) or pos[i] not in SHED_XY:
            continue
        inv = invs[i] if i < len(invs) else {}
        k = u[0]
        if k == "DROP":
            for item, n in inv.items():
                room = max(0, cap - sum(shed.values()))
                shed[item] = shed.get(item, 0) + min(int(n), room)
        elif k == "PLACE" and len(u) > 1 and u[1] not in S.ANIMALS:
            n = min(int(u[2]) if len(u) > 2 else 1, int(inv.get(u[1], 0)))
            room = max(0, cap - sum(shed.values()))
            shed[u[1]] = shed.get(u[1], 0) + min(n, room)
        elif k == "PICKUP" and len(u) > 1:
            n = min(int(u[2]) if len(u) > 2 else 1, shed.get(u[1], 0))
            shed[u[1]] = shed.get(u[1], 0) - n
    return shed


def _bought(tmpl, buys, shed):
    """Product purchases that will be in the shed before the sells (if P precedes S in the template), respecting capacity 100."""  # noqa: E501
    if tmpl.index("P") > tmpl.index("S"):
        return {}
    room = max(0, 100 - sum(int(v) for v in shed.values()))
    out = {}
    for item, q in buys:
        n = min(int(q), room)
        out[item] = out.get(item, 0) + n
        room -= n
    return out


def market_orders(obs, choice, shed=None):
    """choice: bin indices per S.MARKET_HEADS -> orders (at most 10) in the group-template order. shed = projected shed."""  # noqa: E501
    shed = obs["private"]["shed"] if shed is None else shed
    groups = {g: [] for g in "SPDAHL"}
    tmpl = S.ORDER_TEMPLATES[0]
    sells, first, sfirst = [], None, None
    for (kind, item, bins), bi in zip(S.MARKET_HEADS, choice):
        q = bins[bi]
        if kind == "ORDER":
            tmpl = S.ORDER_TEMPLATES[q]
            continue
        if kind == "SELL_FIRST":
            first = S.PRODUCTS[q - 1] if q > 0 else None
            continue
        if kind == "SEED_FIRST":
            sfirst = S.CROPS[q - 1] if q > 0 else None
            continue
        if q == 0:
            continue
        if kind == "SELL":
            sells.append((item, q))
        elif kind == "HIRE":
            groups["H"] += [["HIRE"]] * q
        elif kind == "BUY_LAND":
            groups["L"].append(["BUY_LAND"])
        else:
            groups[S.GROUP_OF[kind]].append([kind, item, q])
    bought = _bought(tmpl, [(o[1], o[2]) for o in groups["P"]], shed)
    for item, q in sells:
        n = int(shed.get(item, 0)) + bought.get(item, 0) if q == -1 else q
        if n > 0:
            groups["S"].append(["SELL", item, n])
    if first is not None:
        groups["S"].sort(key=lambda o: o[1] != first)
    if sfirst is not None:
        groups["D"].sort(key=lambda o: o[1] != sfirst)
    orders = []
    for g in tmpl:
        orders += groups[g]
    return orders[: S.MAX_ORDERS]


def execute(obs, unit_choices, market_choice):
    """unit_choices: list of (target, operation) per unit in the order [farmer, hired workers...]."""
    pos = unit_positions(obs)
    acts = []
    for i, p in enumerate(pos):
        if i < len(unit_choices) and i < S.MAX_UNITS:
            t, o = unit_choices[i]
            acts.append(unit_action(p, t, o))
        else:
            acts.append(["PASS"])
    shed = projected_shed(obs, acts)
    return {"farmer": acts[0], "hands": acts[1:], "market": market_orders(obs, market_choice, shed)}


def _template_of(firsts):
    for i, t in enumerate(S.ORDER_TEMPLATES):
        idx = [t.index(g) for g in firsts]
        if idx == sorted(idx):
            return i
    return 0


def market_labels(obs, action):
    """Engine orders -> bin indices per S.MARKET_HEADS (orders of the same kind are summed; group order -> template)."""
    tot, hires, land, firsts, buys, sold, seeds = {}, 0, 0, [], [], [], []
    for m in [m for m in (action.get("market") or []) if m][: S.MAX_ORDERS]:
        k = m[0]
        g = S.GROUP_OF.get(k)
        if g is None:
            continue
        if g not in firsts:
            firsts.append(g)
        if k == "HIRE":
            hires += 1
        elif k == "BUY_LAND":
            land = 1
        else:
            q = int(m[2]) if len(m) > 2 else 1
            tot[(k, m[1])] = tot.get((k, m[1]), 0) + q
            if k == "BUY_PRODUCT":
                buys.append((m[1], q))
            if k == "SELL" and m[1] not in sold:
                sold.append(m[1])
            if k == "BUY_SEED" and m[1] not in seeds:
                seeds.append(m[1])
    ti = _template_of(firsts)
    ua = [action.get("farmer") or ["PASS"]] + list(action.get("hands") or [])
    shed = projected_shed(obs, ua)
    bought = _bought(S.ORDER_TEMPLATES[ti], buys, shed)
    out = []
    for kind, item, bins in S.MARKET_HEADS:
        if kind == "ORDER":
            out.append(ti)
        elif kind == "SELL_FIRST":
            out.append(S.PRODUCTS.index(sold[0]) + 1 if len(sold) > 1 and sold[0] in S.PRODUCTS else 0)
        elif kind == "SEED_FIRST":
            out.append(S.CROPS.index(seeds[0]) + 1 if len(seeds) > 1 and seeds[0] in S.CROPS else 0)
        elif kind == "HIRE":
            out.append(min(hires, len(bins) - 1))
        elif kind == "BUY_LAND":
            out.append(land)
        else:
            q = tot.get((kind, item), 0)
            if kind == "SELL":
                have = int(shed.get(item, 0)) + bought.get(item, 0)
                if q <= 0:
                    out.append(0)
                elif have > 0 and q >= have:
                    out.append(len(bins) - 1)  # "all"
                else:
                    out.append(_floor_bin(bins[:-1], q))
            else:
                out.append(_floor_bin(bins, q) if q > 0 else 0)
    return out


def unit_labels(traj):
    """Hindsight unit labels: for each step t, a list of (target, operation) per unit of observation t."""
    T = len(traj)
    labels = [None] * T
    day_of = [traj[t][0]["day"] for t in range(T)]
    for t in range(T):
        obs, act = traj[t]
        pos = unit_positions(obs)
        ua = [act.get("farmer") or ["PASS"]] + list(act.get("hands") or [])
        lab = []
        for i, p in enumerate(pos):
            u = ua[i] if i < len(ua) else ["PASS"]
            op = engine_to_op(u)
            if op is not None:
                lab.append((S.tile_id(*p), op))
                continue
            # movement: target = end of the shortest-path segment from p (the turning point) or the tile of this
            # unit's next non-move operation on the same day, if it comes before the turning point
            tgt, top = None, S.OP_ID["STAY"]
            last = p
            for t2 in range(t + 1, T):
                if day_of[t2] != day_of[t]:
                    break
                o2, a2 = traj[t2]
                pos2 = unit_positions(o2)
                if i >= len(pos2):
                    break
                q = pos2[i]
                if abs(q[0] - p[0]) + abs(q[1] - p[1]) != t2 - t:  # the path stopped being shortest
                    tgt = S.tile_id(*last)
                    break
                last = q
                ua2 = [a2.get("farmer") or ["PASS"]] + list(a2.get("hands") or [])
                u2 = ua2[i] if i < len(ua2) else ["PASS"]
                op2 = engine_to_op(u2)
                if op2 is not None:
                    tgt, top = S.tile_id(*q), op2
                    break
            if tgt is None:
                tgt = S.tile_id(*last)
            # our step (x first) would diverge from the teacher's move: target = the tile the teacher stepped to
            if step_toward(p, tgt) != u[0]:
                dx, dy = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}[u[0]]
                nx, ny = p[0] + dx, p[1] + dy
                tgt, top = (
                    (S.tile_id(nx, ny), S.OP_ID["STAY"])
                    if 0 <= nx < S.N and 0 <= ny < S.N
                    else (S.tile_id(*p), S.OP_ID["STAY"])
                )
            lab.append((tgt, top))
        labels[t] = lab
    return labels
