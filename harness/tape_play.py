#!/usr/bin/env python3
"""Games against tapes (recorded actions from live episodes) on kagsim with a pinned shop schedule.

World = a recorded episode: seed, shop schedule (as in the record), action tapes of both seats.
World file (replays/worlds/<pool>/ep<ID>.json.gz, built by harness/tape_worlds.py):
    {"ep", "seed", "shops", "names", "rewards", "actions": [tape of seat 0, tape of seat 1], ...}
Tape of seat s: actions[s][t] = action at step t (= steps[t+1][s].action of the record), t = 0..718.

play_vs_tape(world, tape_seat, agent_path): the tape plays in its recorded seat, the live agent in the other one.
Shops are pinned (kagsim.Game(seed, 720, shops)): the schedule is not reshuffled because our farm
differs from the recorded opponent's. Weeds are not pinned: if the tape is in seat 1, its weeds depend on the number
of empty tiles on our farm (farms roll the dice in order 0 -> 1), so the tape may diverge slightly from the record.

match=True: the live agent in seat s is compared with the recorded actions of seat s; the game stops at the first
mismatch (this checks that the agent's code = the code that played the record). Returns first_mismatch
(None = all steps matched).

    python harness/tape_play.py WORLD.json.gz                       # tape vs tape, printed next to the recorded rewards
    python harness/tape_play.py WORLD.json.gz AGENT.py [TAPE_SEAT]  # + AGENT against the tape of TAPE_SEAT (default 1)

Requires kagsim (scripts/build_engines.sh); playing a live agent also needs kaggle-environments.
"""

import contextlib
import gzip
import io
import json

from kagsim_play import _base_config, _load, fast_structify

PASS = {"farmer": ["PASS"], "hands": [], "market": []}
_WORLDS = {}  # path -> world (per-process cache: callers usually play several games on one world in a row)


def load_world(path):
    """Read a world file (.json.gz), cached per process (the cache is cleared when it exceeds 64 worlds)."""
    if path not in _WORLDS:
        if len(_WORLDS) > 64:
            _WORLDS.clear()
        with gzip.open(path, "rt") as f:
            _WORLDS[path] = json.load(f)
    return _WORLDS[path]


def canon(a):
    """Canonical JSON of an action, used to compare a live agent's action with the recorded one."""
    return json.dumps(a, sort_keys=True, separators=(",", ":"), default=list)


def replay_tapes(seed, tapes, shops=None, steps=720):
    """Tape against tape; missing or None steps are played as PASS.

    Returns (rewards, shop opening order at the end of the game). Without shops the engine draws the order itself."""
    import kagsim

    g = kagsim.Game(seed, steps, shops) if shops is not None else kagsim.Game(seed, steps)
    t = 0
    while not g.done:
        a = [tp[t] if t < len(tp) and tp[t] is not None else PASS for tp in tapes]
        g.step(a[0], a[1])
        t += 1
    town = g.observe(0).get("town") or {}
    return [g.reward(0), g.reward(1)], list(town.get("unlocked_shops") or [])


TEL_KEYS = (
    "sell_revenue",
    "total_spend",
    "sold_units",
    "hire_paid",
    "animals_escaped",
    "plants_dry",
    "shed_discarded_units",
    "sell_dead_units",
    "silent_loss_coins",
    "dead_actions",
)


def _farm_snap(f):
    """Farm summary from the visible state: money, plantings per crop, animals per species, weeds, land, units."""
    plants, animals, weeds = {}, {}, 0
    for row in f["tiles"]:
        for c in row:
            if isinstance(c, dict):
                if c["kind"] == "PLANT":
                    plants[c["crop"]] = plants.get(c["crop"], 0) + 1
                elif c["kind"] == "WEED":
                    weeds += 1
                elif c.get("animal"):
                    animals[c["animal"]] = animals.get(c["animal"], 0) + 1
    return {
        "money": f["money"],
        "plants": plants,
        "animals": animals,
        "weeds": weeds,
        "quads": len(f["unlocked_quadrants"]),
        "hands": len(f["hands"]),
    }


def play_vs_tape(world, tape_seat, agent_path, pin=True, match=False, steps=720, trace=False):
    """agent_path == "@tape": seat me plays its own recorded actions (the record's reference under the same conditions).
    trace=True: the output additionally gets daily summaries of both farms (end of each day, step % 24 == 23), the final
    kagsim telemetry of both sides and revenue/sold units per product: the telemetry increment of a turn is split
    among the products of that turn's SELL orders (first 10 orders) in proportion to quantity x price before the turn;
    rev_exact is the share of revenue from turns with a single product sold (there the split is exact)."""
    import kagsim

    me = 1 - tape_seat
    tape = world["actions"][tape_seat]
    ref = world["actions"][me] if match else None
    own = world["actions"][me] if agent_path == "@tape" else None
    cfg = dict(_base_config())
    cfg["seed"] = world["seed"]
    g = kagsim.Game(world["seed"], steps, world["shops"]) if pin else kagsim.Game(world["seed"], steps)
    fn, status, t, first_mismatch = None, "ACTIVE", 0, None
    if trace:
        days = []
        rev = [{}, {}]
        units = [{}, {}]
        exact = [0.0, 0.0]
        prev = [g.telemetry(s) for s in (0, 1)]
    while not g.done:
        raw = g.observe(me)
        if own is not None:
            act = own[t] if t < len(own) and own[t] is not None else PASS
        else:
            obs = fast_structify(raw)
            conf = fast_structify(cfg)
            conf["__raw_path__"] = agent_path
            with io.StringIO() as o, io.StringIO() as e, contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
                try:
                    if fn is None:
                        fn = _load(agent_path)
                    args = [obs, conf]
                    if hasattr(fn, "__code__"):
                        args = args[: fn.__code__.co_argcount]
                    act = fn(*args)
                except Exception:
                    status = "ERROR"
        if status == "ERROR":
            break
        if match and (t >= len(ref) or canon(act or {}) != canon(ref[t])):
            first_mismatch = t
            break
        ta = tape[t] if t < len(tape) and tape[t] is not None else PASS
        pair = [None, None]
        pair[me], pair[tape_seat] = act or {}, ta
        g.step(pair[0], pair[1])
        if trace:
            price = raw["market"]["prices"]
            for s in (0, 1):
                cur = g.telemetry(s)
                d_rev, d_u = cur["sell_revenue"] - prev[s]["sell_revenue"], cur["sold_units"] - prev[s]["sold_units"]
                prev[s] = cur
                if d_rev <= 0 and d_u <= 0:
                    continue
                sells = {}
                for m in list((pair[s] or {}).get("market") or [])[:10]:
                    if isinstance(m, (list, tuple)) and len(m) >= 3 and m[0] == "SELL":
                        sells[m[1]] = sells.get(m[1], 0) + max(0, int(m[2]))
                wts = {k: q * max(1.0, float(price.get(k, 1))) for k, q in sells.items() if q > 0}
                tot = sum(wts.values())
                if not tot:
                    wts, tot = {"?": 1.0}, 1.0
                if len(wts) == 1:
                    exact[s] += d_rev
                for k, wv in wts.items():
                    rev[s][k] = rev[s].get(k, 0.0) + d_rev * wv / tot
                    units[s][k] = units[s].get(k, 0.0) + d_u * wv / tot
            if t % 24 == 23:
                fs = g.observe(0)["farms"]
                days.append([_farm_snap(fs[0]), _farm_snap(fs[1])])
        t += 1
    out = {"steps_played": t}
    if match:
        out["first_mismatch"] = first_mismatch
    if status == "ERROR":
        out.update(rewards=None, status=["ERROR" if s == me else "DONE" for s in (0, 1)])
    elif not g.done:
        out.update(rewards=None, status=["ABORTED", "ABORTED"])
    else:
        out.update(rewards=[g.reward(0), g.reward(1)], status=["DONE", "DONE"])
    if trace:
        tel = [g.telemetry(s) for s in (0, 1)]
        out["trace"] = {
            "days": days,
            "tel": [{k: tel[s].get(k) for k in TEL_KEYS} for s in (0, 1)],
            "rev": [{k: round(v, 1) for k, v in rev[s].items()} for s in (0, 1)],
            "units": [{k: round(v, 1) for k, v in units[s].items()} for s in (0, 1)],
            "rev_exact": [round(exact[s] / max(1e-9, tel[s]["sell_revenue"]), 3) for s in (0, 1)],
        }
    return out


if __name__ == "__main__":
    import sys
    import time

    w = load_world(sys.argv[1])
    t0 = time.time()
    r = replay_tapes(w["seed"], w["actions"], w["shops"])
    print("tape vs tape:", r, "record:", w["rewards"], f"{time.time() - t0:.2f}s")
    if len(sys.argv) > 2:
        t0 = time.time()
        print(play_vs_tape(w, int(sys.argv[3]) if len(sys.argv) > 3 else 1, sys.argv[2]), f"{time.time() - t0:.2f}s")
