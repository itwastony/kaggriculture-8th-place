#!/usr/bin/env python3
"""Recording of action streams for parity checks: games of live agents on the ORIGINAL kagsim
(kagsim.Game), optionally with "chaos" (actions mutated into garbage/illegal ones) or
with a random agent in one of the seats. For each game it saves the seed, the pinned
shops, both action streams (exactly the objects that went into kagsim.Game.step) and
the final banks.

    PYTHONHASHSEED=0 python engines/kagsim_cuda/record_streams.py      # -> engines/kagsim_cuda/data/streams.pkl.gz

Game modes (default counts --live 130 --chaos 54 --garbage 30 --random2 6, i.e. 220 games):
  live    - two live agents as is;
  chaos   - live agents, some unit commands/orders replaced with random ones (including
            garbage: unknown op/item, n out of range, extra hired workers, >10 orders,
            non-lists, a non-dict action as a whole); the agent sees the consequences;
  garbage - one seat is a random agent (legal and garbage actions), the other is live;
  random2 - both seats random.

Output (--out, default engines/kagsim_cuda/data/streams.pkl.gz): a gzip pickle, a list of game records sorted by job id
(the job fields + acts, rewards, agent_errors, steps, secs). The file is not shipped; it is the input of
engines/kagsim_cuda/parity.py, gpu_parity.py and diag_state.py (--streams, default: this path), and of
engines/kagsim_cuda/check/emu_dev_test.py, engines/kagrl_cuda/gpu_parity.py and the step_raw API check of
engines/kagrl_cuda/parity_cpu.py (fixed to this path). The live agents are loaded from their main.py: by default
(--pairs shipped) the bots shipped in this repository (caretaker, pass_agent; SHIPPED_PAIRS); --pairs competition gives
the job list of the competition-time streams (COMPETITION_PAIRS: our bots that are not released and public league bots).
The seeds, shops and chaos parameters of the jobs are the same for both pair sets. Missing agent files stop the run
before any game.
Agents may depend on the hash seed (e.g. by iterating over sets of strings), so set PYTHONHASHSEED in the environment
for reproducible games: the setdefault in main() does not change the hash seed of the running interpreter, nor of
fork-started workers. Requires kagsim and kaggle-environments.
"""

import argparse
import contextlib
import copy
import gzip
import io
import os
import pickle
import random
import sys
import time
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "harness"))

OPS = [
    "PASS",
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "PICKUP",
    "DROP",
    "PLACE",
    "PLANT",
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "DIG",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "FEED",
    "COLLECT_FERTILIZER",
    "CARE",
]
BAD_OPS = ["", "FLY", "pass", None, 5, "PLANTX", "INVALID"]
ITEMS = [
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
    "GOOSE",
    "COW",
    "SHEEP",
]
BAD_ITEMS = ["", "DOG", None, 3, "wheat"]
NS = [
    1,
    1,
    1,
    2,
    3,
    5,
    10,
    100,
    0,
    -1,
    -5,
    32767,
    32768,
    65536 + 7,
    -40000,
    2**31 + 5,
    2**40,
    2**70,
    1.5,
    "3",
    None,
    True,
]
MOPS = ["HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"]
BAD_MOPS = ["", "STEAL", None, 7]


def rand_unit(rng, junk=0.15):
    """Random unit command; with probability ~junk the op/item/n are invalid, sometimes the whole command is."""
    if rng.random() < junk * 0.3:
        return rng.choice([None, "NORTH", 5, {"op": 1}, ("NORTH",), [], ("PLANT", "WHEAT")])
    op = rng.choice(OPS) if rng.random() >= junk else rng.choice(BAD_OPS)
    k = rng.choice([1, 1, 2, 2, 3, 3, 4])
    v = [op]
    if k >= 2:
        v.append(rng.choice(ITEMS) if rng.random() >= junk else rng.choice(BAD_ITEMS))
    if k >= 3:
        v.append(rng.choice(NS) if rng.random() < 0.5 + junk else rng.choice([1, 2, 5, 20]))
    if k >= 4:
        v.append("extra")
    return v


def rand_order(rng, junk=0.15):
    """Random market order [op, item, n]; with probability ~junk invalid parts or a malformed order."""
    if rng.random() < junk * 0.3:
        return rng.choice([[], None, "SELL", ["SELL"], ["SELL", "WHEAT"], ("SELL", "WHEAT", 3), ["HIRE", "x", 5]])
    mop = rng.choice(MOPS) if rng.random() >= junk else rng.choice(BAD_MOPS)
    it = rng.choice(ITEMS) if rng.random() >= junk else rng.choice(BAD_ITEMS)
    # mostly small n (so that chaos does not spend the whole bank in one turn), sometimes garbage/huge
    n = rng.choice(NS + [1000, 50, 20, 7]) if rng.random() < 0.2 + junk else rng.choice([1, 1, 2, 3, 5])
    return [mop, it, n]


def random_agent_action(rng, junk):
    """Random agent: many hires/purchases/plantings (to reach rich
    states) + a share of garbage."""
    nh = rng.choice([0, 0, 1, 2, 4, 8, 12])
    units = [rand_unit(rng, junk) for _ in range(1 + nh)]
    market = []
    for _ in range(rng.choice([0, 1, 2, 3, 5, 10, 12])):
        r = rng.random()
        if r < 0.2:
            market.append(["HIRE"])
        elif r < 0.35:
            market.append(["BUY_SEED", rng.choice(ITEMS[:5]), rng.choice([1, 3, 10])])
        elif r < 0.45:
            market.append(["BUY_LAND"])
        else:
            market.append(rand_order(rng, junk))
    act = {"farmer": units[0], "hands": units[1:], "market": market}
    if rng.random() < junk * 0.05:
        return rng.choice([None, 5, "x", [], {}, {"farmer": "NORTH"}])
    return act


def mutate(act, rng, p):
    """Chaos on top of a live agent's action."""
    if rng.random() < p * 0.05:
        return rng.choice([None, 5, "x", [], {}, {"farmer": "NORTH"}, {"market": []}])
    a = act if isinstance(act, dict) else {}
    units = [a.get("farmer")] + list(a.get("hands") or [])
    for i in range(len(units)):
        if rng.random() < p:
            units[i] = rand_unit(rng)
    if rng.random() < p:
        units += [rand_unit(rng) for _ in range(rng.randint(1, 6))]  # extra/phantom hired workers
    if rng.random() < p * 0.5 and len(units) > 1:
        units = units[: rng.randint(1, len(units))]  # hired workers without commands
    market = list(a.get("market") or [])
    for i in range(len(market)):
        if rng.random() < p:
            market[i] = rand_order(rng)
    if rng.random() < p:
        for _ in range(rng.randint(1, 8)):
            market.insert(rng.randint(0, len(market)), rand_order(rng))
    if rng.random() < p * 0.1:
        market = [rand_order(rng) for _ in range(rng.randint(10, 20))]  # > max_orders and > 16 slots
    out = {"farmer": units[0], "hands": units[1:], "market": market}
    if rng.random() < p * 0.05:
        del out["farmer"]
    return out


def play(job):
    """One game on the original kagsim; returns a record for the parity check."""
    import kagsim
    from kagsim_play import _base_config, _load, fast_structify

    rng = random.Random(job["rng_seed"])
    seed, shops = job["seed"], job["shops"]
    game = kagsim.Game(seed, 720, shops) if shops is not None else kagsim.Game(seed, 720)
    cfg = dict(_base_config())
    cfg["seed"] = seed
    fns = [None, None]
    for p in (0, 1):
        if job["agents"][p] != "random":
            fns[p] = _load(os.path.join(ROOT, job["agents"][p]))
    acts_rec = [[], []]
    errors = [0, 0]
    t0 = time.time()
    while not game.done:
        acts = [None, None]
        for p in (0, 1):
            if job["agents"][p] == "random":
                acts[p] = random_agent_action(rng, job["junk"][p])
                continue
            obs = fast_structify(game.observe(p))
            conf = fast_structify(cfg)
            conf["__raw_path__"] = os.path.join(ROOT, job["agents"][p])
            fn = fns[p]
            args = [obs, conf]
            if hasattr(fn, "__code__"):
                args = args[: fn.__code__.co_argcount]
            try:
                with (
                    io.StringIO() as so,
                    io.StringIO() as se,
                    contextlib.redirect_stdout(so),
                    contextlib.redirect_stderr(se),
                ):
                    a = fn(*args)
            except Exception:
                errors[p] += 1
                a = {}
            a = copy.deepcopy(a if a is not None else {})
            if job["chaos"][p] > 0:
                a = mutate(a, rng, job["chaos"][p])
            acts[p] = a
        acts = [copy.deepcopy(acts[0]), copy.deepcopy(acts[1])]
        acts_rec[0].append(copy.deepcopy(acts[0]))
        acts_rec[1].append(copy.deepcopy(acts[1]))
        game.step(acts[0], acts[1])
    return {
        **job,
        "acts": acts_rec,
        "rewards": [game.reward(0), game.reward(1)],
        "agent_errors": errors,
        "steps": len(acts_rec[0]),
        "secs": time.time() - t0,
    }


# Live agent pairs (seat 0, seat 1); paths relative to the repository root.
# Default: the bots shipped in this repository. Chaos games take pair (3 * i) % len, so the length of a pair list must
# not be a multiple of 3 (otherwise chaos games would use only some of the pairs).
SHIPPED_PAIRS = [
    ("agents/caretaker/main.py", "agents/pass_agent/main.py"),
    ("agents/caretaker/main.py", "agents/caretaker/main.py"),
    ("agents/pass_agent/main.py", "agents/caretaker/main.py"),
    ("agents/caretaker/main.py", "agents/caretaker/main.py"),
]
# The pairs of the competition-time streams (--pairs competition): c3bc, v4_look12 and v19_wheat148 are our bots that
# are not released; the pub0923_* bots are public league bots (tools/fetch_league_bots.py), except pub0923_boatlee_v29
# and pub0923_lynn_v44, which are not listed in agents/sources.json.
COMPETITION_PAIRS = [
    ("agents/c3bc/main.py", "agents/v4_look12/main.py"),
    ("agents/v4_look12/main.py", "agents/c3bc/main.py"),
    ("agents/c3bc/main.py", "agents/pub0923_arsg_herdsafe3/main.py"),
    ("agents/pub0923_tetsutani_dp/main.py", "agents/c3bc/main.py"),
    ("agents/pub0923_ahmed_v54/main.py", "agents/v4_look12/main.py"),
    ("agents/caretaker/main.py", "agents/c3bc/main.py"),
    ("agents/pub0923_kaito_v58/main.py", "agents/pub0923_thomas_2945/main.py"),
    ("agents/pub0923_salemali_2900/main.py", "agents/pub0923_goodpjw_melon/main.py"),
    ("agents/pub0923_boatlee_v29/main.py", "agents/pub0923_lynn_v44/main.py"),
    ("agents/v19_wheat148/main.py", "agents/pub0923_statma_tdp/main.py"),
]
PAIR_SETS = {"shipped": SHIPPED_PAIRS, "competition": COMPETITION_PAIRS}


def make_jobs(n_live, n_chaos, n_garbage, n_random2, base_seed, pairs=SHIPPED_PAIRS):
    """Deterministic job list (live, chaos, garbage, random2 games, in this order) drawn from random.Random(base_seed).

    Every 4th game pins a random shop order (1..8 shop ids 0..7); each job has its own game seed and rng_seed. The draws
    do not depend on pairs (the live agent pairs), only the agent paths of the jobs do."""
    rng = random.Random(base_seed)
    jobs = []

    def shops_for(k):
        # every 4th game has pinned shops (random length 1..8, id 0..7)
        if k % 4 != 3:
            return None
        return [rng.randrange(8) for _ in range(rng.randint(1, 8))]

    k = 0
    for i in range(n_live):
        a, b = pairs[i % len(pairs)]
        jobs.append(
            dict(
                mode="live",
                agents=[a, b],
                chaos=[0, 0],
                junk=[0, 0],
                seed=rng.randrange(1, 2**31),
                shops=shops_for(k),
                rng_seed=rng.randrange(2**31),
            )
        )
        k += 1
    for i in range(n_chaos):
        a, b = pairs[(i * 3) % len(pairs)]
        p = [0.005, 0.02, 0.06][i % 3]
        ch = [[p, 0], [0, p], [p, p]][(i // 3) % 3]
        jobs.append(
            dict(
                mode="chaos",
                agents=[a, b],
                chaos=ch,
                junk=[0, 0],
                seed=rng.randrange(1, 2**31),
                shops=shops_for(k),
                rng_seed=rng.randrange(2**31),
            )
        )
        k += 1
    for i in range(n_garbage):
        live = pairs[i % len(pairs)][i % 2]
        junk = [0.0, 0.15, 0.5][i % 3]
        ag = ["random", live] if i % 2 == 0 else [live, "random"]
        jb = [junk, 0] if i % 2 == 0 else [0, junk]
        jobs.append(
            dict(
                mode="garbage",
                agents=ag,
                chaos=[0, 0],
                junk=jb,
                seed=rng.randrange(1, 2**31),
                shops=shops_for(k),
                rng_seed=rng.randrange(2**31),
            )
        )
        k += 1
    for _ in range(n_random2):
        jobs.append(
            dict(
                mode="random2",
                agents=["random", "random"],
                chaos=[0, 0],
                junk=[0.1, 0.4],
                seed=rng.randrange(1, 2**31),
                shops=shops_for(k),
                rng_seed=rng.randrange(2**31),
            )
        )
        k += 1
    for j, job in enumerate(jobs):
        job["id"] = j
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "streams.pkl.gz"))
    ap.add_argument("--live", type=int, default=130)
    ap.add_argument("--chaos", type=int, default=54)
    ap.add_argument("--garbage", type=int, default=30)
    ap.add_argument("--random2", type=int, default=6)
    ap.add_argument("--base-seed", type=int, default=20260925)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument(
        "--pairs",
        choices=sorted(PAIR_SETS),
        default="shipped",
        help="live agent pairs: shipped (caretaker, pass_agent; default) or competition (needs bots that are not in "
        "this repository)",
    )
    args = ap.parse_args()
    os.environ.setdefault("PYTHONHASHSEED", "0")
    jobs = make_jobs(args.live, args.chaos, args.garbage, args.random2, args.base_seed, PAIR_SETS[args.pairs])
    missing = sorted(
        {a for j in jobs for a in j["agents"] if a != "random" and not os.path.exists(os.path.join(ROOT, a))}
    )
    if missing:
        sys.exit(f"agents not found: {', '.join(missing)} (--pairs {args.pairs}; see agents/sources.json)")
    t0 = time.time()
    with Pool(args.procs) as pool:
        recs = []
        for r in pool.imap_unordered(play, jobs):
            recs.append(r)
            if len(recs) % 20 == 0:
                print(f"{len(recs)}/{len(jobs)} games, {time.time() - t0:.0f} s", flush=True)
    recs.sort(key=lambda r: r["id"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with gzip.open(args.out, "wb") as f:
        pickle.dump(recs, f, protocol=pickle.HIGHEST_PROTOCOL)
    errs = sum(1 for r in recs if sum(r["agent_errors"]) > 0)
    print(f"recorded {len(recs)} games in {time.time() - t0:.0f} s -> {args.out}; games with agent exceptions: {errs}")
    for r in recs:
        if sum(r["agent_errors"]):
            print("  agent exceptions:", r["id"], r["mode"], r["agents"], r["agent_errors"])


if __name__ == "__main__":
    main()
