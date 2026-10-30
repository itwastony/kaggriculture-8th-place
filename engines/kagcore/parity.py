#!/usr/bin/env python3
"""Cross-check of C++ kagcore (engines/kagcore) against Python rl/encode.py and rl/act.py on kagsim games.

Mode A (agents): a game of two live agents on the canonical kagsim.Game; in parallel, the same game in
  kagcore.GameBatch (same dict actions). At every step, for both seats:
    - obs: kagsim.Game.observe(p) == GameBatch.observe(0, p) (the states match);
    - encode: C++ encode_into(state) == rl.encode.encode(kagsim obs) bit-exact across the 7 arrays
      (tmask is compared with np.packbits(tmask.reshape(16,-1), axis=-1));
    - execute: K random choices (from tmask/mmask + a share of arbitrary ones) ->
      conv_action_raw(rl.act.execute(obs, ...)) == GameBatch.execute_raw(...) (the whole engine action).
  At the end, the rewards of both seats match to the dollar.
Mode B (trajectory via execute): "rand" / "randH" seats play random choices (in the first two hours of each day the
  market of "randH" only hires 12 workers, to push the number of units past 16); the canonical kagsim.Game receives the
  dict from Python rl.act.execute, while GameBatch gets the same choices executed by C++ execute (step / step_vs).
  The same obs/encode/execute checks at every step + rewards at the end.

    python engines/kagcore/parity.py --procs 8            # 20 games A + 16 games B
    python engines/kagcore/parity.py --quick              # quick run (2 + 2 games, 100 steps)
    python engines/kagcore/parity.py --agents caretaker pub0923_ahmed_v54   # other agents from agents/<name>/

Agents are loaded from agents/<name>/main.py. By default (--agents caretaker pass_agent) they are the two bots shipped
in this repository: mode A plays every ordered pair of --agents in turn (with the defaults: caretaker vs caretaker,
caretaker vs pass_agent, pass_agent vs caretaker, pass_agent vs pass_agent; --quick plays the first two), mode B uses
them as the opponents of the random seats (each agent on both seats). The seeds, the random roles and all comparisons
do not depend on the agents. --competition-agents reproduces the competition-time job list, whose agents include our
bots that are not released (c3bc, v4_look12, bcr_tdp_v4) and public league bots (tools/fetch_league_bots.py). Missing
agent folders stop the run before any game. Requires the kagcore module (engines/kagcore/build.sh), kagsim and
kaggle-environments. Writes the full summary to --out (default engines/kagcore/parity_result.json); exit code 1 on any
mismatch.
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "build"))
sys.path.insert(0, ROOT)

ARRS = ("tiles", "units", "unit_pos", "unit_mask", "glob", "tmask", "mmask")
RANDOM_ROLES = ("rand", "randH")
# Default agents: the bots shipped in this repository.
SHIPPED_AGENTS = ("caretaker", "pass_agent")
# The agents of the competition-time runs (--competition-agents): c3bc, v4_look12 and bcr_tdp_v4 are our bots that are
# not released; soil_v7 and pub0923_* are public league bots (tools/fetch_league_bots.py).
COMPETITION_PAIRS = [
    ("c3bc", "v4_look12"),
    ("v4_look12", "c3bc"),
    ("c3bc", "bcr_tdp_v4"),
    ("pub0923_arsg_herdsafe3", "c3bc"),
    ("v4_look12", "pub0923_tetsutani_dp"),
    ("soil_v7", "c3bc"),
    ("c3bc", "pub0923_tetsutani_dp"),
    ("bcr_tdp_v4", "v4_look12"),
    ("pub0923_tetsutani_dp", "pub0923_arsg_herdsafe3"),
    ("v4_look12", "soil_v7"),
]
COMPETITION_OPPS = ["c3bc", "v4_look12", "bcr_tdp_v4", "pub0923_arsg_herdsafe3"]


def agent_path(name):
    """agents/<name>/main.py under the repository root."""
    return os.path.join(ROOT, "agents", name, "main.py")


class CEnc:
    """C++ encode buffers for n rows."""

    def __init__(self, n):
        import kagcore as K

        self.n = n
        self.buf = {
            "tiles": np.empty((n, K.NT, K.FT), np.float32),
            "units": np.empty((n, K.MAXU, K.FU), np.float32),
            "unit_pos": np.empty((n, K.MAXU), np.int64),
            "unit_mask": np.empty((n, K.MAXU), bool),
            "glob": np.empty((n, K.FG), np.float32),
            "tmask": np.empty((n, K.MAXU, K.TMASK_BYTES), np.uint8),
            "mmask": np.empty((n, K.MB), bool),
        }

    def run(self, gb, seats, games):
        """encode_into for row i = (games[i], seats[i]) of GameBatch gb; returns the filled buffers."""
        b = self.buf
        gb.encode_into(
            np.asarray(seats, np.int32),
            b["tiles"],
            b["units"],
            b["unit_pos"],
            b["unit_mask"],
            b["glob"],
            b["tmask"],
            b["mmask"],
            games=np.asarray(games, np.int32),
        )
        return b


class Stats:
    """Comparison counters, coverage of the visited states/actions and the first 20 errors of one worker."""

    def __init__(self):
        self.c = {
            k: 0
            for k in (
                "games",
                "steps",
                "enc_calls",
                "enc_arrays",
                "enc_bytes",
                "enc_mismatch",
                "obs_compared",
                "obs_mismatch",
                "exec_compared",
                "exec_mismatch",
                "reward_compared",
                "reward_mismatch",
            )
        }
        self.cov = {
            "max_units": 0,
            "states_units_gt16": 0,
            "states_with_animals": 0,
            "states_inv_nonempty": 0,
            "exec_10_orders": 0,
            "unit_ops": set(),
            "order_ops": set(),
            "sell_all_nonzero": 0,
            "execute_moves": 0,
            "execute_ops": 0,
        }
        self.errors = []

    def err(self, **kw):
        if len(self.errors) < 20:
            self.errors.append(kw)

    def as_dict(self):
        cov = dict(self.cov)
        cov["unit_ops"] = sorted(cov["unit_ops"])
        cov["order_ops"] = sorted(cov["order_ops"])
        return {"counts": self.c, "coverage": cov, "errors": self.errors}


def compare_encode(py, cb, i, st, ctx, n_units_total):
    """Compare the Python encoding py with row i of the C++ buffers cb (bytes, dtype, shape); count coverage."""
    st.c["enc_calls"] += 1
    for k in ARRS:
        a = np.packbits(py["tmask"].reshape(16, -1), axis=-1) if k == "tmask" else py[k]
        c = cb[k][i]
        st.c["enc_arrays"] += 1
        st.c["enc_bytes"] += a.nbytes
        if a.dtype != c.dtype or a.shape != c.shape or a.tobytes() != c.tobytes():
            st.c["enc_mismatch"] += 1
            d = {"field": k, "py_dtype": str(a.dtype), "c_dtype": str(c.dtype), "shape": [list(a.shape), list(c.shape)]}
            if a.shape == c.shape and a.dtype == c.dtype:
                ab = np.frombuffer(a.tobytes(), np.uint8)
                cb_ = np.frombuffer(c.tobytes(), np.uint8)
                bad = np.flatnonzero(ab != cb_) // a.itemsize
                j = int(bad[0])
                d.update(
                    n_bad_elems=int(len(np.unique(bad))),
                    first=j,
                    py=repr(a.reshape(-1)[j].item()),
                    c=repr(c.reshape(-1)[j].item()),
                )
            st.err(kind="encode", **ctx, **d)
    # coverage
    nu = int(py["unit_mask"].sum())
    st.cov["max_units"] = max(st.cov["max_units"], nu)
    if py["tiles"][:, 11:14].any():
        st.cov["states_with_animals"] += 1
    if py["units"][:, 3:15].any():
        st.cov["states_inv_nonempty"] += 1
    if n_units_total > 16:
        st.cov["states_units_gt16"] += 1


def rand_choices(rng, py, hire_rush=False):
    """Random choices: units - from tmask (in place / towards the shed / any legal one) or arbitrary; market -
    biased towards zeros, "all" for SELL. hire_rush: only HIRE=12 (+ a random template) - inflates the number of units
    beyond 16 (checks the MAX_UNITS truncation and PASS for the extra hired workers)."""
    from rl import spec as S

    tm = py["tmask"]
    pos = py["unit_pos"]
    pairs, uc = [], np.zeros(16, np.int32)
    for u in range(16):
        r = rng.random()
        if not py["unit_mask"][u] or r >= 0.9:
            t, o = int(rng.integers(S.NT)), int(rng.integers(S.NOPS))
        elif r < 0.45:
            t = int(pos[u])
            o = int(rng.choice(np.flatnonzero(tm[u, t])))
        elif r < 0.6:
            t = int(rng.choice(S.SHED_IDX))
            o = int(rng.choice(np.flatnonzero(tm[u, t])))
        else:
            flat = np.flatnonzero(tm[u])
            k = int(rng.choice(flat))
            t, o = divmod(k, S.NOPS)
        pairs.append((t, o))
        uc[u] = t * S.NOPS + o
    mc = np.zeros(24, np.int32)
    for h, (kind, _item, bins) in enumerate(S.MARKET_HEADS):
        nb = len(bins)
        if kind in ("ORDER", "SELL_FIRST", "SEED_FIRST"):
            mc[h] = rng.integers(nb)
        elif kind == "SELL" and rng.random() < 0.12:
            mc[h] = nb - 1
        elif rng.random() < (0.8 if kind in ("HIRE", "BUY_LAND") else 0.7):
            mc[h] = 0
        else:
            mc[h] = rng.integers(1, nb)
    if hire_rush:
        keep = mc[21]
        mc[:] = 0
        mc[19] = 12
        mc[21] = keep
    return uc, pairs, mc


def check_execute(gb, b, p, obs, py, rng, st, ctx, kagcore, act, hire_rush=False):
    """One random choice set for game b, seat p: Python execute (via conv_action_raw) vs C++ execute_raw.

    Returns (unit choices, market choices, the Python action dict)."""
    uc, pairs, mc = rand_choices(rng, py, hire_rush)
    d = act.execute(obs, pairs, [int(x) for x in mc])
    tp = kagcore.conv_action_raw(d)
    tc = gb.execute_raw(b, p, uc, mc)
    st.c["exec_compared"] += 1
    if tp != tc:
        st.c["exec_mismatch"] += 1
        st.err(kind="execute", **ctx, py=repr(tp), c=repr(tc), uc=uc.tolist(), mc=mc.tolist())
    n_units, units, n_orders, orders = tp
    if n_orders == 10:
        st.cov["exec_10_orders"] += 1
    for op, _arg, _n in units:
        st.cov["unit_ops"].add(op)
        if 1 <= op <= 4:
            st.cov["execute_moves"] += 1
        elif op != 0:
            st.cov["execute_ops"] += 1
    for op, _item, _n in orders:
        st.cov["order_ops"].add(op)
    for h in range(9):
        if mc[h] == 31 and any(o[0] == 6 and o[1] == h for o in orders):
            st.cov["sell_all_nonzero"] += 1
    return uc, mc, d


def game_agents(spec):
    """Mode A: seed, agents a/b (names), K random executes per step and seat, steps."""
    import kagcore
    import kagsim

    from rl import act
    from rl import encode as E
    from rl.record import _base_config, _load, call_agent

    seed, na, nb, K, steps = spec["seed"], spec["a"], spec["b"], spec["K"], spec["steps"]
    st = Stats()
    rng = np.random.default_rng(seed * 7919 + 17)
    paths = [agent_path(na), agent_path(nb)]
    fns = [_load(paths[0]), _load(paths[1])]
    cfg = dict(_base_config())
    cfg["seed"] = seed
    canon = kagsim.Game(seed, 720)
    gb = kagcore.GameBatch([seed], 720)
    ce = CEnc(2)
    zu, zm = np.zeros((1, 2, 16), np.int32), np.zeros((1, 2, 24), np.int32)
    t0 = time.time()
    while not canon.done and canon.step_count < steps:
        cb = ce.run(gb, [0, 1], [0, 0])
        acts = [None, None]
        for p in (0, 1):
            ctx = {"mode": "A", "seed": seed, "a": na, "b": nb, "step": canon.step_count, "seat": p}
            obs = canon.observe(p)
            st.c["obs_compared"] += 1
            if obs != gb.observe(0, p):
                st.c["obs_mismatch"] += 1
                st.err(kind="obs", **ctx)
            py = E.encode(obs)
            compare_encode(py, cb, p, st, ctx, len(obs["farms"][p]["hands"]) + 1)
            for _ in range(K):
                check_execute(gb, 0, p, obs, py, rng, st, ctx, kagcore, act)
            acts[p] = call_agent(fns[p], obs, cfg, paths[p]) or {}
        canon.step(acts[0], acts[1])
        gb.step(zu, zm, [(acts[0], acts[1])])
        st.c["steps"] += 1
    for p in (0, 1):
        st.c["reward_compared"] += 1
        if canon.reward(p) != gb.reward(0, p):
            st.c["reward_mismatch"] += 1
            st.err(kind="reward", mode="A", seed=seed, seat=p, py=canon.reward(p), c=gb.reward(0, p))
    st.c["games"] += 1
    out = st.as_dict()
    out["spec"] = spec
    out["rewards"] = [canon.reward(0), canon.reward(1)]
    out["sec"] = round(time.time() - t0, 1)
    return out


def game_random(spec):
    """Mode B: a batch of games in one GameBatch. roles[b] = (role of seat 0, role of seat 1): 'rand', 'randH' or an
    agent name. use_vs: every game in the batch has exactly one random seat -> step via step_vs, otherwise step."""
    import kagcore
    import kagsim

    from rl import act
    from rl import encode as E
    from rl.record import _base_config, _load, call_agent

    seeds, roles, steps, use_vs = spec["seeds"], spec["roles"], spec["steps"], spec["use_vs"]
    B = len(seeds)
    st = Stats()
    rng = np.random.default_rng(seeds[0] * 104729 + 3)
    canon = [kagsim.Game(s, 720) for s in seeds]
    gb = kagcore.GameBatch(seeds, 720)
    ce = CEnc(2 * B)
    fns, paths, cfgs = {}, {}, []
    for b in range(B):
        c = dict(_base_config())
        c["seed"] = seeds[b]
        cfgs.append(c)
        for p in (0, 1):
            if roles[b][p] not in ("rand", "randH"):
                paths[(b, p)] = agent_path(roles[b][p])
                fns[(b, p)] = _load(paths[(b, p)])
    t0 = time.time()
    for _ in range(steps):
        if all(g.done for g in canon):
            break
        cb = ce.run(gb, [0] * B + [1] * B, list(range(B)) * 2)
        uc = np.zeros((B, 2, 16), np.int32)
        mc = np.zeros((B, 2, 24), np.int32)
        actions = [[None, None] for _ in range(B)]
        for b in range(B):
            if canon[b].done:
                continue
            cacts = [None, None]
            for p in (0, 1):
                ctx = {"mode": "B", "seed": seeds[b], "roles": list(roles[b]), "step": canon[b].step_count, "seat": p}
                obs = canon[b].observe(p)
                st.c["obs_compared"] += 1
                if obs != gb.observe(b, p):
                    st.c["obs_mismatch"] += 1
                    st.err(kind="obs", **ctx)
                py = E.encode(obs)
                compare_encode(py, cb, p * B + b, st, ctx, len(obs["farms"][p]["hands"]) + 1)
                if roles[b][p] in ("rand", "randH"):
                    rush = roles[b][p] == "randH" and obs["hour"] < 2
                    u, m, d = check_execute(gb, b, p, obs, py, rng, st, ctx, kagcore, act, rush)
                    uc[b, p], mc[b, p] = u, m
                    cacts[p] = d  # canonical game: dict from Python execute
                else:
                    a = call_agent(fns[(b, p)], obs, cfgs[b], paths[(b, p)]) or {}
                    cacts[p] = a
                    actions[b][p] = a  # GameBatch: the same opponent dict
            canon[b].step(cacts[0], cacts[1])
            st.c["steps"] += 1
        if use_vs:
            seats = np.array([0 if roles[b][0] in ("rand", "randH") else 1 for b in range(B)], np.int32)
            uc1 = uc[np.arange(B), seats]
            mc1 = mc[np.arange(B), seats]
            opp = [actions[b][1 - seats[b]] if actions[b][1 - seats[b]] is not None else {} for b in range(B)]
            gb.step_vs(seats, uc1, mc1, opp)
        else:
            gb.step(uc, mc, [tuple(a) for a in actions])
    for b in range(B):
        for p in (0, 1):
            st.c["reward_compared"] += 1
            if canon[b].reward(p) != gb.reward(b, p):
                st.c["reward_mismatch"] += 1
                st.err(kind="reward", mode="B", seed=seeds[b], seat=p, py=canon[b].reward(p), c=gb.reward(b, p))
    st.c["games"] += B
    out = st.as_dict()
    out["spec"] = spec
    out["rewards"] = [[g.reward(0), g.reward(1)] for g in canon]
    out["sec"] = round(time.time() - t0, 1)
    return out


def run(spec):
    """Pool worker: one mode A game or one mode B batch."""
    return game_agents(spec) if spec["mode"] == "A" else game_random(spec)


def make_specs(quick, K, pairs, opps):
    """Job list: mode A games over the agent pairs (cycled), then mode B batches (rand vs rand, then rand vs the agents
    in opps, cycled). The seeds and roles do not depend on which agents are given."""
    steps = 100 if quick else 720
    n_a = 2 if quick else 20
    specs = []
    for i in range(n_a):
        a, b = pairs[i % len(pairs)]
        specs.append({"mode": "A", "seed": 910000 + 37 * i, "a": a, "b": b, "K": K, "steps": steps})
    # mode B: 2 batches of 4 "rand vs rand" (step) and 2 batches of 4 "rand vs agent" (step_vs, our seat 0/1)
    nb = 1 if quick else 2
    for j in range(nb):
        seeds = [920000 + 101 * j + 13 * k for k in range(4 if not quick else 1)]
        roles = [("randH", "rand") if k % 2 == 0 else ("rand", "randH") for k in range(len(seeds))]
        specs.append({"mode": "B", "seeds": seeds, "roles": roles, "steps": steps, "use_vs": False})
    for j in range(nb):
        seeds = [930000 + 101 * j + 13 * k for k in range(4 if not quick else 1)]
        me = "randH" if j == 0 else "rand"
        roles = [(me, opps[k % len(opps)]) if k % 2 == 0 else (opps[k % len(opps)], me) for k in range(len(seeds))]
        specs.append({"mode": "B", "seeds": seeds, "roles": roles, "steps": steps, "use_vs": True})
    return specs


def missing_agents(specs):
    """Agent names used by specs whose agents/<name>/main.py does not exist."""
    names = set()
    for s in specs:
        names |= {s["a"], s["b"]} if s["mode"] == "A" else {r for rr in s["roles"] for r in rr}
    return sorted(n for n in names - set(RANDOM_ROLES) if not os.path.exists(agent_path(n)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--K", type=int, default=2, help="random executes per (step, seat) in mode A")
    ap.add_argument("--out", default=os.path.join(HERE, "parity_result.json"))
    who = ap.add_mutually_exclusive_group()
    who.add_argument(
        "--agents",
        nargs="+",
        default=list(SHIPPED_AGENTS),
        help="code agents (folders under agents/) that generate the states: mode A plays every ordered pair, mode B "
        "uses them as opponents (default: the bots shipped in this repository)",
    )
    who.add_argument(
        "--competition-agents",
        action="store_true",
        help="the agent pairs of the competition-time runs (needs bots that are not in this repository)",
    )
    args = ap.parse_args()
    if args.competition_agents:
        pairs, opps = COMPETITION_PAIRS, COMPETITION_OPPS
    else:
        if set(args.agents) & set(RANDOM_ROLES):
            ap.error(f"--agents: {' and '.join(RANDOM_ROLES)} are random roles, not agent folders")
        pairs = [(a, b) for a in args.agents for b in args.agents]
        opps = [a for a in args.agents for _ in (0, 1)]  # opponent k % len(opps): each agent on both seats
    specs = make_specs(args.quick, args.K, pairs, opps)
    missing = missing_agents(specs)
    if missing:
        sys.exit(
            "agents not found: "
            + ", ".join(os.path.relpath(agent_path(n), ROOT) for n in missing)
            + "\nShipped bots: caretaker, pass_agent. Public league bots: "
            "python tools/fetch_league_bots.py --only NAME (see agents/sources.json); "
            "c3bc, v4_look12 and bcr_tdp_v4 are not released."
        )
    t0 = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(args.procs) as pool:
        res = pool.map(run, specs, chunksize=1)
    tot = {}
    cov = {
        "max_units": 0,
        "states_units_gt16": 0,
        "states_with_animals": 0,
        "states_inv_nonempty": 0,
        "exec_10_orders": 0,
        "unit_ops": set(),
        "order_ops": set(),
        "sell_all_nonzero": 0,
        "execute_moves": 0,
        "execute_ops": 0,
    }
    errors = []
    by_mode = {"A": {}, "B": {}}
    for r in res:
        m = r["spec"]["mode"]
        for k, v in r["counts"].items():
            tot[k] = tot.get(k, 0) + v
            by_mode[m][k] = by_mode[m].get(k, 0) + v
        for k, v in r["coverage"].items():
            if k == "max_units":
                cov[k] = max(cov[k], v)
            elif isinstance(v, list):
                cov[k] |= set(v)
            else:
                cov[k] += v
        errors += r["errors"]
    cov["unit_ops"] = sorted(cov["unit_ops"])
    cov["order_ops"] = sorted(cov["order_ops"])
    summary = {
        "total": tot,
        "by_mode": by_mode,
        "coverage": cov,
        "errors": errors[:50],
        "games": [{"spec": r["spec"], "rewards": r["rewards"], "sec": r["sec"]} for r in res],
        "wall_sec": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(
        json.dumps(
            {"total": tot, "by_mode": by_mode, "coverage": cov, "wall_sec": summary["wall_sec"]},
            indent=1,
            ensure_ascii=False,
        )
    )
    mism = tot["enc_mismatch"] + tot["obs_mismatch"] + tot["exec_mismatch"] + tot["reward_mismatch"]
    print(f"MISMATCHES: {mism}  (details: {args.out})")
    if errors:
        print(json.dumps(errors[:5], indent=1, ensure_ascii=False))
    sys.exit(1 if mism else 0)


if __name__ == "__main__":
    main()
