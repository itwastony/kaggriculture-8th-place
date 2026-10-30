#!/usr/bin/env python3
"""Parity check of the CPU build of kagrl (build/libkagrl_cpu.so - the same kagrl.cu that goes to nvcc) against
references (no GPU needed).

References (at every step of every game):
  E1 kagcore.GameBatch (engines/kagcore: vendor kagsim engine + C++ encode/execute, checked against Python by
     engines/kagcore/parity.py):
     encode_into - 7 arrays byte for byte; execute_raw - the whole engine action; step; rewards; done; step_counts.
  E2 Python: kagsim.Game (vendor) + rl/encode.py + rl/act.py: encode(observe(p)) byte for byte (tmask via
     np.packbits), conv_action_raw(rl.act.execute(obs, ...)) == engine action, rewards to the dollar.
  E3 kaghd_cpu.RefBatch (vendor sim.hpp), stepped with THE SAME actions that kagrl returned (acts_out):
     canonical serialization of the full State (7645 bytes) byte for byte - check of the sim_hd engine inside kagrl.
Modes (who chooses the seats' actions):
  R  random choices: rand/randH (rand_choices from engines/kagcore/parity.py; randH - HIRE=12 in hours 0-1, >16 units)
     and kern (the kagrl random_choices kernel); step via act_step; int32 and int64 (numpy) indices on alternate steps.
  P  policy: randomly initialized rl/model.Net (d=64, 2 layers, vtower=1), Gumbel sampling as in
     rl/appo.py (rl.ppo.sample) over OUR encodings; uc/mc - torch.long (CPU tensors) passed straight to act_step.
  A  both seats - code agents (like mode A of engines/kagcore/parity.py): step via act_step with raw_mask=1 (ready
     actions kaghd_cpu.encode_action(dict)); at every step additionally K sets of random choices via execute
     (no step) - check of execute on rich states (animals, inventory, >10 orders, up to 16+ units).
  L  league: our seat - rand or policy, the other - a code agent via raw_mask; the kagcore reference - step with a dict
     for the opponent's seat.
Plus API checks: encode row layouts (games/seats), partial reset by mask with pinned shops,
step_raw over recorded streams (m < n), determinism/legality of random_choices, finished games.

    bash scripts/build_engines.sh                              # once: kagsim + kagcore
    bash engines/kagsim_cuda/build_cpu.sh                      # once: kaghd_cpu (engines/kagsim_cuda/build/)
    bash engines/kagrl_cuda/build_cpu.sh                       # once: engines/kagrl_cuda/build/libkagrl_cpu.so
    PYTHONHASHSEED=0 python engines/kagsim_cuda/record_streams.py   # once: the streams file of the step_raw check
    python engines/kagrl_cuda/parity_cpu.py --procs 4          # full run
    python engines/kagrl_cuda/parity_cpu.py --quick            # 100 steps, 2 games each

Needs the CPU builds of kagrl, kagsim_cuda (kaghd_cpu) and kagcore, kagsim, kaggle-environments and torch; a missing
build stops the run before any game, with the command that makes it. The code
agents of modes A and L are, by default, the bots shipped in this repository (agents/caretaker, agents/pass_agent:
SHIPPED_A_PAIRS, SHIPPED_L_ROLES); --competition-agents uses the tables of the competition-time runs
(COMPETITION_A_PAIRS, COMPETITION_L_ROLES), which include our bots that are not released (c3bc, v4_look12,
bcr_tdp_v4) and public league bots (soil_v7, pub0923_*; tools/fetch_league_bots.py). Seeds and the other roles are the
same for both tables. The step_raw API check reads the first 5 games of engines/kagsim_cuda/data/streams.pkl.gz
(STREAMS), which is not shipped: create it with the record_streams.py command above. Missing agent folders or a
missing streams file stop the run before any game. Writes --out (default engines/kagrl_cuda/data/parity_cpu_result.json,
with compiler, platform and library versions); exit code 1 on any mismatch.
"""

import argparse
import gzip
import importlib.util
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (
    os.path.join(ROOT, "engines", "kagsim_cuda", "build"),
    os.path.join(ROOT, "engines", "kagcore", "build"),
    os.path.join(ROOT, "engines", "kagcore"),
    HERE,
    ROOT,
):
    if p not in sys.path:
        sys.path.insert(0, p)

ENC = ("tiles", "units", "unit_pos", "unit_mask", "glob", "tmask", "mmask")
STREAMS = os.path.join(ROOT, "engines", "kagsim_cuda", "data", "streams.pkl.gz")
CHOICE_ROLES = ("rand", "randH", "kern", "pol")  # seats played by random choices or the random policy
# Code agents (agents/<name>/main.py) of modes A (pairs, game k of batch j: (4j + k) % 6) and L (seat roles, same
# indexing). Default: the bots shipped in this repository, in the slots where the competition-time tables had agents.
SHIPPED_A_PAIRS = [
    ("caretaker", "caretaker"),
    ("caretaker", "pass_agent"),
    ("pass_agent", "caretaker"),
    ("caretaker", "caretaker"),
    ("caretaker", "pass_agent"),
    ("pass_agent", "caretaker"),
]
SHIPPED_L_ROLES = [
    ("pol", "caretaker"),
    ("caretaker", "rand"),
    ("caretaker", "pol"),
    ("randH", "caretaker"),
    ("pol", "pass_agent"),
    ("caretaker", "kern"),
]
# --competition-agents: the tables of the competition-time runs; c3bc, v4_look12 and bcr_tdp_v4 are our bots that are
# not released, soil_v7 and pub0923_* are public league bots (tools/fetch_league_bots.py).
COMPETITION_A_PAIRS = [
    ("c3bc", "v4_look12"),
    ("v4_look12", "pub0923_tetsutani_dp"),
    ("bcr_tdp_v4", "c3bc"),
    ("pub0923_arsg_herdsafe3", "c3bc"),
    ("soil_v7", "v4_look12"),
    ("c3bc", "bcr_tdp_v4"),
]
COMPETITION_L_ROLES = [
    ("pol", "c3bc"),
    ("v4_look12", "rand"),
    ("bcr_tdp_v4", "pol"),
    ("randH", "pub0923_tetsutani_dp"),
    ("pol", "pub0923_arsg_herdsafe3"),
    ("c3bc", "kern"),
]
# every role that is played by a code agent (the workers test membership in this tuple)
AGENTS = tuple(
    sorted(
        {n for t in (SHIPPED_A_PAIRS, SHIPPED_L_ROLES, COMPETITION_A_PAIRS, COMPETITION_L_ROLES) for r in t for n in r}
        - set(CHOICE_ROLES)
    )
)


def agent_path(name):
    """agents/<name>/main.py under the repository root."""
    return os.path.join(ROOT, "agents", name, "main.py")


def act_tuple(row):
    """int32[170] (layout of batch_common.hpp) -> tuple in the format of kagcore.execute_raw / conv_action_raw."""
    nu, no = int(row[0]), int(row[121])
    us = tuple((int(row[1 + 3 * i]), int(row[2 + 3 * i]), int(row[3 + 3 * i])) for i in range(nu))
    os_ = tuple((int(row[122 + 3 * o]), int(row[123 + 3 * o]), int(row[124 + 3 * o])) for o in range(no))
    return (nu, us, no, os_)


class Stats:
    """Comparison counters (KEYS), coverage of the visited states/actions and the first 30 errors of one worker."""

    KEYS = (
        "games",
        "steps",
        "enc_rows_vs_kagcore",
        "enc_mismatch_vs_kagcore",
        "enc_rows_vs_python",
        "enc_mismatch_vs_python",
        "exec_vs_kagcore",
        "exec_mismatch_vs_kagcore",
        "exec_vs_python",
        "exec_mismatch_vs_python",
        "state_cmp",
        "state_mismatch",
        "reward_cmp",
        "reward_mismatch",
        "flag_cmp",
        "flag_mismatch",
        "final_reward_cmp",
        "final_reward_mismatch",
        "api_checks",
        "api_fail",
    )

    def __init__(self):
        self.c = {k: 0 for k in self.KEYS}
        self.cov = {
            "max_units": 0,
            "states_units_gt16": 0,
            "states_with_animals": 0,
            "states_inv_nonempty": 0,
            "exec_10_orders": 0,
            "unit_ops": set(),
            "order_ops": set(),
            "raw_seat_steps": 0,
            "int64_steps": 0,
            "int32_steps": 0,
            "torch_tensor_steps": 0,
            "nonzero_final_banks": 0,
        }
        self.errors = []

    def err(self, **kw):
        if len(self.errors) < 30:
            self.errors.append(kw)

    def as_dict(self):
        cov = dict(self.cov)
        cov["unit_ops"] = sorted(cov["unit_ops"])
        cov["order_ops"] = sorted(cov["order_ops"])
        return {"counts": self.c, "coverage": cov, "errors": self.errors}


def cmp_arrays(st, a, b, key_cnt, key_bad, ctx):
    """Compare two encodings (dicts of the 7 arrays) byte for byte; one count per call, the first differing array is
    reported."""
    st.c[key_cnt] += 1
    for k in ENC:
        x, y = a[k], b[k]
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            st.c[key_bad] += 1
            d = {"field": k, "dtype": [str(x.dtype), str(y.dtype)], "shape": [list(x.shape), list(y.shape)]}
            if x.shape == y.shape and x.dtype == y.dtype:
                j = int(np.flatnonzero(x.reshape(-1) != y.reshape(-1))[0])
                d.update(first=j, ours=repr(x.reshape(-1)[j].item()), ref=repr(y.reshape(-1)[j].item()))
            st.err(kind=key_bad, **ctx, **d)
            return


def py_row(py):
    """rl/encode.py dict -> kagrl format (tmask packed as np.packbits)."""
    d = dict(py)
    d["tmask"] = np.packbits(py["tmask"].reshape(py["tmask"].shape[0], -1), axis=-1)
    return d


def cover(st, py, n_units_total, act):
    """Coverage counters of one state (and of its engine action act, a tuple as from act_tuple, or None)."""
    nu = int(py["unit_mask"].sum())
    st.cov["max_units"] = max(st.cov["max_units"], nu)
    if n_units_total > 16:
        st.cov["states_units_gt16"] += 1
    if py["tiles"][:, 11:14].any():
        st.cov["states_with_animals"] += 1
    if py["units"][:, 3:15].any():
        st.cov["states_inv_nonempty"] += 1
    if act is not None:
        nu_, us, no, os_ = act
        if no == 10:
            st.cov["exec_10_orders"] += 1
        for u in us:
            st.cov["unit_ops"].add(u[0])
        for o in os_:
            st.cov["order_ops"].add(o[0])


# ---------------------------------------------------------------- one run (a batch of games of one mode)


def run_spec(spec):
    """One batch of games of mode R, P, A or L (see the module docstring), compared with all references."""
    import kagcore
    import kaghd_cpu
    import kagsim
    from parity import rand_choices

    from rl import act as A
    from rl import encode as E
    from rl import spec as S
    from rl.record import _base_config, _load, call_agent

    # isort: split
    from kagrl import RLBatch

    mode, seeds, steps, roles = spec["mode"], spec["seeds"], spec["steps"], spec["roles"]
    G = len(seeds)
    st = Stats()
    rng = np.random.default_rng(seeds[0] * 7919 + 11)
    rb = RLBatch(G, 720, backend="cpu")
    rb.reset(np.array(seeds, np.uint64))
    gb = kagcore.GameBatch(seeds, 720)
    ref = kaghd_cpu.RefBatch(G, 720)
    ref.reset(np.array(seeds, np.uint64))
    canon = [kagsim.Game(s, 720) for s in seeds]
    obs = rb.alloc_obs()
    kref = rb.alloc_obs()
    seats2 = np.tile(np.array([0, 1], np.int32), G)
    games2 = np.repeat(np.arange(G, dtype=np.int32), 2)
    acts = np.zeros((G, 2, 170), np.int32)
    xacts = np.zeros((G, 2, 170), np.int32)
    fns, cfgs = {}, []
    for g in range(G):
        c = dict(_base_config())
        c["seed"] = seeds[g]
        cfgs.append(c)
        for p in (0, 1):
            if roles[g][p] in AGENTS:
                fns[(g, p)] = _load(agent_path(roles[g][p]))
    net = None
    if any(r == "pol" for rr in roles for r in rr):
        import torch

        from rl.appo import unpack
        from rl.model import Net
        from rl.ppo import sample

        torch.set_num_threads(1)
        torch.manual_seed(seeds[0])
        net = Net(d=64, layers=2, heads=4, dk=16, vtower=1).eval()
    t0 = time.time()
    for t in range(steps):
        rb.encode(obs)
        gb.encode_into(seats2, *[kref[k] for k in ENC], games=games2)
        for r in range(2 * G):
            cmp_arrays(
                st,
                {k: obs[k][r] for k in ENC},
                {k: kref[k][r] for k in ENC},
                "enc_rows_vs_kagcore",
                "enc_mismatch_vs_kagcore",
                {"mode": mode, "seed": seeds[r // 2], "t": t, "seat": r % 2},
            )
        pyo = [[None, None] for _ in range(G)]
        pye = [[None, None] for _ in range(G)]
        for g in range(G):
            for p in (0, 1):
                o = canon[g].observe(p)
                pyo[g][p] = o
                pye[g][p] = E.encode(o)
                cmp_arrays(
                    st,
                    {k: obs[k][2 * g + p] for k in ENC},
                    py_row(pye[g][p]),
                    "enc_rows_vs_python",
                    "enc_mismatch_vs_python",
                    {"mode": mode, "seed": seeds[g], "t": t, "seat": p},
                )
        use64 = t % 2 == 1
        uc = np.zeros((G, 2, 16), np.int64 if use64 else np.int32)
        mc = np.zeros((G, 2, 24), np.int64 if use64 else np.int32)
        pairs = [[None, None] for _ in range(G)]
        # --- choices
        if any(r == "kern" for rr in roles for r in rr):
            ku = np.zeros((G, 2, 16), np.int32)
            km = np.zeros((G, 2, 24), np.int32)
            rb.random_choices(obs, seeds[0], t, ku, km)
        if net is not None:
            with torch.no_grad():
                b = {k: torch.from_numpy(obs[k]) for k in ENC}
                b["tmask"] = unpack(b["tmask"])
                lg, ml, _ = net(b["tiles"], b["units"], b["unit_pos"], b["unit_mask"], b["glob"])
                puc, pmc = sample(lg, ml, b)
            puc = puc.numpy().reshape(G, 2, 16)
            pmc = pmc.numpy().reshape(G, 2, 24)
        raw = np.zeros((G, 2, 170), np.int32)
        raw_mask = np.zeros((G, 2), np.uint8)
        dicts = [[None, None] for _ in range(G)]
        for g in range(G):
            if canon[g].done:
                continue
            for p in (0, 1):
                role = roles[g][p]
                if role in AGENTS:
                    d = call_agent(fns[(g, p)], pyo[g][p], cfgs[g], agent_path(role)) or {}
                    dicts[g][p] = d
                    raw[g, p] = kaghd_cpu.encode_action(d)
                    raw_mask[g, p] = 1
                    st.cov["raw_seat_steps"] += 1
                    continue
                if role in ("rand", "randH"):
                    u, pr, m = rand_choices(rng, pye[g][p], role == "randH" and pyo[g][p]["hour"] < 2)
                elif role == "kern":
                    u, m = ku[g, p], km[g, p]
                elif role == "pol":
                    u, m = puc[g, p], pmc[g, p]
                else:
                    raise ValueError(role)
                uc[g, p], mc[g, p] = u, m
                pairs[g][p] = [(int(v) // S.NOPS, int(v) % S.NOPS) for v in uc[g, p]]
        # --- A: K sets of random choices via execute (no step) on the current (rich) states
        for _ in range(spec.get("K", 0)):
            xu = np.zeros((G, 2, 16), np.int32)
            xm = np.zeros((G, 2, 24), np.int32)
            xp = [[None, None] for _ in range(G)]
            for g in range(G):
                for p in (0, 1):
                    u, pr, m = rand_choices(rng, pye[g][p])
                    xu[g, p], xm[g, p], xp[g][p] = u, m, pr
            rb.execute(xu, xm, xacts)
            for g in range(G):
                if canon[g].done:
                    continue
                for p in (0, 1):
                    mine = act_tuple(xacts[g, p])
                    tk = gb.execute_raw(g, p, xu[g, p], xm[g, p])
                    tp = kagcore.conv_action_raw(A.execute(pyo[g][p], xp[g][p], [int(v) for v in xm[g, p]]))
                    st.c["exec_vs_kagcore"] += 1
                    st.c["exec_vs_python"] += 1
                    if mine != tk:
                        st.c["exec_mismatch_vs_kagcore"] += 1
                        st.err(
                            kind="exec_kagcore(A)", mode=mode, seed=seeds[g], t=t, seat=p, ours=repr(mine), ref=repr(tk)
                        )
                    if mine != tp:
                        st.c["exec_mismatch_vs_python"] += 1
                        st.err(
                            kind="exec_python(A)", mode=mode, seed=seeds[g], t=t, seat=p, ours=repr(mine), ref=repr(tp)
                        )
                    cover(st, pye[g][p], len(pyo[g][p]["farms"][p]["hands"]) + 1, mine)
        # --- reference actions BEFORE the step (state before the step)
        ref_k = [[None, None] for _ in range(G)]
        ref_p = [[None, None] for _ in range(G)]
        for g in range(G):
            if canon[g].done:
                continue
            for p in (0, 1):
                if raw_mask[g, p]:
                    continue
                ref_k[g][p] = gb.execute_raw(g, p, uc[g, p].astype(np.int32), mc[g, p].astype(np.int32))
                d = A.execute(pyo[g][p], pairs[g][p], [int(v) for v in mc[g, p]])
                dicts[g][p] = d
                ref_p[g][p] = kagcore.conv_action_raw(d)
        # --- step: ours (act_step), kagcore (step with dict for agents), vendor RefBatch (our acts_out), Python
        if mode == "P" and not raw_mask.any():
            import torch

            rb.act_step(torch.from_numpy(uc.astype(np.int64)), torch.from_numpy(mc.astype(np.int64)), acts_out=acts)
            st.cov["torch_tensor_steps"] += 1
        else:
            rb.act_step(
                uc,
                mc,
                raw=raw if raw_mask.any() else None,
                raw_mask=raw_mask if raw_mask.any() else None,
                acts_out=acts,
            )
        st.cov["int64_steps" if use64 or mode == "P" else "int32_steps"] += 1
        for g in range(G):
            if canon[g].done:
                if acts[g].any():
                    st.err(kind="acts_out_not_zero_after_done", seed=seeds[g], t=t)
                    st.c["exec_mismatch_vs_kagcore"] += 1
                continue
            for p in (0, 1):
                mine = act_tuple(acts[g, p])
                if raw_mask[g, p]:
                    tp = kagcore.conv_action_raw(dicts[g][p])
                    st.c["exec_vs_python"] += 1
                    if mine != tp:
                        st.c["exec_mismatch_vs_python"] += 1
                        st.err(kind="raw_passthrough", mode=mode, seed=seeds[g], t=t, seat=p)
                    continue
                st.c["exec_vs_kagcore"] += 1
                st.c["exec_vs_python"] += 1
                if mine != ref_k[g][p]:
                    st.c["exec_mismatch_vs_kagcore"] += 1
                    st.err(
                        kind="exec_kagcore",
                        mode=mode,
                        seed=seeds[g],
                        t=t,
                        seat=p,
                        ours=repr(mine),
                        ref=repr(ref_k[g][p]),
                    )
                if mine != ref_p[g][p]:
                    st.c["exec_mismatch_vs_python"] += 1
                    st.err(
                        kind="exec_python",
                        mode=mode,
                        seed=seeds[g],
                        t=t,
                        seat=p,
                        ours=repr(mine),
                        ref=repr(ref_p[g][p]),
                    )
                cover(st, pye[g][p], len(pyo[g][p]["farms"][p]["hands"]) + 1, mine)
        gacts = [
            tuple(dicts[g][p] if raw_mask[g, p] else None for p in (0, 1)) if raw_mask[g].any() else None
            for g in range(G)
        ]
        gb.step(uc.astype(np.int32), mc.astype(np.int32), gacts if raw_mask.any() else None)
        ref.step(acts)
        for g in range(G):
            if not canon[g].done:
                canon[g].step(dicts[g][0], dicts[g][1])
        st.c["steps"] += int(sum(1 for g in range(G) if not (acts[g] == 0).all()))
        # --- state, rewards, flags
        a_ser, r_ser = rb.serialize(), ref.serialize()
        bad = np.nonzero((a_ser != r_ser).any(axis=1))[0]
        st.c["state_cmp"] += G
        st.c["state_mismatch"] += len(bad)
        for g in bad[:2]:
            st.err(kind="state", mode=mode, seed=seeds[g], t=t, byte=int(np.nonzero(a_ser[g] != r_ser[g])[0][0]))
        rw = rb.rewards()
        rk, rr = gb.rewards(), ref.rewards()
        rp = np.array([[c.reward(0), c.reward(1)] for c in canon])
        st.c["reward_cmp"] += 2 * G
        badr = int(((rw != rk) | (rw != rr) | (rw != rp)).sum())
        st.c["reward_mismatch"] += badr
        if badr:
            st.err(kind="reward", mode=mode, t=t, ours=rw.tolist(), kagcore=rk.tolist(), python=rp.tolist())
        dn, sc = rb.done(), rb.steps()
        st.c["flag_cmp"] += 2 * G
        st.c["flag_mismatch"] += int((dn.astype(bool) != gb.done).sum()) + int((sc != gb.step_counts).sum())
        st.c["flag_mismatch"] += int(
            sum(bool(dn[g]) != canon[g].done or int(sc[g]) != canon[g].step_count for g in range(G))
        )
    fin = rb.rewards()
    rp = np.array([[c.reward(0), c.reward(1)] for c in canon])
    st.c["final_reward_cmp"] += 2 * G
    st.c["final_reward_mismatch"] += int((fin != rp).sum())
    st.cov["nonzero_final_banks"] += int((fin != 0).sum())
    st.c["games"] += G
    out = st.as_dict()
    out["spec"] = {k: v for k, v in spec.items()}
    out["final_banks"] = fin.tolist()
    out["final_steps"] = rb.steps().tolist()
    out["sec"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------- API checks


def run_api(spec):
    """API checks: encode row layouts, partial reset by mask with pinned shops, step_raw, random_choices, finished
    games."""
    import kaghd_cpu
    from kagrl import RLBatch

    st = Stats()
    t0 = time.time()

    def check(ok, what, **kw):
        st.c["api_checks"] += 1
        if not ok:
            st.c["api_fail"] += 1
            st.err(kind="api", what=what, **kw)

    n = 12
    rng = np.random.default_rng(5)
    seeds = (np.arange(n, dtype=np.uint64) * 7919 + 424242).astype(np.uint64)
    rb = RLBatch(n, 720, backend="cpu")
    ref = kaghd_cpu.RefBatch(n, 720)
    rb.reset(seeds)
    ref.reset(seeds)
    obs = rb.alloc_obs()
    uc = np.zeros((n, 2, 16), np.int32)
    mc = np.zeros((n, 2, 24), np.int32)
    acts = np.zeros((n, 2, 170), np.int32)
    for t in range(150):  # reach non-empty states
        rb.encode(obs)
        rb.random_choices(obs, 77, t, uc, mc)
        rb.act_step(uc, mc, acts_out=acts)
        ref.step(acts)
    check(bool((rb.serialize() == ref.serialize()).all()), "state after 150 random steps")
    rb.encode(obs)
    # 1) row layouts: games/seats and seats-only == rows of the "both seats" layout
    R = 40
    games = rng.integers(0, n, R).astype(np.int32)
    seats = rng.integers(0, 2, R).astype(np.int32)
    o2 = rb.alloc_obs(R)
    rb.encode(o2, games=games, seats=seats)
    check(all(o2[k].tobytes() == obs[k][2 * games + seats].tobytes() for k in ENC), "encode games/seats rows")
    s1 = rng.integers(0, 2, n).astype(np.int32)
    o3 = rb.alloc_obs(n)
    rb.encode(o3, seats=s1)
    check(all(o3[k].tobytes() == obs[k][2 * np.arange(n) + s1].tobytes() for k in ENC), "encode seats-only rows")
    try:
        rb.encode(o3, games=games[:n], seats=None)
        check(False, "games without seats must raise")
    except ValueError:
        check(True, "games without seats raises")
    # 2) random_choices: determinism, int32 == int64, legality, dependence on the salt
    u1, m1 = np.zeros((n, 2, 16), np.int32), np.zeros((n, 2, 24), np.int32)
    u2, m2 = np.zeros((n, 2, 16), np.int64), np.zeros((n, 2, 24), np.int64)
    u3, m3 = np.zeros((n, 2, 16), np.int32), np.zeros((n, 2, 24), np.int32)
    rb.random_choices(obs, 1234, 5, u1, m1)
    rb.random_choices(obs, 1234, 5, u2, m2)
    rb.random_choices(obs, 1234, 6, u3, m3)
    check((u1 == u2).all() and (m1 == m2).all(), "random_choices int32 == int64")
    check(not ((u1 == u3).all() and (m1 == m3).all()), "random_choices depends on salt")
    tm = np.unpackbits(obs["tmask"], axis=-1).reshape(2 * n, 16, -1)
    um = obs["unit_mask"]
    uu = u1.reshape(2 * n, 16)
    legal = all((tm[r, u, uu[r, u]] == 1) if um[r, u] else uu[r, u] == 0 for r in range(2 * n) for u in range(16))
    check(legal, "random_choices legal (bit set in tmask; absent units = 0)")
    from rl.model import HEAD_SIZES

    check(
        all(0 <= m1.reshape(2 * n, 24)[r, h] < HEAD_SIZES[h] for r in range(2 * n) for h in range(24)),
        "random_choices bins in range",
    )
    # 3) partial reset by mask + pinned shops in the middle of a game == vendor RefBatch
    mask = (np.arange(n) % 3 == 0).astype(np.uint8)
    seeds2 = seeds + np.uint64(17)
    shops = rng.integers(0, 8, (n, 8)).astype(np.uint8)
    nshops = rng.integers(0, 9, n).astype(np.int32)
    rb.reset(seeds2, mask=mask, shops=shops, nshops=nshops)
    ref.reset(seeds2, mask, shops, nshops)
    check(bool((rb.serialize() == ref.serialize()).all()), "partial reset with mask+shops")
    for t in range(600):
        rb.encode(obs)
        rb.random_choices(obs, 99, t, uc, mc)
        rb.act_step(uc, mc, acts_out=acts)
        ref.step(acts)
    check(
        bool((rb.serialize() == ref.serialize()).all()) and bool((rb.rewards() == ref.rewards()).all()),
        "600 steps after partial reset",
    )
    d0 = rb.done()
    check(bool(d0[mask == 0].all()) and not bool(d0[mask == 1].any()), "done only for non-reset games")
    # 4) finished games: a step does not change the state, acts_out = 0, execute = 0
    for t in range(200):
        rb.encode(obs)
        rb.random_choices(obs, 98, t, uc, mc)
        rb.act_step(uc, mc, acts_out=acts)
        ref.step(acts)
    check(bool(rb.done().all()) and bool((rb.steps() == 719).all()), "all done at step 719")
    s_before = rb.serialize()
    acts[:] = 7
    rb.act_step(uc, mc, acts_out=acts)
    check(bool((rb.serialize() == s_before).all()) and not acts.any(), "act_step on done: no-op, acts_out zeros")
    acts[:] = 7
    rb.execute(uc, mc, acts)
    check(not acts.any(), "execute on done: zeros")
    # 5) step_raw over recorded streams (m < n: game i - stream i % m) == vendor RefBatch
    recs = pickle.load(gzip.open(STREAMS))[:5]
    m = len(recs)
    A_ = np.zeros((719, m, 2, 170), np.int32)
    for j, r in enumerate(recs):
        for p in (0, 1):
            A_[:, j, p] = kaghd_cpu.encode_actions(r["acts"][p])
    s3 = np.array([recs[i % m]["seed"] + (i // m) * 7919 for i in range(n)], np.uint64)
    sh3 = np.zeros((n, 8), np.uint8)
    ns3 = np.zeros(n, np.int32)
    for i in range(m):  # recorded pinned shops (for some of the streams)
        if recs[i]["shops"] is not None:
            sh3[i, : len(recs[i]["shops"])] = recs[i]["shops"]
            ns3[i] = len(recs[i]["shops"])
    rb.reset(s3, shops=sh3, nshops=ns3)
    ref.reset(s3, None, sh3, ns3)
    bad = 0
    for t in range(719):
        rb.step_raw(A_[t])
        ref.step(A_[t])
        bad += int((rb.serialize() != ref.serialize()).any(axis=1).sum())
    check(bad == 0, "step_raw streams (m<n)", bad_states=bad)
    check(
        bool((rb.rewards()[:m] == np.array([r["rewards"] for r in recs])).all()),
        "step_raw streams reproduce recorded banks",
        ours=rb.rewards()[:m].tolist(),
        recorded=[r["rewards"] for r in recs],
    )
    # 6) argument errors
    for what, fn in (
        ("bad uc shape", lambda: rb.act_step(np.zeros((n, 16), np.int32), mc)),
        ("uc/mc dtype mismatch", lambda: rb.act_step(np.zeros((n, 2, 16), np.int64), mc)),
        ("bad obs dtype", lambda: rb.encode({**obs, "tiles": obs["tiles"].astype(np.float64)})),
        ("nshops out of range", lambda: rb.reset(seeds, shops=shops, nshops=nshops + 9)),
    ):
        try:
            fn()
            check(False, what + " must raise")
        except ValueError:
            check(True, what)
    out = st.as_dict()
    out["spec"] = spec
    out["sec"] = round(time.time() - t0, 1)
    return out


def run(spec):
    """Pool worker: run_api for mode API, otherwise run_spec."""
    return run_api(spec) if spec["mode"] == "API" else run_spec(spec)


def make_specs(quick, K, a_pairs, l_roles):
    """Job list: R, P, A (agent pairs a_pairs) and L (seat roles l_roles) batches, then the API checks. The seeds and
    the non-agent roles do not depend on the agent tables."""
    q = quick
    steps = 100 if q else 720
    specs = []
    # R: 4 batches of 6 games (rand/randH/kern in different seat combinations)
    rroles = [("randH", "rand"), ("rand", "kern"), ("kern", "randH"), ("kern", "kern")]
    for j in range(1 if q else 4):
        seeds = [940000 + 101 * j + 13 * k for k in range(2 if q else 6)]
        specs.append(
            {"mode": "R", "seeds": seeds, "roles": [rroles[(k + j) % 4] for k in range(len(seeds))], "steps": steps}
        )
    # P: 3 batches of 6 games, policy on both seats
    for j in range(1 if q else 3):
        seeds = [950000 + 101 * j + 13 * k for k in range(2 if q else 6)]
        specs.append({"mode": "P", "seeds": seeds, "roles": [("pol", "pol")] * len(seeds), "steps": steps})
    # A: 3 batches of 4 games agent vs agent + K random execute calls per step
    for j in range(1 if q else 3):
        seeds = [960000 + 101 * j + 13 * k for k in range(2 if q else 4)]
        specs.append(
            {
                "mode": "A",
                "seeds": seeds,
                "roles": [a_pairs[(4 * j + k) % 6] for k in range(len(seeds))],
                "steps": steps,
                "K": K,
            }
        )
    # L: 3 batches of 4 games - our seat rand/pol, the other - an agent
    for j in range(1 if q else 3):
        seeds = [970000 + 101 * j + 13 * k for k in range(2 if q else 4)]
        specs.append(
            {
                "mode": "L",
                "seeds": seeds,
                "roles": [l_roles[(4 * j + k) % 6] for k in range(len(seeds))],
                "steps": steps,
            }
        )
    specs.append({"mode": "API"})
    return specs


def missing_builds():
    """Messages for the compiled modules this check imports that are not built, each with the command that builds it."""
    out = []
    if importlib.util.find_spec("kaghd_cpu") is None:
        out.append("kaghd_cpu not found (engines/kagsim_cuda/build/): bash engines/kagsim_cuda/build_cpu.sh")
    if importlib.util.find_spec("kagcore") is None:
        out.append("kagcore not found (engines/kagcore/build/): bash scripts/build_engines.sh")
    if importlib.util.find_spec("kagsim") is None:
        out.append("kagsim not installed: bash scripts/build_engines.sh")
    lib = os.path.join(HERE, "build", "libkagrl_cpu.so")
    if not os.path.exists(lib):
        out.append(f"{os.path.relpath(lib, ROOT)} not found: bash engines/kagrl_cuda/build_cpu.sh")
    return out


def missing_inputs(specs):
    """Messages for agent folders and the streams file that the job list needs but that do not exist."""
    names = {r for s in specs for rr in s.get("roles", ()) for r in rr if r in AGENTS}
    out = [
        f"agent not found: {os.path.relpath(agent_path(n), ROOT)} (see agents/sources.json)"
        for n in sorted(names)
        if not os.path.exists(agent_path(n))
    ]
    if any(s["mode"] == "API" for s in specs) and not os.path.exists(STREAMS):
        out.append(
            f"{os.path.relpath(STREAMS, ROOT)} not found (step_raw API check): create it first with "
            "PYTHONHASHSEED=0 python engines/kagsim_cuda/record_streams.py"
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--K", type=int, default=2, help="sets of random execute calls per step in mode A")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "parity_cpu_result.json"))
    ap.add_argument(
        "--competition-agents",
        action="store_true",
        help="code agents of modes A and L as in the competition-time runs (needs unreleased bots); "
        "default: caretaker and pass_agent",
    )
    args = ap.parse_args()
    steps = 100 if args.quick else 720
    if args.competition_agents:
        specs = make_specs(args.quick, args.K, COMPETITION_A_PAIRS, COMPETITION_L_ROLES)
    else:
        specs = make_specs(args.quick, args.K, SHIPPED_A_PAIRS, SHIPPED_L_ROLES)
    missing = missing_builds() + missing_inputs(specs)
    if missing:
        sys.exit("\n".join(missing))
    t0 = time.time()
    with get_context("spawn").Pool(args.procs) as pool:
        res = pool.map(run, specs, chunksize=1)
    tot, by_mode = {k: 0 for k in Stats.KEYS}, {}
    cov = {"max_units": 0, "unit_ops": set(), "order_ops": set()}
    errors = []
    for r in res:
        m = r["spec"]["mode"]
        bm = by_mode.setdefault(m, {k: 0 for k in Stats.KEYS})
        for k, v in r["counts"].items():
            tot[k] += v
            bm[k] += v
        for k, v in r["coverage"].items():
            if k == "max_units":
                cov[k] = max(cov[k], v)
            elif isinstance(v, list):
                cov[k] = cov.get(k, set()) | set(v)
            else:
                cov[k] = cov.get(k, 0) + v
        errors += r["errors"]
    cov["unit_ops"] = sorted(cov["unit_ops"])
    cov["order_ops"] = sorted(cov["order_ops"])
    mism = sum(tot[k] for k in Stats.KEYS if "mismatch" in k) + tot["api_fail"]
    try:
        cxx = subprocess.run(["c++", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    except Exception:  # noqa: BLE001
        cxx = "?"
    import torch

    summary = {
        "ok": mism == 0,
        "mismatches_total": mism,
        "total": tot,
        "by_mode": by_mode,
        "coverage": cov,
        "errors": errors[:50],
        "runs": [
            {
                "spec": r["spec"],
                "sec": r["sec"],
                "final_banks": r.get("final_banks"),
                "final_steps": r.get("final_steps"),
            }
            for r in res
        ],
        "protocol": {
            "build": "bash engines/kagrl_cuda/build_cpu.sh (c++ -x c++ -std=c++17 -O2 -ffp-contract=off -DKAG_EMULATE)",
            "compiler": cxx,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "steps_per_game": steps,
            "loop": "T iterations: encode both seats -> choice -> act_step (a game finishes "
            "after 719 steps; the 720th iteration encodes the final state, its step is a no-op)",
            "references": [
                "kagcore.GameBatch (engines/kagcore/build)",
                "kagsim.Game + rl/encode.py + rl/act.py",
                "kaghd_cpu.RefBatch (vendor sim.hpp), stepped with our acts_out",
            ],
            "policy": "rl.model.Net(d=64, layers=2, heads=4, dk=16, vtower=1), "
            "torch.manual_seed(first seed of the batch), "
            "sampling rl.ppo.sample (Gumbel), fp32 on the CPU",
        },
        "wall_sec": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(
        json.dumps(
            {"ok": summary["ok"], "total": tot, "coverage": cov, "wall_sec": summary["wall_sec"]},
            indent=1,
            ensure_ascii=False,
        )
    )
    print(f"MISMATCHES: {mism}  (details: {args.out})")
    if errors:
        print(json.dumps(errors[:5], indent=1, ensure_ascii=False))
    sys.exit(1 if mism else 0)


if __name__ == "__main__":
    main()
