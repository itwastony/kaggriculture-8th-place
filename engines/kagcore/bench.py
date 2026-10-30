#!/usr/bin/env python3
"""Speed measurement: Python rl.encode/rl.act versus C++ kagcore on real states.

1) One full game of the agents in AGENTS (kagsim) in lockstep with GameBatch; at every step and seat:
   observe (dict), Python encode, Python execute (random choice), C++ encode_into for 1 row (via pybind),
   C++ encode/execute in a pure loop (bench_encode/bench_execute, every 24 steps).
2) Batched encode_into for 64 rows (the final state of that game) - the cost per row without pybind overhead.
3) Speed of GameBatch(64): encode of both seats + execute of both seats + engine step (random choices).

    python engines/kagcore/bench.py

Prints timings only. AGENTS defaults to caretaker vs caretaker (shipped in this repository); the competition-time
measurement used our bots c3bc vs v4_look12, which are not released. Requires the kagcore module
(engines/kagcore/build.sh), kagsim and kaggle-environments.
"""

import os
import statistics as stt
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "build"))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import kagcore  # noqa: E402
import kagsim  # noqa: E402

from rl import act as A  # noqa: E402
from rl import encode as E  # noqa: E402
from rl.record import _base_config, _load, call_agent  # noqa: E402

# isort: split
from parity import CEnc, rand_choices  # noqa: E402  imported after rl, in the original order

ns = time.perf_counter_ns
AGENTS = ("caretaker", "caretaker")  # agents/<name>/main.py of seats 0 and 1 (competition time: c3bc, v4_look12)


def summ(xs, scale=1e-3):
    xs = [x * scale for x in xs]
    return f"mean {stt.mean(xs):8.2f}  median {stt.median(xs):8.2f}  (n={len(xs)})"


def main():
    seed = 910000
    paths = [os.path.join(ROOT, "agents", n, "main.py") for n in AGENTS]
    fns = [_load(p) for p in paths]
    cfg = dict(_base_config())
    cfg["seed"] = seed
    canon = kagsim.Game(seed, 720)
    gb = kagcore.GameBatch([seed], 720)
    ce = CEnc(1)
    rng = np.random.default_rng(1)
    t_obs, t_pyenc, t_pyexe, t_cenc1, t_cexe1, t_step, b_enc, b_exe = [], [], [], [], [], [], [], []
    zu, zm = np.zeros((1, 2, 16), np.int32), np.zeros((1, 2, 24), np.int32)
    while not canon.done:
        acts = [None, None]
        for p in (0, 1):
            a = ns()
            obs = canon.observe(p)
            t_obs.append(ns() - a)
            a = ns()
            py = E.encode(obs)
            t_pyenc.append(ns() - a)
            a = ns()
            ce.run(gb, [p], [0])
            t_cenc1.append(ns() - a)
            uc, pairs, mc = rand_choices(rng, py)
            mcl = [int(x) for x in mc]
            a = ns()
            A.execute(obs, pairs, mcl)
            t_pyexe.append(ns() - a)
            a = ns()
            gb.execute_raw(0, p, uc, mc)
            t_cexe1.append(ns() - a)
            if canon.step_count % 24 == 12:
                b_enc.append(kagcore.bench_encode(gb, 0, p, 2000))
                b_exe.append(kagcore.bench_execute(gb, 0, p, uc, mc, 20000))
            acts[p] = call_agent(fns[p], obs, cfg, paths[p]) or {}
        a = ns()
        canon.step(acts[0], acts[1])
        t_step.append(ns() - a)
        gb.step(zu, zm, [(acts[0], acts[1])])
    assert canon.reward(0) == gb.reward(0, 0) and canon.reward(1) == gb.reward(0, 1)
    print(
        f"game {seed} {AGENTS[0]} vs {AGENTS[1]}: rewards {canon.reward(0):.0f} / {canon.reward(1):.0f}, "
        f"calls per function {len(t_pyenc)}  (us per call)"
    )
    print(f"  Python kagsim.observe(dict)         {summ(t_obs)}")
    print(f"  Python rl.encode.encode             {summ(t_pyenc)}")
    print(f"  Python rl.act.execute               {summ(t_pyexe)}")
    print(f"  Python kagsim.Game.step(dict x2)    {summ(t_step)}")
    print(f"  C++ encode_into, 1 row (pybind)     {summ(t_cenc1)}")
    print(f"  C++ execute_raw (pybind+tuple)      {summ(t_cexe1)}")
    print(f"  C++ encode, pure loop               {summ(b_enc)}")
    print(f"  C++ execute, pure loop              {summ(b_exe)}")

    # batched encode: 64 rows from the final state of the game, both seats
    n = 64
    ceb = CEnc(n)
    seats = [k % 2 for k in range(n)]
    reps = []
    for _ in range(200):
        a = ns()
        ceb.run(gb, seats, [0] * n)
        reps.append((ns() - a) / n)
    print(f"  C++ encode_into, batch of 64 rows   {summ(reps)}  [us per row]")

    # throughput of GameBatch(64): encode 2 seats + execute 2 seats + step
    B = 64
    g64 = kagcore.GameBatch([500000 + k for k in range(B)], 720)
    ce2 = CEnc(2 * B)
    rng = np.random.default_rng(2)
    # random choices, precomputed: any target/operation per unit; market bins uniform within range, then heads 0-20
    # (all except ORDER, SELL_FIRST, SEED_FIRST) are zeroed with probability 0.8
    uc = rng.integers(0, 100 * 56, size=(720, B, 2, 16)).astype(np.int32)
    from rl import spec as S

    nb = np.array([len(b) for _, _, b in S.MARKET_HEADS])
    mc = (rng.random((720, B, 2, 24)) * nb).astype(np.int32)
    mc[..., :21] *= rng.random((720, B, 2, 21)) < 0.2
    seats = [0] * B + [1] * B
    games = list(range(B)) * 2
    a = ns()
    t = 0
    while not g64.done.all():
        ce2.run(g64, seats, games)
        g64.step(uc[t], mc[t])
        t += 1
    dt = ns() - a
    print(
        f"  GameBatch(64): {t} steps, {dt / 1e9:.2f} s; per game-step (encode x2 + execute x2 + engine step) "
        f"{dt / (t * B) / 1e3:.2f} us; {t * B / (dt / 1e9):.0f} game-steps/s on 1 core"
    )


if __name__ == "__main__":
    main()
