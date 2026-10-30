#!/usr/bin/env python3
"""Comparison of the CPU build of the CUDA-capable core (sim_hd.hpp, module kaghd_cpu) with the original kagsim.

What is compared at EVERY step of every game (same seeds, same action streams):
  1. full state: canonical State serialization (all fields, including telemetry and array
     tails; state_visit.hpp), kaghd_cpu.Batch (sim_hd.hpp) vs kaghd_cpu.RefBatch
     (the ORIGINAL vendor/kagsim/sim/sim.hpp, the same batch loop), byte-for-byte;
  2. banks of both seats: sim_hd vs the installed kagsim module (kagsim.Game.reward) and the reference, to the dollar
     (exact float64 equality);
  3. observations of both seats (kagsim.Game.observe: board, positions, market, shops, shed, seeds,
     unit inventories with key ORDER) and telemetry (kagsim.Game.telemetry), sim_hd vs kagsim;
  4. done and the step number.
Actions go into kagsim.Game.step as raw dicts/objects (its conv_action), into sim_hd via
kaghd_cpu.encode_action -> int32[170] -> decode_action, i.e. the encoding is checked as well.

Static checks: price amplitudes (literals == formula), market_price sim_hd == vendor == Python environment
for all 9 items and inventories [-20000, 80000], completeness of the serializer (parsing of struct declarations).

    PYTHONHASHSEED=0 python engines/kagsim_cuda/record_streams.py   # -> engines/kagsim_cuda/data/streams.pkl.gz
    python engines/kagsim_cuda/parity.py --streams engines/kagsim_cuda/data/streams.pkl.gz --xseed 2 --bench 200

The streams file is not shipped: create it first with the record_streams.py command above (by default games of the
shipped bots caretaker and pass_agent; see its docstring). --xseed K replays every stream on K new seeds (without pinned
shops; the final banks are then not compared with the record); --bench N times the CPU batch loops against
kagsim.run_many on the first N streams. Writes the full result to --out (default
engines/kagsim_cuda/data/parity_result.json); exit code 1 on any mismatch. Requires the CPU module
(engines/kagsim_cuda/build_cpu.sh), kagsim and kaggle-environments.
"""

import argparse
import gzip
import json
import os
import pickle
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "build"))
import kaghd_cpu as K  # noqa: E402
import kagsim  # noqa: E402

A = K.ACT_I32


# ------------------------------------------------------------------ static checks
def struct_members(src, name):
    """Names of the top-level data members of `struct name { ... };` in C++ source text (methods are skipped)."""
    m = re.search(r"struct\s+" + name + r"\s*\{(.*?)\n\};", src, re.S)
    body = m.group(1)
    names = []
    depth = 0  # fields only at the top level of the struct body
    for line in body.split("\n"):
        line = line.split("//")[0].strip()
        d0 = depth
        depth += line.count("{") - line.count("}")
        if d0 != 0 or not line or "(" in line or not line.endswith(";"):
            continue
        decl = line[:-1]
        decl = re.sub(r"=\s*[^,]*", "", decl)  # initializers
        parts = decl.split(None, 1)
        if len(parts) < 2:
            continue
        for v in parts[1].split(","):
            v = re.sub(r"\[.*?\]", "", v).strip()
            if v:
                names.append(v)
    return names


def static_checks():
    """Price amplitude literals, market_price of sim_hd / vendor / Python environment, serializer completeness."""
    res = {}
    res["amps_selfcheck_mismatches"] = K.amps_selfcheck()
    lo, hi = -20000, 80000
    ph, pr = K.market_prices(lo, hi), K.ref_market_prices(lo, hi)
    from kaggle_environments.envs.kaggriculture import kaggriculture as KE

    py = np.array([[KE.market_price(it, v) for v in range(lo, hi + 1)] for it in KE.PRODUCTS], np.int32)
    res["price_range"] = [lo, hi]
    res["price_points"] = int(ph.size)
    res["price_mismatch_hd_vs_vendor"] = int((ph != pr).sum())
    res["price_mismatch_hd_vs_python_env"] = int((ph != py).sum())
    src = open(os.path.join(HERE, "sim_hd.hpp")).read()
    vis = open(os.path.join(HERE, "state_visit.hpp")).read()
    missing = []
    for st, pref in (("Tile", "t."), ("Farm", "f."), ("Market", "s.market."), ("State", "s.")):
        for nm in struct_members(src, st):
            if st == "State" and nm in ("farms", "market"):
                continue
            if not re.search(re.escape(pref) + nm + r"\b", vis):
                missing.append(f"{st}.{nm}")
    res["serializer_missing_fields"] = missing
    res["ser_bytes"] = K.SER_BYTES
    res["state_bytes_with_padding"] = K.STATE_BYTES
    return res


# ------------------------------------------------------------------ step-by-step comparison
class Stats:
    """Comparison counters, the first 20 mismatch messages and coverage of one comparison run."""

    def __init__(self):
        self.games = 0
        self.steps = 0  # engine steps (transitions) in total
        self.state_cmp = 0  # full-state comparisons (game x moment)
        self.state_mismatch = 0
        self.reward_cmp = 0
        self.reward_mismatch = 0
        self.obs_cmp = 0  # observe comparisons (game x seat x moment)
        self.obs_mismatch = 0
        self.tel_cmp = 0
        self.tel_mismatch = 0
        self.flag_mismatch = 0  # done / step number
        self.final_vs_recorded_mismatch = 0
        self.examples = []
        # coverage: sums of final telemetry over games and seats, max hired workers/land quadrants
        self.coverage_telemetry_sums = {}
        self.max_hands_seen = 0
        self.max_quadrants_seen = 0
        self.games_with_pinned_shops = 0

    def ex(self, msg):
        if len(self.examples) < 20:
            self.examples.append(msg)

    def total_mismatch(self):
        return (
            self.state_mismatch
            + self.reward_mismatch
            + self.obs_mismatch
            + self.tel_mismatch
            + self.flag_mismatch
            + self.final_vs_recorded_mismatch
        )


def inv_orders(o):
    """Key order of the unit inventories of an observation (dict equality alone ignores it)."""
    return [list(d.items()) for d in o["private"]["inventories"]]


def replay_chunk(recs, seeds, shops, obs_every, st, tag, check_recorded):
    """Replay recs in lockstep on kaghd_cpu.Batch, kaghd_cpu.RefBatch and kagsim.Game, comparing after every step.

    Streams shorter than the game are padded with empty actions; check_recorded compares the final banks with the
    record."""
    n = len(recs)
    T = max(len(r["acts"][0]) for r in recs)
    empty = K.encode_action({})
    acts_T = np.empty((T, n, 2, A), np.int32)
    for g, r in enumerate(recs):
        for p in (0, 1):
            e = K.encode_actions(r["acts"][p])
            acts_T[: len(e), g, p] = e
            acts_T[len(e) :, g, p] = empty
    sd = np.array(seeds, np.uint64)
    sh = np.zeros((n, 8), np.uint8)
    ns = np.zeros(n, np.int32)
    for g, s in enumerate(shops):
        if s is not None:
            sh[g, : len(s)] = s
            ns[g] = len(s)
    hd, ref = K.Batch(n), K.RefBatch(n)
    hd.reset(sd, None, sh, ns)
    ref.reset(sd, None, sh, ns)
    games = [
        kagsim.Game(int(seeds[g]), 720, shops[g]) if shops[g] is not None else kagsim.Game(int(seeds[g]), 720)
        for g in range(n)
    ]
    st.games += n

    def check(t):
        s_hd, s_ref = hd.serialize(), ref.serialize()
        bad = np.nonzero((s_hd != s_ref).any(axis=1))[0]
        st.state_cmp += n
        st.state_mismatch += len(bad)
        for g in bad[:3]:
            off = int(np.nonzero(s_hd[g] != s_ref[g])[0][0])
            st.ex(f"[{tag}] state g={g} t={t} first byte {off}")
        r_hd, r_ref = hd.rewards(), ref.rewards()
        r_ks = np.array([[x.reward(0), x.reward(1)] for x in games])
        rb = np.nonzero(((r_hd != r_ks) | (r_ref != r_ks)).any(axis=1))[0]
        st.reward_cmp += 2 * n
        st.reward_mismatch += len(rb)
        for g in rb[:3]:
            st.ex(f"[{tag}] reward g={g} t={t} hd={r_hd[g]} ref={r_ref[g]} kagsim={r_ks[g]}")
        d_hd, c_hd = hd.done(), hd.step_counts()
        for g in range(n):
            if bool(d_hd[g]) != games[g].done or int(c_hd[g]) != games[g].step_count:
                st.flag_mismatch += 1
                st.ex(f"[{tag}] flags g={g} t={t}")
        if obs_every and (t % obs_every == 0 or t == T):
            for g in range(n):
                for p in (0, 1):
                    o1, o2 = K.observe(hd, g, p), games[g].observe(p)
                    fo = o1["farms"][p]
                    st.max_hands_seen = max(st.max_hands_seen, len(fo["hands"]))
                    st.max_quadrants_seen = max(st.max_quadrants_seen, len(fo["unlocked_quadrants"]))
                    st.obs_cmp += 1
                    if o1 != o2 or inv_orders(o1) != inv_orders(o2):
                        st.obs_mismatch += 1
                        st.ex(f"[{tag}] observe g={g} p={p} t={t}")
                    t1, t2 = K.telemetry(hd, g, p), games[g].telemetry(p)
                    st.tel_cmp += 1
                    if t1 != t2:
                        st.tel_mismatch += 1
                        st.ex(f"[{tag}] telemetry g={g} p={p} t={t}")

    check(0)
    for t in range(T):
        hd.step(acts_T[t])
        ref.step(acts_T[t])
        for g, r in enumerate(recs):
            if not games[g].done:
                a0 = r["acts"][0][t] if t < len(r["acts"][0]) else {}
                a1 = r["acts"][1][t] if t < len(r["acts"][1]) else {}
                games[g].step(a0, a1)
                st.steps += 1
        check(t + 1)
    st.games_with_pinned_shops += sum(1 for s in shops if s is not None)
    for g in range(n):
        for p in (0, 1):
            for k, v in K.telemetry(hd, g, p).items():
                st.coverage_telemetry_sums[k] = st.coverage_telemetry_sums.get(k, 0) + v
    if check_recorded:
        r_hd = hd.rewards()
        for g, r in enumerate(recs):
            if list(r_hd[g]) != list(r["rewards"]):
                st.final_vs_recorded_mismatch += 1
                st.ex(f"[{tag}] final g={g}: hd={r_hd[g]} recorded={r['rewards']}")


def run_replays(recs, chunk, obs_every, st, tag, seed_fn, shops_fn, check_recorded):
    """replay_chunk over recs in chunks; seed_fn / shops_fn give the seed and pinned shops of a record."""
    t0 = time.time()
    for i in range(0, len(recs), chunk):
        part = recs[i : i + chunk]
        replay_chunk(part, [seed_fn(r) for r in part], [shops_fn(r) for r in part], obs_every, st, tag, check_recorded)
        print(
            f"  [{tag}] {min(i + chunk, len(recs))}/{len(recs)} games, mismatches {st.total_mismatch()}, "
            f"{time.time() - t0:.0f} s",
            flush=True,
        )


# ------------------------------------------------------------------ speed
def bench(recs, m, reps, mult=1):
    """Best-of-reps time of the CPU batch loops (sim_hd and vendor) and of kagsim.run_many on the same games, and
    whether their final banks are equal."""
    recs = recs[:m]
    m = len(recs)
    T = max(len(r["acts"][0]) for r in recs)
    empty = K.encode_action({})
    acts_T = np.empty((T, m, 2, A), np.int32)
    for g, r in enumerate(recs):
        for p in (0, 1):
            e = K.encode_actions(r["acts"][p])
            acts_T[: len(e), g, p] = e
            acts_T[len(e) :, g, p] = empty
    # n = m*mult games: game i plays stream i % m on its own seed; without pinned
    # shops (kagsim.run_many does not support them)
    n = m * mult
    seeds = np.array([(int(recs[i % m]["seed"]) + (i // m) * 7919) % (2**31) for i in range(n)], np.uint64)
    out = {"games": n, "distinct_streams": m, "steps_per_game": T, "reps": reps}
    rew = {}
    for name, cls in (("sim_hd_cpu_batch", K.Batch), ("vendor_sim_same_batch_loop", K.RefBatch)):
        best = None
        for _ in range(reps):
            b = cls(n)
            b.reset(seeds)
            dt = b.run(acts_T)
            best = dt if best is None else min(best, dt)
            rew[name] = b.rewards()
        out[name + "_sec"] = best
        out[name + "_games_per_sec"] = n / best
    strm = [(kagsim.Stream(r["acts"][0]), kagsim.Stream(r["acts"][1])) for r in recs]
    jobs = [(strm[i % m][0], strm[i % m][1], int(seeds[i])) for i in range(n)]
    best = None
    for _ in range(reps):
        t0 = time.perf_counter()
        res = kagsim.run_many(jobs, 720, 1)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    rew["kagsim_run_many"] = np.array(res, np.float64)
    out["kagsim_run_many_1thread_sec"] = best
    out["kagsim_run_many_1thread_games_per_sec"] = n / best
    out["bench_final_rewards_equal"] = bool(
        (rew["sim_hd_cpu_batch"] == rew["kagsim_run_many"]).all()
        and (rew["vendor_sim_same_batch_loop"] == rew["kagsim_run_many"]).all()
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default=os.path.join(HERE, "data", "streams.pkl.gz"))
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--obs-every", type=int, default=1, help="observe/telemetry every k steps (1 = every step)")
    ap.add_argument("--xseed", type=int, default=0, help="extra runs of each stream on new seeds")
    ap.add_argument("--xseed-obs-every", type=int, default=1)
    ap.add_argument("--bench", type=int, default=0, help="speed measurement on the first N games")
    ap.add_argument("--bench-reps", type=int, default=3)
    ap.add_argument(
        "--bench-mult", type=int, default=1, help="games in the measurement = bench * mult (streams cycled, new seeds)"
    )
    ap.add_argument("--no-replay", action="store_true", help="only static checks and the speed measurement")
    ap.add_argument("--limit", type=int, default=0, help="take only the first N streams (debugging)")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "parity_result.json"))
    args = ap.parse_args()

    result = {"streams": args.streams}
    print("== static checks")
    sc = static_checks()
    result["static"] = sc
    print(json.dumps(sc, ensure_ascii=False))

    with gzip.open(args.streams, "rb") as f:
        recs = pickle.load(f)
    if args.limit:
        recs = recs[: args.limit]
    modes = {}
    for r in recs:
        modes[r["mode"]] = modes.get(r["mode"], 0) + 1
    result["recorded_games_by_mode"] = modes
    print(f"== streams: {len(recs)} {modes}")

    if not args.no_replay:
        print("== comparison on the recorded seeds (with pinned shops where there were any)")
        st = Stats()
        run_replays(recs, args.chunk, args.obs_every, st, "rec", lambda r: r["seed"], lambda r: r["shops"], True)
        result["recorded"] = {k: v for k, v in st.__dict__.items()}

    if args.xseed and not args.no_replay:
        print(f"== comparison of the same streams on {args.xseed} new seeds each (without pinning shops)")
        st2 = Stats()
        for j in range(1, args.xseed + 1):
            run_replays(
                recs,
                args.chunk,
                args.xseed_obs_every,
                st2,
                f"x{j}",
                lambda r, j=j: (r["seed"] * 7 + j * 1000003 + 12345) % (2**31),
                lambda r: None,
                False,
            )
        result["xseed"] = {k: v for k, v in st2.__dict__.items()}

    if args.bench:
        print(f"== speed (1 thread, {args.bench * args.bench_mult} games)")
        result["bench"] = bench(recs, args.bench, args.bench_reps, args.bench_mult)
        print(json.dumps(result["bench"], ensure_ascii=False, indent=1))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1, default=float)
    print("== SUMMARY")
    for key in ("recorded", "xseed"):
        if key in result:
            s = result[key]
            print(
                f"  {key}: games {s['games']}, steps {s['steps']}, state comparisons {s['state_cmp']} "
                f"(mismatches {s['state_mismatch']}), banks {s['reward_cmp']} ({s['reward_mismatch']}), "
                f"observe {s['obs_cmp']} ({s['obs_mismatch']}), telemetry {s['tel_cmp']} ({s['tel_mismatch']}), "
                f"flags ({s['flag_mismatch']}), final vs record ({s['final_vs_recorded_mismatch']})"
            )
            for e in s["examples"]:
                print("    example:", e)
    ok = (
        sc["amps_selfcheck_mismatches"] == 0
        and sc["price_mismatch_hd_vs_vendor"] == 0
        and sc["price_mismatch_hd_vs_python_env"] == 0
        and not sc["serializer_missing_fields"]
        and (args.no_replay or result["recorded"]["games"] > 0)
        and all(result[k]["examples"] == [] for k in ("recorded", "xseed") if k in result)
        and result.get("bench", {}).get("bench_final_rewards_equal", True)
    )
    print("RESULT:", "OK, 0 mismatches" if ok else "MISMATCHES FOUND")
    print("json:", args.out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
