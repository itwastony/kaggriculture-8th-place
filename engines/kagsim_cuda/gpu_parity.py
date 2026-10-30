#!/usr/bin/env python3
"""Comparison of the GPU build (build/libkagsim_cuda.so) with the CPU build of the same core (build/kaghd_cpu), plus a
GPU speed measurement. Run on a machine with a GPU after bash build_cuda.sh and bash build_cpu.sh.

The CPU build of sim_hd.hpp is checked against the original kagsim by parity.py, so GPU == CPU byte-for-byte at every
step => GPU == kagsim.

Checks:
  1. price amplitude literals == the formula on the host of the GPU library (kag_amps_selfcheck);
  2. market_price on the device == on the host for 9 items x inv in [-20000, 80000] (kag_price_check);
  3. recorded streams (--streams, default data/streams.pkl.gz, not shipped: create it first with
     PYTHONHASHSEED=0 python engines/kagsim_cuda/record_streams.py; --xseed k: k more runs on new seeds, without pinned
     shops): GPU and CPU in lockstep, at EVERY step: canonical serialization of the full State (byte-for-byte), banks,
     done;
  4. measurement: --bench-n games (streams cycled, own seeds), all steps of the streams, actions uploaded to the device
     in advance; the final banks of the first --bench-check games are compared with the CPU build (--bench-n 0 skips).

    python engines/kagsim_cuda/gpu_parity.py --streams engines/kagsim_cuda/data/streams.pkl.gz --xseed 2
    python engines/kagsim_cuda/gpu_parity.py --backend cpu ...   # dry run of the script logic without a GPU

Writes --out (default engines/kagsim_cuda/data/gpu_parity_result.json); exit code 1 on any mismatch.
"""

import argparse
import gzip
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kagsim_cuda as KC  # noqa: E402

K = KC._cpu_module()
A = K.ACT_I32


def encode_block(recs):
    """int32 (T, games, 2, 170) action block of recs; shorter streams are padded with empty actions."""
    T = max(len(r["acts"][0]) for r in recs)
    empty = K.encode_action({})
    acts_T = np.empty((T, len(recs), 2, A), np.int32)
    for g, r in enumerate(recs):
        for p in (0, 1):
            e = K.encode_actions(r["acts"][p])
            acts_T[: len(e), g, p] = e
            acts_T[len(e) :, g, p] = empty
    return acts_T


def make_dev(backend, n, lib):
    """The batch under test: GpuBatch, or CpuBatch for --backend cpu."""
    if backend == "gpu":
        return KC.GpuBatch(n, 720, 0, lib)
    return KC.CpuBatch(n, 720)


def lockstep(recs, seeds, shops, backend, lib, tag, res):
    """Step the tested batch and the CPU batch together over recs, comparing state, banks and done after every step."""
    n = len(recs)
    acts_T = encode_block(recs)
    sd = np.array(seeds, np.uint64)
    sh = np.zeros((n, 8), np.uint8)
    ns = np.zeros(n, np.int32)
    for g, s in enumerate(shops):
        if s is not None:
            sh[g, : len(s)] = s
            ns[g] = len(s)
    dev = make_dev(backend, n, lib)
    cpu = KC.CpuBatch(n, 720)
    dev.reset(sd, None, sh, ns)
    cpu.reset(sd, None, sh, ns)

    def check(t):
        a, b = dev.serialize(), cpu.serialize()
        bad = np.nonzero((a != b).any(axis=1))[0]
        res["state_cmp"] += n
        res["state_mismatch"] += len(bad)
        for g in bad[:3]:
            if len(res["examples"]) < 20:
                res["examples"].append(f"[{tag}] state g={g} t={t} byte {int(np.nonzero(a[g] != b[g])[0][0])}")
        ra, rb = dev.rewards(), cpu.rewards()
        res["reward_cmp"] += 2 * n
        res["reward_mismatch"] += int((ra != rb).any(axis=1).sum())
        res["flag_mismatch"] += int((dev.done() != cpu.done()).sum())

    check(0)
    for t in range(acts_T.shape[0]):
        dev.step(acts_T[t])
        cpu.step(acts_T[t])
        res["steps"] += n
        check(t + 1)
    res["games"] += n
    dev.close()


def bench(recs, backend, lib, n, check_n):
    """Run n games over uploaded actions; returns timings and the final-bank check of the first check_n games."""
    m = len(recs)
    acts_T = encode_block(recs)
    seeds = np.array([(int(recs[i % m]["seed"]) + (i // m) * 7919) % (2**31) for i in range(n)], np.uint64)
    dev = make_dev(backend, n, lib)
    dev.reset(seeds)
    dev.upload(acts_T)
    sec = dev.run_uploaded(0, acts_T.shape[0])
    rew = dev.rewards()
    out = {
        "backend": backend,
        "games": n,
        "distinct_streams": m,
        "steps_per_game": int(acts_T.shape[0]),
        "sec": sec,
        "games_per_sec": n / sec,
        "game_steps_per_sec": n * acts_T.shape[0] / sec,
    }
    k = min(check_n, n)
    cpu = KC.CpuBatch(k, 720)
    cpu.reset(seeds[:k])
    cpu.upload(acts_T)
    cpu.run_uploaded()
    out["checked_games"] = k
    out["final_rewards_mismatch_vs_cpu"] = int((cpu.rewards() != rew[:k]).any(axis=1).sum())
    dev.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default=os.path.join(HERE, "data", "streams.pkl.gz"))
    ap.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--lib", default=None, help="path to libkagsim_cuda.so (default build/)")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--xseed", type=int, default=2)
    ap.add_argument("--bench-n", type=int, default=65536)
    ap.add_argument("--bench-check", type=int, default=2048)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "gpu_parity_result.json"))
    args = ap.parse_args()

    result = {"backend": args.backend}
    if args.backend == "gpu":
        lib = KC.load_lib(args.lib)
        result["amps_selfcheck_mismatches"] = lib.kag_amps_selfcheck()
        bad, fi, fv = KC.price_check(-20000, 80000, args.lib)
        result["device_price_check"] = {
            "range": [-20000, 80000],
            "points": 9 * 100001,
            "mismatches": bad,
            "first_item": fi,
            "first_inv": fv,
        }
        print("amps:", result["amps_selfcheck_mismatches"], "device prices:", result["device_price_check"], flush=True)

    with gzip.open(args.streams, "rb") as f:
        recs = pickle.load(f)
    res = dict(
        games=0, steps=0, state_cmp=0, state_mismatch=0, reward_cmp=0, reward_mismatch=0, flag_mismatch=0, examples=[]
    )
    t0 = time.time()
    for j in range(0, args.xseed + 1):
        for i in range(0, len(recs), args.chunk):
            part = recs[i : i + args.chunk]
            if j == 0:
                seeds, shops = [r["seed"] for r in part], [r["shops"] for r in part]
            else:
                seeds = [(r["seed"] * 7 + j * 1000003 + 12345) % (2**31) for r in part]
                shops = [None] * len(part)
            lockstep(part, seeds, shops, args.backend, args.lib, f"{'rec' if j == 0 else f'x{j}'}", res)
        print(
            f"  pass {j}: games {res['games']}, state mismatches {res['state_mismatch']}, {time.time() - t0:.0f} s",
            flush=True,
        )
    result["lockstep"] = res
    if args.bench_n:
        result["bench"] = bench(recs, args.backend, args.lib, args.bench_n, args.bench_check)
        print(json.dumps(result["bench"], ensure_ascii=False), flush=True)

    ok = (
        res["state_mismatch"] == 0
        and res["reward_mismatch"] == 0
        and res["flag_mismatch"] == 0
        and result.get("amps_selfcheck_mismatches", 0) == 0
        and result.get("device_price_check", {}).get("mismatches", 0) == 0
        and result.get("bench", {}).get("final_rewards_mismatch_vs_cpu", 0) == 0
    )
    result["ok"] = ok
    with open(args.out, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(
        f"lockstep: games {res['games']}, steps (game x step) {res['steps']}, state comparisons {res['state_cmp']} "
        f"(mismatches {res['state_mismatch']}), banks {res['reward_cmp']} ({res['reward_mismatch']}), "
        f"flags ({res['flag_mismatch']})"
    )
    for e in res["examples"]:
        print("  example:", e)
    print("RESULT:", "OK, 0 mismatches" if ok else "MISMATCHES FOUND", "| json:", args.out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
