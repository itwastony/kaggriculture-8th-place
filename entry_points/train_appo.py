#!/usr/bin/env python3
"""Step 4 - APPO league training of the final run d7b (configs/appo_d7b.json).

Each rank = 1 GPU + 26 CPU worker processes; seed 99990000. The number of ranks is set at launch (--nnodes, --nproc);
the configuration does not fix it. The submitted weights are OUT/ema_00110.pt, written after 110 iterations (0-109).
Asynchronous collection and multi-process game workers make the run non-deterministic: a rerun gives a statistically
similar, not identical, policy. See docs/training.md for what to compare.

    # several nodes, one command per node (e.g. under SLURM: scripts/slurm/appo_d7b.sbatch)
    python entry_points/train_appo.py --run d7 --nnodes N --nproc G --node-rank $SLURM_NODEID --rdzv HOST:29591 \\
        --no-pool-src
    # one node with G GPUs (same per-rank settings)
    python entry_points/train_appo.py --run d7 --nnodes 1 --nproc G --no-pool-src
    # installation smoke test on one GPU (2 iterations) with the bots and tape worlds that are present
    python entry_points/train_appo.py --run smoke --preset smoke --no-pool-src --league-available-only
    # override APPO arguments of the config (repeat --set, always in the = form)
    python entry_points/train_appo.py --run d7 --nnodes 1 --nproc G --no-pool-src --set=--gpu-games=64 \\
        --set=--minibatches=72

Inputs: the BC checkpoint (default: the one produced by entry_points/train_bc.py for the same --run if it exists,
otherwise the shipped artifacts/checkpoints/final/bc_d7_ckpt_003000.pt), five fixed past checkpoints (the BC
checkpoint and four in artifacts/checkpoints/past), the league (configs/league_d7.json: 101 code bots in agents/ + 6
tape-world pools in data/pools), and the GPU engine library (KAGRL_LIB). The original run also sampled past opponents
from the snapshot pools of three earlier lines (SNAPSHOT_POOL_DIR/{d1,d4,d6}); they are not published (ASSETS.md), so
pass --no-pool-src: past opponents are then the run's own snapshots plus the five fixed checkpoints (a different
opponent mix).

The public repository ships only two of the 101 league bots (the rest are public Kaggle notebooks and our own earlier
agents, see agents/README.md) and no tape worlds (tools/rebuild_worlds.py downloads them). The script stops if a league
entry is missing; --league-available-only trains with the bots that are present and the tape-pool seats whose world
files exist (a different opponent mix).

Output: OUTPUT_DIR/<run>/appo/ (rl/appo.py: ppo.jsonl, pool_state.jsonl, ckpt_*.pt, ema_*.pt, anchor_*.pt, pool/) and
OUTPUT_DIR/<run>/launch_<preset>.json with the exact command and environment (written by node rank 0).
"""

import argparse
import json
import os
import sys

from common import ROOT, args_list, fill, load_config, run, run_dir, settings


def entry_state(name):
    """'ok', 'partial' (a tape pool with some world files missing) or 'missing'; plus the present pool seats."""
    if name in ("self", "past", "best"):
        return "ok", None
    if not name.startswith("tapes:"):
        return ("ok" if os.path.exists(os.path.join(ROOT, name)) else "missing"), None
    pool = os.path.join(ROOT, name[6:])
    if not os.path.exists(pool):
        return "missing", None
    ws = json.load(open(pool))["worlds"]
    ok = [r for r in ws if os.path.exists(os.path.join(ROOT, r["world"]))]
    return ("ok" if len(ok) == len(ws) else "partial" if ok else "missing"), ok


def available_league(L, rd):
    """Drop missing entries; a partial tape pool is replaced by a copy with only the present seats.

    The copy is RUN/pools/<same file name>, so the opponent key in the logs stays the same."""
    out = []
    for n, w in L:
        st, ok = entry_state(n)
        if st == "missing":
            print(f"league: dropped {n}", flush=True)
            continue
        if st == "partial":
            src = os.path.join(ROOT, n[6:])
            dst = os.path.join(rd, "pools", os.path.basename(src))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            json.dump({**json.load(open(src)), "worlds": ok}, open(dst, "w"))
            print(f"league: {n[6:]}: {len(ok)} seats with world files -> {dst}", flush=True)
            n = "tapes:" + os.path.abspath(dst)
        out.append([n, w])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="d7", help="run name under OUTPUT_DIR")
    ap.add_argument(
        "--preset", default="d7b", choices=["d7b", "smoke"], help="smoke: configs/appo_smoke.json overrides"
    )
    ap.add_argument(
        "--bc-ckpt", default="", help="BC checkpoint (default: RUN/bc/ckpt_003000.pt, else the shipped one)"
    )
    ap.add_argument("--nnodes", type=int, default=None, help="number of nodes (required, except with --preset smoke)")
    ap.add_argument(
        "--nproc", type=int, default=None, help="GPUs per node = ranks per node (required, except with --preset smoke)"
    )
    ap.add_argument("--node-rank", type=int, default=0, help="rank of this node (multi-node)")
    ap.add_argument("--rdzv", default="", help="HOST:PORT of node 0 (multi-node)")
    ap.add_argument(
        "--no-pool-src",
        action="store_true",
        help="train without the snapshot pools of earlier lines; they are not published (ASSETS.md), so this is "
        "required unless you have them",
    )
    ap.add_argument(
        "--league-available-only",
        action="store_true",
        help="drop missing league bots and tape-pool seats without world files (changes the opponent mix)",
    )
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="--ARG=VALUE",
        help="override an argument of rl/appo.py; repeat for several, always in the = form: "
        "--set=--gpu-games=64 --set=--minibatches=72",
    )
    ap.add_argument("--dry", action="store_true", help="print the command without running it")
    x = ap.parse_args()
    s = settings()
    rd = run_dir(x.run)
    cfg = load_config("appo_d7b.json")
    if x.preset == "smoke":
        sm = load_config("appo_smoke.json")
        cfg["nnodes"], cfg["nproc_per_node"] = sm["nnodes"], sm["nproc_per_node"]
        cfg["args"].update(sm["args"])
    bc = x.bc_ckpt or os.path.join(rd, "bc", "ckpt_003000.pt")
    if not os.path.exists(bc):
        bc = os.path.join(s["CHECKPOINT_DIR"], "final", "bc_d7_ckpt_003000.pt")
    league = os.path.join(ROOT, "configs", "league_d7.json")
    if os.path.realpath(s["DATA_DIR"]) != os.path.realpath(os.path.join(ROOT, "data")):
        L = [[n.replace("tapes:data/", "tapes:" + s["DATA_DIR"] + "/", 1), w] for n, w in json.load(open(league))]
        league = os.path.join(rd, "league_d7.json")
        json.dump(L, open(league, "w"), indent=0)
    L = json.load(open(league))
    bad = [(n, st) for n, _ in L for st in [entry_state(n)[0]] if st != "ok"]
    if bad:
        if not x.league_available_only:
            sys.exit(
                f"{len(bad)} of {len(L)} league entries are missing or incomplete, e.g. {bad[:3]} "
                "(see agents/README.md and data/README.md, or pass --league-available-only)"
            )
        os.makedirs(rd, exist_ok=True)
        L = available_league(L, rd)
        league = os.path.join(rd, "league_available.json")
        json.dump(L, open(league, "w"), indent=0)
        print(f"league: {len(L)} entries -> {league}", flush=True)
    ctx = {**s, "RUN": rd, "BC_CKPT": os.path.abspath(bc), "LEAGUE": league}
    cfg = fill(cfg, ctx)
    a = dict(cfg["args"])
    if x.no_pool_src:
        a.pop("--pool-src")
    else:
        for spec in a["--pool-src"].split(","):
            d = spec.split(":")[0]
            if not os.path.isdir(d) or not any(f.endswith(".pt") for f in os.listdir(d)):
                sys.exit(f"snapshot pool {d} is missing (the pools are not published, ASSETS.md): pass --no-pool-src")
    for kv in x.set:
        k, v = kv.split("=", 1)
        a[k] = v
    for f in [bc, *a["--past"].split(","), a["--kagrl-lib"]]:
        if not os.path.exists(f):
            sys.exit(f"missing input: {f}")
    nnodes = x.nnodes or cfg.get("nnodes")
    nproc = x.nproc or cfg.get("nproc_per_node")
    if not nnodes or not nproc:
        sys.exit("pass --nnodes and --nproc (GPUs per node); the configuration does not fix the number of ranks")
    tr = [sys.executable, "-m", "torch.distributed.run", "--nnodes", str(nnodes), "--nproc_per_node", str(nproc)]
    if nnodes > 1:
        if not x.rdzv:
            sys.exit("--rdzv HOST:PORT is required with --nnodes > 1")
        tr += [
            "--rdzv_backend",
            "c10d",
            "--rdzv_endpoint",
            x.rdzv,
            "--rdzv_id",
            f"appo_{x.run}",
            "--node_rank",
            str(x.node_rank),
        ]
    else:
        tr += ["--standalone"]
    cmd = tr + ["-m", "rl.appo", "--out", cfg["out"], *args_list(a)]
    if x.node_rank == 0 and not x.dry:
        os.makedirs(cfg["out"], exist_ok=True)
        json.dump(
            {"cmd": cmd, "env": cfg["env"], "preset": x.preset, "nnodes": nnodes, "nproc": nproc},
            open(os.path.join(rd, f"launch_{x.preset}.json"), "w"),
            indent=1,
        )
    sys.exit(run(cmd, env=cfg["env"], dry=x.dry))


if __name__ == "__main__":
    main()
