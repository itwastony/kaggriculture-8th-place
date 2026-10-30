#!/usr/bin/env python3
"""Step 2 - behaviour-cloning demonstrations for the final line d7 (configs/bcgen_d7.json).

Replays every Sep 30 tape world on kagsim and, for the selected teacher seats, records (observation, hindsight label)
pairs. Deterministic: the same worlds and shard count give byte-identical output. 224 shards, as in the original run
(the shard of a world depends on the shard count, "nshards" in the config; keep 224 to reproduce the train/val split of
step 3, which holds out shards with index divisible by 50). Existing shard files are kept (delete them to regenerate);
when all shards are present the per-team trajectory counts are compared with the original run.

    python entry_points/make_bc_data.py --run d7 --jobs 32
    python entry_points/make_bc_data.py --run d7 --shards 0-3      # only some shards (e.g. one SLURM task per range)
"""

import argparse
import collections
import glob
import gzip
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor

from common import args_list, fill, load_config, run, run_dir, settings


def shard_cmd(cfg, i, out):
    """Command that generates shard i: python -m <module> <config args> --shard i --nshards N --out OUT."""
    return [
        sys.executable,
        "-m",
        cfg["module"],
        *args_list(cfg["args"]),
        "--shard",
        str(i),
        "--nshards",
        str(cfg["nshards"]),
        "--out",
        out,
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="d7", help="run name under OUTPUT_DIR")
    ap.add_argument("--jobs", type=int, default=os.cpu_count(), help="shards generated in parallel (default: all CPUs)")
    ap.add_argument("--shards", default="", help="a-b: only these shards (inclusive)")
    ap.add_argument("--dry", action="store_true", help="print the commands without running them")
    x = ap.parse_args()
    s = settings()
    rd = run_dir(x.run)
    cfg = fill(load_config("bcgen_d7.json"), {**s, "RUN": rd})
    out = cfg["out"]
    os.makedirs(out, exist_ok=True)
    lo, hi = (int(v) for v in x.shards.split("-")) if x.shards else (0, cfg["nshards"] - 1)
    env = {
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "LC_ALL": "C.UTF-8",
    }
    todo = [i for i in range(lo, hi + 1) if not os.path.exists(os.path.join(out, f"bc_{i:05d}.pkl.gz"))]
    print(f"{len(todo)} shards to generate into {out}", flush=True)
    with ThreadPoolExecutor(x.jobs) as ex:
        codes = list(ex.map(lambda i: run(shard_cmd(cfg, i, out), env=env, dry=x.dry), todo))
    if any(codes):
        sys.exit(f"{sum(1 for c in codes if c)} shards failed")
    files = sorted(glob.glob(os.path.join(out, "bc_*.pkl.gz")))
    if len(files) == cfg["nshards"] and not x.dry:
        teams = collections.Counter()
        for f in files:
            for r in pickle.load(gzip.open(f)):
                teams[r["team"]] += 1
        print("trajectories:", sum(teams.values()), dict(teams.most_common()))
        exp = cfg["expected"]
        same = sum(teams.values()) == exp["trajectories"] and dict(teams) == exp["per_team"]
        print("matches the original run (809 trajectories, same per-team counts):", same)


if __name__ == "__main__":
    main()
