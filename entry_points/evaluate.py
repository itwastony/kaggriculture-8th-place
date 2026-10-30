#!/usr/bin/env python3
"""Head-to-head evaluation of two agents on kagsim, as in our checkpoint gates (docs/evaluation.md).

Agent A plays every opponent B on seeds SEED0 .. SEED0+N-1, in both seats (2N games per opponent). Each agent is a
submission archive (.tar.gz) or a path to a main.py; the agent is loaded and called the way kaggle-environments does it
(harness/kagsim_play.py). kagsim is bit-exact with kaggle-environments 1.32.7 and the agents are deterministic, so a
given (A, B, seeds) always gives the same games.

    # the reference number of the original evaluation: d7b_110 vs d6_250, 512 seeds from 98,300,000 -> P = 0.621
    python entry_points/evaluate.py --a submission/rl_d7b_110.tar.gz --b submission/rl_d6_250.tar.gz \\
        --seeds 512 --seed0 98300000
    # a retrained agent against the shipped one, a league bot and the empty farm
    python entry_points/evaluate.py --a runs/d7/my_d7b_110.tar.gz --b submission/rl_d7b_110.tar.gz \\
        agents/caretaker/main.py pass --seeds 64 --jobs 32

P = (wins + draws / 2) / games for A. SE is computed over seeds (the mean of the two seats of a seed is one sample):
the two seats of a seed usually give the same outcome, so a per-game SE would be too small. Bank and margin are A's
final bank and A's bank minus B's, averaged over games. "pass" means agents/pass_agent/main.py (an empty farm).

Output: OUTPUT_DIR/<run>/evaluate/games.jsonl (one line per game) and summary.json; the summary is also printed.
Games run in parallel processes (--jobs), each with one BLAS thread, as in the original evaluation (the script
restarts itself with PYTHONHASHSEED=0 and OMP/OPENBLAS/MKL_NUM_THREADS=1). Games between two networks are slow, so
use many cores (README.md gives the cost of the reference command). On Oct 12 the reference command plus the same 512
seeds against "pass" (2,048 games) were rerun on a clean copy of the code, and every game was identical to the Sep 30
evaluation.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import tarfile
import tempfile
from multiprocessing import Pool

from common import ROOT, run_dir

# PYTHONHASHSEED: some league bots iterate over sets of strings (docs/training.md). One BLAS thread per game process: as
# in the original evaluation, and it keeps the NumPy network's arithmetic the same on every run.
_ENV = {"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
if any(os.environ.get(k) != v for k, v in _ENV.items()):
    os.environ.update(_ENV)
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, os.path.join(ROOT, "harness"))


def resolve(spec, work):
    """Archive -> extracted main.py; 'pass' -> the empty-farm agent; otherwise a main.py path."""
    if spec == "pass":
        return os.path.join(ROOT, "agents", "pass_agent", "main.py")
    path = spec if os.path.isabs(spec) else os.path.join(ROOT, spec)
    if path.endswith((".tar.gz", ".tgz")):
        # one folder per archive path: two archives with the same file name must not overwrite each other
        tag = hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:8]
        d = os.path.join(work, os.path.basename(path).split(".")[0] + "_" + tag)
        os.makedirs(d, exist_ok=True)
        with tarfile.open(path) as t:
            t.extractall(d)
        path = os.path.join(d, "main.py")
    if not os.path.exists(path):
        sys.exit(f"agent not found: {spec}")
    return path


def play(job):
    """One game of job = (opponent key, seed, A's seat, A's main.py, B's main.py) on kagsim."""
    from kagsim_play import play_kagsim

    opp, seed, seat, a, b = job
    rewards, status = play_kagsim(seed, a, b) if seat == 0 else play_kagsim(seed, b, a)
    return {"opp": opp, "seed": seed, "a_seat": seat, "rewards": rewards, "status": status}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="the agent being evaluated (archive or main.py)")
    ap.add_argument("--b", nargs="+", required=True, help="opponents (archives, main.py paths or 'pass')")
    ap.add_argument("--seeds", type=int, default=64, help="number of seeds N; each is played in both seats")
    ap.add_argument("--seed0", type=int, default=98_300_000, help="first seed")
    ap.add_argument("--jobs", type=int, default=os.cpu_count(), help="parallel game processes (default: all CPUs)")
    ap.add_argument("--run", default="evaluate", help="output goes to OUTPUT_DIR/<run>/evaluate")
    x = ap.parse_args()
    work = tempfile.mkdtemp(prefix="eval_")
    a = resolve(x.a, work)
    opps = {b: resolve(b, work) for b in x.b}
    jobs = [(b, s, seat, a, p) for b, p in opps.items() for s in range(x.seed0, x.seed0 + x.seeds) for seat in (0, 1)]
    out = os.path.join(run_dir(x.run), "evaluate")
    os.makedirs(out, exist_ok=True)
    rows = []
    with Pool(x.jobs) as pool, open(os.path.join(out, "games.jsonl"), "w") as f:
        for i, r in enumerate(pool.imap_unordered(play, jobs), 1):
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            if i % max(1, len(jobs) // 20) == 0:
                print(f"{i}/{len(jobs)} games", flush=True)
    summ = {"a": x.a, "seed0": x.seed0, "seeds": x.seeds, "engine": "kagsim", "opponents": {}}
    for b in opps:
        per_seed, banks, margins, bad = {}, [], [], 0
        for r in rows:
            if r["opp"] != b:
                continue
            s = r["a_seat"]
            if r["status"] != ["DONE", "DONE"]:
                bad += 1
            me, op = r["rewards"][s], r["rewards"][1 - s]
            # an agent error (reward None) loses
            me, op = (me if me is not None else -1.0), (op if op is not None else -1.0)
            per_seed.setdefault(r["seed"], []).append(1.0 if me > op else 0.5 if me == op else 0.0)
            banks.append(me)
            margins.append(me - op)
        sv = [sum(v) / len(v) for v in per_seed.values()]
        se = statistics.stdev(sv) / math.sqrt(len(sv)) if len(sv) > 1 else float("nan")
        summ["opponents"][b] = {
            "games": len(banks),
            "P": round(sum(sv) / len(sv), 4),
            "SE_per_seed": round(se, 4),
            "bank_mean": round(statistics.mean(banks)),
            "margin_mean": round(statistics.mean(margins)),
            "not_done": bad,
        }
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
