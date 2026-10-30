#!/usr/bin/env python3
"""Step 6 - inference: play the submission archive in the official engine (kaggle-environments 1.32.7), as Kaggle does.

In this competition "prediction" means playing games: Kaggle extracts the archive, imports main.py and calls
agent(observation, configuration) once per turn (719 calls per game; limit 1 s per turn plus a 60 s reserve per game).
This script does the same locally and reports the outcome, the final banks and the time per agent call.

    # shipped final archive vs "starter", 2 seeds x 2 seats
    python entry_points/predict.py
    python entry_points/predict.py --opp random pass starter --seeds 4
    python entry_points/predict.py --archive runs/d7/rl_d7b_110.tar.gz --opp agents/caretaker/main.py --save-replays

Opponents: the engine's built-in agents ("pass", "random", "starter") or a path to any main.py (e.g. a league bot in
agents/). The built-in "random" agent is not seeded by the engine, so games against it are not repeatable; every other
pairing is deterministic for a given seed. Our agent is called through a thin timing wrapper (the same function object
that Kaggle would call). Needs only numpy + kaggle-environments; no torch, no GPU.

Output: OUTPUT_DIR/<run>/predict/games.jsonl (one line per game), summary.json, and with --save-replays one
replay_<seed>_<seat>_<opp>.json per game (open it with kaggle_environments' renderer or any JSON viewer).
"""

import argparse
import importlib.util
import json
import os
import statistics
import sys
import tarfile
import tempfile
import time

# one BLAS thread, as in our timing (set before numpy is imported)
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

from common import ROOT, run_dir, settings  # noqa: E402


def load_agent(main_py):
    """Import an agent's main.py the way kaggle-environments does (its folder on sys.path) and return agent()."""
    d = os.path.dirname(os.path.abspath(main_py))
    sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(f"agent_{abs(hash(d))}", main_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="predict", help="output goes to OUTPUT_DIR/<run>/predict")
    ap.add_argument("--archive", default="", help="submission .tar.gz (default: SUBMISSION_DIR/rl_d7b_110.tar.gz)")
    ap.add_argument("--opp", nargs="*", default=["starter"], help="pass, random, starter or paths to main.py files")
    ap.add_argument("--seeds", type=int, default=2, help="seeds per opponent; each is played in both seats")
    ap.add_argument("--seed0", type=int, default=900100, help="first seed")
    ap.add_argument("--save-replays", action="store_true", help="also write one replay JSON per game")
    x = ap.parse_args()
    s = settings()
    from kaggle_environments import __version__ as kv
    from kaggle_environments import make

    if kv != "1.32.7":
        print(f"warning: kaggle-environments {kv}; the submissions were validated on 1.32.7", flush=True)
    archive = x.archive or os.path.join(s["SUBMISSION_DIR"], "rl_d7b_110.tar.gz")
    work = tempfile.mkdtemp(prefix="agent_")
    with tarfile.open(archive) as t:
        t.extractall(work)
    t0 = time.time()
    ours = load_agent(os.path.join(work, "main.py"))
    print(f"loaded {archive} in {time.time() - t0:.2f} s", flush=True)
    calls = []

    def timed(obs, cfg):
        a = time.perf_counter()
        out = ours(obs, cfg)
        calls.append(1000 * (time.perf_counter() - a))
        return out

    rd = run_dir(x.run)
    out = os.path.join(rd, "predict")
    os.makedirs(out, exist_ok=True)
    rows = []
    with open(os.path.join(out, "games.jsonl"), "w") as f:
        for opp in x.opp:
            opp_agent = (
                opp
                if opp in ("pass", "random", "starter")
                else load_agent(os.path.join(ROOT, opp) if not os.path.isabs(opp) else opp)
            )
            for k in range(x.seeds):
                seed = x.seed0 + k
                for seat in (0, 1):
                    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
                    n0 = len(calls)
                    env.run([timed, opp_agent] if seat == 0 else [opp_agent, timed])
                    last = env.steps[-1]
                    banks = [float(p.reward or 0) for p in last]
                    ms = calls[n0:]
                    r = {
                        "opp": opp,
                        "seed": seed,
                        "seat": seat,
                        "our_bank": banks[seat],
                        "opp_bank": banks[1 - seat],
                        "win": banks[seat] > banks[1 - seat],
                        "statuses": [p.status for p in last],
                        "calls": len(ms),
                        "ms_mean": round(statistics.mean(ms), 1) if ms else None,
                        "ms_max": round(max(ms), 1) if ms else None,
                    }
                    rows.append(r)
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    print(json.dumps(r), flush=True)
                    if x.save_replays:
                        tag = os.path.basename(os.path.dirname(opp)) if opp.endswith(".py") else opp
                        json.dump(env.toJSON(), open(os.path.join(out, f"replay_{seed}_{seat}_{tag}.json"), "w"))
    srt = sorted(calls)
    summ = {
        "archive": archive,
        "kaggle_environments": kv,
        "games": len(rows),
        "wins": sum(r["win"] for r in rows),
        "all_done": all(r["statuses"] == ["DONE", "DONE"] for r in rows),
        "our_bank_mean": round(statistics.mean(r["our_bank"] for r in rows)),
        "our_bank_min": round(min(r["our_bank"] for r in rows)),
        "agent_calls": len(srt),
        "ms_per_call_mean": round(statistics.mean(srt), 1),
        "ms_per_call_p99": round(srt[int(0.99 * (len(srt) - 1))], 1),
        "ms_per_call_max": round(srt[-1], 1),
    }
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
