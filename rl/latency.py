"""Per-turn latency of an exported NumPy agent (rl/export_agent.py) and agreement of its actions with the torch policy.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m rl.latency --agent runs/d7/export/my_agent/main.py \\
        --ckpt runs/d7/appo/ema_00110.pt --opp agents/caretaker/main.py --seeds 2 --seed0 97000000 --out lat.json

Games on kagsim (as in harness/kagsim_play.py); the agent plays both seats in turn. Turn time = the agent call only (the
observation is already built and structify'ed); loading the agent file (exec of main.py, which loads the weights) is
timed separately. Agreement (gate G2): the NumPy agent's action == the torch policy's action (rl/policy.Policy,
greedy) on the same observation. The default --opp is an earlier league bot that is not in this repository.
Output: --out JSON with the latency percentiles, the agreement count [same, compared] and the banks.
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "harness"))


def _cpu():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--opp", default="agents/c3bc/main.py")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=97_000_000)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--out", required=True)
    x = ap.parse_args()
    import kagsim
    from kagsim_play import _base_config, _load, fast_structify

    pol = None
    if x.ckpt:
        from rl.policy import Policy

        pol = Policy(x.ckpt, threads=1)
    apath, opath = os.path.abspath(x.agent), os.path.abspath(x.opp)
    times, loads, same, cmp_n, banks = [], [], 0, 0, []
    for i in range(x.seeds):
        seed = x.seed0 + i
        cfg = dict(_base_config())
        cfg["seed"] = seed
        for me in (0, 1):
            t0 = time.perf_counter()
            fa = _load(apath)
            loads.append(time.perf_counter() - t0)
            fo = _load(opath)
            game = kagsim.Game(seed, x.steps)
            while not game.done:
                acts = [None, None]
                for p in (0, 1):
                    raw = game.observe(p)
                    obs, conf = fast_structify(raw), fast_structify(cfg)
                    if p == me:
                        conf["__raw_path__"] = apath
                        t0 = time.perf_counter()
                        a = fa(obs, conf)
                        times.append(time.perf_counter() - t0)
                        if pol is not None:
                            cmp_n += 1
                            same += int(pol.act(raw) == a)
                    else:
                        conf["__raw_path__"] = opath
                        a = fo(obs, conf)
                    acts[p] = a
                game.step(acts[0] or {}, acts[1] or {})
            banks.append([seed, me, game.reward(me), game.reward(1 - me)])
            print(
                json.dumps(
                    {"seed": seed, "seat": me, "bank": banks[-1][2], "opp_bank": banks[-1][3], "calls": len(times)}
                ),
                flush=True,
            )
    t = np.array(times) * 1000
    res = {
        "agent": x.agent,
        "ckpt": x.ckpt,
        "opp": x.opp,
        "games": len(banks),
        "calls": int(len(t)),
        "ms_mean": round(float(t.mean()), 3),
        "ms_p50": round(float(np.percentile(t, 50)), 3),
        "ms_p99": round(float(np.percentile(t, 99)), 3),
        "ms_max": round(float(t.max()), 3),
        "load_s": [round(v, 3) for v in loads],
        "threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "cpu": _cpu(),
        "numpy": np.__version__,
        "torch_match": [same, cmp_n],
        "banks": banks,
    }
    json.dump(res, open(x.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "banks"}), flush=True)


if __name__ == "__main__":
    main()
