"""BC demonstrations from recorded games (tape worlds, harness/tape_worlds.py).

Each world is replayed tape against tape on kagsim with the record's seed and shop order; a world whose final banks do
not match the record is skipped. The (observation, action) pairs of the selected seats are labelled with rl/act.py
(hindsight unit labels, market bins) and written in the format of rl/gen_bc.py:
{seed, seat, opp, reward: [me, opp], obs, act, ulab, mlab} + team, ep.

Seats: teams with leaderboard rank <= --max-rank (the world's "lb" field) and/or named in --teams / --teams-file;
--min-quads keeps only seats with at least that many land quadrants at turn --quads-step; --exclude-names skips every
world in which one of these teams plays. One process = one shard: the worlds whose index in the sorted file list
modulo --nshards equals --shard. Output: OUT/bc_<shard>.pkl.gz.

    python -m rl.gen_bc_tapes --worlds 'data/worlds/d0930b/*.json.gz' --teams-file configs/bc_teams_d7.txt \\
        --min-quads 4 --quads-step 300 --exclude-names 'Anton Tikhonov' --shard 0 --nshards 224 --out runs/d7/bcgen/bc
entry_points/make_bc_data.py runs every shard with configs/bcgen_d7.json.
"""

import argparse
import glob
import gzip
import json
import os
import pickle
import sys
import time

from . import act as A

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def replay(w, seats):
    import kagsim

    g = kagsim.Game(int(w["seed"]), 720, w["shops"])
    tr = {s: [] for s in seats}
    t = 0
    while not g.done:
        acts = [
            w["actions"][s][t] if t < len(w["actions"][s]) and w["actions"][s][t] is not None else PASS for s in (0, 1)
        ]
        for s in seats:
            tr[s].append((g.observe(s), acts[s]))
        g.step(acts[0], acts[1])
        t += 1
    return [g.reward(0), g.reward(1)], tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", nargs="+", required=True)
    ap.add_argument("--max-rank", type=int, default=0)
    ap.add_argument("--teams", default="")
    ap.add_argument("--teams-file", default="")  # teams, one per line (names may contain spaces/commas)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--out", required=True)
    # worlds where one of these teams (comma-separated) plays are skipped
    ap.add_argument("--exclude-names", default="")
    ap.add_argument("--worlds-file", default="")  # additional list of worlds (one path per line)
    # a seat is taken only if at turn --quads-step it has >= this many land quadrants
    ap.add_argument("--min-quads", type=int, default=0)
    ap.add_argument("--quads-step", type=int, default=300)
    x = ap.parse_args()
    os.makedirs(x.out, exist_ok=True)
    teams = {t for t in x.teams.split(",") if t}
    if x.teams_file:
        teams |= {line.strip() for line in open(x.teams_file, encoding="utf-8") if line.strip()}
    files = sorted(f for p in x.worlds for f in glob.glob(p))
    if x.worlds_file:
        files = sorted(set(files) | {line.strip() for line in open(x.worlds_file) if line.strip()})
    files = [f for i, f in enumerate(files) if i % x.nshards == x.shard]
    excl = {t for t in x.exclude_names.split(",") if t}
    recs, t0, skip = [], time.time(), {"repro": 0, "noseat": 0, "err": 0, "excl": 0, "quads": 0}
    for f in files:
        try:
            w = json.load(gzip.open(f, "rt"))
            if excl & set(w["names"]):
                skip["excl"] += 1
                continue
            lb = w.get("lb") or [None, None]
            seats = [
                s
                for s in (0, 1)
                if (w["names"][s] in teams)
                or (x.max_rank and lb[s] and lb[s].get("rank") is not None and lb[s]["rank"] <= x.max_rank)
            ]
            if not seats:
                skip["noseat"] += 1
                continue
            rew, tr = replay(w, seats)
            if [round(v) for v in rew] != [round(v) for v in w["rewards"]]:
                skip["repro"] += 1
                continue
            for s in seats:
                traj = tr[s]
                if (
                    x.min_quads
                    and len(traj[min(x.quads_step, len(traj) - 1)][0]["farms"][s].get("unlocked_quadrants") or [])
                    < x.min_quads
                ):
                    skip["quads"] += 1
                    continue
                recs.append(
                    {
                        "seed": int(w["seed"]),
                        "seat": s,
                        "opp": w["names"][1 - s],
                        "team": w["names"][s],
                        "ep": w.get("ep"),
                        "reward": [rew[s], rew[1 - s]],
                        "obs": [o for o, _ in traj],
                        "act": [a for _, a in traj],
                        "ulab": A.unit_labels(traj),
                        "mlab": [A.market_labels(o, a) for o, a in traj],
                    }
                )
        except Exception as e:  # noqa: BLE001
            skip["err"] += 1
            print(f"{f}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    path = os.path.join(x.out, f"bc_{x.shard:05d}.pkl.gz")
    with gzip.open(path + ".tmp", "wb", compresslevel=3) as fh:
        pickle.dump(recs, fh, protocol=5)
    os.rename(path + ".tmp", path)
    print(
        f"shard {x.shard}: worlds {len(files)}, trajectories {len(recs)}, skipped {skip}, {time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
