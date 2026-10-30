"""Pool of tape worlds for rl/appo.py (league entry "tapes:FILE"), built from a world index.

    python -m rl.tape_pool --index DATA/replays/worlds/X.index.jsonl --worlds DATA/replays/worlds/X \\
        --exclude holdout.json --out pool.json

From every world with repro=True the opponent's seat (not "Anton Tikhonov") is taken; worlds where the opponent is a
team from --exclude ("hold_teams") or the episode is in "hold_eps" are dropped (the held-out weak field stays clean).
--all-seats: other teams' episodes (without "Anton Tikhonov") give both seats; a seat is skipped if its team is in
hold_teams or its leaderboard rank is <= --skip-rank (top tapes break against a strong agent, measured 2026-09-24).
Entry: {world, opp_seat, seed, shops, team, lb_score, ep, sub_src}. The output file must not exist yet. At the end, a
summary of the opponents' ratings is printed. The published pools are in data/pools/ (data/README.md).
"""

import argparse
import gzip
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--worlds", required=True)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--all-seats", action="store_true")
    ap.add_argument("--skip-rank", type=int, default=50)
    x = ap.parse_args()
    ex = json.load(open(x.exclude)) if x.exclude else {}
    hold_t, hold_e = set(ex.get("hold_teams", [])), set(ex.get("hold_eps", []))
    rows = [json.loads(line) for line in open(x.index) if line.strip()]
    out, drop = [], {"repro": 0, "not_ours": 0, "hold": 0}
    for r in rows:
        if not r.get("repro"):
            drop["repro"] += 1
            continue
        if "Anton Tikhonov" in r["names"]:
            seats = [1 - r["names"].index("Anton Tikhonov")]
        elif x.all_seats:
            seats = [0, 1]
        else:
            drop["not_ours"] += 1
            continue
        if r["ep"] in hold_e:
            drop["hold"] += 1
            continue
        wf = os.path.join(x.worlds, f"ep{r['ep']}.json.gz")
        w = json.load(gzip.open(wf, "rt"))
        for o in seats:
            lb = (w.get("lb") or [None, None])[o]
            if r["names"][o] in hold_t:
                drop["hold"] += 1
                continue
            if x.all_seats and lb and lb.get("rank") is not None and lb["rank"] <= x.skip_rank:
                drop["top"] = drop.get("top", 0) + 1
                continue
            out.append(
                {
                    "world": os.path.abspath(wf),
                    "opp_seat": o,
                    "seed": int(w["seed"]),
                    "shops": w["shops"],
                    "team": r["names"][o],
                    "lb_score": (lb or {}).get("score"),
                    "ep": r["ep"],
                    "sub_src": w.get("sub_src"),
                }
            )
    json.dump({"worlds": out}, open(x.out, "x"), indent=0)
    sc = sorted(s["lb_score"] for s in out if s["lb_score"] is not None)
    q = (lambda p: sc[min(len(sc) - 1, int(p * len(sc)))]) if sc else (lambda p: None)
    print(
        f"worlds {len(rows)}, seats in pool {len(out)} (teams {len({s['team'] for s in out})}), dropped {drop}; "
        f"opponent ratings: without rating {len(out) - len(sc)}, quantiles 10/50/90% {q(0.1)}/{q(0.5)}/{q(0.9)}"
    )


if __name__ == "__main__":
    main()
