#!/usr/bin/env python3
"""Build tape worlds (for harness/tape_play.py) from recorded episodes: replays/worlds/<pool>/ep<ID>.json.gz.

World = {ep, seed, shops, names, rewards, statuses, actions: [tape 0, tape 1], src, lb, repro} (+ team_src, sub_src
for compact tapes). Every world is checked: tape against tape on kagsim, without pinned shops, must give the recorded
rewards to the dollar (repro=True); otherwise the world is kept with repro=False. shops is the shop opening order of
that replay (8 openings, repeats allowed), i.e. the record's order when repro=True; lb holds the leaderboard entry of
each seat (or None) from the --lb CSV. An index <pool>.index.jsonl (one summary line per world) is written next to
the pool folder; it must not exist yet.

    # compact tapes (tools/fetch_episodes.py)
    python harness/tape_worlds.py compact --src replays/band_tapes replays/band_tapes2 --pool band --lb LB.csv
    # Kaggle daily dataset (zip with full replays)
    python harness/tape_worlds.py zip --src kaggriculture-episodes-2026-09-22.zip --pool top0922 --lb LB.csv

LB.csv is a Kaggle leaderboard export (Rank, Score, TeamName, TeamId, TeamMemberUserNames); configs/empty_lb.csv
(header only) gives lb = [None, None]. Requires kagsim (scripts/build_engines.sh).
"""

import argparse
import csv
import gzip
import json
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tape_play import replay_tapes

ROOT = Path(__file__).resolve().parent.parent


def load_lb(path):
    """Leaderboard CSV -> (by team name, by member user name) dicts of {rank, score, team, team_id}."""
    by_name, by_user = {}, {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rec = {"rank": int(r["Rank"]), "score": float(r["Score"]), "team": r["TeamName"], "team_id": r["TeamId"]}
            by_name[r["TeamName"]] = rec
            for u in (r.get("TeamMemberUserNames") or "").split(","):
                if u.strip():
                    by_user.setdefault(u.strip(), rec)
    return by_name, by_user


def lb_of(name, lb):
    """Leaderboard entry of a seat name (a team name, else a member user name), or None."""
    by_name, by_user = lb
    return by_name.get(name) or by_user.get(name)


def finish(w, pool, lb):
    """Replay check (sets shops and repro), leaderboard lookup, write the world file; returns its index line."""
    r, shops = replay_tapes(w["seed"], w["actions"])
    w["shops"] = shops
    w["repro"] = [float(x) for x in r] == [float(x) for x in w["rewards"]]
    w["lb"] = [lb_of(n, lb) for n in w["names"]]
    out = ROOT / "replays" / "worlds" / pool / f"ep{w['ep']}.json.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as f:
        json.dump(w, f, separators=(",", ":"))
    return {
        "ep": w["ep"],
        "repro": w["repro"],
        "names": w["names"],
        "lb": [x and x["rank"] for x in w["lb"]],
        "n_shops": len(shops),
    }


def from_compact(args):
    """Worker: one compact tape (path, pool, leaderboard CSV) -> index line."""
    path, pool, lbp = args
    d = json.load(gzip.open(path))
    w = {k: d[k] for k in ("ep", "seed", "names", "rewards", "statuses", "actions")}
    w["src"] = str(Path(path).relative_to(ROOT))
    w["team_src"], w["sub_src"] = d.get("team_src"), d.get("sub_src")
    return finish(w, pool, load_lb(lbp))


def from_zip(args):
    """Worker: one full replay inside a zip (zip path, member, pool, leaderboard CSV) -> index line.

    Also records whether the replay's final unlocked_shops equal the replayed shop order (shops_match_record)."""
    zpath, name, pool, lbp = args
    with zipfile.ZipFile(zpath) as z:
        d = json.loads(z.read(name))
    st = d["steps"]
    info = d["info"]
    w = {
        "ep": info["EpisodeId"],
        "seed": info["seed"],
        "names": info["TeamNames"],
        "rewards": [s.get("reward") for s in st[-1]],
        "statuses": [s.get("status") for s in st[-1]],
        "actions": [[st[t + 1][i]["action"] for t in range(len(st) - 1)] for i in (0, 1)],
        "src": f"{os.path.relpath(zpath, ROOT)}:{name}",
    }
    rec = finish(w, pool, load_lb(lbp))
    rec_shops = (st[-1][0]["observation"].get("town") or {}).get("unlocked_shops")
    rec["shops_match_record"] = (
        rec_shops is None
        or list(rec_shops)
        == json.load(gzip.open(ROOT / "replays" / "worlds" / pool / f"ep{w['ep']}.json.gz", "rt"))["shops"]
    )
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["compact", "zip"])
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--lb", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    x = ap.parse_args()
    lbp = os.path.abspath(x.lb)
    if x.mode == "compact":
        seen, jobs = set(), []
        for s in x.src:
            for p in sorted(Path(s).glob("ep*.json.gz")):
                if p.name not in seen:
                    seen.add(p.name)
                    jobs.append((str(p.resolve()), x.pool, lbp))
        fn = from_compact
    else:
        jobs = []
        for s in x.src:
            with zipfile.ZipFile(s) as z:
                jobs += [(os.path.abspath(s), n, x.pool, lbp) for n in z.namelist() if n.endswith(".json")]
        fn = from_zip
    index = ROOT / "replays" / "worlds" / f"{x.pool}.index.jsonl"
    rows = []
    with ProcessPoolExecutor(x.jobs) as ex:
        for r in ex.map(fn, jobs, chunksize=4):
            rows.append(r)
            if len(rows) % 100 == 0:
                print(len(rows), "/", len(jobs), flush=True)
    with open(index, "x") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    bad = [r["ep"] for r in rows if not r["repro"]]
    print(
        json.dumps(
            {
                "pool": x.pool,
                "worlds": len(rows),
                "repro_ok": len(rows) - len(bad),
                "bad": bad[:20],
                "lb_mapped_seats": sum(v is not None for r in rows for v in r["lb"]),
                "shops_mismatch": sum(1 for r in rows if r.get("shops_match_record") is False),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
