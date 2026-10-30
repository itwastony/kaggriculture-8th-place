#!/usr/bin/env python3
"""Compare tape worlds rebuilt from Kaggle replays with the shipped ones (data/worlds/<pool>/ep<ID>.json.gz).

Compared fields: seed, team names, recorded rewards, shop schedule, repro flag, and both seats' actions. Actions are
normalised first ({"farmer", "hands", "market"} with defaults), because worlds built from the daily datasets keep the
raw action dicts while worlds built from compact tapes keep normalised ones. Leaderboard fields (lb) are not compared:
they depend on the leaderboard snapshot passed to harness/tape_worlds.py.

    python tools/compare_worlds.py --rebuilt replays/worlds/rebuilt --shipped data/worlds

--rebuilt is one folder of ep<ID>.json.gz files; --shipped is searched one level down (<pool>/ep<ID>.json.gz), and
worlds are matched by file name. Prints a JSON summary; exit code 1 if any matched world differs (rebuilt worlds that
are not shipped are only counted).
"""

import argparse
import gzip
import json
from pathlib import Path


def norm(a):
    """Action dict with the default PASS farmer, empty hands and empty market filled in."""
    a = a or {}
    return {"farmer": a.get("farmer") or ["PASS"], "hands": a.get("hands") or [], "market": a.get("market") or []}


def key(w):
    """The compared fields of a world, with types and actions normalised."""
    return {
        "seed": int(w["seed"]),
        "names": list(w["names"]),
        "rewards": [float(r) for r in w["rewards"]],
        "shops": list(w["shops"]),
        "repro": bool(w["repro"]),
        "actions": [[norm(a) for a in w["actions"][s]] for s in (0, 1)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuilt", required=True)
    ap.add_argument("--shipped", required=True)
    x = ap.parse_args()
    shipped = {p.name: p for p in Path(x.shipped).glob("*/ep*.json.gz")}
    ok, diff, missing = 0, [], []
    for p in sorted(Path(x.rebuilt).glob("ep*.json.gz")):
        s = shipped.get(p.name)
        if s is None:
            missing.append(p.name)
            continue
        a, b = key(json.load(gzip.open(p, "rt"))), key(json.load(gzip.open(s, "rt")))
        if a == b:
            ok += 1
        else:
            diff.append((p.name, [k for k in a if a[k] != b[k]]))
    print(json.dumps({"identical": ok, "different": len(diff), "not_in_shipped": len(missing), "examples": diff[:10]}))
    raise SystemExit(1 if diff else 0)


if __name__ == "__main__":
    main()
