#!/usr/bin/env python3
"""Rebuild the tape worlds referenced by the pool manifests (data/pools/*.json) from public Kaggle replays.

The worlds themselves (recorded actions of other teams) are not redistributed in this repository; the manifests list
every episode id, seed, shop order and seat that the final run used. For every world folder named in the manifests
(d0927, d0928, d0929, d0930b, topvtop, live*, weaktrain) this script

1. downloads the episodes as compact tapes (tools/fetch_episodes.py, read-only Kaggle API, needs a Kaggle API token);
2. replays both seats in kagsim and keeps a world only if both final banks match the record to the dollar
   (harness/tape_worlds.py compact);
3. copies the accepted worlds to data/worlds/<folder>/ep<ID>.json.gz, where the manifests expect them.

    python tools/rebuild_worlds.py --list                 # folders and episode counts, no downloads
    python tools/rebuild_worlds.py --folders d0930b       # one folder
    python tools/rebuild_worlds.py                        # everything (5,539 world files of 5,465 distinct episodes;
                                                          # Kaggle rate-limits replay downloads, so this is slow)

Then `python entry_points/prepare_data.py --repro 50` re-checks a sample on the training path.
For the Sep 27-29 folders the Kaggle daily episode datasets are an alternative source:
`harness/tape_worlds.py zip --src kaggriculture-episodes-2026-09-27.zip --pool d0927 --lb configs/empty_lb.csv`.

Intermediate files: replays/compact/<folder>/ (downloaded tapes; already present episodes are not downloaded again) and
replays/worlds/<folder>/ + <folder>.index.jsonl (built worlds). The build step is skipped when the index lists every
downloaded tape; otherwise (an interrupted run, tapes downloaded later) the folder is rebuilt and the previous index is
kept as <folder>.index.jsonl.old.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def folders_from_manifests(pattern):
    """{world folder: set of episode ids} over all manifests matching pattern (relative to the repository root)."""
    eps = defaultdict(set)
    for f in sorted(glob.glob(str(ROOT / pattern))):
        for r in json.load(open(f))["worlds"]:
            p = Path(r["world"])
            eps[p.parent.name].add(int(p.name[2:].split(".")[0]))
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="data/pools/*.json")
    ap.add_argument("--folders", nargs="*", default=None, help="only these world folders")
    ap.add_argument("--list", action="store_true", help="print folders and episode counts, then exit")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--sleep", type=float, default=3.0, help="pause between downloads, s")
    x = ap.parse_args()

    eps = folders_from_manifests(x.pools)
    todo = [f for f in sorted(eps) if x.folders is None or f in x.folders]
    for f in todo:
        print(f"{f}: {len(eps[f])} episodes")
    if x.list:
        print(f"total: {sum(len(eps[f]) for f in todo)} episodes")
        return

    py = sys.executable
    for f in todo:
        compact = ROOT / "replays" / "compact" / f
        built = ROOT / "replays" / "worlds" / f
        dest = ROOT / "data" / "worlds" / f
        have = {int(p.name[2:].split(".")[0]) for p in compact.glob("ep*.json.gz")}
        missing = sorted(eps[f] - have)
        if missing:
            subprocess.run(
                [
                    py,
                    str(ROOT / "tools/fetch_episodes.py"),
                    "--out",
                    str(compact),
                    "--sleep",
                    str(x.sleep),
                    "--eps",
                    *map(str, missing),
                ],
                check=True,
            )
        idx = ROOT / "replays" / "worlds" / f"{f}.index.jsonl"
        tapes = {int(p.name[2:].split(".")[0]) for p in compact.glob("ep*.json.gz")}
        indexed = {int(json.loads(line)["ep"]) for line in open(idx) if line.strip()} if idx.exists() else set()
        if not idx.exists() or tapes - indexed:
            if idx.exists():  # tapes downloaded after the last build: rebuild; tape_worlds.py writes a new index
                os.replace(idx, idx.with_name(idx.name + ".old"))
            subprocess.run(
                [
                    py,
                    str(ROOT / "harness/tape_worlds.py"),
                    "compact",
                    "--src",
                    str(compact),
                    "--pool",
                    f,
                    "--lb",
                    str(ROOT / "configs/empty_lb.csv"),
                    "--jobs",
                    str(x.jobs),
                ],
                check=True,
            )
        index = [json.loads(line) for line in open(idx)]
        ok = {int(r["ep"]) for r in index if r["repro"]}
        dest.mkdir(parents=True, exist_ok=True)
        copied = 0
        for e in sorted(eps[f] & ok):
            shutil.copy2(built / f"ep{e}.json.gz", dest / f"ep{e}.json.gz")
            copied += 1
        lost = sorted(eps[f] - ok)
        print(
            json.dumps(
                {
                    "folder": f,
                    "needed": len(eps[f]),
                    "copied": copied,
                    "not_reproduced_or_missing": len(lost),
                    "first_missing": lost[:10],
                }
            )
        )


if __name__ == "__main__":
    main()
