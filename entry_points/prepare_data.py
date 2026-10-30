#!/usr/bin/env python3
"""Step 1 - data. Checks that the tape worlds referenced by the pool manifests exist, and replays them.

    python entry_points/prepare_data.py                 # which world files of each pool are present
    python entry_points/prepare_data.py --repro 50      # + replay the first 50 seats of each complete pool through
                                                        #   kagcore (the exact training path); banks must match the
                                                        #   record to the dollar
    python entry_points/prepare_data.py --repro all     # all 9,954 pool seats (slow)

The worlds are not in the repository; tools/rebuild_worlds.py downloads the public replays and rebuilds them
(data/README.md). Rebuilt worlds are checked by replay, not by hash: their leaderboard field and gzip headers differ
from our copies.
"""

import argparse
import json
import os
import re
import subprocess
import sys

from common import ROOT, settings


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repro", default="0", help="number of seats per pool to replay, or 'all'")
    x = ap.parse_args()
    data = settings()["DATA_DIR"]
    pools = sorted(f for f in os.listdir(os.path.join(data, "pools")) if f.endswith(".json"))
    complete = []
    print(f"{'pool':28s} {'seats':>6s} {'worlds':>6s} {'missing':>7s}")
    for p in pools:
        ws = json.load(open(os.path.join(data, "pools", p)))["worlds"]
        files = {w["world"] for w in ws}
        miss = [f for f in files if not os.path.exists(f if os.path.isabs(f) else os.path.join(ROOT, f))]
        print(f"{p:28s} {len(ws):6d} {len(files):6d} {len(miss):7d}")
        if not miss:
            complete.append(p)
    if len(complete) < len(pools):
        print(
            "some world files are missing: run tools/rebuild_worlds.py (data/README.md); training needs "
            "entry_points/train_appo.py --league-available-only until they are all present"
        )
    if x.repro != "0":
        for p in complete:
            cmd = [sys.executable, "-m", "rl.tape_check", os.path.join(data, "pools", p)]
            if x.repro != "all":
                cmd.append(str(int(x.repro)))
            print("+", " ".join(cmd), flush=True)
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            print(r.stdout.strip() or r.stderr.strip()[-2000:], flush=True)
            m = re.search(r"(\d+)/(\d+) to the dollar", r.stdout)
            if r.returncode != 0 or not m or m.group(1) != m.group(2):
                sys.exit("tape_check failed (is kagcore built? see scripts/build_engines.sh)")
        if not complete:
            sys.exit("no complete pool to replay")
    print("data OK" if len(complete) == len(pools) else f"{len(complete)} of {len(pools)} pools complete")


if __name__ == "__main__":
    main()
