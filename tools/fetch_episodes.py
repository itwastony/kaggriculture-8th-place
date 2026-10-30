#!/usr/bin/env python3
"""Download Kaggle episode replays and store them as compact tapes (input of `harness/tape_worlds.py compact`).

A compact tape is out/ep<ID>.json.gz = {ep, seed, names, rewards, statuses, actions: [seat-0 actions, seat-1 actions],
team_src, sub_src}; the ~30 MB replay JSON is deleted after conversion and existing tapes are not downloaded again.
Only reads the Kaggle API (needs a Kaggle API token, see the Kaggle docs).

    # every episode referenced by the shipped pool manifests (5,465 distinct episodes; Kaggle rate-limits replay
    # downloads, so this is slow and backs off on HTTP 429)
    python tools/fetch_episodes.py --pools data/pools/*.json --out replays/compact
    # a few episodes
    python tools/fetch_episodes.py --eps 114046305 114046318 --out replays/compact

Then rebuild the worlds and compare them with the shipped ones:
    python harness/tape_worlds.py compact --src replays/compact --pool rebuilt --lb configs/empty_lb.csv --jobs 8
    python tools/compare_worlds.py --rebuilt replays/worlds/rebuilt --shipped data/worlds

Daily episode datasets (`kaggle/kaggriculture-episodes-YYYY-MM-DD`, full replays in one zip) are an alternative source
for the Sep 27-29 pools: `harness/tape_worlds.py zip --src <zip> ...`.
"""

import argparse
import ast
import gzip
import json
import os
import time
from pathlib import Path


def get_api(tries=30):
    """One authenticated KaggleApi per process (authentication is retried: it occasionally fails transiently)."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    for i in range(tries):
        try:
            api.authenticate()
            return api
        except SystemExit:
            time.sleep(5 + 5 * i)
    raise RuntimeError("Kaggle: authentication failed")


def compact(d, ep):
    """Full replay (dict) -> compact record: seed, team names, rewards, statuses, both seats' actions per step."""
    info = d["info"] if isinstance(d["info"], dict) else ast.literal_eval(d["info"])
    acts = [[], []]
    for step in d["steps"][1:]:
        for s in (0, 1):
            x = step[s].get("action") or {}
            acts[s].append(
                {"farmer": x.get("farmer", ["PASS"]), "hands": x.get("hands", []), "market": x.get("market", [])}
            )
    return {
        "ep": int(ep),
        "seed": int(info["seed"]),
        "names": info["TeamNames"],
        "rewards": d["rewards"] if isinstance(d["rewards"], list) else ast.literal_eval(d["rewards"]),
        "statuses": d["statuses"] if isinstance(d["statuses"], list) else ast.literal_eval(d["statuses"]),
        "team_src": "",
        "sub_src": "",
        "actions": acts,
    }


def fetch_one(api, ep, out, tmp):
    """Download one replay into tmp and write the compact tape out/ep<ep>.json.gz; True if a new tape was written.

    HTTP 429 is retried (up to 12 attempts; wait Retry-After + 30 s, else 30 s x attempt, at most 600 s); any other
    error skips the episode."""
    f = out / f"ep{ep}.json.gz"
    if f.exists():
        return False
    for i in range(12):
        try:
            api.competition_episode_replay(int(ep), path=str(tmp), quiet=True)
            break
        except Exception as ex:  # noqa: BLE001
            if "429" not in repr(ex) or i == 11:
                print("replay failed", ep, repr(ex)[:120], flush=True)
                return False
            ra = getattr(getattr(ex, "response", None), "headers", {}).get("retry-after")
            wait = int(ra) + 30 if ra and str(ra).isdigit() else min(600, 30 * (i + 1))
            print(f"HTTP 429 on {ep}: waiting {wait} s", flush=True)
            time.sleep(wait)
    rp = tmp / f"episode-{ep}-replay.json"
    if not rp.exists():
        return False
    try:
        rec = compact(json.load(open(rp)), ep)
        with gzip.open(f, "wt") as fh:
            json.dump(rec, fh)
        return True
    finally:
        rp.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="*", default=[], help="pool manifests; all their episode ids are fetched")
    ap.add_argument("--eps", nargs="*", type=int, default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--sleep", type=float, default=3.0, help="pause between downloads, s")
    ap.add_argument("--shard", default="0/1", help="i/n: take every n-th episode starting at i (parallel downloads)")
    a = ap.parse_args()
    eps = set(a.eps)
    for p in a.pools:
        eps |= {int(w["ep"]) for w in json.load(open(p))["worlds"]}
    si, sn = (int(v) for v in a.shard.split("/"))
    eps = [e for i, e in enumerate(sorted(eps)) if i % sn == si]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"_dl{si}"
    tmp.mkdir(exist_ok=True)
    api = get_api()
    got = 0
    for i, ep in enumerate(eps):
        if fetch_one(api, ep, out, tmp):
            got += 1
            time.sleep(a.sleep)
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(eps)} processed, {got} new", flush=True)
    os.rmdir(tmp) if not any(tmp.iterdir()) else None
    print(f"done: {got} new tapes in {out} ({len(eps)} episodes requested)")


if __name__ == "__main__":
    main()
