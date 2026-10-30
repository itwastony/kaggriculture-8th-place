"""BC demonstrations from a code teacher (c3bc) playing a pool of code bots on kagsim (earlier lines r1/r2).

Not used by the final line, which clones recorded top-team games (rl/gen_bc_tapes.py). The teacher and the bots in OPPS
are earlier league bots that are not in this repository (agents/README.md).

One process = one shard: --shard K --games N --out DIR -> DIR/bc_<K>.pkl.gz.
Game: the teacher in a random seat against a random opponent (weighted); against itself both seats are recorded.
A shard holds a list of records {seed, seat, opp, reward: [me, opp], obs, act (raw actions, for relabelling), ulab,
mlab}. Observations are stored raw (kagsim dict); encoding happens in the training data loader (rl/data.py).

    python -m rl.gen_bc --shard 0 --games 20 --out runs/X/bc --seed0 70000000
"""

import argparse
import gzip
import os
import pickle
import random
import sys
import time

from . import act as A
from .record import play_record

TEACHER = "agents/c3bc/main.py"
OPPS = [  # (agent, weight)
    ("agents/c3bc/main.py", 3),
    ("agents/v4_look12/main.py", 2),
    ("agents/l12_combo3/main.py", 1),
    ("agents/bcr_tdp_v4/main.py", 2),
    ("agents/pub0923_tetsutani_dp/main.py", 2),
    ("agents/pub0923_arsg_herdsafe3/main.py", 2),
    ("agents/pub0923_koshinm_0921/main.py", 1),
    ("agents/pub0923_wzb_v15stack/main.py", 1),
    ("agents/pub0923_dmitrii_morewheat/main.py", 1),
    ("agents/pub0923_goodpjw_melon/main.py", 1),
    ("agents/pub0923_aurax7_v7/main.py", 1),
    ("agents/pub0923_jaxa_2802/main.py", 1),
    ("agents/pub0923_guru_v3/main.py", 1),
    ("agents/pub0923_thomas_2945/main.py", 1),
    ("agents/pub0923_statma_tdp/main.py", 1),
]


def one_game(seed, rng, teacher=TEACHER):
    """Play one recorded game; returns the records of the teacher's seat(s)."""
    opp = rng.choices([o for o, _ in OPPS], [w for _, w in OPPS])[0]
    seat = rng.randint(0, 1)
    a, b = (teacher, opp) if seat == 0 else (opp, teacher)
    both = opp == teacher
    rew, trajs = play_record(seed, a, b, seat=seat, record_both=True)
    out = []
    for s in (0, 1) if both else (seat,):
        tr = trajs[s]
        out.append(
            {
                "seed": seed,
                "seat": s,
                "opp": opp,
                "reward": [rew[s], rew[1 - s]],
                "obs": [o for o, _ in tr],
                "act": [a_ for _, a_ in tr],
                "ulab": A.unit_labels(tr),
                "mlab": [A.market_labels(o, a_) for o, a_ in tr],
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed0", type=int, default=70_000_000)
    x = ap.parse_args()
    os.makedirs(x.out, exist_ok=True)
    rng = random.Random(x.seed0 + x.shard)
    recs = []
    t0 = time.time()
    for g in range(x.games):
        seed = x.seed0 + x.shard * 1000 + g
        try:
            recs += one_game(seed, rng)
        except Exception as e:  # noqa: BLE001
            print(f"shard {x.shard} game {g} seed {seed}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    path = os.path.join(x.out, f"bc_{x.shard:05d}.pkl.gz")
    with gzip.open(path + ".tmp", "wb", compresslevel=3) as f:
        pickle.dump(recs, f, protocol=5)
    os.rename(path + ".tmp", path)
    print(
        f"shard {x.shard}: {len(recs)} traj, {sum(len(r['obs']) for r in recs)} steps, {time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
