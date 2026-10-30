"""Replay check of the tape-game path of rl/appo.py: banks must match the record to the dollar.

For every pool entry: kagcore.GameBatch([record seed], 720, record shops); both seats play the record's actions (the
action dict overrides the policy's choice), and the final banks are compared with the record.

    python -m rl.tape_check data/pools/pool_d0930_train.json [N]      # the first N entries (default: all)

entry_points/prepare_data.py --repro runs it for every complete pool. Needs the kagcore module
(scripts/build_engines.sh).
"""

import gzip
import json
import sys

import numpy as np

PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def main():
    import kagcore

    pool = json.load(open(sys.argv[1]))["worlds"]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else len(pool)
    ok, bad = 0, []
    for w in pool[:n]:
        d = json.load(gzip.open(w["world"], "rt"))
        gb = kagcore.GameBatch([int(w["seed"])], 720, w["shops"])
        uc = np.zeros((1, 2, 16), np.int64)
        mc = np.zeros((1, 2, kagcore.NHEADS), np.int64)
        for t in range(720):
            a = [
                d["actions"][s][t] if t < len(d["actions"][s]) and d["actions"][s][t] is not None else PASS
                for s in (0, 1)
            ]
            gb.step(uc, mc, [(a[0], a[1])])
        r = gb.rewards().tolist()[0]
        if r == [float(v) for v in d["rewards"]]:
            ok += 1
        else:
            bad.append((w["ep"], r, d["rewards"]))
    print(
        f"tape vs tape via kagcore with the record's shops: {ok}/{min(n, len(pool))} to the dollar; "
        f"mismatches {bad[:5]}"
    )


if __name__ == "__main__":
    main()
