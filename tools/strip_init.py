"""Turn an APPO/soup checkpoint into a BC starting point (rl/bc_train.py --init loads the state_dict strictly).

Keeps only the keys of the reference BC checkpoint --like (drops the value encoder venc.* and anything else the BC
network does not have); every reference key must be present in --src with the same shape. The output
{model, d, layers, step=0, src} is written to a new file (an existing --out is never overwritten).

    python tools/strip_init.py --src S.pt --like BC.pt --out OUT.pt
"""

import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--like", required=True)
    ap.add_argument("--out", required=True)
    x = ap.parse_args()
    s = torch.load(x.src, map_location="cpu", weights_only=False)
    ref = torch.load(x.like, map_location="cpu", weights_only=False)
    keys = set(ref["model"])
    miss = keys - set(s["model"])
    assert not miss, sorted(miss)[:5]
    m = {k: v for k, v in s["model"].items() if k in keys}
    for k in keys:
        assert m[k].shape == ref["model"][k].shape, k
    with open(x.out, "xb") as f:
        torch.save({"model": m, "d": s["d"], "layers": s["layers"], "step": 0, "src": x.src}, f)
    print({"kept": len(m), "dropped": len(s["model"]) - len(m)})


if __name__ == "__main__":
    main()
