#!/usr/bin/env python3
"""Diagnostics of GPU vs CPU state mismatches reported by gpu_parity.py: which State fields differ.

The byte -> field map is built by parsing state_visit.hpp (put order) and the struct declarations in sim_hd.hpp
(types, array sizes). Resets N games on GPU and CPU with the same seeds, compares the serializations after reset and
after k steps of the recorded streams, and prints all differing fields (game, player, tile/index) with the values of
both sides.

    python engines/kagsim_cuda/diag_state.py --steps 0,1,24 --games 4 [--out diag.json]

Needs a GPU, build/libkagsim_cuda.so, the CPU module and a streams file (--streams, default
engines/kagsim_cuda/data/streams.pkl.gz, not shipped: create it with
PYTHONHASHSEED=0 python engines/kagsim_cuda/record_streams.py).
"""

import argparse
import gzip
import json
import os
import pickle
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kagsim_cuda as KC  # noqa: E402

TYPES = {
    "double": ("<f8", 8),
    "int": ("<i4", 4),
    "int32_t": ("<i4", 4),
    "int16_t": ("<i2", 2),
    "int8_t": ("i1", 1),
    "uint8_t": ("u1", 1),
    "bool": ("u1", 1),
    "TileKind": ("u1", 1),
}
CONST = {"MAX_UNITS": 40, "BOARD": 10, "N_ITEMS": 12, "N_CROPS": 5, "N_PRODUCTS": 9, "MAX_SHOP_INSTANCES": 8}


def decls(src, name):
    """{member: (C++ type, array dims)} of the top-level data members of `struct name` in sim_hd.hpp."""
    m = re.search(r"struct\s+" + name + r"\s*\{(.*?)\n\};", src, re.S)
    out, depth = {}, 0
    for line in m.group(1).split("\n"):
        line = line.split("//")[0].strip()
        d0 = depth
        depth += line.count("{") - line.count("}")
        if d0 != 0 or not line or "(" in line or not line.endswith(";"):
            continue
        decl = re.sub(r"=\s*\{.*?\}+|=\s*[^,;]*", "", line[:-1])
        typ, rest = decl.split(None, 1)
        for v in rest.split(","):
            v = v.strip()
            dims = [CONST.get(d) or int(d) for d in re.findall(r"\[(\w+)\]", v)]
            out[re.sub(r"\[.*", "", v).strip()] = (typ, dims)
    return out


def field_map():
    """Serialization layout: ([(label, numpy dtype, count, byte offset)] in put order, total bytes)."""
    src = open(os.path.join(HERE, "sim_hd.hpp")).read()
    vis = open(os.path.join(HERE, "state_visit.hpp")).read()
    D = {"t": decls(src, "Tile"), "f": decls(src, "Farm"), "s": decls(src, "State"), "m": decls(src, "Market")}
    puts = re.findall(r"put\((\w)\.([\w.]+)\)", vis)
    off = 0
    # explicit unrolling in visit_state order
    tile_f = [nm for who, nm in puts if who == "t"]
    farm_f = [nm for who, nm in puts if who == "f"]
    st_f = [nm for who, nm in puts if who == "s"]
    fields = []

    def add(label, typ, dims):
        nonlocal off
        dt, sz = TYPES[typ]
        cnt = int(np.prod(dims)) if dims else 1
        fields.append((label, dt, cnt, off))
        off += sz * cnt

    for p in range(2):
        pre_tiles = [n for n in farm_f[:1]]
        for nm in pre_tiles:
            add(f"f{p}.{nm}", *D["f"][nm])
        for y in range(10):
            for x in range(10):
                for nm in tile_f:
                    add(f"f{p}.tile[{y}][{x}].{nm}", *D["t"][nm])
        for nm in farm_f[1:]:
            add(f"f{p}.{nm}", *D["f"][nm])
    for nm in st_f:
        if nm.startswith("market."):
            add(nm, *D["m"][nm.split(".")[1]])
        else:
            add(nm, *D["s"][nm])
    return fields, off


def explain(a, b, fields, lim=30):
    """(number of differing bytes, up to lim "field[index] gpu=... cpu=..." lines) for serializations a (GPU), b."""
    diff = np.nonzero(a != b)[0]
    out, seen = [], set()
    for d in diff:
        for label, dt, cnt, off in fields:
            sz = np.dtype(dt).itemsize
            if off <= d < off + sz * cnt:
                k = (d - off) // sz
                key = (label, k)
                if key in seen:
                    break
                seen.add(key)
                va = np.frombuffer(a[off + k * sz : off + (k + 1) * sz].tobytes(), dt)[0]
                vb = np.frombuffer(b[off + k * sz : off + (k + 1) * sz].tobytes(), dt)[0]
                out.append(f"{label}[{k}] gpu={va} cpu={vb}")
                break
        if len(out) >= lim:
            break
    return len(diff), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default=os.path.join(HERE, "data", "streams.pkl.gz"))
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--steps", default="0,1,24,200")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lib", default=None)
    x = ap.parse_args()
    fields, total = field_map()
    K = KC._cpu_module()
    recs = pickle.load(gzip.open(x.streams, "rb"))[: x.games]
    n = len(recs)
    T = max(len(r["acts"][0]) for r in recs)
    empty = K.encode_action({})
    acts = np.empty((T, n, 2, K.ACT_I32), np.int32)
    for g, r in enumerate(recs):
        for p in (0, 1):
            e = K.encode_actions(r["acts"][p])
            acts[: len(e), g, p] = e
            acts[len(e) :, g, p] = empty
    seeds = np.array([r["seed"] for r in recs], np.uint64)
    gpu = KC.GpuBatch(n, 720, 0, x.lib)
    cpu = KC.CpuBatch(n, 720)
    gpu.reset(seeds)
    cpu.reset(seeds)
    res = {
        "ser_bytes_map": total,
        "ser_bytes_gpu": int(gpu.ser_bytes),
        "ser_bytes_cpu": int(cpu.serialize().shape[1]),
        "checks": [],
    }
    print(json.dumps({k: v for k, v in res.items() if k != "checks"}), flush=True)
    want = sorted(int(s) for s in x.steps.split(","))
    t = 0
    for tw in want:
        while t < tw:
            gpu.step(acts[t])
            cpu.step(acts[t])
            t += 1
        a, b = gpu.serialize(), cpu.serialize()
        for g in range(n):
            nd, ex = explain(a[g], b[g], fields)
            res["checks"].append({"t": t, "g": g, "n_diff_bytes": int(nd), "fields": ex})
            print(f"t={t} g={g} differing bytes {nd}", flush=True)
            for e in ex:
                print("   ", e)
    if x.out:
        json.dump(res, open(x.out, "w"), ensure_ascii=False, indent=1)
    gpu.close()


if __name__ == "__main__":
    main()
