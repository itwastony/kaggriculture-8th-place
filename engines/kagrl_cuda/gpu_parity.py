#!/usr/bin/env python3
"""GPU build of kagrl (build/libkagrl_cuda.so) against the CPU build of the same kagrl.cu (build/libkagrl_cpu.so), plus
benchmarks. Run on a machine with a GPU after bash build_cuda.sh and bash build_cpu.sh . Writes JSON lines to stdout
and the summary to --out (default engines/kagrl_cuda/data/gpu_parity_result.json; it also records the torch, CUDA and
GPU names).

The CPU build has been checked against kagcore.GameBatch, Python rl/encode.py + rl/act.py + kagsim and the vendor
engine (parity_cpu.py, 0 mismatches), so GPU == CPU byte for byte => GPU == references.

Check (at EVERY step, 720 iterations per game, games finish after 719 steps):
  P1 "rich" states: --n1 games from the recorded streams engines/kagsim_cuda/data/streams.pkl.gz (a fixed path; not
     shipped: create it first with PYTHONHASHSEED=0 python engines/kagsim_cuda/record_streams.py, by default 220 games
     of the shipped bots, live/chaos/random; the first len(streams) games use the recorded seeds and shops, the rest
     the same streams on new seeds).
     encode 2n rows (7 arrays byte for byte) · random_choices (uc/mc) · execute of these choices without a step (engine
     actions) · step with the recorded actions via act_step(raw_mask=1) · full state (serialization of 7645 bytes),
     banks, done, steps.
  P2 choice path: --n2 games on new seeds; even steps - choice by the random_choices kernel (int64), odd steps - the
     policy (rl/model.Net with random initialization d=64, bf16 autocast, sampling on the GPU; the same indices are
     copied to the CPU build); in odd games seat 1 plays a recorded stream via raw_mask (league). Check as in P1 +
     acts_out; if kagcore can be imported (engines/kagcore/build) - additionally the CPU build against
     kagcore.GameBatch (encode, banks).
Benchmarks (CUDA events; games/s = B / time of a full game or B / (sec per step × 719)):
  (i)   engine: B games, 719 steps: encode 2B rows + random_choices + act_step; breakdown by kernel; block-size sweep.
  (i-b) streams: real states - step_raw of recorded streams + encode + random_choices + execute (no step).
  (ii)  policy: encode + rl/model.Net (--d 384, --layers 8, --vtower 2) bf16 autocast over 2B rows in pieces of
        --fwd-chunk + sampling + logp (= act_compute rl/appo.py) + act_step; engine warm-up of --warm-steps steps
        with random choices, then a window of --policy-steps steps; eager and (--compile 1) torch.compile.
  (iii) rollout: gpu_rollout.Rollout - full self-play of --rollout-B games × 720 with appo-format storage.

    python gpu_parity.py --out res.json                       # with a GPU: check + all benchmarks
    python gpu_parity.py --dry-run                            # no GPU: both sides = CPU build, small sizes
"""

import argparse
import gzip
import json
import os
import pickle
import platform
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import kagrl as KR  # noqa: E402

ENC = KR.OBS_KEYS
STREAMS = os.path.join(ROOT, "engines", "kagsim_cuda", "data", "streams.pkl.gz")


def emit(rec, sink):
    """Append a result record to sink and print it as a JSON line."""
    sink.append(rec)
    print(json.dumps(rec, ensure_ascii=False), flush=True)


def to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x


class Side:
    """One side of the check: RLBatch + buffers on its own device."""

    def __init__(self, n, backend, dev, kind):
        self.rb = KR.RLBatch(n, 720, backend=backend, device=0)
        self.dev, self.kind = dev, kind  # kind: torch (tensors on dev) | numpy
        self.obs = self.rb.alloc_obs(kind=kind, device=dev if kind == "torch" else None)

    def z(self, shape, dt):
        if self.kind == "numpy":
            return np.zeros(shape, dt)
        return torch.zeros(shape, dtype=KR._torch_dtype(dt), device=self.dev)

    def put(self, a):
        """host numpy -> buffer of this side."""
        if self.kind == "numpy":
            return np.ascontiguousarray(a)
        return torch.from_numpy(np.ascontiguousarray(a)).to(self.dev)


def load_streams():
    """Recorded streams + encoder of dict actions into int32[170]: kaghd_cpu.encode_actions, otherwise via
    kagcore.conv_action_raw (both are pybind modules, built by scripts/build_engines.sh)."""
    with gzip.open(STREAMS, "rb") as f:
        recs = pickle.load(f)
    for p in (os.path.join(ROOT, "engines", "kagsim_cuda", "build"), os.path.join(ROOT, "engines", "kagcore", "build")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import kaghd_cpu

        return recs, kaghd_cpu.encode_actions
    except ImportError:
        import kagcore

        def enc(lst):
            out = np.zeros((len(lst), 170), np.int32)
            for t, d in enumerate(lst):
                nu, us, no, os_ = kagcore.conv_action_raw(d)
                out[t, 0], out[t, 121] = nu, no
                for i, u in enumerate(us):
                    out[t, 1 + 3 * i : 4 + 3 * i] = u
                for o, x in enumerate(os_):
                    out[t, 122 + 3 * o : 125 + 3 * o] = x
            return out

        return recs, enc


def stream_block(recs, enc, n):
    """Engine actions int32 [719, n, 2, 170]: game i - stream i % len(recs); seeds/shops."""
    m = len(recs)
    acts_m = np.zeros((719, m, 2, 170), np.int32)
    for j, r in enumerate(recs):
        for p in (0, 1):
            acts_m[:, j, p] = enc(r["acts"][p])
    idx = np.arange(n) % m
    seeds = np.array(
        [
            recs[i % m]["seed"] if i < m else (recs[i % m]["seed"] * 7 + (i // m) * 1000003 + 12345) % 2**31
            for i in range(n)
        ],
        np.uint64,
    )
    shops = np.zeros((n, 8), np.uint8)
    nshops = np.zeros(n, np.int32)
    for i in range(min(n, m)):
        if recs[i]["shops"] is not None:
            shops[i, : len(recs[i]["shops"])] = recs[i]["shops"]
            nshops[i] = len(recs[i]["shops"])
    return acts_m, idx, seeds, shops, nshops


def cmp_obs(a, b, res, tag, t):
    """Compare the 7 observation buffers of sides a and b row by row; count mismatching rows in res."""
    for k in ENC:
        x, y = to_np(a.obs[k]), to_np(b.obs[k])
        bad = np.nonzero((x.reshape(x.shape[0], -1) != y.reshape(y.shape[0], -1)).any(axis=1))[0]
        res["enc_rows"] += x.shape[0]
        res["enc_mismatch"] += len(bad)
        if len(bad) and len(res["examples"]) < 20:
            res["examples"].append(f"[{tag}] encode {k} row {int(bad[0])} t={t}")


def cmp_state(a, b, res, tag, t):
    """Compare the full-state serializations, banks, done flags and step counts of sides a and b."""
    sa, sb = a.rb.serialize(), b.rb.serialize()
    bad = np.nonzero((sa != sb).any(axis=1))[0]
    res["state_cmp"] += sa.shape[0]
    res["state_mismatch"] += len(bad)
    if len(bad) and len(res["examples"]) < 20:
        res["examples"].append(
            f"[{tag}] state game {int(bad[0])} t={t} byte {int(np.nonzero(sa[bad[0]] != sb[bad[0]])[0][0])}"
        )
    ra, rb_ = a.rb.rewards(), b.rb.rewards()
    res["reward_cmp"] += ra.size
    res["reward_mismatch"] += int((ra != rb_).sum())
    res["flag_mismatch"] += int((a.rb.done() != b.rb.done()).sum()) + int((a.rb.steps() != b.rb.steps()).sum())


def cmp_arr(x, y, res, key, tag, t):
    """Compare two arrays row by row; counts go to res[key + "_cmp"] / res[key + "_mismatch"]."""
    x, y = to_np(x), to_np(y)
    res[key + "_cmp"] = res.get(key + "_cmp", 0) + x.shape[0]
    bad = int((x.reshape(x.shape[0], -1) != y.reshape(y.shape[0], -1)).any(axis=1).sum())
    res[key + "_mismatch"] = res.get(key + "_mismatch", 0) + bad
    if bad and len(res["examples"]) < 20:
        res["examples"].append(f"[{tag}] {key} t={t}: {bad} mismatches")


def new_res():
    return dict(
        games=0,
        steps=0,
        enc_rows=0,
        enc_mismatch=0,
        state_cmp=0,
        state_mismatch=0,
        reward_cmp=0,
        reward_mismatch=0,
        flag_mismatch=0,
        examples=[],
    )


# ---------------------------------------------------------------- parity check


def parity_p1(n, dev_backend, dev, recs, enc, res):
    """Check P1 (recorded streams, see the module docstring)."""
    acts_m, idx, seeds, shops, nshops = stream_block(recs, enc, n)
    D, C = Side(n, dev_backend, dev, "torch"), Side(n, "cpu", torch.device("cpu"), "numpy")
    for s in (D, C):
        s.rb.reset(
            s.put(seeds.view(np.int64)) if s.kind == "torch" else seeds, shops=s.put(shops), nshops=s.put(nshops)
        )
    uD, mD, xD, aD = (
        D.z((n, 2, 16), np.int32),
        D.z((n, 2, 24), np.int32),
        D.z((n, 2, 170), np.int32),
        D.z((n, 2, 170), np.int32),
    )
    uC, mC, xC, aC = (
        C.z((n, 2, 16), np.int32),
        C.z((n, 2, 24), np.int32),
        C.z((n, 2, 170), np.int32),
        C.z((n, 2, 170), np.int32),
    )
    ones = np.ones((n, 2), np.uint8)
    oD, oC = D.put(ones), C.put(ones)
    cmp_state(D, C, res, "P1", -1)
    for t in range(720):
        D.rb.encode(D.obs)
        C.rb.encode(C.obs)
        cmp_obs(D, C, res, "P1", t)
        D.rb.random_choices(D.obs, 4242, t, uD, mD)
        C.rb.random_choices(C.obs, 4242, t, uC, mC)
        cmp_arr(uD, uC, res, "rand_uc", "P1", t)
        cmp_arr(mD, mC, res, "rand_mc", "P1", t)
        D.rb.execute(uD, mD, xD)
        C.rb.execute(uC, mC, xC)
        cmp_arr(xD, xC, res, "execute", "P1", t)
        raw = acts_m[min(t, 718)][idx]
        D.rb.act_step(uD, mD, raw=D.put(raw), raw_mask=oD, acts_out=aD)
        C.rb.act_step(uC, mC, raw=C.put(raw), raw_mask=oC, acts_out=aC)
        cmp_arr(aD, aC, res, "acts_out", "P1", t)
        cmp_state(D, C, res, "P1", t)
        res["steps"] += n
    res["games"] += n
    res["p1_final_banks_first4"] = D.rb.rewards()[:4].tolist()
    res["p1_recorded_banks_first4"] = [recs[i]["rewards"] for i in range(min(4, n))]


def parity_p2(n, dev_backend, dev, recs, enc, res, net_d=64):
    """Check P2 (random kernel choices and a random policy, see the module docstring)."""
    from rl.appo import unpack
    from rl.model import Net
    from rl.ppo import sample

    acts_m, idx, _, _, _ = stream_block(recs, enc, n)
    seeds = (np.arange(n, dtype=np.uint64) * 104729 + np.uint64(990001)).astype(np.uint64)
    D, C = Side(n, dev_backend, dev, "torch"), Side(n, "cpu", torch.device("cpu"), "numpy")
    for s in (D, C):
        s.rb.reset(s.put(seeds.view(np.int64)) if s.kind == "torch" else seeds)
    kg = None
    try:
        sys.path.insert(0, os.path.join(ROOT, "engines", "kagcore", "build"))
        import kagcore

        kg = kagcore.GameBatch([int(s) for s in seeds], 720)
        kbuf = C.rb.alloc_obs()
        seats2 = np.tile(np.array([0, 1], np.int32), n)
        games2 = np.repeat(np.arange(n, dtype=np.int32), 2)
        res["kagcore_check"] = True
    except Exception as e:  # noqa: BLE001
        res["kagcore_check"] = f"skipped: {e!r}"
    tdev = dev
    torch.manual_seed(12345)
    net = Net(d=net_d, layers=2, heads=4, dk=16, vtower=1).to(tdev).eval()
    amp = dict(device_type=tdev.type, dtype=torch.bfloat16)
    rmask = np.zeros((n, 2), np.uint8)
    rmask[1::2, 1] = 1  # odd games: seat 1 - recorded stream (league)
    rD, rC = D.put(rmask), C.put(rmask)
    uD, mD, aD = D.z((n, 2, 16), np.int64), D.z((n, 2, 24), np.int64), D.z((n, 2, 170), np.int32)
    uC, mC, aC = C.z((n, 2, 16), np.int64), C.z((n, 2, 24), np.int64), C.z((n, 2, 170), np.int32)
    for t in range(720):
        D.rb.encode(D.obs)
        C.rb.encode(C.obs)
        cmp_obs(D, C, res, "P2", t)
        if kg is not None:
            kg.encode_into(seats2, *[kbuf[k] for k in ENC], games=games2)
            for k in ENC:
                bad = int((kbuf[k].reshape(2 * n, -1) != C.obs[k].reshape(2 * n, -1)).any(axis=1).sum())
                res["kagcore_enc_rows"] = res.get("kagcore_enc_rows", 0) + 2 * n
                res["kagcore_enc_mismatch"] = res.get("kagcore_enc_mismatch", 0) + bad
        if t % 2 == 0:
            D.rb.random_choices(D.obs, 777, t, uD, mD)
            C.rb.random_choices(C.obs, 777, t, uC, mC)
            cmp_arr(uD, uC, res, "rand_uc", "P2", t)
            cmp_arr(mD, mC, res, "rand_mc", "P2", t)
            uc_host, mc_host = to_np(uD).copy(), to_np(mD).copy()
        else:
            with torch.no_grad():
                b = {k: D.obs[k] for k in ENC}
                b["tmask"] = unpack(b["tmask"])
                with torch.autocast(**amp):
                    lg, ml, _ = net(b["tiles"], b["units"], b["unit_pos"], b["unit_mask"], b["glob"])
                puc, pmc = sample(lg, ml, b)
            uc_host = puc.cpu().numpy().reshape(n, 2, 16)
            mc_host = pmc.cpu().numpy().reshape(n, 2, 24)
            uD.copy_(puc.view(n, 2, 16))
            mD.copy_(pmc.view(n, 2, 24))
            uC[:], mC[:] = uc_host, mc_host
        raw = acts_m[min(t, 718)][idx]
        D.rb.act_step(uD, mD, raw=D.put(raw), raw_mask=rD, acts_out=aD)
        C.rb.act_step(uC, mC, raw=C.put(raw), raw_mask=rC, acts_out=aC)
        cmp_arr(aD, aC, res, "acts_out", "P2", t)
        cmp_state(D, C, res, "P2", t)
        if kg is not None:
            acts = [(None, recs[idx[g]]["acts"][1][min(t, 718)]) if rmask[g, 1] else None for g in range(n)]
            kg.step(uc_host.astype(np.int32), mc_host.astype(np.int32), acts)
            res["kagcore_bank_cmp"] = res.get("kagcore_bank_cmp", 0) + 2 * n
            res["kagcore_bank_mismatch"] = res.get("kagcore_bank_mismatch", 0) + int(
                (kg.rewards() != C.rb.rewards()).sum()
            )
        res["steps"] += n
    res["games"] += n
    res["p2_final_banks_first4"] = D.rb.rewards()[:4].tolist()


# ---------------------------------------------------------------- benchmarks


class Ev:
    def __init__(self, cuda):
        self.cuda = cuda

    def __call__(self):
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e
        return time.perf_counter()

    def dt(self, a, b):
        if self.cuda:
            b.synchronize()
            return a.elapsed_time(b) / 1000.0
        return b - a


def bench_engine(B, backend, dev, block, sink, tag="engine"):
    """Benchmark (i): engine with random choices, CUDA block size `block`."""
    rb = KR.RLBatch(B, 720, backend=backend)
    KR.set_block(rb.lib, block)
    obs = rb.alloc_obs(kind="torch", device=dev)
    uc = torch.zeros(B, 2, 16, dtype=torch.int64, device=dev)
    mc = torch.zeros(B, 2, 24, dtype=torch.int64, device=dev)
    rb.reset(torch.arange(B, dtype=torch.int64, device=dev) * 7919 + 31337)
    ev = Ev(dev.type == "cuda")
    if dev.type == "cuda":
        torch.cuda.synchronize()
    marks = []
    for t in range(719):  # no synchronization inside the loop; events are read at the end
        e0 = ev()
        rb.encode(obs)
        e1 = ev()
        rb.random_choices(obs, 5, t, uc, mc)
        e2 = ev()
        rb.act_step(uc, mc)
        marks.append((e0, e1, e2, ev()))
    sec = ev.dt(marks[0][0], marks[-1][3])
    tt = {
        "encode": sum(ev.dt(a, b) for a, b, _, _ in marks),
        "random": sum(ev.dt(b, c) for _, b, c, _ in marks),
        "act_step": sum(ev.dt(c, d) for _, _, c, d in marks),
    }
    done = rb.done()
    bank = rb.rewards()
    rec = {
        "bench": tag,
        "B": B,
        "block": block,
        "steps": 719,
        "sec": round(sec, 4),
        "games_per_sec": round(B / sec, 1),
        "game_steps_per_sec": round(B * 719 / sec),
        "ms_per_step": round(sec / 719 * 1e3, 3),
        "ms_per_step_by_kernel": {k: round(v / 719 * 1e3, 3) for k, v in tt.items()},
        "all_done": bool(done.all()),
        "mean_bank": float(bank.mean()),
        "nonzero_banks": int((bank != 0).sum()),
        "what": "encode 2B rows + random_choices + act_step (execute+step), random legal choices",
    }
    emit(rec, sink)
    rb.close()
    return rec


def bench_streams(B, backend, dev, recs, enc, sink):
    """Benchmark (i-b): recorded streams."""
    acts_m, idx, seeds, shops, nshops = stream_block(recs, enc, min(B, len(recs)))
    m = acts_m.shape[1]
    rb = KR.RLBatch(B, 720, backend=backend)
    KR.set_block(rb.lib, 128)
    obs = rb.alloc_obs(kind="torch", device=dev)
    uc = torch.zeros(B, 2, 16, dtype=torch.int32, device=dev)
    mc = torch.zeros(B, 2, 24, dtype=torch.int32, device=dev)
    xo = torch.zeros(B, 2, 170, dtype=torch.int32, device=dev)
    sd = np.array([(int(recs[i % m]["seed"]) + (i // m) * 7919) % 2**31 for i in range(B)], np.int64)
    rb.reset(torch.from_numpy(sd).to(dev))
    A = torch.from_numpy(acts_m).to(dev)  # [719, m, 2, 170], game i - stream i % m
    ev = Ev(dev.type == "cuda")
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = ev()
    for t in range(719):
        rb.encode(obs)
        rb.random_choices(obs, 9, t, uc, mc)
        rb.execute(uc, mc, xo)
        rb.step_raw(A[t])
    t1 = ev()
    sec = ev.dt(t0, t1)
    bank = rb.rewards()
    rec = {
        "bench": "streams",
        "B": B,
        "distinct_streams": m,
        "steps": 719,
        "sec": round(sec, 4),
        "games_per_sec": round(B / sec, 1),
        "ms_per_step": round(sec / 719 * 1e3, 3),
        "mean_bank": float(bank.mean()),
        "nonzero_banks": int((bank != 0).sum()),
        "what": "encode 2B + random_choices + execute (no step) + step_raw of recorded actions",
    }
    emit(rec, sink)
    rb.close()
    return rec


def load_net(args, dev):
    """rl/model.Net from --ckpt if that file exists (non-strict load), else randomly initialised; (net, source)."""
    from rl.model import Net

    torch.manual_seed(0)
    src = "random-init"
    if args.ckpt and os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        net = Net(d=ck.get("d", args.d), layers=ck.get("layers", args.layers), vtower=ck.get("vtower", args.vtower))
        miss, unexp = net.load_state_dict(ck["model"], strict=False)
        src = f"{args.ckpt} (missing {len(miss)}, unexpected {len(unexp)})"
    else:
        net = Net(d=args.d, layers=args.layers, vtower=args.vtower)
    return net.to(dev).eval(), src


def bench_policy(B, backend, dev, net, net_src, args, sink, compiled=None):
    """Benchmark (ii): engine + policy forward/sampling."""
    from gpu_rollout import make_act_fns

    from rl.appo import unpack

    rb = KR.RLBatch(B, 720, backend=backend)
    KR.set_block(rb.lib, 128)
    obs = rb.alloc_obs(kind="torch", device=dev)
    uc = torch.zeros(B, 2, 16, dtype=torch.int64, device=dev)
    mc = torch.zeros(B, 2, 24, dtype=torch.int64, device=dev)
    rb.reset(torch.arange(B, dtype=torch.int64, device=dev) * 104729 + 5)
    for t in range(args.warm_steps):  # game warm-up (engine, random choices)
        rb.encode(obs)
        rb.random_choices(obs, 3, t, uc, mc)
        rb.act_step(uc, mc)
    amp = dict(device_type=dev.type, dtype=torch.bfloat16)
    act_fn = compiled or make_act_fns(amp)[0]
    R = 2 * B
    ch = min(args.fwd_chunk, R)
    u2, m2 = uc.view(R, 16), mc.view(R, 24)
    ev = Ev(dev.type == "cuda")

    def one_step():
        e0 = ev()
        rb.encode(obs)
        e1 = ev()
        with torch.no_grad():
            for i0 in range(0, R, ch):
                b = {k: obs[k][i0 : i0 + ch] for k in ENC}
                b["tmask"] = unpack(b["tmask"])
                u_, m_, lpu, lpm, v = act_fn(net, b)
                u2[i0 : i0 + ch] = u_
                m2[i0 : i0 + ch] = m_
        e2 = ev()
        rb.act_step(uc, mc)
        e3 = ev()
        return e0, e1, e2, e3

    for _ in range(2):  # warm-up (memory allocation, cuBLAS autotuning)
        one_step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    marks = [one_step() for _ in range(args.policy_steps)]
    tot = ev.dt(marks[0][0], marks[-1][3])
    enc = sum(ev.dt(a, b) for a, b, _, _ in marks)
    pol = sum(ev.dt(b, c) for _, b, c, _ in marks)
    act = sum(ev.dt(c, d) for _, _, c, d in marks)
    S_ = args.policy_steps
    rec = {
        "bench": "policy" + ("_compiled" if compiled else "_eager"),
        "B": B,
        "rows_per_step": R,
        "fwd_chunk": ch,
        "net": {"d": net.d, "layers": len(net.enc.layers), "vtower": net.vtower, "src": net_src},
        "measured_steps": S_,
        "start_step": args.warm_steps + 2,
        "sec_per_step": round(tot / S_, 5),
        "ms_per_step_encode": round(enc / S_ * 1e3, 3),
        "ms_per_step_policy": round(pol / S_ * 1e3, 3),
        "ms_per_step_act_step": round(act / S_ * 1e3, 3),
        "policy_rows_per_sec": round(R * S_ / pol),
        "est_games_per_sec": round(B / (tot / S_ * 719), 2),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev.type == "cuda" else None,
    }
    emit(rec, sink)
    rb.close()
    return rec


def bench_rollout(B, backend, dev, net, net_src, args, sink):
    """Benchmark (iii): gpu_rollout.Rollout."""
    from gpu_rollout import Rollout

    ro = Rollout(B, T=720, backend=backend, device=0 if dev.type == "cuda" else 0, chunk=args.fwd_chunk)
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    seeds = np.arange(B, dtype=np.uint64) * 7 + np.uint64(123456)
    t0 = time.time()
    out = ro.run(net, seeds)
    sec = time.time() - t0
    st = out["store"]
    store_gb = sum(v.numel() * v.element_size() for v in st.values()) / 2**30
    rec = {
        "bench": "rollout",
        "B": B,
        "T": 720,
        "NA": out["NA"],
        "sec": round(sec, 2),
        "games_per_sec": round(B / sec, 2),
        "info": {k: round(v, 2) for k, v in out["info"].items()},
        "store_gb": round(store_gb, 2),
        "stats": out["stats"],
        "R_mean": float(out["R"].mean()),
        "net_src": net_src,
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev.type == "cuda" else None,
    }
    emit(rec, sink)
    del out, st
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="no GPU: device = CPU build, small sizes")
    ap.add_argument("--n1", type=int, default=256)
    ap.add_argument("--n2", type=int, default=256)
    ap.add_argument("--bench-B", default="4096,16384,65536")
    ap.add_argument("--blocks", default="64,128,256", help="sweep of threads per block for engine at the maximum B")
    ap.add_argument("--policy-B", default="4096,16384,65536")
    ap.add_argument("--policy-steps", type=int, default=8)
    ap.add_argument("--warm-steps", type=int, default=300)
    ap.add_argument("--fwd-chunk", type=int, default=8192)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--vtower", type=int, default=2)
    ap.add_argument("--ckpt", default="", help="policy checkpoint (otherwise random initialization)")
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--rollout-B", default="1024")
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "gpu_parity_result.json"))
    args = ap.parse_args()

    if args.dry_run:
        backend, dev = "cpu", torch.device("cpu")
        args.n1, args.n2 = min(args.n1, 24), min(args.n2, 16)
        args.bench_B, args.policy_B, args.rollout_B, args.blocks = "64", "8", "4", "128"
        args.policy_steps, args.warm_steps, args.fwd_chunk = 2, 20, 8
        args.d, args.layers, args.vtower, args.compile = 64, 2, 1, 0
    else:
        backend, dev = "gpu", torch.device("cuda:0")
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = True
    sink = []
    result = {
        "backend": backend,
        "args": vars(args),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if backend == "gpu" else None,
        "cuda_runtime": torch.version.cuda,
    }
    lib = KR.load_lib(backend)
    result["dims_device"] = lib.dims
    result["dims_cpu"] = KR.load_lib("cpu").dims
    result["kernel_info"] = KR.kernel_info(lib)
    print(json.dumps({"kernel_info": result["kernel_info"]}), flush=True)
    dims_ok = all(result["dims_device"][k] == result["dims_cpu"][k] for k in KR.DIM_NAMES if k not in ("EMULATED",))
    recs, enc = load_streams()
    ok = dims_ok
    if not args.skip_parity:
        t0 = time.time()
        res = new_res()
        parity_p1(args.n1, backend, dev, recs, enc, res)
        print(json.dumps({"parity": "P1", **{k: v for k, v in res.items() if k != "examples"}}), flush=True)
        parity_p2(args.n2, backend, dev, recs, enc, res)
        res["sec"] = round(time.time() - t0, 1)
        result["parity"] = res
        emit({"parity": "P1+P2", **{k: v for k, v in res.items()}}, sink)
        ok = ok and all(v == 0 for k, v in res.items() if k.endswith("mismatch"))
    result["parity_ok"] = bool(ok)
    # (i) engine
    Bs = [int(b) for b in args.bench_B.split(",") if b]
    for B in Bs:
        bench_engine(B, backend, dev, 128, sink)
    for blk in [int(b) for b in args.blocks.split(",") if b]:
        if blk != 128:
            bench_engine(max(Bs), backend, dev, blk, sink, tag="engine_block_sweep")
    for B in Bs:
        bench_streams(B, backend, dev, recs, enc, sink)
    # (ii) full-size policy
    net, net_src = load_net(args, dev)
    PB = [int(b) for b in args.policy_B.split(",") if b]
    for B in PB:
        try:
            bench_policy(B, backend, dev, net, net_src, args, sink)
        except torch.OutOfMemoryError as e:  # noqa: PERF203
            emit({"bench": "policy_eager", "B": B, "error": f"OOM: {e}"[:300]}, sink)
            torch.cuda.empty_cache()
    if args.compile:
        try:
            from gpu_rollout import make_act_fns

            amp = dict(device_type=dev.type, dtype=torch.bfloat16)
            cfn = torch.compile(make_act_fns(amp)[0], dynamic=False)
            for B in PB:
                bench_policy(B, backend, dev, net, net_src, args, sink, compiled=cfn)
        except Exception as e:  # noqa: BLE001
            emit({"bench": "policy_compiled", "error": repr(e)[:500]}, sink)
    # (iii) full self-play with storage
    for B in [int(b) for b in args.rollout_B.split(",") if b]:
        try:
            bench_rollout(B, backend, dev, net, net_src, args, sink)
        except torch.OutOfMemoryError as e:
            emit({"bench": "rollout", "B": B, "error": f"OOM: {e}"[:300]}, sink)
        if backend == "gpu":
            torch.cuda.empty_cache()
    result["records"] = sink
    result["ok"] = bool(ok)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("RESULT:", "OK - 0 mismatches" if ok else "MISMATCHES FOUND", "| json:", args.out, flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
