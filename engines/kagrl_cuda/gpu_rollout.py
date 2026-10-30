#!/usr/bin/env python3
"""Self-play entirely on the GPU: engine + encode + execute (kagrl) + policy (rl/model.Net) without CPU workers.

Trajectories are in the format read by the learner in rl/appo.py (make_store): for seat row j and step t
    tiles/units/glob  bf16 [T, NA, ...]      (the actor sees float32, the storage holds bf16, as in appo)
    unit_pos int64 [T,NA,16], unit_mask bool [T,NA,16], tmask uint8 [T,NA,16,700] (packed), mmask bool [T,NA,489]
    uc int64 [T,NA,16] (target*56+op), mc int64 [T,NA,24], lpu f32 [T,NA,16], lpm f32 [T,NA,24], val f32 [T,NA]
and the game result R [NA] (float32, CPU) = outcome (+1/0/-1) + margin_w * clip(margin / margin_scale, -1, 1), as in
appo.
Rows: self-play - all 2n seats (row 2g+s = seat s of game g, NA = 2n); with a past version of the opponent
(opp_net, opp_games, me_seat) - the learner seats of all games in increasing row order; opponent seats are not stored.
Sampling and logp - exactly rl/appo.py act_compute/past_compute (autocast bf16, rl.ppo.sample = Gumbel-max).
Loop of T iterations (like the appo worker): encode -> policy -> act_step; a game finishes after T-1 steps,
the last iteration encodes the final state (its step is a no-op).

    from gpu_rollout import Rollout, make_store
    ro = Rollout(n_games, T=720, backend="gpu", device=torch.cuda.current_device(), chunk=4096)
    out = ro.run(act_net, seeds, store=st)          # st[k][:, :out["NA"]] filled, out["R"], out["stats"]

Self-test without a GPU (CPU build libkagrl_cpu.so, torch CPU; the replay check also needs engines/kagcore):
    python engines/kagrl_cuda/gpu_rollout.py --selftest --B 8 --T 64
It writes --out (default engines/kagrl_cuda/data/rollout_selftest.json); exit code 1 if a check fails.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kagrl import RLBatch  # noqa: E402

from rl import spec as S  # noqa: E402
from rl.appo import FLOATS, H, buffers, unpack  # noqa: E402
from rl.model import Net, market_logp, unit_logp  # noqa: E402
from rl.ppo import ENC, sample  # noqa: E402


def make_store(T, NS, device):
    """Game storage = rl/appo.py make_store (the same keys, shapes [T, NS, ...] and dtypes)."""
    proto = buffers(1)
    st = {
        k: torch.zeros((T, NS) + tuple(v.shape[1:]), dtype=torch.bfloat16 if k in FLOATS else v.dtype, device=device)
        for k, v in proto.items()
    }
    st["lpu"] = torch.zeros(T, NS, S.MAX_UNITS, device=device)
    st["lpm"] = torch.zeros(T, NS, H, device=device)
    st["val"] = torch.zeros(T, NS, device=device)
    return st


def make_act_fns(amp):
    """act_compute / past_compute - verbatim as in rl/appo.py main() (closure over amp)."""

    def act_compute(net_, b):
        with torch.autocast(**amp):
            lg, ml, v = net_(b["tiles"], b["units"], b["unit_pos"], b["unit_mask"], b["glob"])
        uc, mc = sample(lg, ml, b)
        lpu = unit_logp(lg, b["tmask"], b["unit_mask"], uc // S.NOPS, uc % S.NOPS)
        lpm = market_logp(ml, b["mmask"], mc)
        return uc, mc, lpu, lpm, v

    def past_compute(net_, b):
        with torch.autocast(**amp):
            plg, pml, _ = net_(b["tiles"], b["units"], b["unit_pos"], b["unit_mask"], b["glob"], value=False)
        return sample(plg, pml, b)

    return act_compute, past_compute


class _Timer:
    """Sum of stage times: on the GPU - CUDA events (one synchronization at the end), on the CPU - perf_counter."""

    def __init__(self, cuda):
        self.cuda = cuda
        self.ev = {}
        self.acc = {}

    def mark(self, name):
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.ev.setdefault(name, []).append(e)
        else:
            self.ev.setdefault(name, []).append(time.perf_counter())

    def spans(self, pairs):
        """pairs: {result: (start, end)} -> seconds (sum over all steps)."""
        if self.cuda:
            torch.cuda.synchronize()
        out = {}
        for k, (a, b) in pairs.items():
            A, B = self.ev.get(a, []), self.ev.get(b, [])
            if self.cuda:
                out[k] = sum(x.elapsed_time(y) for x, y in zip(A, B)) / 1000.0
            else:
                out[k] = sum(y - x for x, y in zip(A, B))
        return out


class Rollout:
    """n games on RLBatch (backend gpu/cpu) with the observation/choice buffers reused across run() calls."""

    def __init__(
        self, n, T=720, backend="gpu", device=0, chunk=4096, amp_dtype=torch.bfloat16, lib_path=None, pad_chunks=False
    ):
        self.n, self.T, self.chunk, self.pad = int(n), int(T), int(chunk), bool(pad_chunks)
        # game length = T, as in the appo worker (kagcore.GameBatch(seeds, T))
        self.rb = RLBatch(self.n, self.T, backend=backend, device=device, lib_path=lib_path)
        self.dev = torch.device(f"cuda:{device}" if backend == "gpu" else "cpu")
        self.obs = self.rb.alloc_obs(kind="torch", device=self.dev)
        self.uc = torch.zeros(self.n, 2, S.MAX_UNITS, dtype=torch.long, device=self.dev)
        self.mc = torch.zeros(self.n, 2, H, dtype=torch.long, device=self.dev)
        self.bank = torch.zeros(self.n, 2, dtype=torch.float64, device=self.dev)
        self.amp = dict(device_type=self.dev.type, dtype=amp_dtype)
        self.act_fn, self.past_fn = make_act_fns(self.amp)

    def _chunks(self, rows, NA):
        """(i0, i1, idx) over pieces of chunk rows; idx = None -> slice [i0:i1] (contiguous rows), otherwise indices
        (with pad_chunks the last piece is padded up to chunk by repeating its first row - a fixed shape for
        torch.compile, the extra outputs are discarded)."""
        for i0 in range(0, NA, self.chunk):
            i1 = min(NA, i0 + self.chunk)
            if rows is None and not (self.pad and i1 - i0 < self.chunk and NA > self.chunk):  # noqa: SIM300  NA = rows
                yield i0, i1, None
                continue
            idx = rows[i0:i1] if rows is not None else torch.arange(i0, i1, device=self.dev)
            if self.pad and i1 - i0 < self.chunk and NA > self.chunk:  # noqa: SIM300  NA is the row count
                idx = torch.cat([idx, idx[:1].repeat(self.chunk - (i1 - i0))])
            yield i0, i1, idx

    @torch.no_grad()
    def run(
        self,
        net,
        seeds,
        store=None,
        opp_net=None,
        opp_games=None,
        me_seat=None,
        margin_w=0.3,
        margin_scale=30000.0,
        act_fn=None,
        past_fn=None,
        record_all=False,
        timing=True,
    ):
        """One batch of n games of T iterations each. net - the playing (and trained) policy; opp_net - a past version
        for games opp_games (bool [n]) on seat 1 - me_seat[g]. Returns a dict: store, NA, R (CPU f32 [NA]),
        rows (encoding row of each storage column), bank (numpy [n,2]), margin, stats, info."""
        n, T, dev = self.n, self.T, self.dev
        act_fn = act_fn or self.act_fn
        past_fn = past_fn or self.past_fn
        t_start = time.time()
        self.rb.reset(seeds)
        rows = prow = None
        if opp_net is not None and opp_games is not None and np.any(opp_games):
            og = np.asarray(opp_games, bool)
            me = np.asarray(me_seat, np.int64)
            learn = [2 * g + s for g in range(n) for s in (0, 1) if not og[g] or s == me[g]]
            rows_np = np.array(learn, np.int64)
            rows = torch.as_tensor(rows_np, device=dev)
            prow = torch.as_tensor(np.array([2 * g + 1 - me[g] for g in range(n) if og[g]], np.int64), device=dev)
        else:
            og = np.zeros(n, bool)
            me = np.zeros(n, np.int64)
            rows_np = np.arange(2 * n, dtype=np.int64)
        NA = len(rows_np)
        st = store if store is not None else make_store(T, NA, dev)
        if st["tiles"].shape[0] < T or st["tiles"].shape[1] < NA:
            raise ValueError(f"store too small: {tuple(st['tiles'].shape[:2])} < ({T}, {NA})")
        uc2, mc2 = self.uc.view(2 * n, S.MAX_UNITS), self.mc.view(2 * n, H)
        uc2.zero_()
        mc2.zero_()
        allu = torch.zeros(T, 2 * n, S.MAX_UNITS, dtype=torch.long, device=dev) if record_all else None
        allm = torch.zeros(T, 2 * n, H, dtype=torch.long, device=dev) if record_all else None
        tm = _Timer(dev.type == "cuda") if timing else None
        for t in range(T):
            if tm:
                tm.mark("e0")
            self.rb.encode(self.obs)
            if tm:
                tm.mark("e1")
            for i0, i1, idx in self._chunks(rows, NA):
                m_ = i1 - i0
                b = {k: (self.obs[k][i0:i1] if idx is None else self.obs[k][idx]) for k in ENC}
                bt = dict(b)
                bt["tmask"] = unpack(b["tmask"])
                uc_, mc_, lpu, lpm, v = act_fn(net, bt)
                if idx is None:
                    uc2[i0:i1] = uc_
                    mc2[i0:i1] = mc_
                else:
                    uc2[idx[:m_]] = uc_[:m_]
                    mc2[idx[:m_]] = mc_[:m_]
                for k in ENC:
                    st[k][t, i0:i1] = b[k][:m_]
                st["uc"][t, i0:i1] = uc_[:m_]
                st["mc"][t, i0:i1] = mc_[:m_]
                st["lpu"][t, i0:i1] = lpu[:m_]
                st["lpm"][t, i0:i1] = lpm[:m_]
                st["val"][t, i0:i1] = v[:m_].float()
            if prow is not None:
                for j0 in range(0, len(prow), self.chunk):
                    pidx = prow[j0 : j0 + self.chunk]
                    pb = {k: self.obs[k][pidx] for k in ENC}
                    pb["tmask"] = unpack(pb["tmask"])
                    puc, pmc = past_fn(opp_net, pb)
                    uc2[pidx] = puc
                    mc2[pidx] = pmc
            if tm:
                tm.mark("p1")
            if record_all:
                allu[t] = uc2
                allm[t] = mc2
            self.rb.act_step(self.uc, self.mc)
            if tm:
                tm.mark("x1")
        self.rb.rewards(out=self.bank)
        bank = self.bank.cpu().numpy()  # the only host synchronization per batch
        g_of, s_of = rows_np // 2, rows_np % 2
        m = bank[g_of, s_of] - bank[g_of, 1 - s_of]
        R = torch.as_tensor(np.sign(m) + margin_w * np.clip(m / margin_scale, -1.0, 1.0), dtype=torch.float32)
        stats = {}
        for j in range(NA):
            lab = "past" if og[g_of[j]] else "self"
            a = stats.setdefault(lab, [0, 0.0, 0.0, 0.0])  # as in appo: n, wins (draw 0.5), bank, margin
            a[0] += 1
            a[1] += 1.0 if m[j] > 0 else (0.5 if m[j] == 0 else 0.0)
            a[2] += float(bank[g_of[j], s_of[j]])
            a[3] += float(m[j])
        info = {"t_roll": time.time() - t_start}
        if tm:
            info.update(tm.spans({"t_encode": ("e0", "e1"), "t_policy": ("e1", "p1"), "t_act_step": ("p1", "x1")}))
        out = {"store": st, "NA": NA, "R": R, "rows": rows_np, "bank": bank, "margin": m, "stats": stats, "info": info}
        if record_all:
            out["all_uc"], out["all_mc"] = allu, allm
        return out


# ---------------------------------------------------------------- self-test (CPU build + torch CPU)


def _replay_check(res, seeds, T, opp):
    """Replay of the batch with the reference kagcore.GameBatch (vendor engine + kagcore encode/execute) using the
    choices from the storage/record: encodings at every step == storage (features - after casting to bf16), final
    banks == rollout banks to the dollar. Returns counters."""
    sys.path.insert(0, os.path.join(ROOT, "engines", "kagcore", "build"))
    import kagcore

    n = len(seeds)
    gb = kagcore.GameBatch([int(s) for s in seeds], T)
    buf = {
        "tiles": np.empty((2 * n, 100, 30), np.float32),
        "units": np.empty((2 * n, 16, 16), np.float32),
        "unit_pos": np.empty((2 * n, 16), np.int64),
        "unit_mask": np.empty((2 * n, 16), bool),
        "glob": np.empty((2 * n, 77), np.float32),
        "tmask": np.empty((2 * n, 16, 700), np.uint8),
        "mmask": np.empty((2 * n, 489), bool),
    }
    seats2 = np.tile(np.array([0, 1], np.int32), n)
    games2 = np.repeat(np.arange(n, dtype=np.int32), 2)
    st, rows, NA = res["store"], res["rows"], res["NA"]
    cnt = {"steps": 0, "rows_cmp": 0, "rows_mismatch": 0, "choice_cmp": 0, "choice_mismatch": 0}
    for t in range(T):
        gb.encode_into(seats2, *[buf[k] for k in ENC], games=games2)
        for k in ENC:
            ref = torch.from_numpy(buf[k][rows])
            if k in FLOATS:
                ref = ref.to(torch.bfloat16)
            ours = st[k][t, :NA].cpu()
            cnt["rows_cmp"] += NA
            if not torch.equal(
                ours.view(torch.uint8) if ours.dtype == torch.bfloat16 else ours,
                ref.view(torch.uint8) if ref.dtype == torch.bfloat16 else ref,
            ):
                cnt["rows_mismatch"] += int(sum(not torch.equal(ours[j], ref[j]) for j in range(NA)))
        uc = res["all_uc"][t].cpu().numpy().astype(np.int32)
        mc = res["all_mc"][t].cpu().numpy().astype(np.int32)
        cnt["choice_cmp"] += NA
        cnt["choice_mismatch"] += int((st["uc"][t, :NA].cpu().numpy() != uc[rows]).any(axis=1).sum())
        cnt["choice_mismatch"] += int((st["mc"][t, :NA].cpu().numpy() != mc[rows]).any(axis=1).sum())
        gb.step(uc.reshape(n, 2, 16), mc.reshape(n, 2, 24))
        cnt["steps"] += 1
    cnt["bank_ref"] = gb.rewards().tolist()
    cnt["bank_mismatch"] = int((gb.rewards() != res["bank"]).sum())
    cnt["done_ref"] = gb.done.tolist()
    return cnt


def _logp_check(net, res, amp, T):
    """Recompute logp of the chosen actions with the same network from the storage (bf16 features, as in the appo
    learner) against the lpu/lpm recorded by the actor: maximum |difference| (in appo this is the PPO ratio on the
    first minibatch, ~1)."""
    st, NA = res["store"], res["NA"]
    worst_u = worst_m = worst_v = 0.0
    with torch.no_grad():
        for t in range(0, T, max(1, T // 8)):
            b = {k: st[k][t, :NA] for k in ENC}
            b["tmask"] = unpack(b["tmask"])
            with torch.autocast(**amp):
                lg, ml, v = net(b["tiles"], b["units"], b["unit_pos"], b["unit_mask"], b["glob"])
            uc, mc = st["uc"][t, :NA], st["mc"][t, :NA]
            lpu = unit_logp(lg, b["tmask"], b["unit_mask"], uc // S.NOPS, uc % S.NOPS)
            lpm = market_logp(ml, b["mmask"], mc)
            worst_u = max(worst_u, float((lpu - st["lpu"][t, :NA]).abs().max()))
            worst_m = max(worst_m, float((lpm - st["lpm"][t, :NA]).abs().max()))
            worst_v = max(worst_v, float((v.float() - st["val"][t, :NA]).abs().max()))
    return {"max_abs_dlogp_units": worst_u, "max_abs_dlogp_market": worst_m, "max_abs_dvalue": worst_v}


def selftest(B, T, chunk, d, layers, seed):
    """Self-play and vs-past rollouts of a small random network on the CPU build, checked by replay; returns report."""
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    net = Net(d=d, layers=layers, heads=4, dk=16, vtower=1).eval()
    opp = Net(d=d, layers=layers, heads=4, dk=16, vtower=1).eval()
    seeds = np.arange(B, dtype=np.uint64) * 7919 + np.uint64(seed)
    ro = Rollout(B, T=T, backend="cpu", chunk=chunk)
    report = {
        "B": B,
        "T": T,
        "chunk": chunk,
        "net": {"d": d, "layers": layers, "heads": 4, "dk": 16, "vtower": 1},
        "seeds": [int(s) for s in seeds],
    }
    # 1) self-play
    res = ro.run(net, seeds, record_all=True)
    st = res["store"]
    ref = make_store(1, 1, "cpu")
    fmt = all(st[k].dtype == ref[k].dtype and tuple(st[k].shape[2:]) == tuple(ref[k].shape[2:]) for k in ref)
    report["selfplay"] = {
        "NA": res["NA"],
        "store_format_matches_appo": bool(fmt and set(st) == set(ref)),
        "R": res["R"].tolist(),
        "stats": res["stats"],
        "info": res["info"],
        "bank": res["bank"].tolist(),
        "replay": _replay_check(res, seeds, T, None),
        "logp_recompute": _logp_check(net, res, ro.amp, T),
    }
    # 2) half of the games - past version (opp) on seat 1 - me
    og = np.arange(B) % 2 == 0
    me = (np.arange(B) // 2) % 2
    res2 = ro.run(net, seeds + np.uint64(1), opp_net=opp, opp_games=og, me_seat=me, record_all=True)
    report["vs_past"] = {
        "NA": res2["NA"],
        "rows": res2["rows"].tolist(),
        "stats": res2["stats"],
        "info": res2["info"],
        "bank": res2["bank"].tolist(),
        "replay": _replay_check(res2, seeds + np.uint64(1), T, opp),
    }
    ok = (
        report["selfplay"]["store_format_matches_appo"]
        and all(
            r["replay"]["rows_mismatch"] == 0
            and r["replay"]["bank_mismatch"] == 0
            and r["replay"]["choice_mismatch"] == 0
            for r in (report["selfplay"], report["vs_past"])
        )
        and res2["NA"] == 2 * B - int(og.sum())
    )
    report["ok"] = bool(ok)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=6, help="rows per forward (small - to test the chunking)")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=880000)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "rollout_selftest.json"))
    x = ap.parse_args()
    if not x.selftest:
        ap.error("run the module only with --selftest (for training - import Rollout, see README.md)")
    t0 = time.time()
    rep = selftest(x.B, x.T, x.chunk, x.d, x.layers, x.seed)
    rep["wall_sec"] = round(time.time() - t0, 1)
    rep["protocol"] = (
        "CPU build libkagrl_cpu.so + torch CPU (autocast bf16 on the CPU); network rl/model.Net with "
        "random initialization (torch.manual_seed); check: replay with kagcore.GameBatch "
        "using the choices from the rollout - encodings of every step == storage (features after bf16), banks to "
        "the dollar; logp recomputed with the same network from the storage"
    )
    os.makedirs(os.path.dirname(x.out), exist_ok=True)
    with open(x.out, "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    short = {k: v for k, v in rep.items() if k not in ("selfplay", "vs_past")}
    short["selfplay"] = {
        k: rep["selfplay"][k] for k in ("NA", "store_format_matches_appo", "stats", "info", "logp_recompute")
    }
    short["selfplay"]["replay"] = {k: v for k, v in rep["selfplay"]["replay"].items() if k != "done_ref"}
    short["vs_past"] = {k: rep["vs_past"][k] for k in ("NA", "stats", "info")}
    short["vs_past"]["replay"] = {k: v for k, v in rep["vs_past"]["replay"].items() if k != "done_ref"}
    print(json.dumps(short, indent=1, ensure_ascii=False))
    print("RESULT:", "OK" if rep["ok"] else "ERROR", "| json:", x.out)
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
