"""Asynchronous PPO (APPO): game collection (actor thread) runs concurrently with the update (main thread) on one GPU.

Iteration i: the actor plays games with the weights published after update i-2 while the trainer learns on the games of
i-1 (policy lag = 1 update). The PPO ratio is computed against the policy that actually played (its logp is stored
during collection), as in APPO/IMPALA with clipping. Iteration time = max(collection, update) instead of the sum
(rl/ppo.py).

Storage: the legal-action mask is bit-packed (8x less memory and CPU->GPU copying); features are stored in bf16 on the
GPU (under autocast the first Linear receives bf16 anyway, so nothing is lost). Two game buffers (the actor writes one,
the trainer reads the other), synchronized with CUDA events.

    torchrun --standalone --nproc_per_node NPROC -m rl.appo --init BC.pt --out runs/X/out --iters 300

entry_points/train_appo.py builds the full command from configs/appo_d7b.json. Everything not listed below (league,
reward, losses, continuation with --ref/--past) is as in rl/ppo.py. Outputs in --out: ppo.jsonl (one JSON line per
iteration), pool_state.jsonl, ckpt_NNNNN.pt (every --save-every iterations), ema_NNNNN.pt (with --ema),
anchor_NNNNN.pt (with --anchor-gate) and the version pool (--pool-dir, default OUT/pool).

Stability package (all flags are off by default, giving the previous behaviour):
  --anchor-gate G: moving anchor (Orbit Wars 1st place: KL + value distillation to the "previous best", replaced based
     on win rate). The KL anchor (ref) = the last accepted version; the league entry "best" = games against the anchor.
     Replacement happens when, over --anchor-window iterations (all games against the current anchor),
     P(vs anchor) >= G with >= --anchor-min-games games and P(code-bot field) >= field P at the previous replacement
     - --anchor-field-tol. The accepted weights are the ones that played the last iteration of the window; they are
     saved to OUT/anchor_<iter>.pt.
  --vd C: value distillation to the anchor: C * MSE(value, anchor value) (anchor values come from the same pass as the
     trunk cache).
  --margin-w-final / --margin-decay-iters: the weight of the margin reward term goes linearly to final (counted from
     --start-iter).
  --ema TAU: EMA of the weights after every optimizer step; OUT/ema_<iter>.pt next to ckpt (for evaluation).
  --past-greedy 1: "past" snapshots play argmax (Generals.io: past versions play argmax); the "best" anchor samples,
     like the learner, so that the gates compare equal modes.
  --tr-joint-mult M: stop the iteration's updates if the joint k3 KL(playing||current) over all action components
     (units + market heads: log-ratio = sum over components) on a minibatch > M * median of the means of past
     iterations.
  --pfsp FLOOR: PFSP (AlphaStar): code-bot opponent weight = base league weight * max(FLOOR, f(P)), f = 4P(1-P)
     (--pfsp-mode var) or (1-P)^2 (hard: more often the ones we lose to), P is the moving (0.7/0.3 over iterations) win
     rate against it; self/past/best keep their base weights. 0 = off.
  Log: jkl (joint KL(playing||current), naive estimate), jk3, jkl_ref (joint KL to the anchor), vdl, mw, anchor, pfsp.

GPU games (--gpu-games G, default 0 = off): on each rank, G additional self/past/best games run entirely on the GPU:
engine + encode + execute from engines/kagrl_cuda (kagrl, bit-exact with kagcore/kagsim), the policy is the same
act_net/snap_net/best_net. CPU workers then play ONLY against field code bots (drawn by code-bot weights), the GPU only
self/past/best (by their weights); the row share is set by G (the previous league mix ~ G = W*K * weight(self+past+best)
/ weight(code bots)). The GPU-game step is queued on a separate CUDA stream (lower priority than the actor) right after
the actions are sent to the workers, while the workers compute their step. GPU-game rows occupy storage columns
[NA_cpu, NA_cpu + NA_gpu).

Versioned "past" pool (always on): snapshots are files POOL/vNNNNN.pt (--pool-dir, default OUT/pool; every
--snap-every iterations + the initial weights), --past = static additions; source pools of other runs:
--pool-src DIR:weight:rule (e.g., the published versions of an exploiter). Version selection rule (--past-sample):
uniform | recent (δ-uniform, Bansal 2017: uniform over the latest --pool-recent fraction of the history) | quality
(OpenAI Five: p ~ softmax(q), win over version i: q_i -= --q-lr * P_i / (N p_i) once per iteration, a new version gets
max q) | pfsp (AlphaStar f_hard: (1-P)^--pfsp-pow, P = learner win rate against the version, unknown = 0.5; floor
--pool-floor) | latest. Version statistics (games, wins, P, q) are identical on all ranks, log OUT/pool_state.jsonl (a
restart with --start-iter picks up the last line).
Exploiter (--publish-gate P > 0; AlphaStar main exploiter): a run whose "past" is the latest version of the main run
(--pool-src MAIN/pool:1:latest --pool-own-w 0); a version is published to its own POOL when P(vs past) over
--publish-window iterations >= P (with >= --publish-min games) or after --reset-every iterations, then the run is reset
to the initial weights (Adam and lr warmup start over; the updates of the next two iterations are skipped: their games
were played by the old policy).

Privileged critic (--pcritic L, rl/model.py): the value sees the opponent's full observation (its encode from its own
seat: C++ kagcore always encodes both seats; in GPU games it is row 2g+1-s) and the opponent type. Requires
--kagcore 1.
Fresh pass (enabled by --pcritic / --vtrace / --upgo): the current network recomputes the values of all rows of the
iteration (and the current/playing log-ratio for V-trace) before the update, so the values for GAE are fresh rather
than the actor's lagged ones.
--vtrace 1: value targets and advantages via V-trace (IMPALA; rho/c = min(--vtrace-rho/--vtrace-c, pi/mu), jointly over
action components). --upgo W: advantage + W * UPGO (AlphaStar; Toad Brigade Lux S1): G_t = r_t +
gamma * (G_{t+1} if r_{t+1} + gamma V_{t+2} >= V_{t+1}, else V_{t+1}), A = G_t - V_t.
"""

import argparse
import copy
import glob
import json
import math
import os
import queue
import random
import sys
import threading
import time
import zlib

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from . import encode as E
from . import spec as S
from .model import HEAD_SIZES, NOID, Net, market_logp, unit_logp
from .ppo import ENC, LEAGUE, ROOT, market_terms, sample, unit_terms

NOT_FIELD = ("self", "past", "best")  # not field code bots: P_field and the anchor gates are computed without them
OENC = ("tiles", "units", "unit_pos", "unit_mask", "glob")  # opponent observation for the critic (st["o_" + k])
# index of the opponent's open-land-quadrant flags in glob (encode.py: 12 common, shed, seeds, prices, market inventory,
# shops + their count, own quadrants, opponent quadrants), used by --fast-gate
OQ0 = 12 + len(E.ITEMS) + len(S.CROPS) + 2 * len(S.PRODUCTS) + len(S.SHOPS) + 1 + len(S.QUADS)


def opp_id(name):
    """Opponent type for the critic: self/past/best = 0/1/2, field code bot = crc32(folder name) in [3, NOID)."""
    return NOT_FIELD.index(name) if name in NOT_FIELD else 3 + zlib.crc32(name.encode()) % (NOID - 3)


def okey(n):
    """Opponent key (statistics, PFSP, critic): self/past/best; field code bot = folder name; tape pool "tapes:FILE" ->
    tapes_<file name without extension>."""
    if n in NOT_FIELD:
        return n
    if n.startswith("tapes:"):
        return "tapes_" + os.path.basename(n[6:]).split(".")[0]
    return os.path.basename(os.path.dirname(n))


def load_tapes(path):
    """Tape-world pool: JSON {"worlds": [{"world": world file, "opp_seat": tape seat, "seed", "shops"[, "ref"]}]}.

    Game with a tape: the record's seed and shops, opponent = the record's actions in its seat, the learner is in the
    other seat. "ref" (optional) = the record's bank in the learner's seat: reference for the reward relative to the
    record (--rec-w); "ref_opp" = the record's bank in the tape's seat (for --rec-mode margin)."""
    ws = json.load(open(path))["worlds"]
    assert ws and all(("seed" in w and "shops" in w) for w in ws), f"{path}: entries need seed and shops"
    return [
        (
            w["world"],
            int(w["opp_seat"]),
            int(w["seed"]),
            list(w["shops"]),
            float(w["ref"]) if w.get("ref") is not None else None,
            float(w["ref_opp"]) if w.get("ref_opp") is not None else None,
        )
        for w in ws
    ]


class Pool:
    """Versioned pool of "past" opponent networks. Groups: (name, weight, rule, directory or None, static files).
    Per-version statistics (key = path): n games, w learner wins, p (moving win rate), q (quality)."""

    # min_age: listing() skips files younger than this many seconds (pool files are written atomically,
    # see save_version)
    def __init__(self, groups, recent, q_lr, pfsp_pow, floor, min_age=0.0):
        self.groups, self.recent, self.q_lr, self.pow, self.floor, self.min_age = (
            groups,
            recent,
            q_lr,
            pfsp_pow,
            floor,
            min_age,
        )
        self.stat = {}

    def listing(self):
        out, now = [], time.time()
        for name, w, rule, d, files in self.groups:
            fs = list(files)
            if d:
                fs += sorted(f for f in glob.glob(os.path.join(d, "*.pt")) if now - os.path.getmtime(f) > self.min_age)
            out.append((name, w, rule, fs))
        return out

    def weights(self, rule, fs):
        if rule == "quality":
            known = [self.stat[f]["q"] for f in fs if f in self.stat and self.stat[f]["q"] is not None]
            q0 = max(known) if known else 0.0
            q = [self.stat[f]["q"] if f in self.stat and self.stat[f]["q"] is not None else q0 for f in fs]
            m = max(q)
            return [math.exp(v - m) for v in q]
        if rule == "pfsp":
            return [
                max(
                    self.floor,
                    (1.0 - (self.stat[f]["p"] if f in self.stat and self.stat[f]["p"] is not None else 0.5))
                    ** self.pow,
                )
                for f in fs
            ]
        if rule == "recent":
            k = max(1, math.ceil(len(fs) * self.recent))
            return [0.0] * (len(fs) - k) + [1.0] * k
        if rule == "latest":
            return [0.0] * (len(fs) - 1) + [1.0]
        return [1.0] * len(fs)

    def pick(self, rng, listing):
        gs = [g for g in listing if g[1] > 0 and g[3]]
        if not gs:
            raise RuntimeError(f"past pool is empty: {[(g[0], g[1], len(g[3])) for g in listing]}")
        name, _, rule, fs = rng.choices(gs, [g[1] for g in gs])[0]
        w = self.weights(rule, fs)
        i = rng.choices(range(len(fs)), w)[0]
        return fs[i], w[i] / sum(w), len(fs), name

    def update(self, per):
        """per: {key: [games, wins, selection p, group N]}, summed over ranks for the iteration (same on all ranks).

        quality: OpenAI Five step once per iteration (not per game: a version plays many games per iteration):
        q_i -= q_lr * P_i / (N p_i), P_i = win rate over the iteration; a new version starts at max q (before this
        iteration's steps)."""
        known = [v["q"] for v in self.stat.values() if v["q"] is not None]
        q0 = max(known) if known else 0.0
        for k, (n, w, p, N) in per.items():
            if not n:
                continue
            st = self.stat.setdefault(k, {"n": 0, "w": 0.0, "p": None, "q": None})
            if st["q"] is None:
                st["q"] = q0
            st["n"] += n
            st["w"] += w
            st["p"] = w / n if st["p"] is None else 0.7 * st["p"] + 0.3 * w / n
            st["q"] -= self.q_lr * (w / n) / max(1e-6, N * p)


H = len(HEAD_SIZES)
# --land-inject: market heads "land" and "group order", template "sells, then land", own-quadrant flags in glob
HL = [k for k, _, _ in S.MARKET_HEADS].index("BUY_LAND")
HO = [k for k, _, _ in S.MARKET_HEADS].index("ORDER")
ORD_SL = S.ORDER_TEMPLATES.index("SLPHDA")
QOWN0 = OQ0 - len(S.QUADS)
LAND4 = 4000.0  # price of the 4th land quadrant (engine 1.32.7 LAND_PRICES)
PK = S.NT * S.NOPS // 8  # bytes of packed mask per unit
FLOATS = ("tiles", "units", "glob")


def buffers(n):
    return {
        "tiles": torch.zeros(n, S.NT, E.FT),
        "units": torch.zeros(n, S.MAX_UNITS, E.FU),
        "unit_pos": torch.zeros(n, S.MAX_UNITS, dtype=torch.long),
        "unit_mask": torch.zeros(n, S.MAX_UNITS, dtype=torch.bool),
        "glob": torch.zeros(n, E.FG),
        "tmask": torch.zeros(n, S.MAX_UNITS, PK, dtype=torch.uint8),
        "mmask": torch.zeros(n, E.MB, dtype=torch.bool),
        "uc": torch.zeros(n, S.MAX_UNITS, dtype=torch.long),
        "mc": torch.zeros(n, H, dtype=torch.long),
    }


def unpack(p):
    """[..., 16, PK] uint8 -> [..., 16, NT, NOPS] bool (np.packbits, most significant bit first)."""
    sh = torch.arange(7, -1, -1, device=p.device, dtype=torch.uint8)
    return ((p.unsqueeze(-1) >> sh) & 1).reshape(*p.shape[:-1], S.NT, S.NOPS).bool()


# ---------------------------------------------------------------- worker (CPU, its own K games)


def _worker(off, K, T, bufs, conn, use_kc=0):
    os.chdir(ROOT)
    torch.set_num_threads(1)
    import kagsim

    from . import act as A
    from .record import _base_config, _load, call_agent

    if use_kc:
        # C++ encode/execute (engines/kagcore): bit-exact with rl/encode.py and rl/act.py
        import kagcore
    nb = {k: v.numpy() for k, v in bufs.items()}
    cfg0 = dict(_base_config())
    tcache = {}  # world file -> actions of both seats (tapes)
    while True:
        specs = conn.recv()
        if specs is None:
            return
        fns, cfgs, tapes = [], [], []
        for sp in specs:
            c = dict(cfg0)
            c["seed"] = sp["seed"]
            cfgs.append(c)
            fns.append(_load(sp["opp"]) if sp["opp"] else None)
            tp = None
            if sp.get("tape"):
                wf, ts = sp["tape"]
                if wf not in tcache:
                    import gzip

                    if len(tcache) > 256:
                        tcache.clear()
                    with gzip.open(wf, "rt") as fh:
                        tcache[wf] = json.load(fh)["actions"]
                tp = tcache[wf][ts]
            tapes.append(tp)
        te = to = tx = 0.0
        if use_kc:
            nsp = len(specs)
            n2 = 2 * nsp
            v = {k: nb[k][off : off + n2] for k in ENC + ("uc", "mc")}
            per = any(sp.get("shops") is not None for sp in specs)
            if per:  # the record's shops differ per game: one GameBatch per game
                gbs = [kagcore.GameBatch([sp["seed"]], T, sp.get("shops")) for sp in specs]
                s01, g00 = np.array([0, 1], np.int32), np.zeros(2, np.int32)
                vk = [{k: v[k][2 * j : 2 * j + 2] for k in v} for j in range(nsp)]
            else:
                gb = kagcore.GameBatch([sp["seed"] for sp in specs], T)
                seats2 = np.tile(np.array([0, 1], np.int32), nsp)
                games2 = np.repeat(np.arange(nsp, dtype=np.int32), 2)
            for t in range(T):
                t0 = time.perf_counter()
                if per:
                    for j, g in enumerate(gbs):
                        u = vk[j]
                        g.encode_into(
                            s01,
                            u["tiles"],
                            u["units"],
                            u["unit_pos"],
                            u["unit_mask"],
                            u["glob"],
                            u["tmask"],
                            u["mmask"],
                            games=g00,
                        )
                else:
                    gb.encode_into(
                        seats2,
                        v["tiles"],
                        v["units"],
                        v["unit_pos"],
                        v["unit_mask"],
                        v["glob"],
                        v["tmask"],
                        v["mmask"],
                        games=games2,
                    )
                conn.send(1)
                t1 = time.perf_counter()
                opp_act = [None] * nsp
                for k, sp in enumerate(specs):
                    if sp["opp"] or tapes[k] is not None:
                        me = sp["seats"][0]
                        if tapes[k] is not None:
                            a = tapes[k][t] if t < len(tapes[k]) and tapes[k][t] is not None else {}
                        else:
                            try:
                                ob = gbs[k].observe(0, 1 - me) if per else gb.observe(k, 1 - me)
                                a = call_agent(fns[k], ob, cfgs[k], sp["opp"]) or {}
                            # a crashed opponent = PASS, as on the server before ERROR
                            except Exception:  # noqa: BLE001
                                a = {}
                        opp_act[k] = (a, None) if me == 1 else (None, a)
                t2 = time.perf_counter()
                conn.recv()
                t3 = time.perf_counter()
                if per:
                    for j, g in enumerate(gbs):
                        g.step(vk[j]["uc"].reshape(1, 2, S.MAX_UNITS), vk[j]["mc"].reshape(1, 2, H), [opp_act[j]])
                else:
                    gb.step(v["uc"].reshape(nsp, 2, S.MAX_UNITS), v["mc"].reshape(nsp, 2, H), opp_act)
                t4 = time.perf_counter()
                te += t1 - t0
                to += t2 - t1
                tx += t4 - t3
            rw = [g.rewards().tolist()[0] for g in gbs] if per else gb.rewards().tolist()
            conn.send((rw, (te, to, tx)))
            continue
        assert not any(tp is not None for tp in tapes), "tape opponents require --kagcore 1"
        games = [kagsim.Game(sp["seed"], T) for sp in specs]
        obs = [[None, None] for _ in specs]
        for _ in range(T):
            t0 = time.perf_counter()
            for k, sp in enumerate(specs):
                for s in sp["seats"]:
                    o = games[k].observe(s)
                    obs[k][s] = o
                    e = E.encode(o)
                    slot = off + 2 * k + s
                    for key in ENC:
                        if key == "tmask":
                            nb["tmask"][slot] = np.packbits(e["tmask"].reshape(S.MAX_UNITS, -1), axis=-1)
                        else:
                            nb[key][slot] = e[key]
            conn.send(1)
            t1 = time.perf_counter()
            opp_act = [None] * len(specs)
            for k, sp in enumerate(specs):
                if sp["opp"]:
                    try:
                        opp_act[k] = call_agent(fns[k], games[k].observe(1 - sp["seats"][0]), cfgs[k], sp["opp"]) or {}
                    # a crashed opponent = PASS, as on the server before ERROR
                    except Exception:  # noqa: BLE001
                        opp_act[k] = {}
            t2 = time.perf_counter()
            conn.recv()
            t3 = time.perf_counter()
            for k, sp in enumerate(specs):
                acts = [{}, {}]
                for s in sp["seats"]:
                    slot = off + 2 * k + s
                    uc = nb["uc"][slot]
                    try:
                        acts[s] = A.execute(
                            obs[k][s],
                            [(int(v) // S.NOPS, int(v) % S.NOPS) for v in uc],
                            [int(v) for v in nb["mc"][slot]],
                        )
                    except Exception:  # noqa: BLE001
                        acts[s] = {}
                if sp["opp"]:
                    acts[1 - sp["seats"][0]] = opp_act[k]
                games[k].step(acts[0], acts[1])
            t4 = time.perf_counter()
            te += t1 - t0
            to += t2 - t1
            tx += t4 - t3
        conn.send(([[games[k].reward(0), games[k].reward(1)] for k in range(len(specs))], (te, to, tx)))


# ---------------------------------------------------------------- training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--start-iter", type=int, default=0)
    ap.add_argument("--workers", type=int, default=26)
    ap.add_argument("--games", type=int, default=8)  # games per worker
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--seed0", type=int, default=90_000_000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lr-warmup", type=int, default=0)  # optimizer steps: linear warmup of the policy lr from 0
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatches", type=int, default=64)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--ent", type=float, default=1e-3)
    ap.add_argument("--kl-ref", type=float, default=0.02)
    ap.add_argument("--kl-ref-final", type=float, default=None)
    ap.add_argument("--kl-decay-iters", type=int, default=200)
    # --late-step S > 0: policy gradient only on steps >= S (end of the game); on steps < S only KL to --ref with
    # weight --kl-early
    ap.add_argument("--late-step", type=int, default=0)
    ap.add_argument("--kl-early", type=float, default=1.0)
    # "fast" opponent (--fast-gate 1): policy gradient only in games where the opponent has >= --fast-quads open land
    # quadrants before step --fast-by (216 = start of day 9), from the trigger step until --fast-until; outside the
    # window only KL to --ref with weight --kl-early
    ap.add_argument("--fast-gate", type=int, default=0)
    ap.add_argument("--fast-quads", type=int, default=3)
    ap.add_argument("--fast-by", type=int, default=216)
    ap.add_argument("--fast-until", type=int, default=432)
    ap.add_argument("--margin-w", type=float, default=0.3)
    ap.add_argument("--margin-scale", type=float, default=30000.0)
    # weight of the own-bank reward term (clip(bank/scale, 0, 2))
    ap.add_argument("--bank-w", type=float, default=0.0)
    ap.add_argument("--bank-scale", type=float, default=100000.0)
    # tape worlds with a "ref" reference (the record's bank in our seat):
    # R = (1-w)·r(bank − opponent bank) + w·r(bank − reference), r(d) = sign(d) + mw·clip(d/margin_scale):
    # "beat the top player's record in its seat" does not saturate once the win is secured
    ap.add_argument("--rec-w", type=float, default=0.0)
    # bank: d = bank − ref; margin: d = (bank − opponent bank) − (ref − ref_opp), i.e. margin relative to the record's
    # margin (in our runs the bank variant lost pressure on the opponent's market: higher own bank, lower margin)
    ap.add_argument("--rec-mode", default="bank", choices=["bank", "margin"])
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--width", dest="d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--league", default="")
    ap.add_argument("--snap-every", type=int, default=10)
    ap.add_argument("--vf-warmup", type=int, default=0)
    ap.add_argument("--vf-lam", type=float, default=1.0)
    ap.add_argument("--vf-lr", type=float, default=1e-4)
    ap.add_argument("--vtower", type=int, default=2)
    ap.add_argument("--ref", default="")
    # "BUY_LAND,BUY_ANIMAL": KL to --ref only on market heads with these prefixes, units without KL (farm-building
    # teacher, kickstarting)
    ap.add_argument("--ref-heads", default="")
    # 1: "best" anchor = init rather than --ref (when --ref is a teacher)
    ap.add_argument("--best-init", type=int, default=0)
    ap.add_argument("--past", default="")
    # 1: no overlap (the actor waits for the update to finish), as a control
    ap.add_argument("--sync", type=int, default=0)
    ap.add_argument("--kagcore", type=int, default=1)  # 1: workers on C++ kagcore (encode/execute/step), 0: Python
    # 1: anchor trunk once per iteration, only its heads in each step
    ap.add_argument("--ref-cache", type=int, default=1)
    ap.add_argument("--compile", type=int, default=1)  # 1: torch.compile of the trainer's forward+losses
    ap.add_argument("--dual-clip", type=float, default=0.0)  # >1: dual-clip PPO (JueWu: c=3), 0 = off
    # >0: stop the iteration's updates if KL(playing||current) of any head (k3 over the chosen actions) > threshold
    ap.add_argument("--tr-kl", type=float, default=0.0)
    ap.add_argument("--anchor-gate", type=float, default=0.0)  # >0: moving anchor, threshold on P(vs anchor)
    ap.add_argument("--anchor-window", type=int, default=3)
    ap.add_argument("--anchor-min-games", type=int, default=2000)
    ap.add_argument("--anchor-field-tol", type=float, default=0.03)
    # field P of the initial anchor (for the first replacement)
    ap.add_argument("--anchor-field0", type=float, default=0.0)
    ap.add_argument("--vd", type=float, default=0.0)  # weight of value distillation to the anchor
    ap.add_argument("--margin-w-final", type=float, default=None)
    ap.add_argument("--margin-decay-iters", type=int, default=100)
    ap.add_argument("--ema", type=float, default=0.0)  # 0 = off; otherwise tau (0.999)
    ap.add_argument("--past-greedy", type=int, default=0)
    ap.add_argument("--tr-joint-mult", type=float, default=0.0)
    ap.add_argument("--tr-joint-warm", type=int, default=5)  # iterations of history before the joint stop is enabled
    ap.add_argument("--pfsp", type=float, default=0.0)
    ap.add_argument("--pfsp-mode", default="var")  # var: 4P(1-P) (AlphaStar f_var) | hard: (1-P)^2 (f_hard)
    ap.add_argument("--gpu-games", type=int, default=0)  # >0: self/past/best games per rank on the kagrl GPU engine
    ap.add_argument("--kagrl-lib", default="")  # libkagrl_cuda.so (default engines/kagrl_cuda/build/)
    ap.add_argument("--pool-dir", default="")  # where to write own versions (default OUT/pool)
    # weight of the own group in version selection (0 = do not play against itself)
    ap.add_argument("--pool-own-w", type=float, default=1.0)
    ap.add_argument("--pool-src", default="")  # pools of other runs: DIR:weight:rule[,...]
    ap.add_argument("--past-sample", default="uniform")  # uniform | recent | quality | pfsp | latest
    ap.add_argument("--pool-recent", type=float, default=0.5)  # recent: fraction of the latest versions
    ap.add_argument("--q-lr", type=float, default=0.1)  # quality: step per iteration (see Pool.update)
    ap.add_argument("--pfsp-pow", type=float, default=2.0)  # pfsp: f_hard (1-P)^pow (AlphaStar p=2)
    ap.add_argument("--pool-floor", type=float, default=0.05)  # pfsp: minimum version weight
    ap.add_argument("--publish-gate", type=float, default=0.0)  # >0: exploiter mode (publish + reset)
    ap.add_argument("--publish-window", type=int, default=3)
    ap.add_argument("--publish-min", type=int, default=500)
    ap.add_argument("--reset-every", type=int, default=0)  # exploiter: publish + reset at least this often (iterations)
    ap.add_argument("--pcritic", type=int, default=0)  # layers of the critic's opponent-observation encoder (0 = off)
    ap.add_argument("--vtrace", type=int, default=0)
    ap.add_argument("--vtrace-rho", type=float, default=1.0)
    ap.add_argument("--vtrace-c", type=float, default=1.0)
    ap.add_argument("--upgo", type=float, default=0.0)
    # --land-inject S >= 0: environment event: during steps [S, S + --land-window) the learner gets the 4th land
    # quadrant bought (when 3 are open and money >= 4000)
    ap.add_argument("--land-inject", type=int, default=-1)
    ap.add_argument("--land-window", type=int, default=72)
    x = ap.parse_args()
    assert not x.pcritic or x.kagcore, "--pcritic: only kagcore encodes the opponent observation (both seats)"
    PC = bool(x.pcritic)
    CUR = bool(x.pcritic or x.vtrace or x.upgo)
    assert not x.vd or x.ref_cache, "--vd computes the anchor value in the trunk-cache pass (--ref-cache 1)"

    # RL_NO_CUDNN_SDP=1: attention without cuDNN, for machines whose system cuDNN differs from the one torch ships
    # (CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH)
    if os.environ.get("RL_NO_CUDNN_SDP"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dev = torch.device("cuda")
    torch.manual_seed(4321 + rank)
    torch.set_num_threads(4)
    league = (
        (json.load(open(x.league[1:])) if x.league.startswith("@") else json.loads(x.league)) if x.league else LEAGUE
    )
    tpools = {n: load_tapes(n[6:]) for n, _ in league if n.startswith("tapes:")}  # tape-world pools (CPU games)
    assert not tpools or x.kagcore, "tape opponents require --kagcore 1"
    assert x.rec_mode == "bank" or all(t[5] is not None for p_ in tpools.values() for t in p_ if t[4] is not None), (
        "--rec-mode margin: tape worlds with ref need ref_opp"
    )

    ck = None
    if x.init:
        ck = torch.load(x.init, map_location="cpu", weights_only=False)
        d, layers, vt = ck.get("d", x.d), ck.get("layers", x.layers), ck.get("vtower", x.vtower)
        net = Net(d=d, layers=layers, vtower=vt, pcritic=x.pcritic)
        miss, unexp = net.load_state_dict(ck["model"], strict=False)
        # the opponent critic is not needed without --pcritic
        unexp = [k for k in unexp if not (k.startswith("vopp.") and not x.pcritic)]
        assert not unexp and all(k.startswith(("venc.", "vnorm.", "vopp.")) for k in miss), (miss, unexp)
    else:
        d, layers, vt = x.d, x.layers, x.vtower
        net = Net(d=d, layers=layers, vtower=vt, pcritic=x.pcritic)
    net = net.to(dev)
    if x.ref:
        ref = Net(d=d, layers=layers, vtower=vt, pcritic=x.pcritic)
        ref.load_state_dict(torch.load(x.ref, map_location="cpu", weights_only=False)["model"], strict=False)
        ref = ref.to(dev).eval()
    else:
        ref = copy.deepcopy(net).eval()
    act_net = copy.deepcopy(net).eval()
    snap_net = copy.deepcopy(net).eval()
    best_net = copy.deepcopy(net if x.best_init else ref).eval()  # anchor for "best" games in the actor thread
    if x.ref_heads:  # KL to the teacher only on the selected market heads
        pre_ = tuple(q for q in x.ref_heads.split(",") if q)
        hn_ = [f"{g}:{i_}" if i_ is not None else g for g, i_, _ in S.MARKET_HEADS]
        khw = torch.tensor([1.0 if n_.startswith(pre_) else 0.0 for n_ in hn_], device=dev)
        kwu = 0.0
        assert float(khw.sum()) > 0 and not x.vd, "--ref-heads: no heads, or --vd is on"
        print("KL to the teacher on heads:", [n_ for n_ in hn_ if n_.startswith(pre_)], flush=True)
    else:
        khw, kwu = None, 1.0
    for m_ in (ref, act_net, snap_net, best_net):
        for p in m_.parameters():
            p.requires_grad_(False)
    meta = {"d": d, "layers": layers, "vtower": vt, "pcritic": x.pcritic}
    pool_dir = x.pool_dir or os.path.join(x.out, "pool")
    groups = [("own", x.pool_own_w, x.past_sample, pool_dir, [os.path.abspath(q) for q in x.past.split(",") if q])]
    for i_, sp_ in enumerate(q for q in x.pool_src.split(",") if q):
        d_, w_, r_ = sp_.split(":")
        groups.append((f"src{i_}", float(w_), r_, os.path.abspath(d_), []))
    pool = Pool(groups, x.pool_recent, x.q_lr, x.pfsp_pow, x.pool_floor)
    init_sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()} if x.publish_gate > 0 else None

    def save_version(sd_, step_, kind_):
        """Version into the pool: a hidden temporary file, then a rename (readers never see a partial file)."""
        os.makedirs(pool_dir, exist_ok=True)
        fn = os.path.join(pool_dir, f"v{step_:05d}.pt")
        if os.path.exists(fn):
            return fn
        tmp = os.path.join(pool_dir, f".v{step_:05d}.{os.getpid()}.tmp")
        torch.save(
            {
                "model": {k: v.detach().cpu() for k, v in sd_.items()},
                **meta,
                "step": step_,
                "kind": kind_,
                "run": os.path.abspath(x.out),
            },
            tmp,
        )
        os.rename(tmp, fn)
        return fn

    model = torch.nn.parallel.DistributedDataParallel(net, device_ids=[dev.index])
    VPRE = ("venc.", "vnorm.", "value.", "vopp.")
    vpar = [p for n, p in net.named_parameters() if n.startswith(VPRE)]
    ppar = [p for n, p in net.named_parameters() if not n.startswith(VPRE)]
    opt = torch.optim.AdamW(
        [{"params": ppar, "lr": x.lr}, {"params": vpar, "lr": x.vf_lr}], weight_decay=0.0, betas=(0.9, 0.99), eps=1e-6
    )
    if ck is not None and "opt" in ck:
        opt.load_state_dict(ck["opt"])
        opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = x.lr, x.vf_lr
    amp = dict(device_type="cuda", dtype=torch.bfloat16)

    W, K, T = x.workers, x.games, x.T
    NS = W * K * 2
    G = x.gpu_games
    NSG = NS + 2 * G  # storage columns: CPU-worker seats + up to 2G GPU-game seats
    bufs = buffers(NS)
    for v in bufs.values():
        v.share_memory_()
    ctx = mp.get_context("spawn")
    conns, procs = [], []
    for w in range(W):
        a, b = ctx.Pipe()
        p = ctx.Process(target=_worker, args=(w * 2 * K, K, T, bufs, b, x.kagcore), daemon=True)
        p.start()
        conns.append(a)
        procs.append(p)

    def make_store():
        st = {
            k: torch.zeros((T, NSG) + tuple(v.shape[1:]), dtype=torch.bfloat16 if k in FLOATS else v.dtype, device=dev)
            for k, v in bufs.items()
        }
        st["lpu"] = torch.zeros(T, NSG, S.MAX_UNITS, device=dev)
        st["lpm"] = torch.zeros(T, NSG, H, device=dev)
        st["val"] = torch.zeros(T, NSG, device=dev)
        if PC:  # opponent observation of every row + its type (per iteration)
            for k in OENC:
                v = bufs[k]
                st["o_" + k] = torch.zeros(
                    (T, NSG) + tuple(v.shape[1:]), dtype=torch.bfloat16 if k in FLOATS else v.dtype, device=dev
                )
            st["oid"] = torch.zeros(NSG, dtype=torch.long, device=dev)
        return st

    stores = [make_store(), make_store()]
    if G:  # GPU engine: G games, observation row 2g+s = seat s of game g
        sys.path.insert(0, os.path.join(ROOT, "engines", "kagrl_cuda"))
        from kagrl import RLBatch

        gdev = torch.cuda.current_device()
        grb = RLBatch(G, T, backend="gpu", device=gdev, lib_path=x.kagrl_lib or None)
        gobs = grb.alloc_obs(kind="torch", device=dev)
        guc = torch.zeros(2 * G, S.MAX_UNITS, dtype=torch.long, device=dev)
        gmc = torch.zeros(2 * G, H, dtype=torch.long, device=dev)
        gbank = torch.zeros(G, 2, dtype=torch.float64, device=dev)
        gstream = torch.cuda.Stream(priority=0)  # lower than the actor (-1): GPU games fill the gaps
    if rank == 0:
        os.makedirs(x.out, exist_ok=True)
        log = open(os.path.join(x.out, "ppo.jsonl"), "a")
        plog = open(os.path.join(x.out, "pool_state.jsonl"), "a")
        if x.pool_own_w > 0 and x.publish_gate <= 0:
            save_version(net.state_dict(), x.start_iter, "init")
    sp_path = os.path.join(x.out, "pool_state.jsonl")
    if x.start_iter > 0 and os.path.exists(sp_path):  # restart: version statistics = last line of the log
        lines = [line for line in open(sp_path) if line.strip()]
        if lines:
            pool.stat = json.loads(lines[-1]).get("stat", {})
    dist.barrier()
    if rank == 0:
        print(
            f"APPO: ranks {world}, workers {W}x{K} games per rank, slots {NS}, T {T}, d {d} layers {layers} "
            f"vtower {vt} pcritic {x.pcritic}, sync {x.sync}; pool {[(g[0], g[1], g[2], g[3]) for g in groups]}, "
            f"vtrace {x.vtrace}, upgo {x.upgo}, publish {x.publish_gate}",
            flush=True,
        )

    names = [n for n, _ in league]
    wts = [w_ for _, w_ in league]
    lw = {"w": list(wts)}  # current draw weights (PFSP changes them between iterations)
    pkey = [okey(n) for n in names]
    prate = {}  # PFSP: moving win rate against a code bot
    assert ("best" in names) == (x.anchor_gate > 0), "league entry 'best' is required exactly when --anchor-gate > 0"
    anchor = {"w": None, "ver": 0}  # anchor published to the actor (weights+event), version
    ema = [p.detach().clone() for p in net.parameters()] if x.ema else None
    its = range(x.start_iter, x.start_iter + x.iters)
    astream = torch.cuda.Stream(priority=-1)  # actor: high priority (small kernels do not wait for the update)
    ready = queue.Queue()  # actor -> trainer: finished iteration
    free = [threading.Semaphore(1), threading.Semaphore(1)]
    done_ev = [None, None]  # event "trainer finished reading the buffer"
    pub = {"w": None}  # published weights (+ event)
    pub_lock = threading.Lock()
    upd_done = threading.Semaphore(0)  # for --sync 1
    err = []

    def act_compute(net_, b):
        with torch.autocast(**amp):
            lg, ml, v = net_(b["tiles"], b["units"], b["unit_pos"], b["unit_mask"], b["glob"])
        uc, mc = sample(lg, ml, b)
        lpu = unit_logp(lg, b["tmask"], b["unit_mask"], uc // S.NOPS, uc % S.NOPS)
        lpm = market_logp(ml, b["mmask"], mc)
        return uc, mc, lpu, lpm, v

    def past_compute(net_, b, greedy):
        with torch.autocast(**amp):
            plg, pml, _ = net_(b["tiles"], b["units"], b["unit_pos"], b["unit_mask"], b["glob"], value=False)
        return sample(plg, pml, b, greedy=greedy)

    # "past" plays argmax with --past-greedy; "best" always samples, like the learner (the anchor gates compare equal
    # modes)
    PG = bool(x.past_greedy)

    def safe(fn, eager, *a):
        """Compiled call; if the main thread is compiling at that moment (torch does not allow the call), run eager."""
        try:
            return fn(*a)
        except RuntimeError as e:
            if "symbolically trace a dynamo-optimized" not in str(e):
                raise
            return eager(*a)

    def bucket(n, buckets):
        for q in buckets:
            if n <= q:
                return q
        return n

    def brange(n, p, extra=0, step=32):
        """Shapes for a seat count of extra + Binomial(n, p) within ±4 sigma, as multiples of step."""
        m_, s_ = n * p, (n * p * (1 - p)) ** 0.5
        return sorted(
            {
                -(-(extra + q) // step) * step
                for q in range(max(0, int(m_ - 4 * s_)), int(m_ + 4 * s_) + 2)
                if extra + q > 0
            }
        )

    nfw = {n: w_ for n, w_ in zip(names, wts) if n in NOT_FIELD}
    GBUCKETS, GPBUCKETS, GBBUCKETS = [], [], []
    if G:
        assert 0 < sum(nfw.values()) < sum(wts), (
            "--gpu-games: the league needs both self/past/best (GPU) and code bots (CPU)"
        )
        ps_, pp_, pb_ = (nfw.get(n, 0) / sum(nfw.values()) for n in ("self", "past", "best"))
        # CPU: code bots only, one learner seat per game; GPU: G + number of self games, past/best binomial
        ABUCKETS = [W * K] if x.compile else []
        PBUCKETS = []
        if x.compile:
            GBUCKETS = brange(G, ps_, G)
            GPBUCKETS = brange(G, pp_) if pp_ > 0 else []
            GBBUCKETS = brange(G, pb_) if pb_ > 0 else []
        if rank == 0:
            print(
                f"GPU games: {G} per rank (self {ps_:.2f} / past {pp_:.2f} / best {pb_:.2f}), CPU games: {W * K} "
                f"with code bots; GPU row share ~{(G * (1 + ps_)) / (G * (1 + ps_) + W * K):.2f}",
                flush=True,
            )
    else:
        # actor shapes: number of learner seats ~ 208 + number of self games (binomial ~30±5), past seats ~22±5
        ABUCKETS = [q for q in (224, 256, 288) if q <= NS] if x.compile else []
        PBUCKETS = [32, 48] if x.compile else []

    def zb(q):  # zero batch of shape q (for compilation)
        b = {k: torch.zeros((q,) + tuple(bufs[k].shape[1:]), dtype=bufs[k].dtype, device=dev) for k in ENC}
        b["tmask"] = unpack(b["tmask"])
        return b

    act_c, past_c = act_compute, past_compute
    if x.compile:  # compile here, in the main thread: the actor thread only calls
        torch._dynamo.config.cache_size_limit = 64
        if hasattr(torch._dynamo.config, "recompile_limit"):
            torch._dynamo.config.recompile_limit = 64
        act_c = torch.compile(act_compute, dynamic=False)
        past_c = torch.compile(past_compute, dynamic=False)
        tc = time.time()
        with torch.no_grad():
            for q in sorted(set(ABUCKETS) | set(GBUCKETS)):
                act_c(act_net, zb(q))
            for q in sorted(set(PBUCKETS) | set(GPBUCKETS)):
                past_c(snap_net, zb(q), PG)
            if "best" in names:  # a different module = its own graph variant, also before the actor starts
                for q in sorted(set(PBUCKETS) | set(GBBUCKETS)):
                    past_c(best_net, zb(q), False)
        torch.cuda.synchronize()
        if rank == 0:
            print(
                f"actor compiled: shapes {ABUCKETS} + past {PBUCKETS}; GPU {GBUCKETS} + past {GPBUCKETS} + best "
                f"{GBBUCKETS}, {time.time() - tc:.0f} s",
                flush=True,
            )

    def land_inj(mc, g, t, cnt):
        """--land-inject: executed copy of the learner seats' orders with a land purchase and the template "sells, then
        land" (as in export_agent --land-steps). The actions stored for training and their logp are unchanged: the
        purchase is an environment event."""
        if x.land_inject < 0 or not (x.land_inject <= t < x.land_inject + x.land_window) or not len(mc):
            return mc
        g = g[: len(mc)].float()
        q = g[:, QOWN0 : QOWN0 + 4].sum(1)
        ok = (q > 2.5) & (q < 3.5) & (g[:, 5] * 1e5 >= LAND4 - 0.5)
        cnt.add_(ok.sum())
        m = mc.clone()
        m[:, HL] = torch.where(ok, torch.ones_like(m[:, HL]), m[:, HL])
        m[:, HO] = torch.where(ok, torch.full_like(m[:, HO], ORD_SL), m[:, HO])
        return m

    def q4_count(g, n):  # learner seats with 4 open land quadrants (glob of the last step)
        return (g[:n, QOWN0 : QOWN0 + 4].float().sum(1) > 3.5).sum()

    def pad(i, q):  # indices padded to shape q by repeating the first one
        return torch.cat([i, i[:1].repeat(q - len(i))]) if len(i) and q > len(i) else i

    best_ver = [0]  # anchor version loaded into best_net (actor thread)
    snap_key = [None]  # pool version loaded into snap_net (actor thread)

    def actor():
        try:
            for it in its:
                bi = it % 2
                free[bi].acquire()
                if x.sync and it > x.start_iter:
                    upd_done.acquire()
                t0 = time.time()
                st = stores[bi]
                with torch.cuda.stream(astream):
                    if done_ev[bi] is not None:
                        astream.wait_event(done_ev[bi])
                    with pub_lock:
                        pw = pub["w"]
                    if pw is not None:
                        astream.wait_event(pw[1])
                        act_net.load_state_dict(pw[0])
                    with pub_lock:
                        aw, aver = anchor["w"], anchor["ver"]
                    if aw is not None and aver != best_ver[0]:
                        astream.wait_event(aw[1])
                        best_net.load_state_dict(aw[0], strict=False)
                        best_ver[0] = aver
                    rng = random.Random((x.seed0 + it) * 7919 + rank)
                    for try_ in range(60):  # another run's pool (exploiter) may have no versions yet
                        try:
                            pv, pv_p, pv_n, pv_g = pool.pick(rng, pool.listing())
                            break
                        except RuntimeError:
                            if try_ == 59:
                                raise
                            time.sleep(10)
                    if pv != snap_key[0]:  # version from disk (the file is written atomically, see save_version)
                        snap_net.load_state_dict(
                            torch.load(pv, map_location="cpu", weights_only=False)["model"], strict=False
                        )
                        snap_key[0] = pv
                    astream.synchronize()
                    active, slot_opp, past_idx, best_idx = [], [], [], []
                    slot_ref = []  # record reference for the slot (tape world with "ref") or None
                    # balancing: games against a code-bot opponent (Python, the most expensive part on the CPU) are
                    # dealt round-robin, grouped by agent, so every worker gets an equal share (the step waits for the
                    # slowest worker)
                    cw = lw["w"]
                    # G: CPU plays code bots only
                    ccw = [0.0 if G and n in NOT_FIELD else w_ for n, w_ in zip(names, cw)]
                    draws = [(rng.choices(names, ccw)[0], rng.randint(0, 1)) for _ in range(W * K)]
                    order = sorted(range(W * K), key=lambda j: (draws[j][0] in NOT_FIELD, draws[j][0]))
                    grid = [[None] * K for _ in range(W)]
                    for pos, j in enumerate(order):
                        grid[pos % W][pos // W] = draws[j]
                    for w in range(W):
                        specs = []
                        for k in range(K):
                            seed = x.seed0 + ((it * world + rank) * W + w) * K + k
                            opp, me = grid[w][k]
                            if opp == "self":
                                specs.append({"seed": seed, "seats": [0, 1], "opp": None})
                                learn = [0, 1]
                            elif opp in ("past", "best"):
                                specs.append({"seed": seed, "seats": [0, 1], "opp": None})
                                learn = [me]
                                (past_idx if opp == "past" else best_idx).append(w * 2 * K + 2 * k + 1 - me)
                            elif opp in tpools:
                                # tape: the record's world, the learner takes the record's opposite seat
                                wf, ts, wseed, wshops, wref, wrefo = rng.choice(tpools[opp])
                                me = 1 - ts
                                specs.append(
                                    {"seed": wseed, "shops": wshops, "seats": [me], "opp": None, "tape": (wf, ts)}
                                )
                                learn = [me]
                            else:
                                specs.append({"seed": seed, "seats": [me], "opp": opp})
                                learn = [me]
                            for s in learn:
                                active.append(w * 2 * K + 2 * k + s)
                                slot_opp.append(okey(opp))
                                slot_ref.append((wref, wrefo) if opp in tpools and wref is not None else None)
                        conns[w].send(specs)
                    idx = torch.tensor(active)
                    pidx = torch.tensor(past_idx, dtype=torch.long)
                    NA = len(active)
                    # the actor batch is padded by repetition to a precompiled shape (otherwise eager)
                    PA, PP = bucket(NA, ABUCKETS), bucket(len(past_idx), PBUCKETS)
                    afn = act_c if PA in ABUCKETS else act_compute
                    pfn = past_c if PP in PBUCKETS else past_compute
                    aidx = torch.cat([idx, idx[:1].repeat(PA - NA)])
                    if PC:  # the row's opponent = the other seat of the same game (slot ^ 1)
                        oaidx = aidx ^ 1
                        st["oid"][:NA] = torch.tensor([opp_id(n) for n in slot_opp], dtype=torch.long, device=dev)
                    ppidx = torch.cat([pidx, pidx[:1].repeat(PP - len(past_idx))]) if len(past_idx) else pidx
                    bidx = torch.tensor(best_idx, dtype=torch.long)
                    PB = bucket(len(best_idx), PBUCKETS)
                    bfn = past_c if PB in PBUCKETS else past_compute
                    bbidx = torch.cat([bidx, bidx[:1].repeat(PB - len(best_idx))]) if len(best_idx) else bidx
                    NG, gl, glab = 0, [], []
                    if G:  # GPU games: only self/past/best, by their league weights
                        gw_ = [w_ if n in NOT_FIELD else 0.0 for n, w_ in zip(names, cw)]
                        gp_, gb_ = [], []
                        for g in range(G):
                            opp, me = rng.choices(names, gw_)[0], rng.randint(0, 1)
                            if opp == "self":
                                gl += [2 * g, 2 * g + 1]
                                glab += ["self", "self"]
                            else:
                                gl.append(2 * g + me)
                                glab.append(opp)
                                (gp_ if opp == "past" else gb_).append(2 * g + 1 - me)
                        NG = len(gl)
                        gi = torch.tensor(gl, dtype=torch.long, device=dev)
                        if PC:
                            goi = gi ^ 1
                            st["oid"][NA : NA + NG] = torch.tensor(
                                [opp_id(n) for n in glab], dtype=torch.long, device=dev
                            )
                        gpi = torch.tensor(gp_, dtype=torch.long, device=dev)
                        gbi = torch.tensor(gb_, dtype=torch.long, device=dev)
                        QA, QP, QB = bucket(NG, GBUCKETS), bucket(len(gp_), GPBUCKETS), bucket(len(gb_), GBBUCKETS)
                        gafn = act_c if QA in GBUCKETS else act_compute
                        gpfn = past_c if QP in GPBUCKETS else past_compute
                        gbfn = past_c if QB in GBBUCKETS else past_compute
                        gai, gpai, gbai = pad(gi, QA), pad(gpi, QP), pad(gbi, QB)
                        gseeds = np.array(
                            [x.seed0 + 700_000_000 + (it * world + rank) * G + g for g in range(G)], np.int64
                        )
                        gstream.wait_stream(astream)  # act/snap/best weights and indices were loaded on astream
                        with torch.cuda.stream(gstream):
                            grb.reset(gseeds)
                    tw = tg = tgq = 0.0
                    # --land-inject: rows with substitution (CPU / GPU), seats with 4 quadrants
                    linj = torch.zeros((), device=dev)
                    linjg = torch.zeros((), device=dev)
                    q4c = torch.zeros((), device=dev)
                    q4g = torch.zeros((), device=dev)
                    for t in range(T):
                        ta = time.time()
                        for c in conns:
                            if not c.poll(900):
                                raise RuntimeError("worker did not respond within 900 s")
                            c.recv()
                        tb = time.time()
                        tw += tb - ta
                        b = {k: bufs[k][aidx].to(dev) for k in ENC}
                        # opponent observation is read before sending actions (after that the workers write the next
                        # step into bufs)
                        ob_ = {k: bufs[k][oaidx].to(dev) for k in OENC} if PC else None
                        bt = dict(b)
                        bt["tmask"] = unpack(b["tmask"])
                        with torch.no_grad():
                            uc, mc, lpu, lpm, v = (o[:NA] for o in safe(afn, act_compute, act_net, bt))
                        bufs["uc"][idx] = uc.cpu()
                        bufs["mc"][idx] = land_inj(mc, b["glob"], t, linj).cpu()
                        if x.land_inject >= 0 and t == T - 1:
                            q4c.add_(q4_count(b["glob"], NA))
                        if len(past_idx):
                            pb = {k: bufs[k][ppidx].to(dev) for k in ENC}
                            pb["tmask"] = unpack(pb["tmask"])
                            with torch.no_grad():
                                puc, pmc = (o[: len(past_idx)] for o in safe(pfn, past_compute, snap_net, pb, PG))
                            bufs["uc"][pidx] = puc.cpu()
                            bufs["mc"][pidx] = pmc.cpu()
                        if len(best_idx):
                            bb = {k: bufs[k][bbidx].to(dev) for k in ENC}
                            bb["tmask"] = unpack(bb["tmask"])
                            with torch.no_grad():
                                buc, bmc = (o[: len(best_idx)] for o in safe(bfn, past_compute, best_net, bb, False))
                            bufs["uc"][bidx] = buc.cpu()
                            bufs["mc"][bidx] = bmc.cpu()
                        for c in conns:
                            c.send(1)
                        tg += time.time() - tb
                        for k in ENC:
                            st[k][t, :NA] = b[k][:NA]
                        if PC:
                            for k in OENC:
                                st["o_" + k][t, :NA] = ob_[k][:NA]
                        st["uc"][t, :NA] = uc
                        st["mc"][t, :NA] = mc
                        st["lpu"][t, :NA] = lpu
                        st["lpm"][t, :NA] = lpm
                        st["val"][t, :NA] = v.float()
                        if G:  # GPU-game step is queued on gstream while the workers compute
                            tq_ = time.time()
                            with torch.cuda.stream(gstream), torch.no_grad():
                                grb.encode(gobs)
                                gb = {k: gobs[k][gai] for k in ENC}
                                gbt = dict(gb)
                                gbt["tmask"] = unpack(gb["tmask"])
                                uc_, mc_, lpu_, lpm_, v_ = (o[:NG] for o in safe(gafn, act_compute, act_net, gbt))
                                guc[gi] = uc_
                                gmc[gi] = land_inj(mc_, gb["glob"], t, linjg)
                                if x.land_inject >= 0 and t == T - 1:
                                    q4g.add_(q4_count(gb["glob"], NG))
                                for ii, ai, fn_, net_, gr_ in (
                                    (gpi, gpai, gpfn, snap_net, PG),
                                    (gbi, gbai, gbfn, best_net, False),
                                ):
                                    if len(ii):
                                        ob = {k: gobs[k][ai] for k in ENC}
                                        ob["tmask"] = unpack(ob["tmask"])
                                        puc_, pmc_ = (o[: len(ii)] for o in safe(fn_, past_compute, net_, ob, gr_))
                                        guc[ii] = puc_
                                        gmc[ii] = pmc_
                                for k in ENC:
                                    st[k][t, NA : NA + NG] = gb[k][:NG]
                                if PC:
                                    for k in OENC:
                                        st["o_" + k][t, NA : NA + NG] = gobs[k][goi]
                                st["uc"][t, NA : NA + NG] = uc_
                                st["mc"][t, NA : NA + NG] = mc_
                                st["lpu"][t, NA : NA + NG] = lpu_
                                st["lpm"][t, NA : NA + NG] = lpm_
                                st["val"][t, NA : NA + NG] = v_.float()
                                grb.act_step(guc, gmc)
                            tgq += time.time() - tq_
                    tge = time.time()
                    gbk = None
                    if G:
                        with torch.cuda.stream(gstream):
                            grb.rewards(out=gbank)
                        astream.wait_stream(gstream)
                        gbk = gbank.cpu().numpy()  # on astream (waits for gstream): all GPU games are finished
                    tge = time.time() - tge
                    ev = torch.cuda.Event()
                    ev.record(astream)
                rews, wt = [], np.zeros(3)
                for c in conns:
                    if not c.poll(900):
                        raise RuntimeError("worker did not return the result")
                    r_, t_ = c.recv()
                    rews.append(r_)
                    wt += np.array(t_)
                R = torch.zeros(NA + NG)
                stats = {}
                mw = (
                    x.margin_w
                    if x.margin_w_final is None
                    else x.margin_w
                    + (x.margin_w_final - x.margin_w) * min(1.0, (it - x.start_iter) / max(1, x.margin_decay_iters))
                )

                # rfun/add close over this iteration's mw, R, recs and stats and are only called below, within the
                # same iteration
                def rfun(d):
                    return (1.0 if d > 0 else (-1.0 if d < 0 else 0.0)) + mw * max(-1.0, min(1.0, d / x.margin_scale))  # noqa: B023

                # tape worlds with a reference: games, wins over the record, sum of (bank − reference)
                recs = [0, 0.0, 0.0]

                def add(j, me, op, lab, ref=None):
                    m = me - op
                    R[j] = rfun(m)  # noqa: B023
                    if ref is not None:
                        dr = me - ref[0] if x.rec_mode == "bank" else m - (ref[0] - ref[1])
                        recs[0] += 1  # noqa: B023
                        recs[1] += 1.0 if dr > 0 else (0.5 if dr == 0 else 0.0)  # noqa: B023
                        recs[2] += dr  # noqa: B023
                        if x.rec_w:
                            R[j] = (1.0 - x.rec_w) * R[j] + x.rec_w * rfun(dr)  # noqa: B023
                    if x.bank_w:  # own bank: rewards the economy even when the win is already guaranteed
                        R[j] += x.bank_w * max(0.0, min(2.0, me / x.bank_scale))  # noqa: B023
                    st_ = stats.setdefault(lab, [0, 0.0, 0.0, 0.0])  # noqa: B023
                    st_[0] += 1
                    st_[1] += 1.0 if m > 0 else (0.5 if m == 0 else 0.0)
                    st_[2] += me
                    st_[3] += m

                for j, sl in enumerate(active):
                    w, r = divmod(sl, 2 * K)
                    k, s = divmod(r, 2)
                    add(j, rews[w][k][s], rews[w][k][1 - s], slot_opp[j], slot_ref[j])
                for j, (r, lab) in enumerate(zip(gl, glab)):
                    g, s = divmod(r, 2)
                    add(NA + j, float(gbk[g, s]), float(gbk[g, 1 - s]), lab)
                inf = {
                    "t_roll": time.time() - t0,
                    "p_wait": tw,
                    "p_gpu": tg,
                    "w_enc": wt[0] / W,
                    "w_opp": wt[1] / W,
                    "w_exec": wt[2] / W,
                    "mw": mw,
                    "aver": best_ver[0],
                    "pv": pv,
                    "pv_p": pv_p,
                    "pv_n": pv_n,
                    "pv_g": pv_g,
                    "rec": list(recs),
                }
                if G:
                    inf.update(na_gpu=NG, g_enq=tgq, g_end=tge)
                if x.land_inject >= 0:  # substitutions (row-steps) and % of learner seats with 4 quadrants at the end
                    inf.update(land_inj=float(linj + linjg), q4_end_pct=100.0 * float(q4c + q4g) / max(1, NA + NG))
                ready.put((it, NA + NG, R, stats, ev, inf, pw))
        except Exception as e:  # noqa: BLE001
            err.append(e)
            ready.put(None)
            raise

    def fwd_terms(mb, rcb, uc, mc):
        """Policy forward (+ anchor heads from the cache or a full anchor forward) and per-component loss terms."""
        with torch.autocast(**amp):
            if PC:
                lg, ml, v = model(
                    mb["tiles"],
                    mb["units"],
                    mb["unit_pos"],
                    mb["unit_mask"],
                    mb["glob"],
                    opp={k: mb["o_" + k] for k in OENC},
                    oid=mb["oid"],
                )
            else:
                lg, ml, v = model(mb["tiles"], mb["units"], mb["unit_pos"], mb["unit_mask"], mb["glob"])
            with torch.no_grad():
                if rcb is not None:
                    rlg, rml = ref.heads_from_cache(rcb[0], rcb[1], rcb[2], mb["unit_pos"])
                else:
                    rlg, rml, _ = ref(
                        mb["tiles"], mb["units"], mb["unit_pos"], mb["unit_mask"], mb["glob"], value=False
                    )
        lpu, eu, ku = unit_terms(lg, rlg, mb, uc)
        lpm, em, km = market_terms(ml, rml, mb, mc)
        return lpu, eu, ku, lpm, em, km, v

    def advantages(val, R, lrho):
        """-> (policy advantage, value target, UPGO advantage or None) [T, NA]. Reward only at the end.
        lrho=None: GAE(lam) / GAE(vf_lam) target (as before). Otherwise V-trace (IMPALA): vs_t - V_t = rho_t delta_t +
        gamma c_t lam (vs_{t+1} - V_{t+1}); advantage A_t = r_t + gamma vs_{t+1} - V_t (policy lam = --lam)."""
        NA_ = val.shape[1]
        z = torch.zeros(NA_, device=dev)
        adv = torch.zeros_like(val)
        ret = torch.zeros_like(val)
        upg = torch.zeros_like(val) if x.upgo else None
        rho = c = None
        if lrho is not None:
            ex = torch.exp(lrho.clamp(max=20.0))
            rho, c = ex.clamp(max=x.vtrace_rho), ex.clamp(max=x.vtrace_c)
        gp, gv, gu = z.clone(), z.clone(), z.clone()
        for t in reversed(range(T)):
            last = t == T - 1
            nv = z if last else val[t + 1]
            r_t = R if last else z
            delta = r_t + x.gamma * nv - val[t]
            if rho is None:
                gp = delta + x.gamma * x.lam * gp
                gv = delta + x.gamma * x.vf_lam * gv
                adv[t] = gp
            else:
                adv[t] = delta + x.gamma * gp  # gp here = vs_{t+1} - V_{t+1} (not yet updated)
                gp = rho[t] * delta + x.gamma * x.lam * c[t] * gp
                gv = rho[t] * delta + x.gamma * x.vf_lam * c[t] * gv
            ret[t] = gv + val[t]
            if upg is not None:  # UPGO: follow G while the next action is not worse than average
                if last:
                    gu = R.clone()
                else:
                    r1 = R if t + 1 == T - 1 else z
                    q1 = r1 + x.gamma * (val[t + 2] if t + 2 < T else z)
                    gu = r_t + x.gamma * torch.where(q1 >= val[t + 1], gu, val[t + 1])
                upg[t] = gu - val[t]
        return adv, ret, upg

    PAD = 256  # the minibatch is padded to a multiple of PAD: few distinct shapes
    if x.compile:  # dynamic=False: with dynamic shapes the step was slower
        torch._dynamo.config.cache_size_limit = 64
        if hasattr(torch._dynamo.config, "recompile_limit"):
            torch._dynamo.config.recompile_limit = 64
        if hasattr(torch._dynamo.config, "automatic_dynamic_shapes"):
            torch._dynamo.config.automatic_dynamic_shapes = False
    fwd_c = torch.compile(fwd_terms, dynamic=False) if x.compile else fwd_terms
    # with the default --workers/--games, learner seats ~208 + self games (binomial ~30±5) -> minibatch T*NA/64 ~
    # 2450-3000 -> shapes 2560/2816/3072. Compilation (forward and backward) happens here, before the actor starts:
    # compiling in the main thread breaks calls to already compiled functions in the actor thread. Other shapes run
    # eager.
    LBUCKETS = (2560, 2816, 3072) if x.compile else ()
    if x.compile and G:  # NA = W*K + G + Binomial(G, p_self): separate minibatch shapes
        LBUCKETS = tuple(sorted({-(-(T * na // x.minibatches) // PAD) * PAD for na in brange(G, ps_, W * K + G, 1)}))
    if x.compile:
        tc = time.time()
        net.train()
        for q in LBUCKETS:
            z = torch.zeros(q, dtype=torch.long, device=dev)
            mb = {k: stores[0][k][z, z] for k in ENC}
            mb["tmask"] = unpack(mb["tmask"])
            if PC:
                mb.update({"o_" + k: stores[0]["o_" + k][z, z] for k in OENC})
                mb["oid"] = stores[0]["oid"][z]
            rcb = (
                (
                    torch.zeros(q, S.NT, d, dtype=torch.bfloat16, device=dev),
                    torch.zeros(q, S.MAX_UNITS, d, dtype=torch.bfloat16, device=dev),
                    torch.zeros(q, 2 * d, dtype=torch.bfloat16, device=dev),
                )
                if x.ref_cache
                else None
            )
            outs = fwd_c(mb, rcb, stores[0]["uc"][z, z], stores[0]["mc"][z, z])
            (sum(o.float().sum() for o in outs) * 0.0).backward()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if rank == 0:
            print(f"trainer compiled: shapes {LBUCKETS}, {time.time() - tc:.0f} s", flush=True)
    rc = None
    skip_upd = set()  # exploiter: iterations without an update (games from before the reset)
    pubwin = []  # exploiter: window [games vs past, wins]
    last_reset = [x.start_iter]
    jk3_hist = []  # per-iteration minibatch mean of the joint k3 (max over ranks)
    awin = []  # anchor-gate window: [games vs anchor, wins, field games, wins]
    afield = [x.anchor_field0]  # field P at the last anchor replacement
    th = threading.Thread(target=actor, daemon=True)
    th.start()
    nstep = 0
    tprev = time.time()
    for it in its:
        tq = time.time()
        item = ready.get()
        if item is None:
            raise RuntimeError(f"actor crashed: {err}")
        _, NA, R, stats, ev, info, played = item
        t_lwait = time.time() - tq
        torch.cuda.current_stream().wait_event(ev)
        st = stores[it % 2]
        t1 = time.time()
        R = R.to(dev)
        gate_tab, fastlog = None, None  # [T, NA]: 1 = step on which the policy receives a gradient
        if x.fast_gate:  # opponent land quadrants: glob[OQ0:OQ0+4] (encode.py: NW/NE/SW/SE flags)
            oq = st["glob"][:, :NA, OQ0 : OQ0 + 4].float().sum(-1)
            tix = torch.arange(T, device=dev).unsqueeze(1).expand(T, NA)
            first = torch.where(oq >= x.fast_quads - 0.5, tix, torch.full_like(tix, T)).min(0).values
            fv = first < x.fast_by
            gate_tab = ((tix >= first.unsqueeze(0)) & (tix < x.fast_until) & fv.unsqueeze(0)).float()
            nf = int(fv.sum())
            fastlog = {
                "n": nf,
                "frac": round(nf / max(1, NA), 4),
                "P": round(float((R[:NA][fv] > 0).float().mean()), 3) if nf else None,
                "first_med": int(first[fv].median()) if nf else None,
                "steps": int(gate_tab.sum()),
            }
        elif x.late_step > 0:
            gate_tab = (torch.arange(T, device=dev) >= x.late_step).float().unsqueeze(1).expand(T, NA)
        lrho = None
        t_cur = time.time()
        if CUR:  # fresh pass with the current network: values (+ log pi/mu for V-trace)
            val = torch.zeros(T, NA, device=dev)
            lrho = torch.zeros(T, NA, device=dev) if x.vtrace else None
            g = max(1, 2048 // NA)
            with torch.no_grad(), torch.autocast(**amp):
                for t0_ in range(0, T, g):
                    t1_ = min(T, t0_ + g)
                    n_ = t1_ - t0_
                    b = {k: st[k][t0_:t1_, :NA].flatten(0, 1) for k in ENC}
                    kw = {}
                    if PC:
                        kw = {
                            "opp": {k: st["o_" + k][t0_:t1_, :NA].flatten(0, 1) for k in OENC},
                            "oid": st["oid"][:NA].repeat(n_),
                        }
                    lg_, ml_, v_ = net(b["tiles"], b["units"], b["unit_pos"], b["unit_mask"], b["glob"], **kw)
                    val[t0_:t1_] = v_.float().view(n_, NA)
                    if x.vtrace:
                        uc_ = st["uc"][t0_:t1_, :NA].flatten(0, 1)
                        mc_ = st["mc"][t0_:t1_, :NA].flatten(0, 1)
                        um_ = b["unit_mask"].float()
                        lpu_ = unit_logp(lg_, unpack(b["tmask"]), b["unit_mask"], uc_ // S.NOPS, uc_ % S.NOPS)
                        lpm_ = market_logp(ml_, b["mmask"], mc_)
                        olu_ = st["lpu"][t0_:t1_, :NA].flatten(0, 1)
                        olm_ = st["lpm"][t0_:t1_, :NA].flatten(0, 1)
                        lr_ = ((lpu_.float() - olu_) * um_).sum(-1) + (lpm_.float() - olm_).sum(-1)
                        lrho[t0_:t1_] = lr_.view(n_, NA)
        else:
            val = st["val"][:, :NA]
        t_cur = time.time() - t_cur
        adv, ret, upg = advantages(val, R, lrho)
        if upg is not None:
            adv = adv + x.upgo * upg
        an = (adv - adv.mean()) / (adv.std() + 1e-8)
        rv = R.unsqueeze(0).expand(T, NA)
        ev_all = 1 - (rv - val).var() / (rv.var() + 1e-8)
        ev_early = 1 - (rv[:144] - val[:144]).var() / (rv[:144].var() + 1e-8)
        net.train()
        kf = (
            x.kl_ref
            if x.kl_ref_final is None
            else x.kl_ref + (x.kl_ref_final - x.kl_ref) * min(1.0, it / max(1, x.kl_decay_iters))
        )
        N = T * NA
        mbs = N // x.minibatches
        if x.ref_cache:  # anchor trunk once per iteration (KL to the anchor is exact)
            if rc is None or rc["t"].shape[1] < NA:
                cap = min(NSG, (NA + 31) // 32 * 32)
                rc = None
                torch.cuda.empty_cache()
                rc = {
                    "t": torch.empty(T, cap, S.NT, d, dtype=torch.bfloat16, device=dev),
                    "u": torch.empty(T, cap, S.MAX_UNITS, d, dtype=torch.bfloat16, device=dev),
                    "p": torch.empty(T, cap, 2 * d, dtype=torch.bfloat16, device=dev),
                }
            g = max(1, 2048 // NA)
            rvals = torch.zeros(T, NA, device=dev) if x.vd else None
            with torch.no_grad(), torch.autocast(**amp):
                for t0_ in range(0, T, g):
                    t1_ = min(T, t0_ + g)
                    b = {
                        k: st[k][t0_:t1_, :NA].flatten(0, 1)
                        for k in ("tiles", "units", "unit_pos", "unit_mask", "glob")
                    }
                    xr, padr = ref.trunk(b["tiles"], b["units"], b["unit_pos"], b["unit_mask"], b["glob"])
                    ct, cu, cp = ref.head_cache(xr)
                    n_ = t1_ - t0_
                    rc["t"][t0_:t1_, :NA] = ct.view(n_, NA, S.NT, d)
                    rc["u"][t0_:t1_, :NA] = cu.view(n_, NA, S.MAX_UNITS, d)
                    rc["p"][t0_:t1_, :NA] = cp.view(n_, NA, 2 * d)
                    if x.vd:
                        if PC:
                            rv_ = ref.value_head(
                                xr,
                                padr,
                                {k: st["o_" + k][t0_:t1_, :NA].flatten(0, 1) for k in OENC},
                                st["oid"][:NA].repeat(n_),
                            )
                        else:
                            rv_ = ref.value_head(xr, padr)
                        rvals[t0_:t1_] = rv_.float().view(n_, NA)
        t_rc = time.time() - t1
        acc = torch.zeros(9, device=dev)  # metrics accumulate on the GPU and go to the CPU once per iteration
        jacc = torch.zeros(5, device=dev)  # joint: jkl, jk3, jkl_ref, vdl, jk3 (max over ranks)
        jthr = (
            x.tr_joint_mult * float(np.median(jk3_hist[-10:]))
            if x.tr_joint_mult > 0 and len(jk3_hist) >= x.tr_joint_warm
            else 0.0
        )
        hk = torch.zeros(H + 1, device=dev)  # KL(old||new) per market head + units
        hr = torch.zeros(H + 1, device=dev)  # KL to the anchor per market head + units
        tr_host = torch.zeros(1, pin_memory=True)
        tr_ev, tr_stop, n_done, tr_top = None, it in skip_upd, 0, torch.zeros(H + 1, device=dev)
        for _ in range(x.epochs):
            if tr_stop:
                break
            perm = torch.randperm(N, device=dev)
            for i in range(x.minibatches):
                if tr_ev is not None:  # KL of the previous minibatch (all ranks see the same maximum)
                    tr_ev.synchronize()
                    if float(tr_host[0]) > 1.0:
                        tr_stop = True
                        break
                j = perm[i * mbs : (i + 1) * mbs]
                # static shape for compile: pad by repeating own rows; their outputs are cut off before the losses
                # (no gradient)
                if x.compile and mbs % PAD:
                    j = torch.cat([j, j[: PAD - mbs % PAD]])
                tt, aa = j // NA, j % NA
                mb = {k: st[k][tt, aa] for k in ENC}
                mb["tmask"] = unpack(mb["tmask"])
                if PC:
                    mb.update({"o_" + k: st["o_" + k][tt, aa] for k in OENC})
                    mb["oid"] = st["oid"][aa]
                uc, mc = st["uc"][tt, aa], st["mc"][tt, aa]
                if x.ref_cache:
                    rcb = (rc["t"][tt, aa], rc["u"][tt, aa], rc["p"][tt, aa])
                else:
                    rcb = None
                fwd_fn = fwd_c if j.shape[0] in LBUCKETS else fwd_terms
                lpu, eu, ku, lpm, em, km, v = (o[:mbs] for o in fwd_fn(mb, rcb, uc, mc))
                tt, aa = tt[:mbs], aa[:mbs]
                um = mb["unit_mask"][:mbs].float()
                A_ = an[tt, aa].unsqueeze(-1)
                olu, olm = st["lpu"][tt, aa], st["lpm"][tt, aa]
                ru = torch.exp(lpu - olu)
                rm = torch.exp(lpm - olm)
                su = torch.min(ru * A_, ru.clamp(1 - x.clip, 1 + x.clip) * A_)
                sm = torch.min(rm * A_, rm.clamp(1 - x.clip, 1 + x.clip) * A_)
                # dual-clip (JueWu, AAAI-20): for A<0 the objective is not below c*A (huge r on stale data)
                if x.dual_clip > 0:
                    su = torch.where(A_ < 0, torch.maximum(su, x.dual_clip * A_), su)
                    sm = torch.where(A_ < 0, torch.maximum(sm, x.dual_clip * A_), sm)
                su = su * um
                ncomp = um.sum(-1) + H
                pg = -((su.sum(-1) + sm.sum(-1)) / ncomp).mean()
                vl = ((v.float() - ret[tt, aa]) ** 2).mean()
                ent = ((eu.sum(-1) + em.sum(-1)) / ncomp).mean()
                klv = (ku.sum(-1) * kwu + (km if khw is None else km * khw).sum(-1)) / ncomp
                klr = klv.mean()
                # only the window is trained (end of game / games with a fast opponent): outside the window only KL to
                # --ref with weight --kl-early
                if gate_tab is not None:
                    late = gate_tab[tt, aa]
                    pg = -(((su.sum(-1) + sm.sum(-1)) / ncomp) * late).sum() / late.sum().clamp(min=1.0)
                    pol = pg - x.ent * ent + ((kf * late + x.kl_early * (1.0 - late)) * klv).mean()
                else:
                    pol = pg - x.ent * ent + kf * klr
                vdl = ((v.float() - rvals[tt, aa]) ** 2).mean() if x.vd else torch.zeros((), device=dev)
                with torch.no_grad():  # joint log-ratio = sum over action components
                    lrj = (((lpu - olu) * um).sum(-1) + (lpm - olm).sum(-1)).float().clamp(max=20.0)
                    jk3 = (torch.exp(lrj) - 1 - lrj).mean()
                    jk3m = jk3.view(1).clone()
                    dist.all_reduce(jk3m, op=dist.ReduceOp.MAX)
                    jacc += torch.stack([(-lrj).mean(), jk3, (ku.sum(-1) + km.sum(-1)).mean(), vdl.detach(), jk3m[0]])
                # KL(playing||current) over the chosen actions (they come from the playing policy):
                # k3 = (r-1) - log r, r = pi_new/pi_play (Schulman 2020)
                if x.tr_kl > 0 or jthr > 0:
                    with torch.no_grad():
                        lr_u = (lpu - olu).float()
                        lr_m = (lpm - olm).float()
                        k3u = ((torch.exp(lr_u) - 1 - lr_u) * um).sum() / um.sum().clamp(min=1)
                        k3m = (torch.exp(lr_m) - 1 - lr_m).mean(0)
                        trv = torch.cat([k3m, k3u.view(1)])
                        tr_top = torch.maximum(tr_top, trv)
                        # fraction of the threshold: per head (--tr-kl) and joint (--tr-joint-mult); stop if any > 1
                        trm = (
                            torch.stack(
                                [
                                    trv.max() / x.tr_kl if x.tr_kl > 0 else torch.zeros((), device=dev),
                                    jk3 / jthr if jthr > 0 else torch.zeros((), device=dev),
                                ]
                            )
                            .max()
                            .view(1)
                        )
                        dist.all_reduce(trm, op=dist.ReduceOp.MAX)
                        tr_host.copy_(trm, non_blocking=True)
                        tr_ev = torch.cuda.Event()
                        tr_ev.record()
                loss = x.vf * vl + x.vd * vdl + (0.0 if it < x.vf_warmup else 1.0) * pol
                # warmup of both groups: an --init checkpoint without Adam state starts a fresh Adam, whose first steps
                # are ~lr*sign(g)
                if x.lr_warmup:
                    wf = min(1.0, (nstep + 1) / x.lr_warmup)
                    opt.param_groups[0]["lr"] = x.lr * wf
                    opt.param_groups[1]["lr"] = x.vf_lr * wf
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(ppar, 1.0)
                torch.nn.utils.clip_grad_norm_(vpar, 1.0)
                opt.step()
                if ema is not None:
                    torch._foreach_lerp_(ema, [p.detach() for p in net.parameters()], 1.0 - x.ema)
                n_done += 1
                if it >= x.vf_warmup:
                    nstep += 1
                with torch.no_grad():
                    dlu = ((olu - lpu) * um).sum(-1) / um.sum(-1).clamp(min=1)
                    dlm = olm - lpm
                    approx_kl = ((((olu - lpu) * um).sum(-1) + dlm.sum(-1)) / ncomp).mean()
                    cf = (
                        (((ru - 1).abs() > x.clip).float() * um).sum() + ((rm - 1).abs() > x.clip).float().sum()
                    ) / ncomp.sum()
                    acc += torch.stack(
                        [
                            pg.detach(),
                            vl.detach(),
                            ent.detach(),
                            klr.detach(),
                            approx_kl,
                            cf,
                            gn,
                            torch.zeros((), device=dev),
                            torch.zeros((), device=dev),
                        ]
                    )
                    hk[:H] += dlm.mean(0)
                    hk[H] += dlu.mean()
                    hr[:H] += km.mean(0)
                    hr[H] += (ku.sum(-1) / um.sum(-1).clamp(min=1)).mean()
        nmb = max(1, n_done)
        acc /= nmb
        jacc /= nmb
        j_ = jacc.tolist()
        jk3_hist.append(j_[4])
        hk /= nmb
        hr /= nmb
        torch.cuda.current_stream().synchronize()
        done_ev[it % 2] = None  # the trainer finished reading (synchronized): the buffer is free
        free[it % 2].release()
        t_upd = time.time() - t1
        w_ = {k: v.detach().clone() for k, v in net.state_dict().items()}
        pe = torch.cuda.Event()
        pe.record()
        with pub_lock:
            pub["w"] = (w_, pe)
        if x.sync:
            upd_done.release()
        if (it + 1) % x.snap_every == 0 and x.publish_gate <= 0 and rank == 0:
            save_version(net.state_dict(), it + 1, "snap")
        a_ = acc.tolist()
        hk_, hr_ = hk.tolist(), hr.tolist()
        agg = dict(zip(("pg", "vl", "ent", "kl_ref", "approx_kl", "clipfrac", "gnorm"), a_[:7]))
        heads = [f"{g}:{i_}" if i_ is not None else g for g, i_, _ in S.MARKET_HEADS] + ["units"]
        agg["akl_max"], agg["akl_max_head"] = max(hk_), heads[int(np.argmax(hk_))]
        agg["klr_max"], agg["klr_max_head"] = max(hr_), heads[int(np.argmax(hr_))]
        agg["lr"] = opt.param_groups[0]["lr"]
        agg["n_mb"] = n_done
        if x.tr_kl > 0 or jthr > 0:
            tt_ = tr_top.tolist()
            agg["tr_max"], agg["tr_head"] = max(tt_), heads[int(np.argmax(tt_))]
        agg.update(jkl=j_[0], jk3=j_[1], jkl_ref=j_[2], vdl=j_[3], jk3_rmax=j_[4], jthr=jthr, tr_stop=int(tr_stop))
        gathered = [None] * world
        dist.all_gather_object(gathered, (stats, info))
        tot = {}  # identical on all ranks: the anchor-replacement decision is consistent
        for st_, _ in gathered:
            for n, v in st_.items():
                a = tot.setdefault(n, [0, 0.0, 0.0, 0.0])
                for q in range(4):
                    a[q] += v[q]
        fields = [n for n in tot if n not in NOT_FIELD]
        pw = sum(tot[n][1] for n in fields) / max(1, sum(tot[n][0] for n in fields))
        per = {}  # pool versions for the iteration: learner games/wins
        for st_, inf_ in gathered:
            s_ = st_.get("past")
            if s_ and s_[0]:
                a = per.setdefault(inf_["pv"], [0, 0.0, inf_["pv_p"], inf_["pv_n"]])
                a[0] += s_[0]
                a[1] += s_[1]
        if it not in skip_upd:
            pool.update(per)
        pub_ev = None
        if x.publish_gate > 0:  # exploiter: publish a version + reset to the initial weights
            if it not in skip_upd:
                s_ = tot.get("past", [0, 0.0])
                pubwin.append((s_[0], s_[1]))
            wn_ = [sum(a[q] for a in pubwin[-x.publish_window :]) for q in (0, 1)]
            pp_ = wn_[1] / max(1, wn_[0])
            beat = len(pubwin) >= x.publish_window and wn_[0] >= x.publish_min and pp_ >= x.publish_gate
            tout = x.reset_every > 0 and it + 1 - last_reset[0] >= x.reset_every
            if beat or tout:
                fn_ = save_version(net.state_dict(), it + 1, "exploiter") if rank == 0 else None
                net.load_state_dict(init_sd)
                opt.state.clear()
                nstep = 0
                if ema is not None:
                    for e_, p_ in zip(ema, net.parameters()):
                        e_.copy_(p_.detach())
                skip_upd.update((it + 1, it + 2))  # their games may have been played by pre-reset weights (lag 1-2)
                last_reset[0] = it + 1
                pubwin.clear()
                pub_ev = {"file": fn_, "P": round(pp_, 4), "n": wn_[0], "why": "beat" if beat else "timeout"}
                w_ = {k: v.detach().clone() for k, v in net.state_dict().items()}  # the actor gets the reset weights
                pe = torch.cuda.Event()
                pe.record()
                with pub_lock:
                    pub["w"] = (w_, pe)
        if x.pfsp > 0:  # identical on all ranks (tot is shared)
            for k_, v in tot.items():
                if k_ not in NOT_FIELD and v[0]:
                    p_ = v[1] / v[0]
                    prate[k_] = p_ if k_ not in prate else 0.7 * prate[k_] + 0.3 * p_
            fpf = (lambda p_: (1 - p_) ** 2) if x.pfsp_mode == "hard" else (lambda p_: 4 * p_ * (1 - p_))
            lw["w"] = [
                w0 if k_ in NOT_FIELD or k_ not in prate else w0 * max(x.pfsp, fpf(prate[k_]))
                for w0, k_ in zip(wts, pkey)
            ]
        gate, promoted = None, False
        if x.anchor_gate > 0:
            cur = [0, 0.0, 0, 0.0]  # games against the CURRENT anchor (the version at this rank's actor) and the field
            for st_, inf_ in gathered:
                if inf_["aver"] != anchor["ver"]:
                    continue
                b_ = st_.get("best", [0, 0.0])
                cur[0] += b_[0]
                cur[1] += b_[1]
                for n, v in st_.items():
                    if n not in NOT_FIELD:
                        cur[2] += v[0]
                        cur[3] += v[1]
            awin.append(cur)
            wn = [sum(c[q] for c in awin[-x.anchor_window :]) for q in range(4)]
            pb, pf = wn[1] / max(1, wn[0]), wn[3] / max(1, wn[2])
            gate = {
                "n": wn[0],
                "P_best": round(pb, 4),
                "P_field_w": round(pf, 4),
                "field_ref": round(afield[0], 4),
                "iters": min(len(awin), x.anchor_window),
                "ver": anchor["ver"],
            }
            if (
                len(awin) >= x.anchor_window
                and wn[0] >= x.anchor_min_games
                and pb >= x.anchor_gate
                and pf >= afield[0] - x.anchor_field_tol
                and played is not None
            ):
                ref.load_state_dict(played[0], strict=False)  # the weights that played the last iteration of the window
                with pub_lock:
                    anchor["w"] = played
                    anchor["ver"] += 1
                afield[0] = pf
                awin.clear()
                promoted = True
                if rank == 0:
                    torch.save(
                        {
                            "model": {k: v.detach().cpu() for k, v in played[0].items()},
                            **meta,
                            "step": it,
                            "kind": "ppo-anchor",
                            "gate": gate,
                        },
                        os.path.join(x.out, f"anchor_{it:05d}.pt"),
                    )
        if rank == 0:
            vs = {
                n: {"n": v[0], "P": round(v[1] / v[0], 3), "bank": round(v[2] / v[0]), "margin": round(v[3] / v[0])}
                for n, v in sorted(tot.items())
            }
            rt_ = [sum(inf_.get("rec", [0, 0, 0])[q] for _, inf_ in gathered) for q in range(3)]
            rec_ = {"n": rt_[0], "P": round(rt_[1] / rt_[0], 3), "d": round(rt_[2] / rt_[0])} if rt_[0] else None
            tnow = time.time()
            rec = {
                "iter": it,
                "t_iter": round(tnow - tprev, 1),
                "t_upd": round(t_upd, 1),
                "t_rc": round(t_rc, 1),
                "t_lwait": round(t_lwait, 1),
                "fast": fastlog,
                **{k: round(v, 1) for k, v in info.items() if not k.startswith("pv") and k != "rec"},
                "mw": round(info["mw"], 4),
                "rec": rec_,
                "games_rank0": NA,
                "kl_coef": round(kf, 5),
                "P_field": round(pw, 4),
                "R_mean": round(float(R.mean()), 4),
                "v_mean": round(float(val.mean()), 4),
                "ev": round(float(ev_all), 4),
                "ev_early": round(float(ev_early), 4),
                **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in agg.items()},
                "anchor": gate,
                "promoted": promoted,
                "t_cur": round(t_cur, 1),
                "skip": it in skip_upd,
                "pv": {os.path.basename(k_): [v[0], round(v[1] / max(1, v[0]), 3)] for k_, v in per.items()},
                "published": pub_ev,
                "pfsp": {k_: round(w_, 3) for k_, w_ in zip(pkey, lw["w"])} if x.pfsp > 0 else None,
                "vs": vs,
            }
            tprev = tnow
            log.write(json.dumps(rec) + "\n")
            log.flush()
            plog.write(json.dumps({"iter": it, "stat": pool.stat}) + "\n")
            plog.flush()
            print(json.dumps({k: v for k, v in rec.items() if k != "vs"}), flush=True)
            if (it + 1) % x.save_every == 0 or it + 1 == x.start_iter + x.iters:
                torch.save(
                    {"model": net.state_dict(), **meta, "step": it + 1, "kind": "ppo", "opt": opt.state_dict()},
                    os.path.join(x.out, f"ckpt_{it + 1:05d}.pt"),
                )
                if ema is not None:
                    esd = {n: e.detach().cpu() for (n, _), e in zip(net.named_parameters(), ema)}
                    torch.save(
                        {"model": esd, **meta, "step": it + 1, "kind": "ppo-ema", "tau": x.ema},
                        os.path.join(x.out, f"ema_{it + 1:05d}.pt"),
                    )
    th.join(timeout=60)
    for c in conns:
        c.send(None)
    for p in procs:
        p.join(timeout=30)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
