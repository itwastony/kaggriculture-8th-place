"""Synchronous PPO on top of the BC policy (the predecessor of rl/appo.py, which imports its helpers from here).

Rank = 1 GPU (torchrun); each rank has W CPU workers, each worker K parallel kagsim games. All games run synchronously
for T=720 steps: at every step the workers encode the observations of the RL seats into a shared buffer (shared
memory), the rank evaluates the policy on the GPU in one batch and hands out the samples, the workers execute and step.
While the GPU computes, the workers compute the turns of the code-bot opponents (overlap).

    torchrun --standalone --nproc_per_node NPROC -m rl.ppo --init BC.pt --out runs/X/out --iters 300
    python -m rl.ppo --device cpu --init '' --width 64 --layers 2 --workers 2 --games 2 --T 48 --iters 2 \\
        --out runs/ppo_smoke

Opponents (league LEAGUE, or --league JSON): code bots of the field + the current policy ("self": both seats are
trained) + past versions ("past": snapshots every --snap-every iterations, plus the starting policy, or the checkpoints
given by --past; not trained). Most bots of the default LEAGUE are not shipped in this repository (agents/README.md):
pass --league with the bots you have.
Reward at the end of the game: outcome (+1 / 0 / -1) + margin_w * clip(margin / margin_scale, -1, 1); 0 during the game.
Losses: PPO-clip on every action component (unit, market head) with a shared advantage (GAE), value MSE, entropy, KL to
a frozen anchor (the starting policy, or --ref; coefficient --kl-ref).
Log: OUT/ppo.jsonl; checkpoints OUT/ckpt_<iter>.pt (BC format: rl/policy.py reads them).
"""

import argparse
import copy
import json
import os
import random
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from . import encode as E
from . import spec as S
from .model import HEAD_SIZES, Net, market_logp, unit_logp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENC = ("tiles", "units", "unit_pos", "unit_mask", "glob", "tmask", "mmask")
H = len(HEAD_SIZES)
# (agent, weight); "self" = the current policy, "past" = a snapshot of a past version (the first one is the starting
# policy). The code bots are the field we trained against; most of them are not in this repository (agents/README.md).
LEAGUE = [
    ("self", 4),
    ("past", 3),
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


def buffers(n):
    return {
        "tiles": torch.zeros(n, S.NT, E.FT),
        "units": torch.zeros(n, S.MAX_UNITS, E.FU),
        "unit_pos": torch.zeros(n, S.MAX_UNITS, dtype=torch.long),
        "unit_mask": torch.zeros(n, S.MAX_UNITS, dtype=torch.bool),
        "glob": torch.zeros(n, E.FG),
        "tmask": torch.zeros(n, S.MAX_UNITS, S.NT, S.NOPS, dtype=torch.bool),
        "mmask": torch.zeros(n, E.MB, dtype=torch.bool),
        "uc": torch.zeros(n, S.MAX_UNITS, dtype=torch.long),
        "mc": torch.zeros(n, H, dtype=torch.long),
    }


# ---------------------------------------------------------------- worker (CPU, its own K games)


def _worker(off, K, T, bufs, conn):
    os.chdir(ROOT)
    torch.set_num_threads(1)
    import kagsim

    from . import act as A
    from .record import _base_config, _load, call_agent

    nb = {k: v.numpy() for k, v in bufs.items()}
    cfg0 = dict(_base_config())
    while True:
        specs = conn.recv()
        if specs is None:
            return
        games, fns, cfgs = [], [], []
        for sp in specs:
            games.append(kagsim.Game(sp["seed"], T))
            c = dict(cfg0)
            c["seed"] = sp["seed"]
            cfgs.append(c)
            fns.append(_load(sp["opp"]) if sp["opp"] else None)
        obs = [[None, None] for _ in specs]
        for _ in range(T):
            for k, sp in enumerate(specs):
                for s in sp["seats"]:
                    o = games[k].observe(s)
                    obs[k][s] = o
                    e = E.encode(o)
                    slot = off + 2 * k + s
                    for key in ENC:
                        nb[key][slot] = e[key]
            conn.send(1)
            opp_act = [None] * len(specs)
            for k, sp in enumerate(specs):
                if sp["opp"]:
                    try:
                        opp_act[k] = call_agent(fns[k], games[k].observe(1 - sp["seats"][0]), cfgs[k], sp["opp"]) or {}
                    # a crashed opponent = PASS, as on the server before ERROR
                    except Exception:  # noqa: BLE001
                        opp_act[k] = {}
            conn.recv()
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
        conn.send([[games[k].reward(0), games[k].reward(1)] for k in range(len(specs))])


# ---------------------------------------------------------------- action selection


def sample(lg, ml, b, greedy=False):
    """Unit codes [B,16] and market bins [B,H] by Gumbel-max sampling over the legal actions (argmax if greedy)."""
    B = lg.shape[0]
    flat = lg.float().masked_fill(~b["tmask"], -1e9).view(B, S.MAX_UNITS, -1)
    if not greedy:
        flat = flat - torch.log(-torch.log(torch.rand_like(flat).clamp_(1e-20, 1.0)))
    uc = flat.argmax(-1)
    mlm = ml.float().masked_fill(~b["mmask"], -1e9)
    if not greedy:
        mlm = mlm - torch.log(-torch.log(torch.rand_like(mlm).clamp_(1e-20, 1.0)))
    mc, o = [], 0
    for n in HEAD_SIZES:
        mc.append(mlm[:, o : o + n].argmax(-1))
        o += n
    return uc, torch.stack(mc, -1)


def kl_to(lg, ml, rlg, rml, b):
    """KL(pi || pi_ref) per unit [B,16] and per market head [B,H]."""
    B = lg.shape[0]
    lp = F.log_softmax(lg.float().masked_fill(~b["tmask"], -1e9).view(B, S.MAX_UNITS, -1), -1)
    lq = F.log_softmax(rlg.float().masked_fill(~b["tmask"], -1e9).view(B, S.MAX_UNITS, -1), -1)
    ku = (lp.exp() * (lp - lq)).sum(-1) * b["unit_mask"]
    a = ml.float().masked_fill(~b["mmask"], -1e9)
    r = rml.float().masked_fill(~b["mmask"], -1e9)
    km, o = [], 0
    for n in HEAD_SIZES:
        p, q = F.log_softmax(a[:, o : o + n], -1), F.log_softmax(r[:, o : o + n], -1)
        km.append((p.exp() * (p - q)).sum(-1))
        o += n
    return ku, torch.stack(km, -1)


def unit_terms(lg, rlg, b, uc):
    """log p of the chosen action, entropy and KL(pi || pi_ref) per unit [B,16].

    One log_softmax per output (saves memory and time)."""
    B = lg.shape[0]
    m = ~b["tmask"].reshape(B, S.MAX_UNITS, -1)
    um = b["unit_mask"]
    lp = F.log_softmax(lg.float().reshape(B, S.MAX_UNITS, -1).masked_fill(m, -1e9), -1)
    lpu = lp.gather(-1, uc.clamp(min=0).unsqueeze(-1)).squeeze(-1) * um
    p = lp.exp()
    eu = -(p * lp).sum(-1) * um
    lq = F.log_softmax(rlg.float().reshape(B, S.MAX_UNITS, -1).masked_fill(m, -1e9), -1)
    ku = (p * (lp - lq)).sum(-1) * um
    return lpu, eu, ku


def market_terms(ml, rml, b, mc):
    """log p of the chosen bin, entropy and KL(pi || pi_ref) per market head [B,H]."""
    a = ml.float().masked_fill(~b["mmask"], -1e9)
    r = rml.float().masked_fill(~b["mmask"], -1e9)
    lps, ents, kls, o = [], [], [], 0
    for h, n in enumerate(HEAD_SIZES):
        p_, q_ = F.log_softmax(a[:, o : o + n], -1), F.log_softmax(r[:, o : o + n], -1)
        e_ = p_.exp()
        lps.append(p_.gather(-1, mc[:, h : h + 1]).squeeze(-1))
        ents.append(-(e_ * p_).sum(-1))
        kls.append((e_ * (p_ - q_)).sum(-1))
        o += n
    return torch.stack(lps, -1), torch.stack(ents, -1), torch.stack(kls, -1)


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
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatches", type=int, default=64)  # per epoch (the same on all ranks — DDP synchronization)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--ent", type=float, default=1e-3)
    ap.add_argument("--kl-ref", type=float, default=0.02)
    ap.add_argument("--kl-ref-final", type=float, default=None)  # linearly to this value over --kl-decay-iters
    ap.add_argument("--kl-decay-iters", type=int, default=200)
    ap.add_argument("--margin-w", type=float, default=0.3)
    ap.add_argument("--margin-scale", type=float, default=30000.0)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--width", dest="d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--league", default="")  # JSON [[path | "self" | "past", weight], ...] instead of LEAGUE
    ap.add_argument("--snap-every", type=int, default=10)
    ap.add_argument("--vf-warmup", type=int, default=0)  # first N iterations: only the value is trained
    ap.add_argument("--vf-lam", type=float, default=1.0)  # λ for the value target (1 = game outcome, Monte-Carlo)
    ap.add_argument("--vf-lr", type=float, default=1e-4)  # lr of the value parameters (the tower is fresh)
    ap.add_argument("--vtower", type=int, default=2)  # value tower layers (0 = MLP on the detached trunk)
    # >0: adaptive policy lr from the iteration's approx_kl (as in rl_games)
    ap.add_argument("--target-kl", type=float, default=0.0)
    ap.add_argument("--lr-max", type=float, default=3e-4)
    ap.add_argument("--lr-min", type=float, default=1e-6)
    ap.add_argument("--ref", default="")  # KL anchor (default = --init); when continuing = BC
    # comma-separated checkpoints: the starting pool of "past" snapshots
    ap.add_argument("--past", default="")
    ap.add_argument("--profile", type=int, default=0)  # 1: time breakdown by stage (with GPU synchronization)
    x = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl" if x.device == "cuda" else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
        if x.device == "cuda":
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    dev = torch.device("cuda") if x.device == "cuda" else torch.device("cpu")
    torch.manual_seed(4321 + rank)
    torch.set_num_threads(4)
    league = json.loads(x.league) if x.league else LEAGUE

    if x.init:
        ck = torch.load(x.init, map_location="cpu", weights_only=False)
        d, layers, vt = ck.get("d", x.d), ck.get("layers", x.layers), ck.get("vtower", x.vtower)
        net = Net(d=d, layers=layers, vtower=vt)
        miss, unexp = net.load_state_dict(ck["model"], strict=False)
        assert not unexp and all(k.startswith(("venc.", "vnorm.")) for k in miss), (miss, unexp)
    else:
        d, layers, vt = x.d, x.layers, x.vtower
        net = Net(d=d, layers=layers, vtower=vt)
    net = net.to(dev)
    if x.ref:
        rk = torch.load(x.ref, map_location="cpu", weights_only=False)
        ref = Net(d=d, layers=layers, vtower=vt)
        ref.load_state_dict(rk["model"], strict=False)
        ref = ref.to(dev).eval()
    else:
        ref = copy.deepcopy(net).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    snap_net = copy.deepcopy(net).eval()
    for p in snap_net.parameters():
        p.requires_grad_(False)
    snaps = [{k: v.detach().cpu().clone() for k, v in net.state_dict().items()}]
    if x.past:  # continuation: pool = the given checkpoints (+ the current policy)
        snaps = [torch.load(q, map_location="cpu", weights_only=False)["model"] for q in x.past.split(",")] + snaps
    model = (
        torch.nn.parallel.DistributedDataParallel(net, device_ids=[dev.index] if dev.type == "cuda" else None)
        if ddp
        else net
    )
    vpar = [p for n, p in net.named_parameters() if n.startswith(("venc.", "vnorm.", "value."))]
    ppar = [p for n, p in net.named_parameters() if not n.startswith(("venc.", "vnorm.", "value."))]
    opt = torch.optim.AdamW(
        [{"params": ppar, "lr": x.lr}, {"params": vpar, "lr": x.vf_lr}], weight_decay=0.0, betas=(0.9, 0.99), eps=1e-6
    )
    if x.init and "opt" in ck:  # continuation: Adam state and lr
        opt.load_state_dict(ck["opt"])
        for g_ in opt.param_groups[1:]:
            g_["lr"] = x.vf_lr
        if not x.target_kl:
            opt.param_groups[0]["lr"] = x.lr
    amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")

    def sync():
        if x.profile and dev.type == "cuda":
            torch.cuda.synchronize()

    W, K, T = x.workers, x.games, x.T
    NS = W * K * 2
    bufs = buffers(NS)
    for v in bufs.values():
        v.share_memory_()
    ctx = mp.get_context("spawn")
    conns, procs = [], []
    for w in range(W):
        a, b = ctx.Pipe()
        p = ctx.Process(target=_worker, args=(w * 2 * K, K, T, bufs, b), daemon=True)
        p.start()
        conns.append(a)
        procs.append(p)

    store = {k: torch.zeros((T,) + tuple(v.shape), dtype=v.dtype, device=dev) for k, v in bufs.items()}
    store["lpu"] = torch.zeros(T, NS, S.MAX_UNITS, device=dev)
    store["lpm"] = torch.zeros(T, NS, H, device=dev)
    store["val"] = torch.zeros(T, NS, device=dev)
    if rank == 0:
        os.makedirs(x.out, exist_ok=True)
        log = open(os.path.join(x.out, "ppo.jsonl"), "a")
        print(
            f"ranks {world}, workers {W}x{K} games per rank, slots {NS}, T {T}, d {d} layers {layers} vtower {vt}",
            flush=True,
        )

    def recv_all():
        for c in conns:
            if not c.poll(900):
                raise RuntimeError("worker did not respond within 900 s")
            c.recv()

    names = [n for n, _ in league]
    wts = [w_ for _, w_ in league]
    for it in range(x.start_iter, x.start_iter + x.iters):
        t0 = time.time()
        rng = random.Random((x.seed0 + it) * 7919 + rank)
        specs_all, active, slot_opp, past_idx = [], [], [], []
        si = rng.randrange(len(snaps))
        snap_net.load_state_dict(snaps[si], strict=False)
        for w in range(W):
            specs = []
            for k in range(K):
                seed = x.seed0 + ((it * world + rank) * W + w) * K + k
                opp = rng.choices(names, wts)[0]
                me = rng.randint(0, 1)
                if opp == "self":
                    specs.append({"seed": seed, "seats": [0, 1], "opp": None})
                    learn = [0, 1]
                elif opp == "past":
                    specs.append({"seed": seed, "seats": [0, 1], "opp": None})
                    learn = [me]
                    past_idx.append(w * 2 * K + 2 * k + 1 - me)
                else:
                    specs.append({"seed": seed, "seats": [me], "opp": opp})
                    learn = [me]
                for s in learn:
                    active.append(w * 2 * K + 2 * k + s)
                    slot_opp.append(opp if opp in ("self", "past") else os.path.basename(os.path.dirname(opp)))
            specs_all.append(specs)
            conns[w].send(specs)
        idx = torch.tensor(active)
        pidx = torch.tensor(past_idx, dtype=torch.long)
        NA = len(active)
        net.eval()
        tw = tg = 0.0
        for t in range(T):
            ta = time.time()
            recv_all()
            tb = time.time()
            tw += tb - ta
            b = {k: bufs[k][idx].to(dev, non_blocking=True) for k in ENC}
            with torch.no_grad(), torch.autocast(**amp):
                lg, ml, v = net(b["tiles"], b["units"], b["unit_pos"], b["unit_mask"], b["glob"])
            with torch.no_grad():
                uc, mc = sample(lg, ml, b)
                lpu = unit_logp(lg, b["tmask"], b["unit_mask"], uc // S.NOPS, uc % S.NOPS)
                lpm = market_logp(ml, b["mmask"], mc)
            bufs["uc"][idx] = uc.cpu()
            bufs["mc"][idx] = mc.cpu()
            if len(past_idx):
                pb = {k: bufs[k][pidx].to(dev, non_blocking=True) for k in ENC}
                with torch.no_grad(), torch.autocast(**amp):
                    plg, pml, _ = snap_net(
                        pb["tiles"], pb["units"], pb["unit_pos"], pb["unit_mask"], pb["glob"], value=False
                    )
                with torch.no_grad():
                    puc, pmc = sample(plg, pml, pb)
                bufs["uc"][pidx] = puc.cpu()
                bufs["mc"][pidx] = pmc.cpu()
            for c in conns:
                c.send(1)
            tg += time.time() - tb
            for k in ENC:
                store[k][t, :NA] = b[k]
            store["uc"][t, :NA] = uc
            store["mc"][t, :NA] = mc
            store["lpu"][t, :NA] = lpu
            store["lpm"][t, :NA] = lpm
            store["val"][t, :NA] = v.float()
        rews = []
        for c in conns:
            if not c.poll(900):
                raise RuntimeError("worker did not return the result")
            rews.append(c.recv())
        t_roll = time.time() - t0
        # game results per active slot
        R = torch.zeros(NA)
        stats = {}
        for j, sl in enumerate(active):
            w, r = divmod(sl, 2 * K)
            k, s = divmod(r, 2)
            me, op = rews[w][k][s], rews[w][k][1 - s]
            m = me - op
            R[j] = (1.0 if m > 0 else (-1.0 if m < 0 else 0.0)) + x.margin_w * max(-1.0, min(1.0, m / x.margin_scale))
            st = stats.setdefault(slot_opp[j], [0, 0.0, 0.0, 0.0])
            st[0] += 1
            st[1] += 1.0 if m > 0 else (0.5 if m == 0 else 0.0)
            st[2] += me
            st[3] += m
        R = R.to(dev)
        # GAE (reward only at the last step): advantages with λ=lam, value target with λ=vf_lam
        val = store["val"][:, :NA]
        adv = torch.zeros(T, NA, device=dev)
        advv = torch.zeros(T, NA, device=dev)
        last = torch.zeros(NA, device=dev)
        lastv = torch.zeros(NA, device=dev)
        for t in reversed(range(T)):
            nv = val[t + 1] if t + 1 < T else torch.zeros(NA, device=dev)
            r_t = R if t == T - 1 else 0.0
            delta = r_t + x.gamma * nv - val[t]
            last = delta + x.gamma * x.lam * last
            lastv = delta + x.gamma * x.vf_lam * lastv
            adv[t] = last
            advv[t] = lastv
        ret = advv + val
        an = (adv - adv.mean()) / (adv.std() + 1e-8)
        # fraction of the outcome variance explained by the value (over all steps and over the opening t<144)
        rv = R.unsqueeze(0).expand(T, NA)
        ev_all = float(1 - (rv - val).var() / (rv.var() + 1e-8))
        ev_early = float(1 - (rv[:144] - val[:144]).var() / (rv[:144].var() + 1e-8))
        # update
        t1 = time.time()
        net.train()
        kf = (
            x.kl_ref
            if x.kl_ref_final is None
            else x.kl_ref + (x.kl_ref_final - x.kl_ref) * min(1.0, it / max(1, x.kl_decay_iters))
        )
        N = T * NA
        mbs = N // x.minibatches
        agg = {}
        for _ in range(x.epochs):
            perm = torch.randperm(N, device=dev)
            for i in range(x.minibatches):
                j = perm[i * mbs : (i + 1) * mbs]
                tt, aa = j // NA, j % NA
                mb = {k: store[k][tt, aa] for k in ENC}
                sync()
                tq0 = time.time()
                with torch.autocast(**amp):
                    lg, ml, v = model(mb["tiles"], mb["units"], mb["unit_pos"], mb["unit_mask"], mb["glob"])
                    with torch.no_grad():
                        rlg, rml, _ = ref(
                            mb["tiles"], mb["units"], mb["unit_pos"], mb["unit_mask"], mb["glob"], value=False
                        )
                sync()
                tq1 = time.time()
                uc, mc = store["uc"][tt, aa], store["mc"][tt, aa]
                lpu, eu, ku = unit_terms(lg, rlg, mb, uc)
                lpm, em, km = market_terms(ml, rml, mb, mc)
                um = mb["unit_mask"].float()
                A_ = an[tt, aa].unsqueeze(-1)
                ru = torch.exp(lpu - store["lpu"][tt, aa])
                rm = torch.exp(lpm - store["lpm"][tt, aa])
                su = torch.min(ru * A_, ru.clamp(1 - x.clip, 1 + x.clip) * A_) * um
                sm = torch.min(rm * A_, rm.clamp(1 - x.clip, 1 + x.clip) * A_)
                ncomp = um.sum(-1) + H
                pg = -((su.sum(-1) + sm.sum(-1)) / ncomp).mean()
                vl = ((v.float() - ret[tt, aa]) ** 2).mean()
                ent = ((eu.sum(-1) + em.sum(-1)) / ncomp).mean()
                klr = ((ku.sum(-1) + km.sum(-1)) / ncomp).mean()
                pol = pg - x.ent * ent + kf * klr
                # critic warm-up: the policy stays in the graph with weight 0 (otherwise DDP waits for gradients of
                # unused parameters)
                loss = x.vf * vl + (0.0 if it < x.vf_warmup else 1.0) * pol
                sync()
                tq2 = time.time()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(ppar, 1.0)
                torch.nn.utils.clip_grad_norm_(vpar, 1.0)
                opt.step()
                sync()
                tq3 = time.time()
                if x.profile:
                    for kk, vv in (("p_fwd", tq1 - tq0), ("p_loss", tq2 - tq1), ("p_bwd", tq3 - tq2)):
                        agg[kk] = agg.get(kk, 0.0) + vv
                with torch.no_grad():
                    old = torch.cat([store["lpu"][tt, aa] * um, store["lpm"][tt, aa]], -1)
                    new = torch.cat([lpu * um, lpm], -1)
                    approx_kl = ((old - new).sum(-1) / ncomp).mean()
                    cf = (((ru - 1).abs() > x.clip).float() * um).sum() + ((rm - 1).abs() > x.clip).float().sum()
                    cf = cf / ncomp.sum()
                for kk, vv in (
                    ("pg", pg),
                    ("vl", vl),
                    ("ent", ent),
                    ("kl_ref", klr),
                    ("approx_kl", approx_kl),
                    ("clipfrac", cf),
                    ("gnorm", gn),
                ):
                    agg[kk] = agg.get(kk, 0.0) + float(vv.detach()) / (x.epochs * x.minibatches)
        t_upd = time.time() - t1
        agg["lr"] = opt.param_groups[0]["lr"]
        if x.profile:
            agg["p_wait"], agg["p_gpu"] = tw, tg
        if x.target_kl and it >= x.vf_warmup:  # adaptive lr: the same on all ranks (mean approx_kl)
            kt = torch.tensor([agg["approx_kl"]], device=dev)
            if ddp:
                dist.all_reduce(kt)
                kt /= world
            g0 = opt.param_groups[0]
            if kt.item() > 2.0 * x.target_kl:
                g0["lr"] = max(x.lr_min, g0["lr"] / 1.5)
            elif kt.item() < 0.5 * x.target_kl:
                g0["lr"] = min(x.lr_max, g0["lr"] * 1.5)
        if (it + 1) % x.snap_every == 0:
            snaps.append({k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
        # summary over all ranks
        gathered = [None] * world
        if ddp:
            dist.all_gather_object(gathered, stats)
        else:
            gathered = [stats]
        if rank == 0:
            tot = {}
            for st in gathered:
                for n, v in st.items():
                    a = tot.setdefault(n, [0, 0.0, 0.0, 0.0])
                    for q in range(4):
                        a[q] += v[q]
            vs = {
                n: {"n": v[0], "P": round(v[1] / v[0], 3), "bank": round(v[2] / v[0]), "margin": round(v[3] / v[0])}
                for n, v in sorted(tot.items())
            }
            fields = [n for n in tot if n not in ("self", "past")]
            pw = sum(tot[n][1] for n in fields) / max(1, sum(tot[n][0] for n in fields))
            rec = {
                "iter": it,
                "t_roll": round(t_roll, 1),
                "t_upd": round(t_upd, 1),
                "games_rank0": NA,
                "kl_coef": round(kf, 5),
                "P_field": round(pw, 4),
                "R_mean": round(float(R.mean()), 4),
                "v_mean": round(float(val.mean()), 4),
                "ev": round(ev_all, 4),
                "ev_early": round(ev_early, 4),
                **{k: round(v, 5) for k, v in agg.items()},
                "vs": vs,
            }
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps({k: v for k, v in rec.items() if k != "vs"}), flush=True)
            if (it + 1) % x.save_every == 0 or it + 1 == x.start_iter + x.iters:
                torch.save(
                    {
                        "model": net.state_dict(),
                        "d": d,
                        "layers": layers,
                        "vtower": vt,
                        "step": it + 1,
                        "kind": "ppo",
                        "opt": opt.state_dict(),
                    },
                    os.path.join(x.out, f"ckpt_{it + 1:05d}.pt"),
                )
    for c in conns:
        c.send(None)
    for p in procs:
        p.join(timeout=30)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
