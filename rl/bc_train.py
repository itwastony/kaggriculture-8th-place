"""Behaviour cloning (+ a value head trained on the outcome of the teacher's game). One GPU, or DDP via torchrun.

    torchrun --standalone --nproc_per_node G -m rl.bc_train --data 'runs/d7/bcgen/bc/bc_*.pkl.gz' --val-mod 50 \\
        --out runs/d7/bc --steps 3000 --bs 2048/G --width 384 --layers 8 --init INIT.pt
entry_points/train_bc.py runs it with configs/bc_d7.json. --bs is per GPU. Loss: cross-entropy on the (target,
operation) pair of every labelled unit + cross-entropy on every market head + 0.25 x value MSE.
Log: OUT/train.jsonl (one record per 100 steps, validation every 1,000 steps), checkpoints OUT/ckpt_<step>.pt.
"""

import argparse
import glob
import json
import math
import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from . import spec as S
from .data import ShardSteps, collate
from .model import HEAD_SIZES, Net, market_logp, unit_logp


def losses(net, b, dev, vw=0.25):
    """Loss and metrics of one batch (accuracies: unit pair, unit operation, market heads, non-zero market bins)."""
    b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
    with torch.autocast("cuda" if dev.type == "cuda" else "cpu", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        lg, ml, v = net(b["tiles"], b["units"], b["unit_pos"], b["unit_mask"], b["glob"])
    lp_u = unit_logp(lg, b["tmask"], b["has_lab"], b["target"], b["op"])  # [B,16]
    n_u = b["has_lab"].sum().clamp(min=1)
    l_unit = -(lp_u.sum() / n_u)
    lp_m = market_logp(ml, b["mmask"], b["mchoice"])  # [B,H]
    l_mkt = -lp_m.mean()
    l_val = ((v.float() - b["win"]) ** 2).mean()
    with torch.no_grad():
        B = lg.shape[0]
        flat = lg.float().masked_fill(~b["tmask"], -1e9).view(B, S.MAX_UNITS, -1).argmax(-1)
        lab = b["target"] * S.NOPS + b["op"]
        acc_u = ((flat == lab) & b["has_lab"]).sum() / n_u
        acc_op = (((flat % S.NOPS) == b["op"]) & b["has_lab"]).sum() / n_u
        o, accs = 0, []
        mlm = ml.float().masked_fill(~b["mmask"], -1e9)
        for h, n in enumerate(HEAD_SIZES):
            accs.append((mlm[:, o : o + n].argmax(-1) == b["mchoice"][:, h]).float().mean())
            o += n
        acc_m = torch.stack(accs).mean()
        nz = b["mchoice"] != 0
        acc_mnz = torch.stack(
            [
                (
                    (mlm[:, sum(HEAD_SIZES[:h]) : sum(HEAD_SIZES[: h + 1])].argmax(-1) == b["mchoice"][:, h]) & nz[:, h]
                ).sum()
                for h in range(len(HEAD_SIZES))
            ]
        ).sum() / nz.sum().clamp(min=1)
    loss = l_unit + l_mkt + vw * l_val
    return loss, {
        "l_unit": l_unit.item(),
        "l_mkt": l_mkt.item(),
        "l_val": l_val.item(),
        "acc_unit": acc_u.item(),
        "acc_op": acc_op.item(),
        "acc_mkt": acc_m.item(),
        "acc_mkt_nz": acc_mnz.item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default="")
    ap.add_argument("--val-mod", type=int, default=0)  # N > 0: shards whose number is divisible by N -> validation
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=512)  # per GPU
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", dest="d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--keep", type=float, default=0.5)  # fraction of a game's steps used per epoch (decorrelation)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--init", default="")
    x = ap.parse_args()
    # RL_NO_CUDNN_SDP=1: attention without cuDNN, for machines whose system cuDNN differs from the one torch ships
    # (CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH)
    if os.environ.get("RL_NO_CUDNN_SDP"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(1234 + rank)
    files = sorted(glob.glob(x.data))
    vfiles = sorted(glob.glob(x.val)) if x.val else []
    if x.val_mod:
        num = lambda p: int(os.path.basename(p).split("_")[1].split(".")[0])  # noqa: E731
        vfiles = [p for p in files if num(p) % x.val_mod == 0]
        files = [p for p in files if num(p) % x.val_mod != 0]
    if rank == 0:
        os.makedirs(x.out, exist_ok=True)
        print(f"train shards {len(files)}, val shards {len(vfiles)}, world {world}", flush=True)
    net = Net(d=x.d, layers=x.layers).to(dev)
    if x.init:
        net.load_state_dict(torch.load(x.init, map_location="cpu")["model"])
    model = torch.nn.parallel.DistributedDataParallel(net, device_ids=[dev.index]) if ddp else net
    opt = torch.optim.AdamW(net.parameters(), lr=x.lr, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / x.steps)))
    )
    dl = DataLoader(
        ShardSteps(files, seed=1, keep=x.keep, rank=rank, world=world),
        batch_size=x.bs,
        num_workers=x.workers,
        collate_fn=collate,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    vdl = (
        DataLoader(ShardSteps(vfiles, seed=2, keep=0.2), batch_size=x.bs, num_workers=4, collate_fn=collate)
        if vfiles and rank == 0
        else None
    )
    log = open(os.path.join(x.out, "train.jsonl"), "a") if rank == 0 else None
    t0, agg = time.time(), {}
    it = iter(dl)
    for step in range(1, x.steps + 1):
        b = next(it)
        model.train()
        loss, m = losses(model, b, dev)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v
        if rank == 0 and step % 100 == 0:
            rec = {
                "step": step,
                "lr": sched.get_last_lr()[0],
                "sec": round(time.time() - t0, 1),
                **{k: round(v / 100, 4) for k, v in agg.items()},
            }
            agg = {}
            if vdl is not None and step % 1000 == 0:
                model.eval()
                vm, n = {}, 0
                with torch.no_grad():
                    for vb in vdl:
                        _, mm = losses(net, vb, dev)
                        for k, v in mm.items():
                            vm[k] = vm.get(k, 0.0) + v
                        n += 1
                        if n >= 20:
                            break
                rec.update({"val_" + k: round(v / n, 4) for k, v in vm.items()})
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
        if rank == 0 and (step % x.save_every == 0 or step == x.steps):
            torch.save(
                {"model": net.state_dict(), "d": x.d, "layers": x.layers, "step": step},
                os.path.join(x.out, f"ckpt_{step:06d}.pt"),
            )
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
