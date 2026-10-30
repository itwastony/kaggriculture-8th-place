#!/usr/bin/env python3
"""Step 3 - behaviour cloning for the final line d7 (configs/bc_d7.json): 3,000 steps x 2,048 states (global batch).

The initial weights are the actor of an intermediate snapshot of the previous line (d6 EMA 140), with the value
towers dropped (tools/strip_init.py). That file is shipped (artifacts/checkpoints/final/bc_d7_init_d6_140_pol.pt);
--remake-init recreates it from artifacts/checkpoints/past/d6_ema_00140.pt instead.

    python entry_points/train_bc.py --run d7 --nproc G         # G GPUs, 2,048 / G states per GPU per step
    python entry_points/train_bc.py --run d7 --nproc 1         # one GPU, same global batch

The global batch is fixed by the config; the per-GPU batch is 2,048 / G (G must divide 2,048).

Output: OUTPUT_DIR/<run>/bc/{train.jsonl, ckpt_001000.pt, ckpt_002000.pt, ckpt_003000.pt}. The shipped result of the
original run is artifacts/checkpoints/final/bc_d7_ckpt_003000.pt. On the original GPU type and number of GPUs, with
the pinned software, our rerun (Oct 12) reproduced it byte for byte; on other hardware compare the validation
accuracies in train.jsonl with docs/training.md (section 3).
"""

import argparse
import os
import shutil
import sys

from common import args_list, fill, load_config, require, run, run_dir, settings


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="d7", help="run name under OUTPUT_DIR")
    ap.add_argument("--nproc", type=int, required=True, help="GPUs on this node")
    ap.add_argument("--bs", type=int, default=None, help="states per GPU per step (default: global batch / --nproc)")
    ap.add_argument(
        "--remake-init",
        action="store_true",
        help="rebuild the initial weights with tools/strip_init.py instead of copying",
    )
    ap.add_argument("--dry", action="store_true", help="print the commands without running them")
    x = ap.parse_args()
    s = settings()
    rd = run_dir(x.run)
    cfg = fill(load_config("bc_d7.json"), {**s, "RUN": rd})
    require([cfg["args"]["--data"]], "run entry_points/make_bc_data.py first")
    init = os.path.join(rd, "bc_init_d6_140_pol.pt")
    if not os.path.exists(init):
        if x.remake_init:
            src, like = cfg["init_from"]["src"], cfg["init_from"]["like"]
            if run([sys.executable, "tools/strip_init.py", "--src", src, "--like", like, "--out", init], dry=x.dry):
                sys.exit("strip_init failed")
        else:
            shutil.copyfile(cfg["init_from"]["shipped_result"], init)
    a = dict(cfg["args"])
    nproc = x.nproc
    if x.bs:
        a["--bs"] = x.bs
    elif cfg["global_batch"] % nproc:
        sys.exit(f"--nproc must divide the global batch {cfg['global_batch']} (or pass --bs)")
    else:
        a["--bs"] = cfg["global_batch"] // nproc
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(nproc),
        "-m",
        "rl.bc_train",
        *args_list(a),
        "--out",
        cfg["out"],
        "--init",
        init,
    ]
    env = {"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "LC_ALL": "C.UTF-8"}
    sys.exit(run(cmd, env=env, dry=x.dry))


if __name__ == "__main__":
    main()
