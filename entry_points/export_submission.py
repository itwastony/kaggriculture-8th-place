#!/usr/bin/env python3
"""Step 5 - build the Kaggle submission archive from a checkpoint (configs/export_d7b.json).

    python entry_points/export_submission.py --run d7                       # the shipped d7b EMA 110 checkpoint
    python entry_points/export_submission.py --run d7 --ckpt runs/d7/appo/ema_00110.pt --name my_agent

The archive holds main.py (agent entry point; NumPy only, no torch), weights.npz (float16) and rl_<sha8>/ (copies of
rl/{spec,encode,act,np_net}.py; the package name is a hash of their source, so that two versions never mix in one
process). Upload it with `kaggle competitions submit kaggriculture -f <archive> -m <message>`.

Output: OUTPUT_DIR/<run>/export/<name>/ (the unpacked agent; replaced if it exists) and OUTPUT_DIR/<run>/<name>.tar.gz;
the sha256 of the archive, main.py and weights.npz are printed. Line 1 of main.py (a comment with the absolute
checkpoint path) is replaced by a neutral comment, and the tar headers carry no local user name.
"""

import argparse
import os
import shutil
import sys
import tarfile

from common import fill, load_config, run, run_dir, settings, sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="d7", help="run name under OUTPUT_DIR")
    ap.add_argument("--ckpt", default="", help="checkpoint to export (default: the ckpt of configs/export_d7b.json)")
    ap.add_argument("--name", default="rl_d7b_110", help="agent folder and archive name")
    x = ap.parse_args()
    s = settings()
    rd = run_dir(x.run)
    cfg = fill(load_config("export_d7b.json"), {**s, "RUN": rd})
    ckpt = os.path.abspath(x.ckpt or cfg["ckpt"])
    root = os.path.join(rd, "export")
    d = os.path.join(root, x.name)
    if os.path.exists(d):
        shutil.rmtree(d)  # only our own previous export of the same name
    cmd = [sys.executable, "-m", "rl.export_agent", "--ckpt", ckpt, "--name", x.name, "--root", root]
    for k, v in cfg["args"].items():
        cmd += [k, str(v)]
    if run(cmd):
        sys.exit("export failed")
    # line 1 of main.py is a comment with the absolute checkpoint path; keep local paths out of the archive
    mp = os.path.join(d, "main.py")
    lines = open(mp).read().split("\n")
    lines[0] = (
        f"# RL agent (transformer policy, NumPy). Checkpoint: {os.path.basename(ckpt)}, "
        f"{cfg['args']['--dtype']} weights."
    )
    open(mp, "w").write("\n".join(lines))
    tar = os.path.join(rd, f"{x.name}.tar.gz")

    def anon(ti):  # no local user names in the archive
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ""
        return ti

    with tarfile.open(tar, "w:gz") as t:
        for f in sorted(os.listdir(d)):
            t.add(os.path.join(d, f), arcname=f, filter=anon)
    print(
        f"archive {tar}\n  sha256 {sha256(tar)}\n  main.py {sha256(mp)}\n"
        f"  weights.npz {sha256(os.path.join(d, 'weights.npz'))}"
    )


if __name__ == "__main__":
    main()
