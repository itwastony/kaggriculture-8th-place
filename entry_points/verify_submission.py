#!/usr/bin/env python3
"""Verify that the shipped submission archives contain what Kaggle received and that they follow from the checkpoints.

    python entry_points/verify_submission.py                 # checks 1-3 (needs numpy + torch; no games)
    python entry_points/verify_submission.py --games 4       # + check 4 (needs kagsim)

1. sha256 of the archive and of its main.py / weights.npz against the recorded values (main.py / weights.npz: recorded
   at submission time; the archive: the published copy, repacked with a neutral owner in the tar headers, see
   ASSETS.md).
2. Re-export from the checkpoint (entry_points/export_submission.py): weights.npz must be identical (same arrays,
   same dtypes, same bytes).
3. Code: main.py and the rl_<sha8>/ package of the re-export must be the same program as in the archive. The shipped
   archive was built before the code comments were translated to English (its comments are in Russian), so the check
   compares Python ASTs, ignoring the package-name hash and the text of string constants that were in Russian.
4. Optional: both agents (shipped archive and re-export) play --games games on kagsim against a code bot, in both seats;
   every action of every step must be identical.

Prints a JSON report and VERIFIED (exit code 0) or VERIFICATION FAILED (exit code 1). The re-exports go to
OUTPUT_DIR/verify_<name>/.
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile

import numpy as np
from common import ROOT, run_dir, settings, sha256

# sha256 prefixes. main/weights: recorded at submission (Sep 30, 2026); tar: the published copy (the uploaded archives
# were 29d5f125 / 73dbc480)
EXPECTED = {
    "rl_d7b_110": {
        "tar": "bef80bb1",
        "main": "8b0d8f57",
        "weights": "5b154ec0",
        "ckpt": "final/d7b_ema_00110.pt",
        "kaggle_submission_id": 56718362,
    },
    "rl_d6_250": {
        "tar": "d89d96eb",
        "main": "dc1b8bbb",
        "weights": "e360529d",
        "ckpt": "second/d6_ema_00250.pt",
        "kaggle_submission_id": 56713726,
    },
}
CYR = re.compile(r"[Ѐ-ӿ]")


def strip_tsell(tree):
    """Remove the terminal-sell option (_TSELL, _TPROD, _last_step, `if _TSELL and ...`) from a main.py AST.

    rl/export_agent.py gained this option between the two exports (rl_d6_250 was exported without it, rl_d7b_110
    with it). It is off unless --term-sell is given (_TSELL = False in both releases' exports)
    and guarded by `if _TSELL and ...`, so it never runs; the d6_250 archive simply predates it. Returns the value
    _TSELL was set to (must be False for the removal to be valid)."""
    val = None

    def keep(n):
        nonlocal val
        if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in ("_TSELL", "_TPROD") for t in n.targets
        ):
            if any(t.id == "_TSELL" for t in n.targets if isinstance(t, ast.Name)):
                val = n.value.value if isinstance(n.value, ast.Constant) else "not a constant"
            return False
        if isinstance(n, ast.FunctionDef) and n.name == "_last_step":
            return False
        # one `return False` per removed construct (kept parallel rather than folded into `return not (...)`)
        if (  # noqa: SIM103
            isinstance(n, ast.If)
            and isinstance(n.test, ast.BoolOp)
            and isinstance(n.test.op, ast.And)
            and isinstance(n.test.values[0], ast.Name)
            and n.test.values[0].id == "_TSELL"
        ):
            return False
        return True

    for node in ast.walk(tree):
        for f in ("body", "orelse", "finalbody"):
            if isinstance(getattr(node, f, None), list):
                setattr(node, f, [n for n in getattr(node, f) if keep(n)])
    return val


def same_program(a_src, b_src, notes=None):
    """Differences between two Python sources (empty list = same program).

    ASTs are compared, ignoring rl_<hex8> names and the text of Cyrillic str constants. If only b has the (disabled)
    _TSELL option, it is removed first; notes (a list) then receives a line saying so."""
    norm = lambda s: re.sub(r"rl_[0-9a-f]{8}", "rl_PKG", s)  # noqa: E731
    ta, tb = ast.parse(norm(a_src)), ast.parse(norm(b_src))
    if "_TSELL" in b_src and "_TSELL" not in a_src:
        v = strip_tsell(tb)
        if v is not False:
            return [f"_TSELL is {v!r}, not False"]
        if notes is not None:
            notes.append(
                "the re-export has the later terminal-sell option (_TSELL = False, never runs); "
                "it was removed before the comparison"
            )
    diffs = []

    def cmp(a, b, path):
        if type(a) is not type(b):
            diffs.append(path)
            return
        if isinstance(a, ast.Constant):
            if a.value != b.value and not (
                isinstance(a.value, str) and isinstance(b.value, str) and (CYR.search(a.value) or CYR.search(b.value))
            ):
                diffs.append(f"{path} line {getattr(a, 'lineno', '?')}")
            return
        for f in a._fields:
            va, vb = getattr(a, f, None), getattr(b, f, None)
            if isinstance(va, list):
                if not isinstance(vb, list) or len(va) != len(vb):
                    diffs.append(f"{path}.{f}")
                    continue
                for i, (p, q) in enumerate(zip(va, vb)):
                    if isinstance(p, ast.AST):
                        cmp(p, q, f"{path}.{f}[{i}]")
                    elif p != q:
                        diffs.append(f"{path}.{f}[{i}]")
            elif isinstance(va, ast.AST):
                cmp(va, vb, f"{path}.{f}")
            elif va != vb:
                diffs.append(f"{path}.{f}")

    cmp(ta, tb, "module")
    return diffs


def pkg_dir(d):
    """The rl_<hex8> package folder inside an unpacked agent."""
    return next(os.path.join(d, p) for p in os.listdir(d) if re.fullmatch(r"rl_[0-9a-f]{8}", p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", nargs="*", default=list(EXPECTED), help="archives to check (default: both)")
    ap.add_argument(
        "--games", type=int, default=0, help="check 4: number of seeds, each played in both seats on kagsim (0 = skip)"
    )
    ap.add_argument("--opp", default="agents/caretaker/main.py", help="opponent of check 4")
    ap.add_argument("--seed0", type=int, default=97_000_000, help="first seed of check 4")
    x = ap.parse_args()
    s = settings()
    ok = True
    report = {}
    for name in x.which:
        e = EXPECTED[name]
        r = report[name] = {}
        tar = os.path.join(s["SUBMISSION_DIR"], f"{name}.tar.gz")
        shipped = tempfile.mkdtemp(prefix=f"{name}_shipped_")
        with tarfile.open(tar) as t:
            t.extractall(shipped)
        r["1_sha256"] = {
            "tar": sha256(tar)[:8] == e["tar"],
            "main": sha256(os.path.join(shipped, "main.py"))[:8] == e["main"],
            "weights": sha256(os.path.join(shipped, "weights.npz"))[:8] == e["weights"],
        }
        rd = run_dir(f"verify_{name}")
        ckpt = os.path.join(s["CHECKPOINT_DIR"], e["ckpt"])
        if subprocess.call(
            [
                sys.executable,
                os.path.join(ROOT, "entry_points", "export_submission.py"),
                "--run",
                f"verify_{name}",
                "--ckpt",
                ckpt,
                "--name",
                name,
            ],
            cwd=ROOT,
        ):
            sys.exit("re-export failed")
        rex = os.path.join(rd, "export", name)
        wa, wb = np.load(os.path.join(shipped, "weights.npz")), np.load(os.path.join(rex, "weights.npz"))
        same_w = sorted(wa.files) == sorted(wb.files) and all(
            wa[k].dtype == wb[k].dtype and wa[k].shape == wb[k].shape and wa[k].tobytes() == wb[k].tobytes()
            for k in wa.files
        )
        r["2_weights_identical"] = {
            "arrays": same_w,
            "file_bytes": sha256(os.path.join(rex, "weights.npz"))[:8] == e["weights"],
            "n_arrays": len(wa.files),
        }
        notes = []
        code = {
            "main.py": same_program(
                open(os.path.join(shipped, "main.py")).read(), open(os.path.join(rex, "main.py")).read(), notes
            )
        }
        for m in ("spec", "encode", "act", "np_net", "__init__"):
            code[f"package/{m}.py"] = same_program(
                open(os.path.join(pkg_dir(shipped), f"{m}.py")).read(),
                open(os.path.join(pkg_dir(rex), f"{m}.py")).read(),
            )
        r["3_same_program"] = {k: (not v) or v[:3] for k, v in code.items()}
        if notes:
            r["3_note"] = notes
        if x.games:
            sys.path.insert(0, ROOT)
            from rl.record import play_record

            mism, games = 0, 0
            for g in range(x.games):
                for seat in (0, 1):
                    a, b = (
                        (os.path.join(shipped, "main.py"), x.opp)
                        if seat == 0
                        else (x.opp, os.path.join(shipped, "main.py"))
                    )
                    c, d = (os.path.join(rex, "main.py"), x.opp) if seat == 0 else (x.opp, os.path.join(rex, "main.py"))
                    ra, ta = play_record(x.seed0 + g, a, b, seat=seat)
                    rb, tb = play_record(x.seed0 + g, c, d, seat=seat)
                    games += 1
                    mism += int(
                        ra != rb
                        or [json.dumps(t[1], sort_keys=True) for t in ta]
                        != [json.dumps(t[1], sort_keys=True) for t in tb]
                    )
            r["4_action_parity"] = {"games": games, "games_with_any_difference": mism}
        good = (
            all(r["1_sha256"].values())
            and r["2_weights_identical"]["arrays"]
            and all(v is True for v in r["3_same_program"].values())
        )
        if x.games:
            good = good and r["4_action_parity"]["games_with_any_difference"] == 0
        r["passed"] = good  # 2.file_bytes is informational: npz bytes also depend on the zlib build
        ok &= good
    print(json.dumps(report, indent=1))
    print("VERIFIED" if ok else "VERIFICATION FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
