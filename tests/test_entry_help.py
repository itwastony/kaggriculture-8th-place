"""Every entry point starts and prints its --help (with the module docstring) without touching data or engines."""

import glob
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = sorted(
    os.path.basename(f) for f in glob.glob(os.path.join(ROOT, "entry_points", "*.py")) if not f.endswith("common.py")
)


def test_all_entry_points_listed():
    assert SCRIPTS == [
        "evaluate.py",
        "export_submission.py",
        "make_bc_data.py",
        "predict.py",
        "prepare_data.py",
        "train_appo.py",
        "train_bc.py",
        "verify_submission.py",
    ]


@pytest.mark.parametrize("script", SCRIPTS)
def test_help(script):
    p = subprocess.run(
        [sys.executable, os.path.join("entry_points", script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert p.returncode == 0, p.stderr
    assert p.stdout.startswith("usage: ")
    first = open(os.path.join(ROOT, "entry_points", script)).read().split('"""')[1].splitlines()[0]
    assert first in p.stdout  # description = the module docstring
