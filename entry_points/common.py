"""Shared helpers for the entry points: repository root, SETTINGS.json, config placeholders, sha256, command logging.

The entry points import it as a top-level module (`from common import ...`): Python puts entry_points/ on sys.path
when a script from that folder is run.
"""

import glob
import hashlib
import json
import os
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def settings():
    """SETTINGS.json without "_" keys; *_DIR and *_LIB values are made absolute (relative ones are relative to ROOT)."""
    s = json.load(open(os.path.join(ROOT, "SETTINGS.json")))
    out = {}
    for k, v in s.items():
        if k.startswith("_"):
            continue
        if k.endswith(("_DIR", "_LIB")):
            v = v if os.path.isabs(v) else os.path.join(ROOT, v)
        out[k] = v
    return out


def load_config(name):
    """configs/<name> as parsed JSON."""
    return json.load(open(os.path.join(ROOT, "configs", name)))


def fill(value, ctx):
    """Replace {KEY} placeholders in strings (recursively in lists/dicts)."""
    if isinstance(value, str):
        for k, v in ctx.items():
            value = value.replace("{" + k + "}", str(v))
        return value
    if isinstance(value, list):
        return [fill(v, ctx) for v in value]
    if isinstance(value, dict):
        return {k: fill(v, ctx) for k, v in value.items()}
    return value


def args_list(d):
    """{"--flag": value, ...} -> ["--flag", "value", ...] in insertion order."""
    out = []
    for k, v in d.items():
        out += [k, str(v)]
    return out


def run_dir(name):
    """OUTPUT_DIR/<name>; created if missing."""
    d = os.path.join(settings()["OUTPUT_DIR"], name)
    os.makedirs(d, exist_ok=True)
    return d


def sha256(path):
    """Hex sha256 of a file, read in 1 MiB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd, env=None, cwd=ROOT, dry=False):
    """Print the command (copy-pasteable) and run it from the repository root."""
    print("+", " ".join(shlex.quote(c) for c in cmd), flush=True)
    if dry:
        return 0
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.call(cmd, env=e, cwd=cwd)


def require(paths, hint):
    """Exit with the list of missing paths (glob patterns allowed) and a hint on how to create them."""
    missing = [p for p in paths if not glob.glob(p)]
    if missing:
        sys.exit("missing: " + ", ".join(missing) + "\n" + hint)
