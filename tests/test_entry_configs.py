"""The configs and the documented commands only use options that the scripts actually define.

Each script's real ArgumentParser is captured by calling its main() with parse_args patched to stop there (main()
builds the parser before doing anything else). Then:
- every argument that an entry point passes from a config file (configs/*.json) is an exact option of the target
  script and parses with the right type;
- every `--option` in the example commands of the module docstrings, scripts/slurm/*.sbatch and the code blocks of
  the Markdown documentation exists.
"""

import argparse
import ast
import glob
import importlib
import json
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
common = importlib.import_module("common")  # entry_points/common.py (conftest.py puts entry_points/ on sys.path)
_parse_known = argparse.ArgumentParser.parse_known_args


class _Captured(Exception):
    def __init__(self, parser):
        super().__init__()
        self.parser = parser


# entry_points/evaluate.py restarts itself unless these are set, entry_points/predict.py sets the thread counts
SCRIPT_ENV = {"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def parser_of(module, monkeypatch):
    """The ArgumentParser that `module`.main() builds (module: "rl.appo", "train_appo", ...)."""
    for k, v in SCRIPT_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(sys, "path", list(sys.path))  # evaluate.py adds harness/ on import
    mod = importlib.import_module(module)

    def capture(self, args=None, namespace=None):
        raise _Captured(self)

    with monkeypatch.context() as m:
        m.setattr(argparse.ArgumentParser, "parse_args", capture)
        try:
            mod.main()
        except _Captured as c:
            return c.parser
    raise AssertionError(f"{module}.main() returned without parsing arguments")


def check_args(parser, argv):
    """argv parses without leftovers and every option is spelled exactly (no argparse prefix matching)."""
    for a in argv:
        if a.startswith("--"):
            assert a.split("=", 1)[0] in parser._option_string_actions, f"unknown option {a}"
    ns, extra = _parse_known(parser, argv)
    assert not extra, f"unparsed arguments {extra}"
    return ns


def ctx():
    s = common.settings()
    return {**s, "RUN": "/tmp/run", "BC_CKPT": "/tmp/bc.pt", "LEAGUE": "/tmp/league.json"}


def filled(name):
    cfg = common.fill(common.load_config(name), ctx())
    left = re.findall(r"\{[A-Z_]+\}", json.dumps(cfg))
    assert not left, f"{name}: unresolved placeholders {left}"
    return cfg


def test_appo_config(monkeypatch):
    pytest.importorskip("torch")
    p = parser_of("rl.appo", monkeypatch)
    cfg = filled("appo_d7b.json")
    ns = check_args(p, ["--out", cfg["out"], *common.args_list(cfg["args"])])
    assert ns.land_inject == 248 and ns.gpu_games == 128 and ns.pcritic == 2
    smoke = dict(cfg["args"], **common.load_config("appo_smoke.json")["args"])
    check_args(p, ["--out", cfg["out"], *common.args_list(smoke)])
    # rl/appo.py asserts that the league has a "best" entry exactly when the moving anchor is on
    league = json.load(open(os.path.join(ROOT, "configs", "league_d7.json")))
    names = [n for n, _ in league]
    assert ("best" in names) == (ns.anchor_gate > 0)
    assert all(w > 0 for _, w in league)
    for n in names:
        if n.startswith("tapes:"):
            assert os.path.exists(os.path.join(ROOT, n[6:])), n


def test_bc_config(monkeypatch):
    pytest.importorskip("torch")
    cfg = filled("bc_d7.json")
    argv = [*common.args_list(cfg["args"]), "--bs", str(cfg["global_batch"]), "--out", cfg["out"], "--init", "x.pt"]
    ns = check_args(parser_of("rl.bc_train", monkeypatch), argv)
    assert ns.val_mod == 50 and ns.steps == 3000


def test_bcgen_config(monkeypatch):
    make_bc_data = importlib.import_module("make_bc_data")
    cfg = filled("bcgen_d7.json")
    cmd = make_bc_data.shard_cmd(cfg, 7, "/tmp/out")
    assert cmd[1:3] == ["-m", "rl.gen_bc_tapes"]
    assert cmd[-6:] == ["--shard", "7", "--nshards", str(cfg["nshards"]), "--out", "/tmp/out"]
    ns = check_args(parser_of("rl.gen_bc_tapes", monkeypatch), cmd[3:])
    assert ns.exclude_names == "Anton Tikhonov" and ns.min_quads == 4


def test_export_config(monkeypatch):
    pytest.importorskip("torch")
    cfg = filled("export_d7b.json")
    argv = ["--ckpt", cfg["ckpt"], "--name", "x", "--root", "/tmp/export", *common.args_list(cfg["args"])]
    ns = check_args(parser_of("rl.export_agent", monkeypatch), argv)
    assert ns.dtype == "float16" and ns.land_inject == "248:72"


# --------------------------------------------------------------- documented commands

ENTRY = {
    f"entry_points/{os.path.basename(f)}": os.path.basename(f)[:-3]
    for f in glob.glob(f"{ROOT}/entry_points/*.py")
    if not f.endswith("common.py")
}
RL_MODULES = ("rl.appo", "rl.ppo", "rl.bc_train", "rl.export_agent", "rl.gen_bc_tapes", "rl.tape_pool", "rl.latency")


def commands(text):
    """(module, [--options]) for every `python entry_points/X.py ...` or `-m rl.X ...` command in text."""
    out = []
    text = re.sub(r"\\\\?\n", " ", text)  # join continuation lines (also "\\" inside docstrings)
    for line in text.splitlines():
        # drop comments, ${VAR:?message} expansions and quotes (sbatch files run commands inside bash -c "...")
        line = re.sub(r"\$\{[^}]*\}", "VALUE", line.split(" #")[0])
        toks = line.replace('"', " ").replace("'", " ").split()
        for i, t in enumerate(toks):
            mod = ENTRY.get(t) or (
                toks[i + 1] if t == "-m" and i + 1 < len(toks) and toks[i + 1] in RL_MODULES else None
            )
            if mod:
                rest = toks[i + (1 if t in ENTRY else 2) :]
                out.append((mod, [r.split("=", 1)[0] for r in rest if r.startswith("--")]))
                break
    return out


def documented_commands():
    files = [*glob.glob(f"{ROOT}/entry_points/*.py"), *glob.glob(f"{ROOT}/scripts/slurm/*.sbatch")]
    files += [f"{ROOT}/rl/{m[3:]}.py" for m in RL_MODULES]
    files += [f"{ROOT}/README.md", *glob.glob(f"{ROOT}/docs/*.md"), *glob.glob(f"{ROOT}/*/README.md")]
    found = []
    for f in sorted(files):
        text = open(f).read()
        if f.endswith(".py"):
            text = ast.get_docstring(ast.parse(text)) or ""
        elif f.endswith(".md"):  # only fenced code blocks: prose mentions of options are not commands
            text = "\n".join(re.findall(r"^```[^\n]*\n(.*?)^```", text, re.M | re.S))
        found += [(os.path.relpath(f, ROOT), mod, opts) for mod, opts in commands(text)]
    return found


DOCUMENTED = documented_commands()


def test_documented_commands_found():
    mods = {m for _, m, _ in DOCUMENTED}
    assert {"train_appo", "train_bc", "make_bc_data", "export_submission", "evaluate", "predict"} <= mods
    assert {"rl.appo", "rl.export_agent"} <= mods


@pytest.mark.parametrize("where,module,opts", DOCUMENTED, ids=[f"{w}:{m}" for w, m, _ in DOCUMENTED])
def test_documented_options_exist(where, module, opts, monkeypatch):
    if module in ("rl.appo", "rl.ppo", "rl.bc_train", "rl.export_agent"):
        pytest.importorskip("torch")
    p = parser_of(module, monkeypatch)
    for o in opts:
        assert o in p._option_string_actions, f"{where}: {module} has no option {o}"
