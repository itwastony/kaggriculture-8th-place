"""entry_points/common.py (SETTINGS.json, config placeholders, helpers), the league checks of
entry_points/train_appo.py and its --set overrides (a --dry run), on temporary files only."""

import hashlib
import importlib
import json
import os
import shlex
import sys

import pytest

common = importlib.import_module("common")  # entry_points/common.py (conftest.py puts entry_points/ on sys.path)
train_appo = importlib.import_module("train_appo")


def test_settings_resolution():
    raw = json.load(open(os.path.join(common.ROOT, "SETTINGS.json")))
    s = common.settings()
    assert not any(k.startswith("_") for k in s)
    assert set(s) == {k for k in raw if not k.startswith("_")}
    for k, v in s.items():
        if k.endswith(("_DIR", "_LIB")):
            assert os.path.isabs(v), k
            if not os.path.isabs(raw[k]):
                assert v == os.path.join(common.ROOT, raw[k])
        else:
            assert v == raw[k], k  # values other than paths are kept as they are


def test_fill_and_args_list():
    ctx = {"RUN": "/r", "N": 3}
    cfg = {"out": "{RUN}/appo", "args": {"--a": "{RUN}/x", "--n": "{N}", "--k": 1.5}, "l": ["{N}", 7, None]}
    assert common.fill(cfg, ctx) == {
        "out": "/r/appo",
        "args": {"--a": "/r/x", "--n": "3", "--k": 1.5},
        "l": ["3", 7, None],
    }
    assert common.fill("{UNKNOWN}", ctx) == "{UNKNOWN}"
    assert common.args_list({"--b": 2, "--a": "x", "--f": 0.5}) == ["--b", "2", "--a", "x", "--f", "0.5"]


def test_run_dir_and_require(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "settings", lambda: {"OUTPUT_DIR": str(tmp_path / "runs")})
    d = common.run_dir("x")
    assert d == str(tmp_path / "runs" / "x") and os.path.isdir(d)
    assert common.run_dir("x") == d  # an existing folder is fine
    (tmp_path / "a.pt").write_bytes(b"")
    common.require([str(tmp_path / "*.pt")], "hint")  # glob patterns are allowed
    with pytest.raises(SystemExit) as e:
        common.require([str(tmp_path / "a.pt"), str(tmp_path / "b.pt")], "make it")
    assert str(e.value) == f"missing: {tmp_path / 'b.pt'}\nmake it"


def test_sha256(tmp_path):
    p = tmp_path / "f.bin"
    data = os.urandom(3 * (1 << 20) + 17)  # several read blocks
    p.write_bytes(data)
    assert common.sha256(str(p)) == hashlib.sha256(data).hexdigest()


def test_run_dry_prints_a_copy_pasteable_command(capsys):
    assert common.run(["python3", "-c", "print('a b')", "--x"], dry=True) == 0
    assert capsys.readouterr().out == "+ python3 -c 'print('\"'\"'a b'\"'\"')' --x\n"


def _tape_pool(root, name, worlds):
    os.makedirs(root / "data" / "pools", exist_ok=True)
    json.dump(
        {"note": "kept", "worlds": [{"world": w, "seed": i} for i, w in enumerate(worlds)]}, open(root / name, "w")
    )


def test_league_entry_state(tmp_path, monkeypatch):
    monkeypatch.setattr(train_appo, "ROOT", str(tmp_path))
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "bot.py").write_text("")
    (tmp_path / "w1.json.gz").write_bytes(b"")
    _tape_pool(tmp_path, "data/pools/full.json", ["w1.json.gz"])
    _tape_pool(tmp_path, "data/pools/part.json", ["w1.json.gz", "w2.json.gz"])
    _tape_pool(tmp_path, "data/pools/none.json", ["w3.json.gz"])
    assert train_appo.entry_state("self") == ("ok", None)
    assert train_appo.entry_state("best") == ("ok", None)
    assert train_appo.entry_state("agents/bot.py") == ("ok", None)
    assert train_appo.entry_state("agents/gone.py") == ("missing", None)
    assert train_appo.entry_state("tapes:data/pools/absent.json") == ("missing", None)
    assert train_appo.entry_state("tapes:data/pools/full.json")[0] == "ok"
    st, ok = train_appo.entry_state("tapes:data/pools/part.json")
    assert st == "partial" and ok == [{"world": "w1.json.gz", "seed": 0}]
    assert train_appo.entry_state("tapes:data/pools/none.json") == ("missing", [])


def test_available_league(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(train_appo, "ROOT", str(tmp_path))
    (tmp_path / "w1.json.gz").write_bytes(b"")
    _tape_pool(tmp_path, "data/pools/part.json", ["w1.json.gz", "w2.json.gz"])
    rd = tmp_path / "run"
    league = [["self", 10], ["agents/gone.py", 5], ["tapes:data/pools/part.json", 35]]
    out = train_appo.available_league(league, str(rd))
    copy = rd / "pools" / "part.json"
    # the copy keeps the file name, so the opponent key in the logs (rl/appo.py okey) does not change
    assert out == [["self", 10], ["tapes:" + str(copy), 35]]
    assert json.load(open(copy)) == {"note": "kept", "worlds": [{"world": "w1.json.gz", "seed": 0}]}
    assert "league: dropped agents/gone.py" in capsys.readouterr().out


def _dry_appo(tmp_path, monkeypatch, capsys, extra):
    """Run train_appo.main() with --dry on temporary inputs; return (exit code, printed command as a list)."""
    s = {
        "DATA_DIR": os.path.join(common.ROOT, "data"),  # the repository's pool manifests; no league rewrite
        "CHECKPOINT_DIR": str(tmp_path / "ckpt"),
        "SNAPSHOT_POOL_DIR": str(tmp_path / "pools"),
        "OUTPUT_DIR": str(tmp_path / "runs"),
        "KAGRL_LIB": str(tmp_path / "libkagrl_cuda.so"),
    }
    monkeypatch.setattr(common, "settings", lambda: s)  # used by run_dir()
    monkeypatch.setattr(train_appo, "settings", lambda: s)
    bc = tmp_path / "bc.pt"
    past = common.fill(common.load_config("appo_d7b.json")["args"]["--past"], {**s, "BC_CKPT": str(bc)})
    for f in [*past.split(","), s["KAGRL_LIB"]]:  # empty stand-ins: only their existence is checked
        os.makedirs(os.path.dirname(f), exist_ok=True)
        open(f, "wb").close()
    argv = ["train_appo.py", "--run", "t", "--nnodes", "1", "--nproc", "1", "--bc-ckpt", str(bc), "--dry"]
    argv += ["--no-pool-src", "--league-available-only", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as e:
        train_appo.main()
    cmd = [line[2:] for line in capsys.readouterr().out.splitlines() if line.startswith("+ ")]
    return e.value.code, shlex.split(cmd[-1]) if cmd else []


def test_train_appo_set_overrides(tmp_path, monkeypatch, capsys):
    code, base = _dry_appo(tmp_path, monkeypatch, capsys, [])
    assert code == 0 and "rl.appo" in base and "--pool-src" not in base
    code, cmd = _dry_appo(tmp_path, monkeypatch, capsys, ["--set=--gpu-games=64", "--set=--minibatches=72"])
    assert code == 0
    # both overrides replace the config values; every other argument is unchanged
    assert cmd[cmd.index("--gpu-games") + 1] == "64" and cmd[cmd.index("--minibatches") + 1] == "72"
    assert base[base.index("--gpu-games") + 1] == "128" and base[base.index("--minibatches") + 1] == "104"
    norm = {"--gpu-games": "128", "--minibatches": "104"}
    assert [norm[cmd[i - 1]] if i and cmd[i - 1] in norm else a for i, a in enumerate(cmd)] == base
    # an argument that the config does not set is appended
    _, cmd = _dry_appo(tmp_path, monkeypatch, capsys, ["--set=--pool-recent=0.25"])
    assert cmd[-2:] == ["--pool-recent", "0.25"]
    # the space form cannot work: argparse reads "--gpu-games=64" as an option of train_appo.py
    code, _ = _dry_appo(tmp_path, monkeypatch, capsys, ["--set", "--gpu-games=64"])
    assert code == 2
