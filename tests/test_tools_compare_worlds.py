"""Tests of tools/compare_worlds.py on synthetic world files: action normalisation, ignored fields, exit codes."""

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cw = _load("_t_compare_worlds", "tools/compare_worlds.py")


def _world(**over):
    w = {
        "ep": 1,
        "seed": "5",
        "names": ["Team A", "Team B"],
        "rewards": [100, 200.5],
        "shops": ["BAKERY"],
        "repro": True,
        "lb": [None, None],
        "actions": [[None, {"farmer": ["NORTH"]}], [{"farmer": ["PASS"], "hands": [], "market": []}, {}]],
    }
    w.update(over)
    return w


def _write(path, w):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(w, f)


def test_norm_fills_defaults():
    assert cw.norm(None) == {"farmer": ["PASS"], "hands": [], "market": []}
    assert cw.norm({"farmer": ["NORTH"], "market": [["SELL", "MILK", 1]]}) == {
        "farmer": ["NORTH"],
        "hands": [],
        "market": [["SELL", "MILK", 1]],
    }


def test_key_normalises_types_and_ignores_lb():
    a = cw.key(_world())
    b = cw.key(_world(seed=5, rewards=[100.0, 200.5], lb=[{"rank": 1}, None]))
    assert a == b
    assert a["actions"][0][0] == {"farmer": ["PASS"], "hands": [], "market": []}


def _run(monkeypatch, capsys, rebuilt, shipped):
    monkeypatch.setattr(sys, "argv", ["compare_worlds.py", "--rebuilt", str(rebuilt), "--shipped", str(shipped)])
    with pytest.raises(SystemExit) as e:
        cw.main()
    return e.value.code, json.loads(capsys.readouterr().out)


def test_main_identical_different_missing(tmp_path, monkeypatch, capsys):
    rebuilt, shipped = tmp_path / "rebuilt", tmp_path / "shipped"
    _write(rebuilt / "ep1.json.gz", _world())
    _write(shipped / "poolA" / "ep1.json.gz", _world(seed=5, lb=[{"rank": 3}, None]))
    _write(rebuilt / "ep9.json.gz", _world(ep=9))  # not shipped: counted only
    code, out = _run(monkeypatch, capsys, rebuilt, shipped)
    assert code == 0
    assert out == {"identical": 1, "different": 0, "not_in_shipped": 1, "examples": []}

    _write(rebuilt / "ep2.json.gz", _world(ep=2, shops=["PET_CAFE"], repro=False))
    _write(shipped / "poolB" / "ep2.json.gz", _world(ep=2))
    code, out = _run(monkeypatch, capsys, rebuilt, shipped)
    assert code == 1
    assert out["identical"] == 1 and out["different"] == 1
    assert out["examples"] == [["ep2.json.gz", ["shops", "repro"]]]
