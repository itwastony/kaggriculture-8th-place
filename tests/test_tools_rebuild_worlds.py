"""Tests of tools/rebuild_worlds.py: the shipped manifests and the rebuild flow with subprocess.run replaced.

No download or game is run: the fake subprocess.run records the two commands (fetch_episodes.py, tape_worlds.py
compact) and writes the files those steps would produce, so only the folder bookkeeping and the copy rule
(only worlds with repro = true that a manifest names) are checked.
"""

import gzip
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rw = _load("_t_rebuild_worlds", "tools/rebuild_worlds.py")


def test_shipped_manifests():
    eps = rw.folders_from_manifests("data/pools/*.json")
    counts = {f: len(v) for f, v in eps.items()}
    assert sorted(counts) == [
        "d0927",
        "d0928",
        "d0929",
        "d0930b",
        "live0927",
        "live0927b",
        "live0927c",
        "live1735",
        "live2010",
        "topvtop",
        "weaktrain",
    ]
    assert sum(counts.values()) == 5539  # the count quoted in the module docstring


def test_list(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["rebuild_worlds.py", "--list", "--folders", "live1735", "live2010"])
    rw.main()
    out = capsys.readouterr().out.splitlines()
    assert out == ["live1735: 5 episodes", "live2010: 11 episodes", "total: 16 episodes"]


def _manifest(root, worlds):
    p = root / "data" / "pools"
    p.mkdir(parents=True)
    (p / "pool_x.json").write_text(json.dumps({"worlds": [{"world": w} for w in worlds]}))


def _gz(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(obj, f)


def test_rebuild_flow(tmp_path, monkeypatch, capsys):
    root = tmp_path
    monkeypatch.setattr(rw, "ROOT", root)
    _manifest(root, ["data/worlds/fa/ep1.json.gz", "data/worlds/fa/ep2.json.gz", "data/worlds/fa/ep3.json.gz"])
    _gz(root / "replays" / "compact" / "fa" / "ep1.json.gz", {"ep": 1})  # already downloaded
    calls = []

    def fake_run(cmd, check):
        assert check is True
        calls.append(cmd)
        if cmd[1].endswith("tools/fetch_episodes.py"):
            for e in cmd[cmd.index("--eps") + 1 :]:
                _gz(Path(cmd[cmd.index("--out") + 1]) / f"ep{e}.json.gz", {"ep": int(e)})
        else:
            built = root / "replays" / "worlds" / "fa"
            index = []
            for e, ok in ((1, True), (2, False), (3, True), (4, True)):  # ep4: built but not in any manifest
                _gz(built / f"ep{e}.json.gz", {"ep": e, "built": True})
                index.append(json.dumps({"ep": e, "repro": ok}))
            (root / "replays" / "worlds" / "fa.index.jsonl").write_text("\n".join(index) + "\n")

    monkeypatch.setattr(rw.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["rebuild_worlds.py", "--jobs", "3", "--sleep", "1.5"])
    rw.main()
    py = sys.executable
    assert calls[0] == [
        py,
        str(root / "tools/fetch_episodes.py"),
        "--out",
        str(root / "replays" / "compact" / "fa"),
        "--sleep",
        "1.5",
        "--eps",
        "2",
        "3",
    ]
    assert calls[1] == [
        py,
        str(root / "harness/tape_worlds.py"),
        "compact",
        "--src",
        str(root / "replays" / "compact" / "fa"),
        "--pool",
        "fa",
        "--lb",
        str(root / "configs/empty_lb.csv"),
        "--jobs",
        "3",
    ]
    assert sorted(p.name for p in (root / "data" / "worlds" / "fa").iterdir()) == ["ep1.json.gz", "ep3.json.gz"]
    report = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert report == {"folder": "fa", "needed": 3, "copied": 2, "not_reproduced_or_missing": 1, "first_missing": [2]}

    calls.clear()  # second run: everything downloaded and the index exists, so only the copy step runs
    rw.main()
    assert calls == []
