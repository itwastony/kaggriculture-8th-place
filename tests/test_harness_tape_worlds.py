"""Tests of harness/tape_worlds.py on synthetic episodes (no engine: replay_tapes is replaced by a stub).

Checked: leaderboard CSV parsing and lookup, the world record written by finish() (shops, repro flag, lb), both input
formats (compact tapes and daily-dataset zips), and main(): de-duplication of compact tapes by file name, the index
file, the printed summary and the refusal to overwrite an existing index.
"""

import gzip
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel, *paths):
    """Import ROOT/rel as module `name`, with ROOT/<paths> on sys.path during the import."""
    added = [str(ROOT / p) for p in paths if str(ROOT / p) not in sys.path]
    sys.path[:0] = added
    try:
        spec = importlib.util.spec_from_file_location(name, ROOT / rel)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for p in added:
            sys.path.remove(p)
    return mod


tw = _load("_t_tape_worlds", "harness/tape_worlds.py", "harness")

LB_CSV = (
    "\ufeffRank,TeamId,TeamName,LastSubmissionDate,Score,SubmissionCount,TeamMemberUserNames\n"
    '1,11,Team A,2026-09-30,2900.5,3,"alice, bob"\n'
    "2,12,Team B,2026-09-30,2800,1,carol\n"
    "3,13,Team C,2026-09-30,2700,1,alice\n"
)
SHOPS = ["BAKERY", "YARN_STORE", "BAKERY"]


@pytest.fixture
def world_root(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    monkeypatch.setattr(tw, "ROOT", root)
    calls = []

    def fake_replay(seed, actions):
        calls.append((seed, actions))
        return [100.0, 250.5], list(SHOPS)

    monkeypatch.setattr(tw, "replay_tapes", fake_replay)
    lb = root / "lb.csv"
    lb.write_text(LB_CSV, encoding="utf-8")
    return root, str(lb), calls


def _read(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def test_load_lb_and_lookup(world_root):
    _, lbp, _ = world_root
    lb = tw.load_lb(lbp)
    assert lb[0]["Team A"] == {"rank": 1, "score": 2900.5, "team": "Team A", "team_id": "11"}
    assert tw.lb_of("Team B", lb)["rank"] == 2
    assert tw.lb_of("bob", lb)["team"] == "Team A"  # member user name, stripped
    assert tw.lb_of("alice", lb)["team"] == "Team A"  # the first team listing a user wins
    assert tw.lb_of("nobody", lb) is None
    assert tw.load_lb(str(ROOT / "configs" / "empty_lb.csv")) == ({}, {})


def test_finish_writes_world(world_root):
    root, lbp, calls = world_root
    w = {"ep": 5, "seed": 77, "names": ["Team B", "x"], "rewards": [100, 250.5], "actions": [[], []]}
    rec = tw.finish(w, "pool1", tw.load_lb(lbp))
    assert calls == [(77, [[], []])]
    assert rec == {"ep": 5, "repro": True, "names": ["Team B", "x"], "lb": [2, None], "n_shops": 3}
    saved = _read(root / "replays" / "worlds" / "pool1" / "ep5.json.gz")
    assert saved["shops"] == SHOPS and saved["repro"] is True
    assert saved["lb"][0]["team_id"] == "12" and saved["lb"][1] is None
    w2 = dict(w, ep=6, rewards=[100, 250])
    assert tw.finish(w2, "pool1", tw.load_lb(lbp))["repro"] is False


def test_from_compact(world_root):
    root, lbp, _ = world_root
    src = root / "replays" / "compact" / "ep7.json.gz"
    src.parent.mkdir(parents=True)
    tape = {
        "ep": 7,
        "seed": 3,
        "names": ["Team A", "Team C"],
        "rewards": [100.0, 250.5],
        "statuses": ["DONE", "DONE"],
        "actions": [[{"farmer": ["PASS"]}], [{"farmer": ["NORTH"]}]],
        "team_src": "t",
        "sub_src": "s",
        "extra": "dropped",
    }
    with gzip.open(src, "wt") as f:
        json.dump(tape, f)
    rec = tw.from_compact((str(src), "band", lbp))
    assert rec["repro"] is True and rec["lb"] == [1, 3]
    w = _read(root / "replays" / "worlds" / "band" / "ep7.json.gz")
    assert w["src"] == "replays/compact/ep7.json.gz"
    assert (w["team_src"], w["sub_src"]) == ("t", "s") and "extra" not in w
    assert w["actions"] == tape["actions"] and w["statuses"] == ["DONE", "DONE"]


def _replay(ep, unlocked=SHOPS):
    def seat(action, reward=0, status="ACTIVE", obs=None):
        return {"action": action, "reward": reward, "status": status, "observation": obs or {}}

    steps = [[seat({}), seat({})]]
    steps.append([seat({"farmer": ["NORTH"]}), seat({"farmer": ["WEST"]})])
    last_obs = {"town": {"unlocked_shops": unlocked}} if unlocked is not None else {}
    steps.append([seat({"farmer": ["PASS"]}, 100.0, "DONE", last_obs), seat(None, 250.5, "DONE")])
    return {"info": {"EpisodeId": ep, "seed": 9, "TeamNames": ["Team A", "Team B"]}, "steps": steps}


def test_from_zip(world_root):
    root, lbp, calls = world_root
    z = root / "daily.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("ep21.json", json.dumps(_replay(21)))
        f.writestr("ep22.json", json.dumps(_replay(22, ["PET_CAFE"])))
        f.writestr("ep23.json", json.dumps(_replay(23, None)))
    rec = tw.from_zip((str(z), "ep21.json", "d", lbp))
    assert rec["shops_match_record"] is True and rec["repro"] is True
    assert calls[-1] == (9, [[{"farmer": ["NORTH"]}, {"farmer": ["PASS"]}], [{"farmer": ["WEST"]}, None]])
    w = _read(root / "replays" / "worlds" / "d" / "ep21.json.gz")
    assert w["src"] == "daily.zip:ep21.json" and w["statuses"] == ["DONE", "DONE"]
    assert tw.from_zip((str(z), "ep22.json", "d", lbp))["shops_match_record"] is False
    assert tw.from_zip((str(z), "ep23.json", "d", lbp))["shops_match_record"] is True  # no record to compare


class _InlineExecutor:
    """ProcessPoolExecutor stand-in that runs the jobs in this process (so the patched globals apply)."""

    def __init__(self, n):
        self.n = n

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, jobs, chunksize=1):
        return map(fn, jobs)


def test_main_compact(world_root, monkeypatch, capsys):
    root, lbp, _ = world_root
    monkeypatch.setattr(tw, "ProcessPoolExecutor", _InlineExecutor)
    for d, eps in (("a", (1, 2)), ("b", (2, 3))):
        (root / d).mkdir()
        for ep in eps:
            rec = {
                "ep": ep,
                "seed": ep,
                "names": ["Team A", "zed"],
                "rewards": [100.0, 250.5] if ep != 3 else [1.0, 2.0],
                "statuses": ["DONE", "DONE"],
                "actions": [[], []],
                "src_dir": d,
            }
            with gzip.open(root / d / f"ep{ep}.json.gz", "wt") as f:
                json.dump(rec, f)
    argv = ["tape_worlds.py", "compact", "--src", str(root / "a"), str(root / "b"), "--pool", "p", "--lb", lbp]
    monkeypatch.setattr(sys, "argv", argv)
    tw.main()
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary == {
        "pool": "p",
        "worlds": 3,
        "repro_ok": 2,
        "bad": [3],
        "lb_mapped_seats": 3,
        "shops_mismatch": 0,
    }
    index = [json.loads(line) for line in (root / "replays" / "worlds" / "p.index.jsonl").read_text().splitlines()]
    assert [r["ep"] for r in index] == [1, 2, 3]
    assert _read(root / "replays" / "worlds" / "p" / "ep2.json.gz")["src"] == "a/ep2.json.gz"  # first --src wins
    with pytest.raises(FileExistsError):
        tw.main()
