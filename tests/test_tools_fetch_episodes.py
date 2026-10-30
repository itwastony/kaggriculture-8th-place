"""Tests of tools/fetch_episodes.py with a fake Kaggle API (no network, no Kaggle package needed).

Checked: the compact tape format (info/rewards/statuses as dicts or as their string form, PASS defaults), fetch_one
(new tape, existing tape, HTTP 429 retry with and without Retry-After, other errors, a missing replay file) and main()
(episode ids from --eps and --pools, --shard selection, removal of the empty download folder).
"""

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


fe = _load("_t_fetch_episodes", "tools/fetch_episodes.py")


def _replay(seed=42, as_str=False):
    info = {"seed": seed, "TeamNames": ["Team A", "Team B"], "EpisodeId": 1}
    rewards, statuses = [123.0, 45.5], ["DONE", "DONE"]
    first = [{"action": None}, {"action": None}]
    steps = [
        first,
        [{"action": {"farmer": ["NORTH"], "market": [["SELL", "MILK", 2]]}}, {"action": None}],
        [{"action": {"hands": [["WATER"]]}}, {}],
    ]
    if as_str:
        info, rewards, statuses = repr(info), repr(rewards), repr(statuses)
    return {"info": info, "rewards": rewards, "statuses": statuses, "steps": steps}


EXPECTED = {
    "ep": 7,
    "seed": 42,
    "names": ["Team A", "Team B"],
    "rewards": [123.0, 45.5],
    "statuses": ["DONE", "DONE"],
    "team_src": "",
    "sub_src": "",
    "actions": [
        [
            {"farmer": ["NORTH"], "hands": [], "market": [["SELL", "MILK", 2]]},
            {"farmer": ["PASS"], "hands": [["WATER"]], "market": []},
        ],
        [{"farmer": ["PASS"], "hands": [], "market": []}] * 2,
    ],
}


@pytest.mark.parametrize("as_str", [False, True])
def test_compact(as_str):
    assert fe.compact(_replay(as_str=as_str), "7") == EXPECTED


class _Resp:
    def __init__(self, headers):
        self.headers = headers


class _HTTPError(Exception):
    def __init__(self, msg, headers=None):
        super().__init__(msg)
        self.response = _Resp(headers or {})


class FakeApi:
    """competition_episode_replay writes episode-<ep>-replay.json; `errors` are raised first, one per call."""

    def __init__(self, errors=(), write=True):
        self.errors = list(errors)
        self.write = write
        self.calls = []

    def competition_episode_replay(self, ep, path, quiet):
        self.calls.append(ep)
        if self.errors:
            raise self.errors.pop(0)
        if self.write:
            (Path(path) / f"episode-{ep}-replay.json").write_text(json.dumps(_replay(seed=ep)))


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(fe.time, "sleep", sleeps.append)
    out, tmp = tmp_path / "out", tmp_path / "out" / "_dl0"
    tmp.mkdir(parents=True)
    return out, tmp, sleeps


def _read(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def test_fetch_one_writes_tape_once(dirs):
    out, tmp, _ = dirs
    api = FakeApi()
    assert fe.fetch_one(api, 7, out, tmp) is True
    assert _read(out / "ep7.json.gz")["seed"] == 7
    assert not (tmp / "episode-7-replay.json").exists()  # the raw replay is deleted
    assert fe.fetch_one(api, 7, out, tmp) is False  # already there: no download
    assert api.calls == [7]


def test_fetch_one_retries_429(dirs, capsys):
    out, tmp, sleeps = dirs
    errors = [_HTTPError("HTTP 429", {"retry-after": "5"}), _HTTPError("429 Too Many Requests"), _HTTPError("429")]
    api = FakeApi(errors)
    assert fe.fetch_one(api, 8, out, tmp) is True
    assert sleeps == [35, 60, 90]  # Retry-After + 30, then 30 s x attempt
    assert api.calls == [8] * 4
    assert "HTTP 429 on 8: waiting 35 s" in capsys.readouterr().out


def test_fetch_one_gives_up(dirs, capsys):
    out, tmp, sleeps = dirs
    assert fe.fetch_one(FakeApi([ValueError("404 not found")]), 9, out, tmp) is False
    assert sleeps == [] and "replay failed 9" in capsys.readouterr().out
    api = FakeApi([_HTTPError("429")] * 12)
    assert fe.fetch_one(api, 10, out, tmp) is False
    assert len(api.calls) == 12 and sleeps[-1] == 330  # 11 waits, the 12th error gives up
    assert fe.fetch_one(FakeApi(write=False), 11, out, tmp) is False  # no replay file appeared
    assert not list(out.glob("ep*.json.gz"))


def test_main_shard_and_pools(tmp_path, monkeypatch, capsys):
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({"worlds": [{"ep": 5}, {"ep": "3"}, {"ep": 9}]}))
    api = FakeApi()
    monkeypatch.setattr(fe, "get_api", lambda: api)
    sleeps = []
    monkeypatch.setattr(fe.time, "sleep", sleeps.append)
    out = tmp_path / "out"
    argv = ["fetch_episodes.py", "--pools", str(pool), "--eps", "1", "9", "--out", str(out), "--shard", "1/2"]
    monkeypatch.setattr(sys, "argv", argv + ["--sleep", "0.5"])
    fe.main()
    assert api.calls == [3, 9]  # sorted ids 1, 3, 5, 9: every 2nd starting at index 1
    assert sorted(p.name for p in out.iterdir()) == ["ep3.json.gz", "ep9.json.gz"]  # the empty _dl1 is removed
    assert sleeps == [0.5, 0.5]
    assert capsys.readouterr().out.strip().endswith(f"done: 2 new tapes in {out} (2 episodes requested)")
