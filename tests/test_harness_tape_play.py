"""Tests of harness/kagsim_play.py and harness/tape_play.py against a stand-in for the kagsim engine.

No engine is run: a fake `kagsim` module is put into sys.modules, and the kaggle-environments helpers
(_base_config, fast_structify) are replaced, so the tests check only the harness logic: PASS padding of tapes,
pinned shops, the "@tape" and live-agent paths, mismatch detection, error statuses and the trace bookkeeping.
"""

import gzip
import importlib.util
import json
import sys
import types
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


kagsim_play = _load("_t_kagsim_play", "harness/kagsim_play.py")
tape_play = _load("_t_tape_play", "harness/tape_play.py", "harness")

PRICES = {"MILK": 100.0, "WOOL": 50.0, "WHEAT": 0.5}


def _farm(money):
    tiles = [[None] * 3 for _ in range(3)]
    tiles[0][0] = {"kind": "PLANT", "crop": "WHEAT"}
    tiles[0][1] = {"kind": "PLANT", "crop": "WHEAT"}
    tiles[1][0] = {"kind": "WEED"}
    tiles[1][1] = {"kind": "PASTURE", "animal": "COW"}
    tiles[2][2] = {"kind": "PASTURE"}
    return {"tiles": tiles, "money": money, "unlocked_quadrants": [0, 1], "hands": [{}, {}]}


class FakeGame:
    """Stand-in for kagsim.Game: finishes after steps - 1 steps; a SELL order of n units earns 7 * n coins."""

    made = []

    def __init__(self, seed, steps, shops=None):
        self.seed, self.steps, self.shops = seed, steps, shops
        self.step_count = 0
        self.actions = []
        self.tel = [{"sell_revenue": 0.0, "sold_units": 0}, {"sell_revenue": 0.0, "sold_units": 0}]
        FakeGame.made.append(self)

    @property
    def done(self):
        return self.step_count >= self.steps - 1

    def step(self, a0, a1):
        for s, a in enumerate((a0, a1)):
            for m in (a or {}).get("market") or []:
                if m[0] == "SELL":
                    self.tel[s]["sell_revenue"] += 7.0 * m[2]
                    self.tel[s]["sold_units"] += m[2]
        if self.step_count == 1:  # revenue without any SELL order of this turn
            self.tel[0]["sell_revenue"] += 5.0
        self.actions.append((a0, a1))
        self.step_count += 1

    def reward(self, p):
        return 1000.0 * (p + 1) + self.step_count

    def observe(self, p):
        return {
            "town": {"unlocked_shops": ["BAKERY", "YARN_STORE"]},
            "market": {"prices": dict(PRICES)},
            "farms": [_farm(10), _farm(20)],
            "seat": p,
        }

    def telemetry(self, s):
        return dict(self.tel[s])


@pytest.fixture
def fake_engine(monkeypatch):
    FakeGame.made = []
    monkeypatch.setitem(sys.modules, "kagsim", types.SimpleNamespace(Game=FakeGame))
    for mod in (kagsim_play, tape_play):
        monkeypatch.setattr(mod, "_base_config", lambda: {"episodeSteps": 720})
        monkeypatch.setattr(mod, "fast_structify", lambda o: o)
    return FakeGame


def _agent(tmp_path, body, name="agent.py"):
    p = tmp_path / name
    p.write_text(body)
    return str(p)


def test_canon_ignores_key_order_and_tuples():
    a = {"market": [("SELL", "MILK", 2)], "farmer": ["PASS"]}
    b = {"farmer": ["PASS"], "market": [["SELL", "MILK", 2]]}
    assert tape_play.canon(a) == tape_play.canon(b)
    assert tape_play.canon(a) != tape_play.canon({"farmer": ["NORTH"], "market": [["SELL", "MILK", 2]]})


def test_farm_snap_counts_visible_farm():
    snap = tape_play._farm_snap(_farm(42))
    assert snap == {
        "money": 42,
        "plants": {"WHEAT": 2},
        "animals": {"COW": 1},
        "weeds": 1,
        "quads": 2,
        "hands": 2,
    }


def test_load_world_is_cached_and_cache_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(tape_play, "_WORLDS", {})
    p = tmp_path / "ep1.json.gz"
    with gzip.open(p, "wt") as f:
        json.dump({"ep": 1, "seed": 5}, f)
    w = tape_play.load_world(str(p))
    assert w == {"ep": 1, "seed": 5}
    assert tape_play.load_world(str(p)) is w
    for k in range(65):
        tape_play._WORLDS[f"other{k}"] = {}
    tape_play.load_world(str(tmp_path / "ep1.json.gz"))  # already cached: no eviction
    assert len(tape_play._WORLDS) == 66
    q = tmp_path / "ep2.json.gz"
    with gzip.open(q, "wt") as f:
        json.dump({"ep": 2}, f)
    tape_play.load_world(str(q))  # a new world with > 64 cached: the cache is cleared first
    assert list(tape_play._WORLDS) == [str(q)]


def test_replay_tapes_pads_with_pass_and_pins_shops(fake_engine):
    tapes = [[{"farmer": ["NORTH"]}, None], [{"farmer": ["WEST"]}]]
    rewards, shops = tape_play.replay_tapes(11, tapes, shops=[1, 2], steps=5)
    g = fake_engine.made[-1]
    assert (g.seed, g.steps, g.shops) == (11, 5, [1, 2])
    assert g.actions[0] == ({"farmer": ["NORTH"]}, {"farmer": ["WEST"]})
    assert g.actions[1] == (tape_play.PASS, tape_play.PASS)  # None step
    assert g.actions[3] == (tape_play.PASS, tape_play.PASS)  # past the end of the tape
    assert len(g.actions) == 4
    assert rewards == [1004.0, 2004.0]
    assert shops == ["BAKERY", "YARN_STORE"]
    tape_play.replay_tapes(11, tapes, steps=3)
    assert fake_engine.made[-1].shops is None


def _world(n=6):
    me = [{"farmer": ["PASS"], "hands": [], "market": [["SELL", "MILK", 1], ["SELL", "WOOL", 3]]}]
    me += [{"farmer": ["PASS"], "hands": [], "market": []} for _ in range(n - 1)]
    tape = [{"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 2]]}]
    tape += [None] * (n - 1)
    return {"seed": 3, "shops": [4, 0], "actions": [me, tape], "rewards": [0, 0]}


def test_play_vs_tape_own_record_with_trace(fake_engine):
    w = _world(30)
    out = tape_play.play_vs_tape(w, 1, "@tape", match=True, steps=30, trace=True)
    g = fake_engine.made[-1]
    assert g.shops == [4, 0]  # pinned by default
    assert out["first_mismatch"] is None
    assert out["steps_played"] == 29
    assert out["rewards"] == [1029.0, 2029.0] and out["status"] == ["DONE", "DONE"]
    tr = out["trace"]
    # seat 0, turn 0: 28 coins split by quantity x price (MILK 1 x 100, WOOL 3 x 50); turn 1: 5 coins, no SELL order
    assert tr["rev"][0] == {"MILK": 11.2, "WOOL": 16.8, "?": 5.0}
    assert tr["units"][0] == {"MILK": 1.6, "WOOL": 2.4, "?": 0.0}
    # seat 1: one product, so the split is exact; the price floor in the weights is 1
    assert tr["rev"][1] == {"WHEAT": 14.0} and tr["units"][1] == {"WHEAT": 2.0}
    assert tr["rev_exact"] == [round(5.0 / 33.0, 3), 1.0]
    assert tr["tel"][0]["sell_revenue"] == 33.0 and tr["tel"][0]["plants_dry"] is None
    assert len(tr["days"]) == 1 and tr["days"][0][1]["money"] == 20


def test_play_vs_tape_unpinned(fake_engine):
    tape_play.play_vs_tape(_world(), 1, "@tape", pin=False, steps=4)
    assert fake_engine.made[-1].shops is None


def test_play_vs_tape_live_agent_and_mismatch(fake_engine, tmp_path):
    path = _agent(
        tmp_path,
        "def agent(obs, conf):\n"
        "    print('noise')\n"
        "    assert conf['__raw_path__'].endswith('agent.py')\n"
        "    return {'farmer': ['NORTH'], 'hands': [], 'market': []}\n",
    )
    out = tape_play.play_vs_tape(_world(), 1, path, steps=6)
    g = fake_engine.made[-1]
    assert out["status"] == ["DONE", "DONE"] and out["steps_played"] == 5
    assert g.actions[0][0] == {"farmer": ["NORTH"], "hands": [], "market": []}
    assert g.actions[0][1]["market"] == [["SELL", "WHEAT", 2]] and g.actions[1][1] == tape_play.PASS
    out = tape_play.play_vs_tape(_world(), 1, path, match=True, steps=6)
    assert out["first_mismatch"] == 0 and out["steps_played"] == 0
    assert out["rewards"] is None and out["status"] == ["ABORTED", "ABORTED"]


def test_play_vs_tape_agent_error(fake_engine, tmp_path):
    path = _agent(tmp_path, "def agent(obs):\n    raise ValueError('boom')\n")
    out = tape_play.play_vs_tape(_world(), 0, path, steps=6)
    assert out["rewards"] is None
    assert out["status"] == ["DONE", "ERROR"]  # the live agent sits in seat 1 - tape_seat


def test_play_kagsim_finished_game(fake_engine, tmp_path, capsys):
    a = _agent(tmp_path, "def agent(obs):\n    print('hidden')\n    return {'farmer': ['EAST']}\n", "a.py")
    b = _agent(tmp_path, "def agent(obs, conf):\n    return None\n", "b.py")
    rewards, status = kagsim_play.play_kagsim(9, a, b, steps=4, shops=["BAKERY"])
    g = fake_engine.made[-1]
    assert (g.seed, g.steps, g.shops) == (9, 4, ["BAKERY"])
    assert g.actions == [({"farmer": ["EAST"]}, {})] * 3
    assert rewards == [1003.0, 2003.0] and status == ["DONE", "DONE"]
    assert "hidden" not in capsys.readouterr().out


def test_play_kagsim_agent_error(fake_engine, tmp_path):
    a = _agent(
        tmp_path,
        "n = [0]\ndef agent(obs, conf):\n    n[0] += 1\n    if n[0] == 3:\n        raise RuntimeError\n    return {}\n",
        "a.py",
    )
    b = _agent(tmp_path, "def agent(obs, conf):\n    return {}\n", "b.py")
    rewards, status = kagsim_play.play_kagsim(9, a, b, steps=10)
    assert len(fake_engine.made[-1].actions) == 2  # the game stops at the step where the agent raised
    assert status == ["ERROR", "ACTIVE"]
    assert rewards == [None, 2002.0]
