"""Tests of engines/kagsim_cuda/record_streams.py without playing games: the job list and the random action generators.

The digests pin the exact draws from random.Random (a change of the job order, the seeds or the generators changes the
recorded streams and therefore every parity run that uses them). They were taken from the version of the file that
recorded the streams of the final checks, whose agent pairs are now COMPETITION_PAIRS (--pairs competition); the default
SHIPPED_PAIRS changes only the agent paths.
"""

import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    saved = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location(name, ROOT / rel)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved  # the module puts harness/ on sys.path
    return mod


rs = _load("_t_record_streams", "engines/kagsim_cuda/record_streams.py")


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=repr).encode()).hexdigest()


def test_make_jobs_golden():
    jobs = rs.make_jobs(130, 54, 30, 6, 20260925, rs.COMPETITION_PAIRS)
    assert len(jobs) == 220
    assert _digest(jobs) == "24bfa27751386ff9205c3857aeeb4fc31ec319a877b723140297491a1137731b"
    small = rs.make_jobs(5, 4, 3, 2, 7, rs.COMPETITION_PAIRS)
    assert _digest(small) == "a66cf2dc50baff194a3abfb4326cb5c042b5db8b00c94cdf45feb26563988d08"
    assert rs.make_jobs(1, 0, 0, 0, 7, rs.COMPETITION_PAIRS) == [
        {
            "mode": "live",
            "agents": ["agents/c3bc/main.py", "agents/v4_look12/main.py"],
            "chaos": [0, 0],
            "junk": [0, 0],
            "seed": 695425565,
            "shops": None,
            "rng_seed": 647892279,
            "id": 0,
        }
    ]


def _agent_free(jobs):
    return [{k: v for k, v in j.items() if k != "agents"} for j in jobs]


def test_shipped_pairs_change_only_agents():
    shipped = rs.make_jobs(130, 54, 30, 6, 20260925)
    competition = rs.make_jobs(130, 54, 30, 6, 20260925, rs.COMPETITION_PAIRS)
    assert _agent_free(shipped) == _agent_free(competition)
    for a, b in zip(shipped, competition):  # the random seats stay where they were
        assert [x == "random" for x in a["agents"]] == [x == "random" for x in b["agents"]]
    used = {x for j in shipped for x in j["agents"] if x != "random"}
    assert used == {"agents/caretaker/main.py", "agents/pass_agent/main.py"}
    assert all((ROOT / x).is_file() for x in used)
    assert len(rs.SHIPPED_PAIRS) % 3 != 0  # chaos games index the pairs with (3 * i) % len
    assert rs.PAIR_SETS == {"shipped": rs.SHIPPED_PAIRS, "competition": rs.COMPETITION_PAIRS}


def test_make_jobs_structure():
    jobs = rs.make_jobs(130, 54, 30, 6, 20260925)
    modes = [j["mode"] for j in jobs]
    assert modes == ["live"] * 130 + ["chaos"] * 54 + ["garbage"] * 30 + ["random2"] * 6
    assert [j["id"] for j in jobs] == list(range(220))
    for k, j in enumerate(jobs):
        assert 1 <= j["seed"] < 2**31 and 0 <= j["rng_seed"] < 2**31
        if k % 4 == 3:  # every 4th game pins 1..8 shops with ids 0..7
            assert 1 <= len(j["shops"]) <= 8 and all(0 <= s < 8 for s in j["shops"])
        else:
            assert j["shops"] is None
    chaos = [j for j in jobs if j["mode"] == "chaos"]
    assert {p for j in chaos for p in j["chaos"]} == {0, 0.005, 0.02, 0.06}
    garbage = [j for j in jobs if j["mode"] == "garbage"]
    assert all(j["agents"].count("random") == 1 for j in garbage)
    assert all(j["agents"] == ["random", "random"] and j["junk"] == [0.1, 0.4] for j in jobs[-6:])


def test_random_actions_golden():
    rng = random.Random(123)
    acts = [rs.random_agent_action(rng, 0.15) for _ in range(50)]
    base = {"farmer": ["NORTH"], "hands": [["WATER"]], "market": [["SELL", "WHEAT", 3]]}
    muts = [rs.mutate(base, rng, 0.3) for _ in range(50)]
    assert _digest([acts, muts]) == "d41405820f88aa793c0352b963f4a060b76e02f2336f373f24267243cbc28e62"
    assert base == {"farmer": ["NORTH"], "hands": [["WATER"]], "market": [["SELL", "WHEAT", 3]]}  # not modified


def test_generators_without_junk_are_legal_shaped():
    rng = random.Random(5)
    for _ in range(200):
        u = rs.rand_unit(rng, junk=0)
        assert isinstance(u, list) and u[0] in rs.OPS and 1 <= len(u) <= 4
        o = rs.rand_order(rng, junk=0)
        assert o[0] in rs.MOPS and o[1] in rs.ITEMS and len(o) == 3
