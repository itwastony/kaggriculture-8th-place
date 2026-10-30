"""Tests of the job lists of engines/kagcore/parity.py and engines/kagrl_cuda/parity_cpu.py, without engines or games.

The default jobs must use only the bots shipped in this repository and keep the seeds and random roles of the
competition-time jobs; the competition-time tables are pinned by their digests (taken from the version of the files
that ran the final checks).
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = {"caretaker", "pass_agent"}


def _load(name, rel):
    saved = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location(name, ROOT / rel)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved  # the modules put the engine build folders on sys.path
    return mod


kp = _load("_t_kagcore_parity", "engines/kagcore/parity.py")
pc = _load("_t_kagrl_parity_cpu", "engines/kagrl_cuda/parity_cpu.py")


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _agents_kagcore(specs):
    out = set()
    for s in specs:
        out |= {s["a"], s["b"]} if s["mode"] == "A" else {r for rr in s["roles"] for r in rr}
    return out - set(kp.RANDOM_ROLES)


def _mask_kagcore(specs):
    """The specs with agent names replaced by AG (random roles kept)."""
    out = []
    for s in specs:
        s = dict(s)
        if s["mode"] == "A":
            s["a"] = s["b"] = "AG"
        else:
            s["roles"] = [[r if r in kp.RANDOM_ROLES else "AG" for r in rr] for rr in s["roles"]]
        out.append(s)
    return out


def _kagcore_default(quick, K=2):
    agents = list(kp.SHIPPED_AGENTS)  # as main() builds them from --agents
    pairs = [(a, b) for a in agents for b in agents]
    opps = [a for a in agents for _ in (0, 1)]
    return kp.make_specs(quick, K, pairs, opps)


def test_kagcore_competition_jobs_pinned():
    full = kp.make_specs(False, 2, kp.COMPETITION_PAIRS, kp.COMPETITION_OPPS)
    quick = kp.make_specs(True, 2, kp.COMPETITION_PAIRS, kp.COMPETITION_OPPS)
    assert [len(full), len(quick)] == [24, 4]
    assert _digest([full, quick]) == "78a47654db391ceda252621a7e087b8dc3f19dbd3ef14cc0ad435fa0852892d4"


def test_kagcore_default_uses_shipped_bots_only():
    for quick in (False, True):
        d = _kagcore_default(quick)
        c = kp.make_specs(quick, 2, kp.COMPETITION_PAIRS, kp.COMPETITION_OPPS)
        assert _agents_kagcore(d) == SHIPPED
        assert _mask_kagcore(d) == _mask_kagcore(c)  # same seeds, steps, K, random roles
        assert kp.missing_agents(d) == []
    quick = _kagcore_default(True)
    assert [(s["a"], s["b"]) for s in quick if s["mode"] == "A"] == [
        ("caretaker", "caretaker"),
        ("caretaker", "pass_agent"),
    ]
    vs = [s["roles"] for s in _kagcore_default(False) if s["mode"] == "B" and s["use_vs"]]
    for roles in vs:  # every game has exactly one random seat; each agent plays both seats
        assert all(sum(r in kp.RANDOM_ROLES for r in rr) == 1 for rr in roles)
        seats = {(r, p) for rr in roles for p, r in enumerate(rr) if r not in kp.RANDOM_ROLES}
        assert seats == {(a, p) for a in SHIPPED for p in (0, 1)}


def test_kagcore_missing_agents_reported():
    c = kp.make_specs(True, 2, kp.COMPETITION_PAIRS, kp.COMPETITION_OPPS)
    assert "c3bc" in kp.missing_agents(c) and "v4_look12" in kp.missing_agents(c)
    assert kp.missing_agents([{"mode": "A", "a": "caretaker", "b": "no_such_bot"}]) == ["no_such_bot"]


def _mask_pc(specs):
    return [
        {**s, "roles": [[r if r in pc.CHOICE_ROLES else "AG" for r in rr] for rr in s["roles"]]} if "roles" in s else s
        for s in specs
    ]


def test_parity_cpu_competition_jobs_pinned():
    full = pc.make_specs(False, 2, pc.COMPETITION_A_PAIRS, pc.COMPETITION_L_ROLES)
    quick = pc.make_specs(True, 2, pc.COMPETITION_A_PAIRS, pc.COMPETITION_L_ROLES)
    assert [len(full), len(quick)] == [14, 5]
    assert _digest([full, quick]) == "d9122ac3657943f4f08e2c40583adb4b81095b72417afb9f32c23a2e31a5de05"


def test_parity_cpu_default_uses_shipped_bots_only():
    for quick in (False, True):
        d = pc.make_specs(quick, 2, pc.SHIPPED_A_PAIRS, pc.SHIPPED_L_ROLES)
        c = pc.make_specs(quick, 2, pc.COMPETITION_A_PAIRS, pc.COMPETITION_L_ROLES)
        assert _mask_pc(d) == _mask_pc(c)  # same seeds, steps, K and choice roles
        used = {r for s in d for rr in s.get("roles", ()) for r in rr} - set(pc.CHOICE_ROLES)
        assert used == SHIPPED
        assert d[-1] == {"mode": "API"}
    for name in pc.AGENTS:  # the workers recognize agents by membership in AGENTS
        assert name not in pc.CHOICE_ROLES
    assert SHIPPED.issubset(pc.AGENTS)
    assert all(sum(r in pc.CHOICE_ROLES for r in rr) == 1 for rr in pc.SHIPPED_L_ROLES)
    assert all(not set(rr) & set(pc.CHOICE_ROLES) for rr in pc.SHIPPED_A_PAIRS)


def test_parity_cpu_missing_inputs(monkeypatch, tmp_path):
    d = pc.make_specs(True, 2, pc.SHIPPED_A_PAIRS, pc.SHIPPED_L_ROLES)
    monkeypatch.setattr(pc, "STREAMS", str(tmp_path / "streams.pkl.gz"))
    msgs = pc.missing_inputs(d)
    assert len(msgs) == 1 and "record_streams.py" in msgs[0]
    (tmp_path / "streams.pkl.gz").write_bytes(b"")
    assert pc.missing_inputs(d) == []
    c = pc.make_specs(True, 2, pc.COMPETITION_A_PAIRS, pc.COMPETITION_L_ROLES)
    assert any("agents/c3bc/main.py" in m for m in pc.missing_inputs(c))


def test_parity_cpu_missing_builds(monkeypatch, tmp_path):
    present = {"kaghd_cpu", "kagcore", "kagsim"}
    monkeypatch.setattr(pc.importlib.util, "find_spec", lambda name: object() if name in present else None)
    monkeypatch.setattr(pc, "HERE", str(tmp_path))
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "libkagrl_cpu.so").write_bytes(b"")
    assert pc.missing_builds() == []
    present.discard("kaghd_cpu")
    msgs = pc.missing_builds()
    assert len(msgs) == 1 and "bash engines/kagsim_cuda/build_cpu.sh" in msgs[0]
    (tmp_path / "build" / "libkagrl_cpu.so").unlink()
    assert any("bash engines/kagrl_cuda/build_cpu.sh" in m for m in pc.missing_builds())
