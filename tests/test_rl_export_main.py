"""rl/export_agent.py end to end on a tiny checkpoint: the generated main.py loads in a fresh folder, picks the greedy
NumPy action, and applies the --land-inject and --late-ckpt options as the submission does."""

import ast
import hashlib
import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")

from rl import act as A  # noqa: E402
from rl import encode as E  # noqa: E402
from rl import export_agent, np_net  # noqa: E402
from rl.model import HEAD_SIZES, Net  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:enable_nested_tensor:UserWarning")
RL = os.path.dirname(os.path.abspath(export_agent.__file__))


@pytest.fixture
def clean_imports():
    """Undo the sys.path entry and the rl_<sha8> package that an exported main.py adds when imported."""
    path, mods = list(sys.path), set(sys.modules)
    yield
    sys.path[:] = path
    for m in set(sys.modules) - mods:
        del sys.modules[m]


def _ckpt(path, seed):
    torch.manual_seed(seed)
    net = Net(d=64, layers=2, heads=8, dk=64, vtower=1)
    with torch.no_grad():
        net.op_bias.normal_(0, 1.0)
        net.market[2].bias.normal_(0, 1.0)
    torch.save({"model": net.state_dict(), "d": 64, "layers": 2, "vtower": 1}, path)
    return str(path)


def _export(tmp_path, monkeypatch, name, *args):
    root = tmp_path / "export"
    monkeypatch.setattr(sys, "argv", ["export_agent", "--name", name, "--root", str(root), *args])
    export_agent.main()
    return root / name


def _load_agent(folder):
    """Import main.py and return its last callable, which is what kaggle-environments calls."""
    spec = importlib.util.spec_from_file_location(f"exported_{folder.name}", folder / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [v for v in vars(mod).values() if callable(v)][-1]


def _expected(weights, o):
    """Greedy (unit choices, market bins) of the NumPy network in `weights` on observation o."""
    return np_net.greedy(np_net.load_weights(str(weights)), E.encode(o), HEAD_SIZES)


def test_package_name_is_the_hash_of_the_frozen_sources(tmp_path, monkeypatch, clean_imports):
    out = _export(tmp_path, monkeypatch, "a", "--ckpt", _ckpt(tmp_path / "c.pt", 0), "--dtype", "float16")
    h = hashlib.sha256()
    for s in ("spec.py", "encode.py", "act.py", "np_net.py"):
        h.update(open(os.path.join(RL, s), "rb").read())
    pkg = "rl_" + h.hexdigest()[:8]
    assert sorted(os.listdir(out)) == ["main.py", pkg, "weights.npz"]
    assert sorted(os.listdir(out / pkg)) == ["__init__.py", "act.py", "encode.py", "np_net.py", "spec.py"]
    for s in ("spec.py", "encode.py", "act.py", "np_net.py"):
        assert (out / pkg / s).read_bytes() == open(os.path.join(RL, s), "rb").read()
    ast.parse((out / "main.py").read_text())
    with pytest.raises(FileExistsError):  # an existing folder is never overwritten
        _export(tmp_path, monkeypatch, "a", "--ckpt", str(tmp_path / "c.pt"))


def test_exported_agent_plays_the_greedy_action(tmp_path, monkeypatch, make_obs, clean_imports):
    ck = _ckpt(tmp_path / "c.pt", 1)
    out = _export(tmp_path, monkeypatch, "sub", "--ckpt", ck, "--dtype", "float16", "--land-inject", "248:72")
    agent = _load_agent(out)
    hands = [(0, 0), (9, 9)]
    for o in (make_obs(), make_obs(step=100, hands=hands, inventories=[{}, {"WHEAT": 2}, {}])):
        units, mc = _expected(out / "weights.npz", o)
        assert agent(o, None) == A.execute(o, units, mc)

    # --land-inject 248:72: with 3 land quadrants and money >= 4000, the 4th quadrant is bought on turns 248..319
    heads = [k for k, _, _ in A.S.MARKET_HEADS]
    for step, quads, money, inject in (
        (250, ("NW", "NE", "SW"), 5000, True),
        (330, ("NW", "NE", "SW"), 5000, False),
        (250, ("NW", "NE"), 5000, False),
        (250, ("NW", "NE", "SW"), 3000, False),
    ):
        o = make_obs(step=step, quads=quads, money=money)
        units, mc = _expected(out / "weights.npz", o)
        if inject:
            mc0, mc = mc, list(mc)
            mc[heads.index("BUY_LAND")] = 1
            mc[heads.index("ORDER")] = A.S.ORDER_TEMPLATES.index("SLPHDA")
            assert mc != mc0  # the policy alone would not have bought: the injection is what is tested
        got = agent(o, None)
        assert got == A.execute(o, units, mc)
        if inject:
            assert ["BUY_LAND"] in got["market"]


def test_late_checkpoint_takes_over(tmp_path, monkeypatch, make_obs, clean_imports):
    early, late = _ckpt(tmp_path / "e.pt", 2), _ckpt(tmp_path / "l.pt", 3)
    out = _export(tmp_path, monkeypatch, "two", "--ckpt", early, "--late-ckpt", late, "--late-from", "100")
    agent = _load_agent(out)
    assert agent.__name__ == "late_agent"
    for step, w in ((50, "weights.npz"), (100, "weights_late.npz"), (99, "weights.npz")):
        o = make_obs(step=step)
        units, mc = _expected(out / w, o)
        assert agent(o, None) == A.execute(o, units, mc)
