"""The NumPy forward pass of the submitted agent (rl/np_net.py) against the torch model it is exported from.

A small randomly initialised Net (d=64, 2 layers; heads=8 and dk=64 as fixed by export_weights) is saved as a
checkpoint, exported to .npz and evaluated on encoded observations by both implementations.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rl import encode as E  # noqa: E402
from rl import np_net  # noqa: E402
from rl import spec as S  # noqa: E402
from rl.model import HEAD_SIZES, Net, market_logp, unit_logp  # noqa: E402

# torch warns that nested tensors are off because the layers are pre-norm; irrelevant here
pytestmark = pytest.mark.filterwarnings("ignore:enable_nested_tensor:UserWarning")


def _net(vtower=0, seed=0):
    torch.manual_seed(seed)
    net = Net(d=64, layers=2, heads=8, dk=64, vtower=vtower)
    with torch.no_grad():  # the zero-initialised biases would hide errors in their use
        net.op_bias.normal_(0, 0.5)
        net.here.normal_(0, 0.5)
    return net.eval()


def _ckpt(tmp_path, net, vtower=0):
    path = tmp_path / "ck.pt"
    torch.save({"model": net.state_dict(), "d": 64, "layers": 2, "vtower": vtower}, path)
    return path


def _batch(e):
    return {k: torch.as_tensor(e[k]).unsqueeze(0) for k in ("tiles", "units", "unit_pos", "unit_mask", "glob")}


def _encodings(make_obs):
    hands = [(0, 0), (9, 9), (4, 5)]
    return [
        E.encode(make_obs()),
        E.encode(
            make_obs(step=300, money=12345, quads=("NW", "NE"), hands=hands, inventories=[{}, {"MILK": 3}, {}, {}])
        ),
    ]


@pytest.mark.parametrize("vtower", [0, 1])
def test_numpy_forward_matches_torch(tmp_path, make_obs, vtower):
    net = _net(vtower)
    out = tmp_path / "w.npz"
    np_net.export_weights(str(_ckpt(tmp_path, net, vtower)), str(out), "float32")
    w = np_net.load_weights(str(out))
    for e in _encodings(make_obs):
        with torch.no_grad():
            lg, ml, v = net(**_batch(e))
        nlg, nml, nv = np_net.forward(w, e)
        um = e["unit_mask"]
        # padded units: torch's fast path may zero their tokens, the NumPy pass does not; only present units matter
        np.testing.assert_allclose(nlg[um], lg[0].numpy()[um], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(nml, ml[0].numpy(), rtol=1e-4, atol=1e-4)
        if vtower:
            assert nv is None  # the value tower is not exported
        else:
            assert nv == pytest.approx(float(v[0]), rel=1e-4, abs=1e-4)


def test_numpy_greedy_matches_torch_argmax(tmp_path, make_obs):
    net = _net(seed=1)
    out = tmp_path / "w.npz"
    np_net.export_weights(str(_ckpt(tmp_path, net)), str(out), "float32")
    w = np_net.load_weights(str(out))
    for e in _encodings(make_obs):
        units, mc = np_net.greedy(w, e, HEAD_SIZES)
        with torch.no_grad():
            lg, ml, _ = net(**_batch(e))
        tm = torch.as_tensor(e["tmask"]).unsqueeze(0)
        flat = lg.masked_fill(~tm, -1e9).view(1, S.MAX_UNITS, -1).argmax(-1)[0]
        n = int(e["unit_mask"].sum())
        assert units[:n] == [(int(c) // S.NOPS, int(c) % S.NOPS) for c in flat[:n]]
        mm = ml[0].masked_fill(~torch.as_tensor(e["mmask"]), -1e9)
        want, o = [], 0
        for h in HEAD_SIZES:
            want.append(int(mm[o : o + h].argmax()))
            o += h
        assert mc == want
        # the greedy choice is legal and is the most likely one under the torch log-probabilities
        assert all(e["tmask"][i, t, op] for i, (t, op) in enumerate(units[:n]))
        t_ = torch.tensor([[t for t, _ in units]])
        o_ = torch.tensor([[op for _, op in units]])
        um = torch.as_tensor(e["unit_mask"]).unsqueeze(0)
        lp = unit_logp(lg, tm, um, t_, o_)[0, :n]
        assert torch.allclose(lp, torch.log_softmax(lg.masked_fill(~tm, -1e9).view(1, 16, -1), -1).amax(-1)[0, :n])
        mlp = market_logp(ml, torch.as_tensor(e["mmask"]).unsqueeze(0), torch.tensor([mc]))
        assert mlp.shape == (1, len(HEAD_SIZES)) and bool((mlp <= 0).all())


def test_export_weights_format(tmp_path):
    net = Net(d=64, layers=2, heads=8, dk=64, vtower=1)
    out = tmp_path / "w.npz"
    np_net.export_weights(str(_ckpt(tmp_path, net, vtower=1)), str(out), "float16")
    z = np.load(out)
    assert z["__meta__"].tolist() == [64, 2, 8, 64]
    assert int(z["__vtower__"]) == 1
    keys = [k for k in z.files if not k.startswith("__")]
    assert keys and all(z[k].dtype == np.float16 for k in keys)
    assert not any(k.startswith(("venc.", "vnorm.", "vopp.")) for k in keys)
    assert set(keys) == {k for k in net.state_dict() if not k.startswith(("venc.", "vnorm.", "vopp."))}
    w = np_net.load_weights(str(out))
    assert (w["__d__"], w["__layers__"], w["__heads__"], w["__dk__"], w["__vtower__"]) == (64, 2, 8, 64, 1)
    assert w["tile_in.weight"].dtype == np.float32


def test_export_weights_is_byte_stable(tmp_path):
    """The same checkpoint exports to the same bytes (the submission check compares weights.npz arrays)."""
    path = _ckpt(tmp_path, _net())
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    np_net.export_weights(str(path), str(a), "float16")
    np_net.export_weights(str(path), str(b), "float16")
    za, zb = np.load(a), np.load(b)
    assert za.files == zb.files
    assert all(za[k].tobytes() == zb[k].tobytes() for k in za.files)
