"""Helpers of the PPO/APPO trainers that do not need games or a GPU: mask packing, sampling, loss terms, opponent
keys and the versioned opponent pool."""

import json
import os
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rl import appo  # noqa: E402
from rl import spec as S  # noqa: E402
from rl.model import HEAD_SIZES, NOID, market_logp, unit_logp  # noqa: E402
from rl.ppo import market_terms, sample, unit_terms  # noqa: E402


def _batch(B=3, seed=0):
    """Random logits and masks shaped like a trainer batch; every unit and head keeps at least one legal action."""
    g = torch.Generator().manual_seed(seed)
    lg = torch.randn(B, S.MAX_UNITS, S.NT, S.NOPS, generator=g)
    ml = torch.randn(B, sum(HEAD_SIZES), generator=g)
    tmask = torch.rand(B, S.MAX_UNITS, S.NT, S.NOPS, generator=g) < 0.05
    tmask[..., 0, 0] = True
    mmask = torch.rand(B, sum(HEAD_SIZES), generator=g) < 0.5
    o = 0
    for n in HEAD_SIZES:
        mmask[:, o] = True
        o += n
    unit_mask = torch.zeros(B, S.MAX_UNITS, dtype=torch.bool)
    unit_mask[:, :5] = True
    return lg, ml, {"tmask": tmask, "mmask": mmask, "unit_mask": unit_mask}


def test_unpack_inverts_numpy_packbits():
    rng = np.random.default_rng(0)
    m = rng.random((2, S.MAX_UNITS, S.NT, S.NOPS)) < 0.3
    packed = np.packbits(m.reshape(2, S.MAX_UNITS, -1), axis=-1)  # as the CPU workers store it
    assert packed.shape[-1] == appo.PK
    out = appo.unpack(torch.from_numpy(packed))
    assert out.dtype == torch.bool and np.array_equal(out.numpy(), m)


def test_sample_greedy_is_masked_argmax():
    lg, ml, b = _batch()
    uc, mc = sample(lg, ml, b, greedy=True)
    flat = lg.masked_fill(~b["tmask"], -1e9).view(lg.shape[0], S.MAX_UNITS, -1)
    assert torch.equal(uc, flat.argmax(-1))
    mm = ml.masked_fill(~b["mmask"], -1e9)
    o = 0
    for h, n in enumerate(HEAD_SIZES):
        assert torch.equal(mc[:, h], mm[:, o : o + n].argmax(-1))
        o += n


def test_sample_draws_only_legal_actions():
    lg, ml, b = _batch(seed=1)
    torch.manual_seed(0)
    uc, mc = sample(lg, ml, b)
    B = lg.shape[0]
    flat_mask = b["tmask"].view(B, S.MAX_UNITS, -1)
    assert flat_mask.gather(-1, uc.unsqueeze(-1)).all()
    o = 0
    for h, n in enumerate(HEAD_SIZES):
        assert b["mmask"][:, o : o + n].gather(-1, mc[:, h : h + 1]).all()
        o += n


def test_loss_terms_agree_with_model_logp():
    lg, ml, b = _batch(seed=2)
    uc, mc = sample(lg, ml, b, greedy=True)
    lpu, eu, ku = unit_terms(lg, lg, b, uc)
    ref = unit_logp(lg, b["tmask"], b["unit_mask"], uc // S.NOPS, uc % S.NOPS)
    assert torch.allclose(lpu, ref, atol=1e-5)
    assert torch.allclose(ku, torch.zeros_like(ku), atol=1e-5)  # KL to itself
    assert bool((eu >= -1e-6).all()) and bool((eu[:, 5:] == 0).all())  # absent units contribute nothing
    lpm, em, km = market_terms(ml, ml, b, mc)
    assert torch.allclose(lpm, market_logp(ml, b["mmask"], mc), atol=1e-5)
    assert torch.allclose(km, torch.zeros_like(km), atol=1e-5)
    _, _, ku2 = unit_terms(lg, torch.randn_like(lg), b, uc)
    assert bool((ku2[:, :5] > 0).all())


def test_opponent_keys():
    assert appo.okey("self") == "self"
    assert appo.okey("agents/caretaker/main.py") == "caretaker"
    assert appo.okey("tapes:data/pools/pool_weaktrain.json") == "tapes_pool_weaktrain"
    assert [appo.opp_id(n) for n in appo.NOT_FIELD] == [0, 1, 2]
    i = appo.opp_id("agents/caretaker/main.py")
    assert 3 <= i < NOID and i == appo.opp_id("agents/caretaker/main.py")


def test_load_tapes(tmp_path):
    p = tmp_path / "pool.json"
    worlds = [
        {"world": "w1.json.gz", "opp_seat": 1, "seed": 5, "shops": ["BAKERY"], "ref": 1000, "ref_opp": 900},
        {"world": "w2.json.gz", "opp_seat": 0, "seed": 6, "shops": []},
    ]
    p.write_text(json.dumps({"worlds": worlds}))
    assert appo.load_tapes(str(p)) == [
        ("w1.json.gz", 1, 5, ["BAKERY"], 1000.0, 900.0),
        ("w2.json.gz", 0, 6, [], None, None),
    ]


def test_pool_selection_rules(tmp_path):
    files = [str(tmp_path / f"v{i:05d}.pt") for i in range(4)]
    pool = appo.Pool([("own", 1.0, "uniform", None, files)], recent=0.5, q_lr=0.1, pfsp_pow=2.0, floor=0.05)
    assert pool.weights("uniform", files) == [1.0] * 4
    assert pool.weights("latest", files) == [0.0, 0.0, 0.0, 1.0]
    assert pool.weights("recent", files) == [0.0, 0.0, 1.0, 1.0]
    # pfsp: unknown versions count as P = 0.5; a version we always beat gets the floor
    pool.stat = {files[0]: {"n": 10, "w": 10.0, "p": 1.0, "q": 0.0}}
    assert pool.weights("pfsp", files) == [0.05, 0.25, 0.25, 0.25]
    f, p, n, name = pool.pick(random.Random(0), pool.listing())
    assert f in files and n == 4 and name == "own" and p == pytest.approx(0.25)


def test_pool_listing_reads_the_directory(tmp_path):
    for i in (2, 1, 3):
        f = tmp_path / f"v{i:05d}.pt"
        f.write_bytes(b"")
        os.utime(f, (1e9, 1e9))
    os.utime(tmp_path / "v00003.pt")  # just written: skipped while younger than min_age
    (tmp_path / ".v00004.123.tmp").write_bytes(b"")  # a version being written is not listed
    pool = appo.Pool([("own", 1.0, "latest", str(tmp_path), ["static.pt"])], 0.5, 0.1, 2.0, 0.05, min_age=3600)
    ((_, _, _, fs),) = pool.listing()
    assert fs == ["static.pt", os.path.join(str(tmp_path), "v00001.pt"), os.path.join(str(tmp_path), "v00002.pt")]
    with pytest.raises(RuntimeError):
        pool.pick(random.Random(0), [("own", 1.0, "latest", [])])


def test_pool_quality_update():
    pool = appo.Pool([], recent=0.5, q_lr=0.1, pfsp_pow=2.0, floor=0.05)
    pool.update({"a": [10, 5.0, 0.5, 2]})
    assert pool.stat["a"]["n"] == 10 and pool.stat["a"]["p"] == 0.5
    assert pool.stat["a"]["q"] == pytest.approx(-0.1 * 0.5 / (2 * 0.5))
    pool.update({"a": [10, 10.0, 0.5, 2], "b": [0, 0.0, 0.5, 2]})  # games-free entries are ignored
    assert "b" not in pool.stat
    assert pool.stat["a"]["p"] == pytest.approx(0.7 * 0.5 + 0.3 * 1.0)
