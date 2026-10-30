"""Invariants of the action space (rl/spec.py), the executor/labeller (rl/act.py) and the encoder (rl/encode.py)."""

import numpy as np
import pytest

from rl import act as A
from rl import encode as E
from rl import spec as S


def test_spec_sizes():
    assert S.NT == 100
    assert S.NOPS == len(S.OPS) == 56
    assert len(set(S.OPS)) == S.NOPS
    assert len(S.MARKET_HEADS) == 24
    assert S.SHED_IDX == (44, 45, 54, 55)
    assert all(sorted(t) == sorted("SPDAHL") for t in S.ORDER_TEMPLATES)
    assert S.SELL_BINS[0] == 0 and S.SELL_BINS[-1] == -1  # 0 = no order, -1 = the whole shed
    assert sum(len(b) for _, _, b in S.MARKET_HEADS) == E.MB


@pytest.mark.parametrize("op", range(56))
def test_op_engine_round_trip(op):
    assert A.engine_to_op(A.op_to_engine(op)) == op


def test_engine_to_op_moves_and_pass():
    for m in A.MOVES:
        assert A.engine_to_op([m]) is None
    assert A.engine_to_op(["PASS"]) == S.OP_ID["STAY"]
    assert A.engine_to_op([]) == S.OP_ID["STAY"]
    # pickup quantities map to the nearest available bin
    assert A.engine_to_op(["PICKUP", "WHEAT", 40]) == S.OP_ID["PICKUP_WHEAT_10"]
    assert A.engine_to_op(["PLACE", "MILK", 7]) == S.OP_ID["PLACE_MILK_ALL"]


def test_step_toward_moves_along_x_first():
    assert A.step_toward((4, 4), S.tile_id(6, 2)) == "EAST"
    assert A.step_toward((6, 4), S.tile_id(6, 2)) == "NORTH"
    assert A.step_toward((6, 2), S.tile_id(6, 2)) is None


def test_execute_moves_then_operates(make_obs):
    o = make_obs(hands=[(0, 0)])
    plant = S.OP_ID["PLANT_WHEAT"]
    a = A.execute(o, [(S.tile_id(4, 4), plant), (S.tile_id(0, 3), S.OP_ID["WATER"])], [0] * len(S.MARKET_HEADS))
    assert a["farmer"] == ["PLANT", "WHEAT"]
    assert a["hands"] == [["SOUTH"]]
    assert a["market"] == []


def _choice(**bins):
    """Market choice with the given bin index per head name ("SELL:MILK", "ORDER", ...), 0 elsewhere."""
    names = [k if i is None else f"{k}:{i}" for k, i, _ in S.MARKET_HEADS]
    c = [0] * len(names)
    for k, v in bins.items():
        c[names.index(k.replace("__", ":"))] = v
    return c


def test_market_orders_and_labels_round_trip(make_obs):
    o = make_obs(shed={"MILK": 10, "WHEAT": 5})
    sell_all = len(S.SELL_BINS) - 1
    choice = _choice(
        SELL__WHEAT=sell_all,
        SELL__MILK=3,
        BUY_SEED__MELON=4,
        HIRE=2,
        BUY_LAND=1,
        ORDER=S.ORDER_TEMPLATES.index("SLPHDA"),
        SELL_FIRST=S.PRODUCTS.index("WHEAT") + 1,
    )
    orders = A.market_orders(o, choice)
    assert orders == [
        ["SELL", "WHEAT", 5],
        ["SELL", "MILK", 3],
        ["BUY_LAND"],
        ["HIRE"],
        ["HIRE"],
        ["BUY_SEED", "MELON", 4],
    ]
    labels = A.market_labels(o, {"market": orders})
    assert labels == choice
    assert A.market_orders(o, labels) == orders


def test_market_orders_capped_at_ten(make_obs):
    o = make_obs()
    orders = A.market_orders(o, _choice(HIRE=len(S.HIRE_BINS) - 1))
    assert len(orders) == S.MAX_ORDERS == 10


def test_unit_labels_hindsight_target(make_obs):
    # the farmer walks east twice and plants on the third turn: every step is labelled (tile of the plant, PLANT_WHEAT)
    traj = []
    for t, (x, act) in enumerate([(4, ["EAST"]), (5, ["EAST"]), (6, ["PLANT", "WHEAT"])]):
        o = make_obs(step=t)
        o["farms"][0]["farmer"] = [x, 4]
        traj.append((o, {"farmer": act, "hands": []}))
    want = (S.tile_id(6, 4), S.OP_ID["PLANT_WHEAT"])
    assert A.unit_labels(traj) == [[want], [want], [want]]


def test_encode_shapes_and_masks(obs):
    e = E.encode(obs)
    assert e["tiles"].shape == (S.NT, E.FT) and e["tiles"].dtype == np.float32
    assert e["units"].shape == (S.MAX_UNITS, E.FU)
    assert e["glob"].shape == (E.FG,)
    assert e["tmask"].shape == (S.MAX_UNITS, S.NT, S.NOPS)
    assert e["mmask"].shape == (E.MB,) and e["mmask"].all()
    assert e["unit_mask"].tolist() == [True] + [False] * (S.MAX_UNITS - 1)
    assert e["unit_pos"][0] == S.tile_id(4, 4)
    m = e["tmask"][0]
    assert m[:, S.OP_ID["STAY"]].all()
    assert m[:, S.OP_ID["PLANT_MELON"]].all()  # every tile is empty
    assert not m[:, S.OP_ID["WATER"]].any()
    pick = S.OP_ID["PICKUP_WHEAT_1"]
    assert m[list(S.SHED_IDX), pick].all() and m[:, pick].sum() == len(S.SHED_IDX)
    assert not m[:, S.OP_ID["DROP"]].any()  # the farmer carries nothing
    assert not e["tmask"][1:].any()  # absent units have no legal action


def test_encode_drop_needs_a_carried_item(make_obs):
    e = E.encode(make_obs(inventories=[{"MILK": 2}]))
    m = e["tmask"][0]
    assert m[list(S.SHED_IDX), S.OP_ID["DROP"]].all()
    assert m[list(S.SHED_IDX), S.OP_ID["PLACE_MILK_ALL"]].all()
    assert not m[list(S.SHED_IDX), S.OP_ID["PLACE_EGG_ALL"]].any()


def test_glob_offsets_used_by_appo(make_obs):
    """rl/appo.py reads the money and the land-quadrant flags straight from glob (--land-inject, --fast-gate)."""
    appo = pytest.importorskip("rl.appo")
    e = E.encode(make_obs(money=4321, quads=("NW", "NE", "SW"), opp_quads=("NW", "SE")))
    g = e["glob"]
    assert g[5] * 1e5 == pytest.approx(4321, abs=1e-2)
    assert g[appo.QOWN0 : appo.QOWN0 + 4].tolist() == [1, 1, 1, 0]
    assert g[appo.OQ0 : appo.OQ0 + 4].tolist() == [1, 0, 0, 1]


def test_encode_is_seat_symmetric(make_obs):
    a = E.encode(make_obs(player=0, quads=("NW", "NE"), opp_quads=("NW",)))
    b = E.encode(make_obs(player=1, quads=("NW", "NE"), opp_quads=("NW",)))
    for k in a:
        assert np.array_equal(a[k], b[k]), k
