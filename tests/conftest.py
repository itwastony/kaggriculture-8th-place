"""Shared pytest setup: the repository root and entry_points/ on sys.path, plus a minimal game observation.

The tests are fast and offline: they do not play games, build or import the game engines, or read submission/ and
artifacts/.
"""

import copy
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY = os.path.join(ROOT, "entry_points")
for _p in (ENTRY, ROOT):  # ROOT ends up first: `import rl`; entry_points/: `import common` as the scripts do
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _make_obs(player=0, step=0, money=3000, quads=("NW",), opp_quads=("NW",), hands=(), shed=None, inventories=None):
    """A minimal observation in the engine's format: empty 10x10 farms, the farmer on (4, 4)."""
    day, hour = divmod(step, 24)

    def farm(q, h):
        return {
            "money": money,
            "tiles": [[None] * 10 for _ in range(10)],
            "farmer": [4, 4],
            "hands": [list(p) for p in h],
            "unlocked_quadrants": list(q),
            "hires_today": 0,
        }

    farms = [farm(quads, hands), farm(opp_quads, ())]
    if player == 1:
        farms.reverse()
    return {
        "player": player,
        "day": day,
        "hour": hour,
        "step": step,
        "farms": farms,
        "private": {
            "shed": dict(shed or {}),
            "seeds": {},
            "inventories": [{} for _ in range(1 + len(hands))] if inventories is None else copy.deepcopy(inventories),
        },
        "market": {"prices": {}, "inventory": {}},
        "town": {"unlocked_shops": []},
    }


@pytest.fixture
def make_obs():
    """Factory of minimal observations (keyword arguments as in _make_obs)."""
    return _make_obs


@pytest.fixture
def obs():
    """A fresh minimal observation (step 0, player 0, one land quadrant, empty shed)."""
    return _make_obs()
