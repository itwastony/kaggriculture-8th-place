"""Static checks of the engine sources (no compiler, no compiled module): tables, struct layout, action layout.

1. The hour sin/cos tables in engines/kagcore/kagcore.hpp are what engines/kagcore/gen_tables.py prints, and every
   value is far from a float32 rounding boundary (so libm differences cannot change it).
2. The data members of the main structs in the CUDA-capable core (engines/kagsim_cuda/sim_hd.hpp) are those of the
   vendor kagsim core they were derived from, and the canonical serializer (state_visit.hpp) writes every field;
   diag_state.field_map() lays the fields out contiguously in SER_BYTES = 7645 bytes.
3. The flat action layout int32[170] of batch_common.hpp, the Python constant ACT_I32 and parity_cpu.act_tuple agree.
"""

import importlib.util
import re
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / "engines" / "kagsim_cuda"


def _load(name, rel, stubs=()):
    """Import ROOT/rel as `name`; sys.path and the stub modules (names in stubs) are restored afterwards."""
    saved_path, saved_mods = list(sys.path), {k: sys.modules.get(k) for k in (*stubs, "kagsim_cuda")}
    for s in stubs:
        sys.modules[s] = types.SimpleNamespace(ACT_I32=170)
    try:
        spec = importlib.util.spec_from_file_location(name, ROOT / rel)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved_path
        for k, v in saved_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod


hd_parity = _load("_t_kagsim_cuda_parity", "engines/kagsim_cuda/parity.py", stubs=("kaghd_cpu", "kagsim"))
diag = _load("_t_diag_state", "engines/kagsim_cuda/diag_state.py")
kc = _load("_t_kagsim_cuda", "engines/kagsim_cuda/kagsim_cuda.py")
pcpu = _load("_t_parity_cpu", "engines/kagrl_cuda/parity_cpu.py")


# ------------------------------------------------------------------ 1. tables


def _hpp_table(src, fn):
    body = re.search(r"float " + fn + r"\(int h\) \{\s*const float t\[24\] = \{(.*?)\};", src, re.S).group(1)
    return [v.strip().rstrip("f") for v in body.split(",")]


def test_hour_tables_match_generator(capsys):
    runpy.run_path(str(ROOT / "engines" / "kagcore" / "gen_tables.py"), run_name="_t_gen_tables")
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4 and lines[0].startswith("// SIN") and lines[2].startswith("// COS")
    src = (ROOT / "engines" / "kagcore" / "kagcore.hpp").read_text()
    for head, table, fn in ((lines[0], lines[1], "hour_sin"), (lines[2], lines[3], "hour_cos")):
        gen = [v.strip().rstrip("f") for v in table.split(",")]
        hpp = _hpp_table(src, fn)
        assert [float.fromhex(v) for v in hpp] == [float.fromhex(v) for v in gen]
        assert float(head.split("=")[1].split()[0]) >= 5e7  # the margin quoted in kagcore.hpp
    hpp_sin = [float.fromhex(v) for v in _hpp_table(src, "hour_sin")]
    exact = [float(np.float32(np.sin(2 * np.pi * h / 24))) for h in range(24)]
    assert hpp_sin == exact


# ------------------------------------------------------------------ 2. structs and serializer


def _src(path):
    return path.read_text()


@pytest.mark.parametrize("struct", ["Tile", "Farm", "Market", "State", "Action", "Config"])
def test_sim_hd_structs_match_vendor(struct):
    hd = hd_parity.struct_members(_src(CUDA / "sim_hd.hpp"), struct)
    vendor = hd_parity.struct_members(_src(ROOT / "vendor" / "kagsim" / "sim" / "sim.hpp"), struct)
    assert hd and hd == vendor


def test_one_line_structs_match_vendor():
    hd, vendor = _src(CUDA / "sim_hd.hpp"), _src(ROOT / "vendor" / "kagsim" / "sim" / "sim.hpp")
    for struct in ("UnitAction", "Order", "CropDef", "AnimalDef"):
        line = re.search(r"^struct " + struct + r" \{.*\};$", hd, re.M).group(0)
        assert line in vendor.splitlines(), struct


def test_serializer_writes_every_field():
    src, vis = _src(CUDA / "sim_hd.hpp"), _src(CUDA / "state_visit.hpp")
    missing = []  # the same rule as parity.static_checks()
    for st, pref in (("Tile", "t."), ("Farm", "f."), ("Market", "s.market."), ("State", "s.")):
        for nm in hd_parity.struct_members(src, st):
            if st == "State" and nm in ("farms", "market"):
                continue
            if not re.search(re.escape(pref) + nm + r"\b", vis):
                missing.append(f"{st}.{nm}")
    assert missing == []


def test_field_map_is_contiguous():
    fields, total = diag.field_map()
    assert total == 7645  # SER_BYTES quoted in engines/kagrl_cuda/README.md
    off = 0
    for _label, dt, cnt, o in fields:
        assert o == off
        off += np.dtype(dt).itemsize * cnt
    assert off == total
    labels = [f[0] for f in fields]
    assert len(labels) == len(set(labels))
    assert labels.count("f0.tile[9][9].kind") == 1 and any(lb.startswith("market.") for lb in labels)


# ------------------------------------------------------------------ 3. action layout


def test_action_layout():
    src = _src(CUDA / "batch_common.hpp")
    hd = _src(CUDA / "sim_hd.hpp")
    max_units = int(re.search(r"constexpr int MAX_UNITS = (\d+);", hd).group(1))
    slots = int(re.search(r"constexpr int MAX_ORDER_SLOTS = (\d+);", hd).group(1))
    assert (max_units, slots) == (40, 16)
    assert re.search(r"ACT_NORD = 1 \+ 3 \* MAX_UNITS;\s*// = 121", src)
    assert re.search(r"ACT_I32 = ACT_ORDERS \+ 3 \* MAX_ORDER_SLOTS; // = 170", src)
    assert 1 + 3 * max_units + 1 + 3 * slots == kc.ACT_I32 == 170


def test_act_tuple():
    row = np.zeros(170, np.int32)
    row[0] = 2
    row[1:7] = [5, 0, 1, 2, 3, 4]
    row[121] = 1
    row[122:125] = [6, 2, 31]
    row[125:128] = [9, 9, 9]  # beyond n_orders: ignored
    assert pcpu.act_tuple(row) == (2, ((5, 0, 1), (2, 3, 4)), 1, ((6, 2, 31),))
    assert pcpu.act_tuple(np.zeros(170, np.int32)) == (0, (), 0, ())
