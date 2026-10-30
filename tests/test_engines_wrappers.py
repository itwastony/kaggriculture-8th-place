"""Tests of the Python wrappers of the batch engines with fake libraries (nothing is compiled or loaded).

engines/kagsim_cuda/kagsim_cuda.py (GpuBatch over a fake ctypes library, CpuBatch over a fake kaghd_cpu module) and
engines/kagrl_cuda/kagrl.py (RLBatch, CPU backend, over a fake ctypes library): library loading and caching, the
ACT_I32 / backend checks, argument validation, the pointers and values passed to the C API, and error propagation.
"""

import ctypes
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kc = _load("_t_kagsim_cuda_wrapper", "engines/kagsim_cuda/kagsim_cuda.py")
kr = _load("_t_kagrl_wrapper", "engines/kagrl_cuda/kagrl.py")


def _arr(ptr, ctype, n):
    """numpy view of n C values at the address of a ctypes.c_void_p."""
    return np.ctypeslib.as_array((ctype * n).from_address(ptr.value))


class _Fn:
    """A C function stand-in: accepts argtypes/restype like a ctypes function and records its calls."""

    def __init__(self, name, impl, calls):
        self.name, self.impl, self.calls = name, impl, calls
        self.argtypes = self.restype = None

    def __call__(self, *args):
        self.calls.append((self.name, args))
        return self.impl(*args)


class FakeLib:
    def __init__(self, impls):
        self.calls = []
        for name, impl in impls.items():
            setattr(self, name, _Fn(name, impl, self.calls))

    def called(self, name):
        return [a for n, a in self.calls if n == name]


# ------------------------------------------------------------------ kagsim_cuda.GpuBatch


KAG_FUNCS = [
    "kag_last_error",
    "kag_act_i32",
    "kag_ser_bytes",
    "kag_sim_bytes",
    "kag_amps_selfcheck",
    "kag_create",
    "kag_destroy",
    "kag_n",
    "kag_sims_dev_ptr",
    "kag_reset_dev",
    "kag_reset_host",
    "kag_step_dev",
    "kag_step_host",
    "kag_rewards_dev",
    "kag_rewards_host",
    "kag_done_dev",
    "kag_done_host",
    "kag_serialize_host",
    "kag_copy_sims_host",
    "kag_upload_actions",
    "kag_run_uploaded",
    "kag_price_check",
    "kag_sync",
]


def _kag_lib(act_i32=170, create=1234):
    state = {"rc": 0, "n": 0}

    def create_fn(n, steps, dev):
        state["n"] = n
        return create

    def rewards(h, p):
        _arr(p, ctypes.c_double, 2 * state["n"])[:] = np.arange(2 * state["n"]) + 0.5
        return 0

    def done(h, p):
        _arr(p, ctypes.c_uint8, state["n"])[:] = 1
        return 0

    def serialize(h, first, count, p):
        _arr(p, ctypes.c_uint8, count * 7645)[:] = first
        return 0

    impls = {name: (lambda *a: 0) for name in KAG_FUNCS}
    impls.update(
        kag_act_i32=lambda: act_i32,
        kag_ser_bytes=lambda: 7645,
        kag_last_error=lambda: b"bad thing",
        kag_create=create_fn,
        kag_step_host=lambda h, a, m: state["rc"],
        kag_rewards_host=rewards,
        kag_done_host=done,
        kag_serialize_host=serialize,
    )
    lib = FakeLib(impls)
    lib.state = state
    return lib


@pytest.fixture
def kag(monkeypatch):
    libs = {}

    def cdll(path):
        libs[path] = libs.get(path) or _kag_lib(act_i32=171 if "bad" in path else 170)
        return libs[path]

    monkeypatch.setattr(kc, "_LIB", {})
    monkeypatch.setattr(ctypes, "CDLL", cdll)
    return libs


def test_load_lib(kag):
    lib = kc.load_lib("x.so")
    assert kc.load_lib("x.so") is lib and len(kag) == 1  # loaded once per path
    assert lib.kag_create.argtypes == [ctypes.c_int] * 3 and lib.kag_create.restype is ctypes.c_void_p
    assert lib.kag_destroy.restype is None
    assert kc.load_lib() is kc.load_lib(str(Path(kc.HERE) / "build" / "libkagsim_cuda.so"))
    with pytest.raises(RuntimeError, match="not 170"):
        kc.load_lib("bad.so")
    assert "bad.so" not in kc._LIB


def test_gpu_batch_host_path(kag):
    b = kc.GpuBatch(3, steps=50, device=1, lib_path="x.so")
    lib = kag["x.so"]
    assert lib.called("kag_create") == [(3, 50, 1)] and b.ser_bytes == 7645

    b.reset([7, 8, 9])
    h, s, m, sh, ns = lib.called("kag_reset_host")[-1]
    assert h == 1234 and (m, sh, ns) == (None, None, None)
    assert _arr(s, ctypes.c_uint64, 3).tolist() == [7, 8, 9]
    b.reset(np.zeros(3), mask=[1, 0, 1], shops=np.ones((3, 8)), nshops=[1, 2, 8])
    _, _, m, sh, ns = lib.called("kag_reset_host")[-1]
    assert _arr(m, ctypes.c_uint8, 3).tolist() == [1, 0, 1] and _arr(ns, ctypes.c_int32, 3).tolist() == [1, 2, 8]
    with pytest.raises(ValueError, match="together"):
        b.reset(np.zeros(3), shops=np.ones((3, 8)))
    with pytest.raises(ValueError, match="seeds: shape"):
        b.reset(np.zeros(4))

    b.step(np.zeros((1, 2, 170)))
    assert lib.called("kag_step_host")[-1][2] == 1
    with pytest.raises(ValueError, match=r"\(m,2,170\)"):
        b.step(np.zeros((1, 2, 169)))
    lib.state["rc"] = 5
    with pytest.raises(RuntimeError, match="step_host: bad thing"):
        b.step(np.zeros((3, 2, 170)))

    assert b.rewards().tolist() == [[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]]
    assert b.done().tolist() == [1, 1, 1]
    ser = b.serialize(first=1)
    assert ser.shape == (2, 7645) and (ser == 1).all()
    with pytest.raises(ValueError, match="acts_T"):
        b.upload(np.zeros((4, 1, 2, 170))[0])
    b.close()
    b.close()
    assert lib.called("kag_destroy") == [(1234,)]


def test_gpu_batch_create_failure(kag, monkeypatch):
    lib = kc.load_lib("x.so")
    monkeypatch.setattr(lib.kag_create, "impl", lambda n, s, d: 0)
    with pytest.raises(RuntimeError, match="kag_create: bad thing"):
        kc.GpuBatch(2, lib_path="x.so")


class _FakeKaghdBatch:
    def __init__(self, n, steps):
        self.n, self.steps, self.log = n, steps, []

    def reset(self, *a):
        self.log.append(("reset", a))

    def step(self, a):
        self.log.append(("step", a.dtype, a.flags.c_contiguous))

    def serialize(self):
        return np.arange(self.n)[:, None] * np.ones((1, 3), np.uint8)

    def run(self, acts):
        return float(len(acts))


def test_cpu_batch(monkeypatch):
    fake = types.SimpleNamespace(SER_BYTES=3, Batch=_FakeKaghdBatch, RefBatch=type("Ref", (_FakeKaghdBatch,), {}))
    monkeypatch.setitem(sys.modules, "kaghd_cpu", fake)
    monkeypatch.setattr(sys, "path", list(sys.path))  # _cpu_module() prepends build/
    b = kc.CpuBatch(4, steps=9)
    assert type(b.b) is _FakeKaghdBatch and b.b.steps == 9 and b.ser_bytes == 3
    assert type(kc.CpuBatch(1, ref=True).b).__name__ == "Ref"
    b.reset([1, 2, 3, 4])
    assert b.b.log[-1][1][0].dtype == np.uint64
    b.step(np.zeros((2, 2, 170), np.int64)[:, :, ::-1])
    assert b.b.log[-1] == ("step", np.int32, True)
    assert b.serialize(1, 2)[:, 0].tolist() == [1, 2]
    b.upload(np.zeros((10, 1, 2, 170)))
    assert b.run_uploaded(2) == 8.0 and b.run_uploaded(2, 5) == 3.0


# ------------------------------------------------------------------ kagrl.RLBatch (CPU backend)

DIMS = {
    "NT": 100,
    "FT": 30,
    "MAXU": 16,
    "FU": 16,
    "FG": 77,
    "TMASK_BYTES": 700,
    "MB": 489,
    "NHEADS": 24,
    "NOPS": 56,
    "ACT_I32": 170,
    "SER_BYTES": 7645,
    "SIM_BYTES": 1,
    "MAX_ORDERS": 10,
    "EMULATED": 1,
    "API_VERSION": 1,
    "_": 0,
}


def _rl_lib(emulated=1):
    def dims(d):
        for i, k in enumerate(kr.DIM_NAMES):
            d[i] = DIMS[k] if k != "EMULATED" else emulated
        return 0

    def rewards(h, p, s):
        _arr(p, ctypes.c_double, 4)[:] = [1.0, 2.0, 3.0, 4.0]
        return 0

    def kernel_info(a):
        for i in range(24):
            a[i] = i
        return 6

    impls = {
        "kagrl_dims": dims,
        "kagrl_last_error": lambda: b"rl error",
        "kagrl_create": lambda n, steps, dev: 99,
        "kagrl_rewards_dev": rewards,
        "kagrl_kernel_info": kernel_info,
        "kagrl_set_block": lambda t: 0 if t % 32 == 0 else 1,
    }
    sig_names = (
        "kagrl_destroy kagrl_n kagrl_sims_ptr kagrl_reset_dev kagrl_encode_dev kagrl_act_step_dev kagrl_execute_dev "
        "kagrl_step_raw_dev kagrl_random_choices_dev kagrl_done_dev kagrl_steps_dev kagrl_serialize_host "
        "kagrl_copy_sims_host kagrl_stream_sync"
    )
    for name in sig_names.split():
        impls[name] = lambda *a: 0
    return FakeLib(impls)


@pytest.fixture
def rl(monkeypatch):
    monkeypatch.setattr(kr, "_LIBS", {})
    monkeypatch.setattr(ctypes, "CDLL", lambda path: _rl_lib(emulated=0 if "gpu" in path else 1))


def test_rl_load_lib_and_backend_checks(rl):
    lib = kr.load_lib("cpu", "cpu.so")
    assert lib.dims == DIMS and kr.load_lib("cpu", "cpu.so") is lib
    assert lib.kagrl_random_choices_dev.argtypes[3] is ctypes.c_uint64
    with pytest.raises(ValueError, match="backend"):
        kr.RLBatch(2, backend="tpu")
    with pytest.raises(RuntimeError, match="does not match backend=cpu"):
        kr.RLBatch(2, backend="cpu", lib_path="gpu.so")
    assert kr.kernel_info(lib)["k_execute"] == {"regs": 8, "local_bytes": 9, "max_threads": 10, "shared_bytes": 11}
    kr.set_block(lib, 64)
    with pytest.raises(RuntimeError, match="rl error"):
        kr.set_block(lib, 33)


def test_rl_batch_cpu(rl):
    b = kr.RLBatch(2, steps=30, lib_path="cpu.so")
    lib = b.lib
    assert lib.called("kagrl_create") == [(2, 30, 0)]
    obs = b.alloc_obs()
    assert obs["tiles"].shape == (4, 100, 30) and obs["tmask"].dtype == np.uint8
    assert obs["unit_pos"].dtype == np.int64 and obs["mmask"].shape == (4, 489)

    b.reset(np.array([5, 6], np.uint64), shops=np.zeros((2, 8)), nshops=[0, 8])
    h, ps, pm, psh, pns, stream = lib.called("kagrl_reset_dev")[-1]
    assert h == 99 and pm is None and stream is None
    assert _arr(ps, ctypes.c_uint64, 2).tolist() == [5, 6] and _arr(pns, ctypes.c_int32, 2).tolist() == [0, 8]
    with pytest.raises(ValueError, match="nshops"):
        b.reset([1, 2], shops=np.zeros((2, 8)), nshops=[0, 9])
    with pytest.raises(ValueError, match="shop ids"):
        b.reset([1, 2], shops=np.full((2, 8), 8), nshops=[1, 1])
    with pytest.raises(ValueError, match="seeds"):
        b.reset([1, 2, 3])

    b.encode(obs)
    assert lib.called("kagrl_encode_dev")[-1][1] == 4  # rows
    with pytest.raises(ValueError, match="games requires seats"):
        b.encode(obs, games=[0, 1])
    with pytest.raises(ValueError, match="seats: values outside 0..1"):
        b.encode(b.alloc_obs(rows=2), seats=[0, 2])
    obs["glob"] = obs["glob"][:, ::2]
    with pytest.raises(ValueError, match="glob"):
        b.encode(obs)

    b.act_step(np.zeros((2, 2, 16), np.int32), np.zeros((2, 2, 24), np.int32))
    assert lib.called("kagrl_act_step_dev")[-1][3] == 4  # itemsize of the choices
    b.act_step([[[0] * 16] * 2] * 2, [[[0] * 24] * 2] * 2)  # lists are converted to int64
    assert lib.called("kagrl_act_step_dev")[-1][3] == 8
    with pytest.raises(ValueError, match="same dtype"):
        b.act_step(np.zeros((2, 2, 16), np.int32), np.zeros((2, 2, 24), np.int64))

    assert b.rewards().tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert b.serialize().shape == (2, 7645)
    b.close()
    assert lib.called("kagrl_destroy") == [(99,)]
