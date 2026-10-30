"""Python wrapper of the Kaggriculture batch engine.

GpuBatch: ctypes over build/libkagsim_cuda.so (batch.cu, C API from batch.cuh).
CpuBatch: the same interface over the CPU module build/kaghd_cpu (cpu_batch.cpp):
the same sim_hd.hpp core code, a loop over games. Needed for comparing GPU with CPU and
for a "dry" run of scripts on a machine without a GPU.

Buffers: numpy (host, synchronous) or torch CUDA tensors (device, asynchronous on the
current torch stream). Formats: batch.cuh / batch_common.hpp:
    seeds uint64 (n,), mask uint8 (n,), shops uint8 (n,8) + nshops int32 (n,),
    acts int32 (m,2,170): game i takes row i % m,
    rewards float64 (n,2), done uint8 (n,).
Encoding of agent action dicts into int32[170]: kaghd_cpu.encode_action(s).

    from kagsim_cuda import CpuBatch            # or GpuBatch (needs build/libkagsim_cuda.so and a GPU)
    b = CpuBatch(4); b.reset(np.arange(4, dtype=np.uint64)); b.step(np.zeros((1, 2, 170), np.int32))

Build: scripts/build_engines.sh. Importing this module needs only numpy; the libraries are loaded on first use.
"""

import ctypes
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ACT_I32 = 170


def _is_torch_cuda(x):
    """True for a torch CUDA tensor (checked by duck typing, torch is not imported)."""
    return hasattr(x, "is_cuda") and hasattr(x, "data_ptr") and bool(x.is_cuda)


def _np(x, dtype, shape=None, name="array"):
    """C-contiguous numpy array of dtype; ValueError if shape is given and differs."""
    a = np.ascontiguousarray(x, dtype=dtype)
    if shape is not None and a.shape != tuple(shape):
        raise ValueError(f"{name}: shape {a.shape}, expected {tuple(shape)}")
    return a


def _ptr(a):
    return ctypes.c_void_p(a.ctypes.data) if a is not None else None


_LIB = {}


def load_lib(path=None):
    """Load the C library (default build/libkagsim_cuda.so) once per path and declare the C signatures.

    RuntimeError if the library was built with an action layout other than int32[170]."""
    path = path or os.path.join(HERE, "build", "libkagsim_cuda.so")
    if path in _LIB:
        return _LIB[path]
    lib = ctypes.CDLL(path)
    P, I, V = ctypes.c_void_p, ctypes.c_int, None  # noqa: E741  short aliases for the signature table below
    sig = {
        "kag_last_error": ([], ctypes.c_char_p),
        "kag_act_i32": ([], I),
        "kag_ser_bytes": ([], I),
        "kag_sim_bytes": ([], I),
        "kag_amps_selfcheck": ([], I),
        "kag_create": ([I, I, I], P),
        "kag_destroy": ([P], V),
        "kag_n": ([P], I),
        "kag_sims_dev_ptr": ([P], P),
        "kag_reset_dev": ([P, P, P, P, P, P], I),
        "kag_reset_host": ([P, P, P, P, P], I),
        "kag_step_dev": ([P, P, I, P], I),
        "kag_step_host": ([P, P, I], I),
        "kag_rewards_dev": ([P, P, P], I),
        "kag_rewards_host": ([P, P], I),
        "kag_done_dev": ([P, P, P], I),
        "kag_done_host": ([P, P], I),
        "kag_serialize_host": ([P, I, I, P], I),
        "kag_copy_sims_host": ([P, I, I, P], I),
        "kag_upload_actions": ([P, P, I, I], I),
        "kag_run_uploaded": ([P, I, I, ctypes.POINTER(ctypes.c_float)], I),
        "kag_price_check": ([I, I, ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(I), ctypes.POINTER(I)], I),
        "kag_sync": ([], I),
    }
    for name, (args, res) in sig.items():
        f = getattr(lib, name)
        f.argtypes = args
        f.restype = res
    if lib.kag_act_i32() != ACT_I32:
        raise RuntimeError("ACT_I32 in the library is not 170")
    _LIB[path] = lib
    return lib


def price_check(lo=-20000, hi=80000, lib_path=None):
    """market_price on the GPU vs the host for all 9 items and inv in [lo, hi]."""
    lib = load_lib(lib_path)
    bad, fi, fv = ctypes.c_longlong(0), ctypes.c_int(0), ctypes.c_int(0)
    rc = lib.kag_price_check(lo, hi, ctypes.byref(bad), ctypes.byref(fi), ctypes.byref(fv))
    if rc:
        raise RuntimeError(lib.kag_last_error().decode())
    return bad.value, fi.value, fv.value


class GpuBatch:
    """n games on the GPU (libkagsim_cuda.so). Host numpy calls are synchronous; torch CUDA tensors run on the current
    torch stream. Errors of the C API are raised as RuntimeError with the library's last error message."""

    backend = "gpu"

    def __init__(self, n, steps=720, device=0, lib_path=None):
        self.lib = load_lib(lib_path)
        self.n = int(n)
        self.h = self.lib.kag_create(self.n, int(steps), int(device))
        if not self.h:
            raise RuntimeError("kag_create: " + self.lib.kag_last_error().decode())
        self.ser_bytes = self.lib.kag_ser_bytes()

    def _ck(self, rc, what):
        if rc != 0:
            raise RuntimeError(f"{what}: {self.lib.kag_last_error().decode()}")

    @staticmethod
    def _stream():
        import torch

        return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    def reset(self, seeds, mask=None, shops=None, nshops=None):
        """Reset games: seeds (n,); mask (n,) resets only games with mask != 0; shops (n,8) + nshops (n,) pin the first
        nshops shops of each game."""
        if (shops is None) != (nshops is None):
            raise ValueError("shops and nshops must be given together")
        if _is_torch_cuda(seeds):
            p = lambda t: None if t is None else ctypes.c_void_p(t.data_ptr())  # noqa: E731
            self._ck(
                self.lib.kag_reset_dev(self.h, p(seeds), p(mask), p(shops), p(nshops), self._stream()), "reset_dev"
            )
            return
        n = self.n
        s = _np(seeds, np.uint64, (n,), "seeds")
        m = None if mask is None else _np(mask, np.uint8, (n,), "mask")
        sh = None if shops is None else _np(shops, np.uint8, (n, 8), "shops")
        ns = None if nshops is None else _np(nshops, np.int32, (n,), "nshops")
        self._ck(self.lib.kag_reset_host(self.h, _ptr(s), _ptr(m), _ptr(sh), _ptr(ns)), "reset_host")

    def step(self, acts):
        """One step of all games; acts int32 (m,2,170), game i takes row i % m."""
        if _is_torch_cuda(acts):
            if acts.dim() != 3 or acts.shape[1] != 2 or acts.shape[2] != ACT_I32 or not acts.is_contiguous():
                raise ValueError("acts: contiguous int32 (m,2,170)")
            self._ck(
                self.lib.kag_step_dev(self.h, ctypes.c_void_p(acts.data_ptr()), int(acts.shape[0]), self._stream()),
                "step_dev",
            )
            return
        a = np.ascontiguousarray(acts, dtype=np.int32)
        if a.ndim != 3 or a.shape[1:] != (2, ACT_I32):
            raise ValueError("acts: int32 (m,2,170)")
        self._ck(self.lib.kag_step_host(self.h, _ptr(a), a.shape[0]), "step_host")

    def rewards(self, out=None):
        if out is not None and _is_torch_cuda(out):
            self._ck(self.lib.kag_rewards_dev(self.h, ctypes.c_void_p(out.data_ptr()), self._stream()), "rewards_dev")
            return out
        r = np.empty((self.n, 2), np.float64)
        self._ck(self.lib.kag_rewards_host(self.h, _ptr(r)), "rewards_host")
        return r

    def done(self, out=None):
        if out is not None and _is_torch_cuda(out):
            self._ck(self.lib.kag_done_dev(self.h, ctypes.c_void_p(out.data_ptr()), self._stream()), "done_dev")
            return out
        d = np.empty(self.n, np.uint8)
        self._ck(self.lib.kag_done_host(self.h, _ptr(d)), "done_host")
        return d

    def serialize(self, first=0, count=None):
        """uint8 (count, ser_bytes): canonical serialization (state_visit.hpp) of games first..first+count-1."""
        count = self.n - first if count is None else count
        out = np.empty((count, self.ser_bytes), np.uint8)
        self._ck(self.lib.kag_serialize_host(self.h, first, count, _ptr(out)), "serialize")
        return out

    def upload(self, acts_T):
        """Copy a whole action sequence int32 (T,m,2,170) to the device for run_uploaded."""
        a = np.ascontiguousarray(acts_T, dtype=np.int32)
        if a.ndim != 4 or a.shape[2:] != (2, ACT_I32):
            raise ValueError("acts_T: int32 (T,m,2,170)")
        self._ck(self.lib.kag_upload_actions(self.h, _ptr(a), a.shape[0], a.shape[1]), "upload")
        self._up_T = a.shape[0]

    def run_uploaded(self, t0=0, t1=None):
        """Steps t0..t1-1 over the uploaded actions; returns seconds (cudaEvent)."""
        t1 = self._up_T if t1 is None else t1
        ms = ctypes.c_float(0)
        self._ck(self.lib.kag_run_uploaded(self.h, t0, t1, ctypes.byref(ms)), "run_uploaded")
        return ms.value / 1000.0

    def close(self):
        if getattr(self, "h", None):
            self.lib.kag_destroy(self.h)
            self.h = None

    def __del__(self):
        try:  # noqa: SIM105  plain try/except: __del__ may run at interpreter shutdown, when modules are torn down
            self.close()
        except Exception:
            pass


def _cpu_module():
    b = os.path.join(HERE, "build")
    if b not in sys.path:
        sys.path.insert(0, b)
    import kaghd_cpu

    return kaghd_cpu


class CpuBatch:
    """The same interface as GpuBatch, over the CPU build of the same core (kaghd_cpu.Batch)."""

    backend = "cpu"

    def __init__(self, n, steps=720, ref=False):
        K = _cpu_module()
        self.K = K
        self.n = int(n)
        self.b = (K.RefBatch if ref else K.Batch)(self.n, int(steps))
        self.ser_bytes = K.SER_BYTES

    def reset(self, seeds, mask=None, shops=None, nshops=None):
        self.b.reset(np.asarray(seeds, np.uint64), mask, shops, nshops)

    def step(self, acts):
        self.b.step(np.ascontiguousarray(acts, dtype=np.int32))

    def rewards(self, out=None):
        return self.b.rewards()

    def done(self, out=None):
        return self.b.done()

    def serialize(self, first=0, count=None):
        s = self.b.serialize()
        count = self.n - first if count is None else count
        return s[first : first + count]

    def upload(self, acts_T):
        self._up = np.ascontiguousarray(acts_T, dtype=np.int32)

    def run_uploaded(self, t0=0, t1=None):
        """Steps t0..t1-1 over the uploaded actions; returns seconds of engine work."""
        t1 = self._up.shape[0] if t1 is None else t1
        return self.b.run(self._up[t0:t1])

    def close(self):
        pass
