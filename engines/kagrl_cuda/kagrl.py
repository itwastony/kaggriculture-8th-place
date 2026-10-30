"""Python wrapper (ctypes) for the batched Kaggriculture RL library (kagrl.cu, C API kagrl.cuh).

    from kagrl import RLBatch
    rb = RLBatch(n, backend="gpu")                 # build/libkagrl_cuda.so, buffers - torch CUDA tensors
    rb = RLBatch(n, backend="cpu")                 # build/libkagrl_cpu.so (same kagrl.cu, emulated), numpy/torch CPU
    obs = rb.alloc_obs(kind="torch", device="cuda")  # 7 observation buffers for 2n rows (both seats)
    rb.reset(seeds)
    rb.encode(obs)                                 # row 2*g+s = seat s of game g
    rb.act_step(uc, mc)                            # uc [n,2,16] / [2n,16], mc [n,2,24] / [2n,24]; int32 or int64
    bank = rb.rewards(out)                         # [n,2] float64

The interface is the same for both builds. GPU: all operations are asynchronous on the current torch stream
(torch.cuda.current_stream()), the host does not wait, except serialize()/rewards() without out. CPU: synchronous.
Buffers are checked (dtype, number of elements, contiguity, device) without synchronization; index values
(games/seats/shops) are checked only for host arrays.

Build: engines/kagrl_cuda/build_cpu.sh (CPU emulation) or build_cuda.sh (GPU), see engines/kagrl_cuda/README.md.
Importing this module needs only numpy; a library is loaded when the first RLBatch of its backend is created.
"""

import ctypes
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_PATHS = {
    "cpu": os.path.join(HERE, "build", "libkagrl_cpu.so"),
    "gpu": os.path.join(HERE, "build", "libkagrl_cuda.so"),
}
DIM_NAMES = (
    "NT",
    "FT",
    "MAXU",
    "FU",
    "FG",
    "TMASK_BYTES",
    "MB",
    "NHEADS",
    "NOPS",
    "ACT_I32",
    "SER_BYTES",
    "SIM_BYTES",
    "MAX_ORDERS",
    "EMULATED",
    "API_VERSION",
    "_",
)
OBS_KEYS = ("tiles", "units", "unit_pos", "unit_mask", "glob", "tmask", "mmask")

_LIBS = {}


def load_lib(backend="cpu", path=None):
    """Load the library of backend (or path) once per path, declare the C signatures and read lib.dims (kagrl_dims)."""
    path = path or LIB_PATHS[backend]
    if path in _LIBS:
        return _LIBS[path]
    lib = ctypes.CDLL(path)
    P, I, V, U64 = ctypes.c_void_p, ctypes.c_int, None, ctypes.c_uint64  # noqa: E741  aliases for the table below
    sig = {
        "kagrl_dims": ([P], I),
        "kagrl_last_error": ([], ctypes.c_char_p),
        "kagrl_create": ([I, I, I], P),
        "kagrl_destroy": ([P], V),
        "kagrl_n": ([P], I),
        "kagrl_sims_ptr": ([P], P),
        "kagrl_reset_dev": ([P, P, P, P, P, P], I),
        "kagrl_encode_dev": ([P, I, P, P, P, P, P, P, P, P, P, P], I),
        "kagrl_act_step_dev": ([P, P, P, I, P, P, P, P], I),
        "kagrl_execute_dev": ([P, P, P, I, P, P], I),
        "kagrl_step_raw_dev": ([P, P, I, P], I),
        "kagrl_random_choices_dev": ([P, P, P, U64, U64, P, P, I, P], I),
        "kagrl_rewards_dev": ([P, P, P], I),
        "kagrl_done_dev": ([P, P, P], I),
        "kagrl_steps_dev": ([P, P, P], I),
        "kagrl_serialize_host": ([P, I, I, P, P], I),
        "kagrl_copy_sims_host": ([P, I, I, P, P], I),
        "kagrl_stream_sync": ([P], I),
        "kagrl_set_block": ([I], I),
        "kagrl_kernel_info": ([P], I),
    }
    for name, (args, res) in sig.items():
        f = getattr(lib, name)
        f.argtypes = args
        f.restype = res
    d = (ctypes.c_int32 * 16)()
    lib.kagrl_dims(d)
    lib.dims = dict(zip(DIM_NAMES, list(d)))
    _LIBS[path] = lib
    return lib


KERNELS = ("k_encode", "k_act_step", "k_execute", "k_random", "k_step_raw", "k_reset")


def kernel_info(lib):
    """Kernel attributes of the GPU build: {kernel: {regs, local_bytes, max_threads, shared_bytes}} (zeros in the CPU
    build)."""
    a = (ctypes.c_int32 * 24)()
    if lib.kagrl_kernel_info(a) != 6:
        raise RuntimeError(lib.kagrl_last_error().decode())
    return {
        k: {"regs": a[4 * i], "local_bytes": a[4 * i + 1], "max_threads": a[4 * i + 2], "shared_bytes": a[4 * i + 3]}
        for i, k in enumerate(KERNELS)
    }


def set_block(lib, threads):
    """Set the threads per CUDA block of all kernels (32..1024, a multiple of 32); one thread handles one row/game."""
    if lib.kagrl_set_block(int(threads)) != 0:
        raise RuntimeError(lib.kagrl_last_error().decode())


def _is_torch(x):
    return hasattr(x, "data_ptr") and hasattr(x, "is_cuda")


_NP2T = None


def _torch_dtype(np_dtype):
    global _NP2T
    import torch

    if _NP2T is None:
        _NP2T = {
            np.dtype(np.float32): torch.float32,
            np.dtype(np.float64): torch.float64,
            np.dtype(np.int64): torch.int64,
            np.dtype(np.int32): torch.int32,
            np.dtype(np.uint8): torch.uint8,
            np.dtype(np.bool_): torch.bool,
            np.dtype(np.uint64): getattr(torch, "uint64", torch.int64),
        }
    return _NP2T[np.dtype(np_dtype)]


class RLBatch:
    """n games of the batched RL engine: encode observations, execute policy choices, step, read banks and states."""

    def __init__(self, n, steps=720, backend="cpu", device=0, lib_path=None):
        if backend not in ("cpu", "gpu"):
            raise ValueError("backend: cpu | gpu")
        self.backend = backend
        self.lib = load_lib(backend, lib_path)
        self.dims = self.lib.dims
        if (self.dims["EMULATED"] == 1) != (backend == "cpu"):
            raise RuntimeError(f"library {lib_path} does not match backend={backend}")
        self.n = int(n)
        self.device = int(device)
        self.h = self.lib.kagrl_create(self.n, int(steps), self.device)
        if not self.h:
            raise RuntimeError("kagrl_create: " + self.lib.kagrl_last_error().decode())
        d = self.dims
        self.obs_spec = {
            "tiles": (np.float32, (d["NT"], d["FT"])),
            "units": (np.float32, (d["MAXU"], d["FU"])),
            "unit_pos": (np.int64, (d["MAXU"],)),
            "unit_mask": (np.bool_, (d["MAXU"],)),
            "glob": (np.float32, (d["FG"],)),
            "tmask": (np.uint8, (d["MAXU"], d["TMASK_BYTES"])),
            "mmask": (np.bool_, (d["MB"],)),
        }

    # ------------------------------------------------------------ internals
    def _ck(self, rc, what):
        if rc != 0:
            raise RuntimeError(f"{what}: {self.lib.kagrl_last_error().decode()}")

    def _stream(self):
        if self.backend == "cpu":
            return None
        import torch

        return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    def _buf(self, x, dtypes, numel, name, out=False):
        """Buffer pointer: dtype from dtypes (numpy types), exactly numel elements, C-contiguous, on the right device.
        Returns (ctypes.c_void_p, itemsize), or (None, 0) for None."""
        if x is None:
            return None, 0
        if _is_torch(x):
            if self.backend == "gpu":
                if not x.is_cuda or x.device.index != self.device:
                    raise ValueError(f"{name}: a CUDA tensor on cuda:{self.device} is required")
            elif x.is_cuda:
                raise ValueError(f"{name}: the CPU build accepts only host buffers")
            if x.dtype not in [_torch_dtype(t) for t in dtypes]:
                raise ValueError(f"{name}: dtype {x.dtype}, expected {[np.dtype(t).name for t in dtypes]}")
            if not x.is_contiguous() or x.numel() != numel:
                raise ValueError(
                    f"{name}: a contiguous tensor of {numel} elements is required (shape {tuple(x.shape)})"
                )
            return ctypes.c_void_p(x.data_ptr()), x.element_size()
        if self.backend == "gpu":
            raise ValueError(f"{name}: the GPU build accepts only torch CUDA tensors")
        if not isinstance(x, np.ndarray) or x.dtype not in [np.dtype(t) for t in dtypes]:
            raise ValueError(f"{name}: a numpy array of dtype {[np.dtype(t).name for t in dtypes]} is required")
        if not x.flags.c_contiguous or x.size != numel or (out and not x.flags.writeable):
            raise ValueError(
                f"{name}: a C-contiguous{' writable' if out else ''} array of {numel} elements is required"
            )
        return ctypes.c_void_p(x.ctypes.data), x.itemsize

    def _host_in(self, x, dtype):
        """Input array: for the CPU build - numpy/torch CPU as is (dtype conversion - by copy); for the GPU -
        torch CUDA as is, a host array is copied to the device (the temporary tensor lives until the end of the call;
        freeing in the caching allocator is ordered on the current stream - safe)."""
        if x is None:
            return None
        if _is_torch(x):
            if self.backend == "gpu" and not x.is_cuda:
                x = x.to(f"cuda:{self.device}")
            return x.contiguous()
        a = np.ascontiguousarray(x, dtype=dtype)
        if self.backend == "gpu":
            import torch

            if a.dtype == np.uint64:
                a = a.view(np.int64)
            return torch.from_numpy(a).to(f"cuda:{self.device}")
        return a

    # ------------------------------------------------------------ API
    def alloc_obs(self, rows=None, kind="numpy", device=None):
        """Observation buffers for rows rows (default 2n - both seats): numpy or torch (device)."""
        rows = 2 * self.n if rows is None else int(rows)
        out = {}
        for k, (dt, sh) in self.obs_spec.items():
            if kind == "numpy":
                out[k] = np.zeros((rows,) + sh, dt)
            else:
                import torch

                dev = device or (f"cuda:{self.device}" if self.backend == "gpu" else "cpu")
                out[k] = torch.zeros((rows,) + sh, dtype=_torch_dtype(dt), device=dev)
        return out

    def reset(self, seeds, mask=None, shops=None, nshops=None):
        """Reset games: seeds (n,); mask (n,) resets only games with mask != 0; shops (n,8) ids 0..7 + nshops (n,)
        0..8 pin the first nshops shops of each game."""
        if (shops is None) != (nshops is None):
            raise ValueError("shops and nshops must be given together")
        n = self.n
        s = self._host_in(seeds, np.uint64)
        m = self._host_in(mask, np.uint8)
        sh = self._host_in(shops, np.uint8)
        ns = self._host_in(nshops, np.int32)
        if isinstance(ns, np.ndarray):
            if ns.size != n or (ns < 0).any() or (ns > 8).any():
                raise ValueError("nshops: n values in 0..8")
            if (sh.reshape(n, 8) >= 8).any():
                raise ValueError("shops: shop ids 0..7")
        ps, _ = self._buf(s, (np.uint64, np.int64), n, "seeds")
        pm, _ = self._buf(m, (np.uint8, np.bool_), n, "mask")
        psh, _ = self._buf(sh, (np.uint8,), 8 * n, "shops")
        pns, _ = self._buf(ns, (np.int32,), n, "nshops")
        self._ck(self.lib.kagrl_reset_dev(self.h, ps, pm, psh, pns, self._stream()), "reset")

    def encode(self, obs, games=None, seats=None):
        """obs - dict of 7 buffers (alloc_obs). Rows: games=seats=None - 2n rows, r = 2*g+s; seats without games -
        n rows, seat seats[g]; otherwise row r = (games[r], seats[r])."""
        if games is not None and seats is None:
            raise ValueError("games requires seats")
        rows = obs["tiles"].shape[0]
        g = self._host_in(games, np.int32)
        s = self._host_in(seats, np.int32)
        for a, nm, hi in ((g, "games", self.n), (s, "seats", 2)):
            if isinstance(a, np.ndarray) and a.size and (a.min() < 0 or a.max() >= hi):
                raise ValueError(f"{nm}: values outside 0..{hi - 1}")
        pg, _ = self._buf(g, (np.int32,), rows, "games")
        pst, _ = self._buf(s, (np.int32,), rows, "seats")
        p = {
            k: self._buf(obs[k], (dt,), rows * int(np.prod(sh)), k, out=True)[0]
            for k, (dt, sh) in self.obs_spec.items()
        }
        self._ck(
            self.lib.kagrl_encode_dev(
                self.h,
                rows,
                pg,
                pst,
                p["tiles"],
                p["units"],
                p["unit_pos"],
                p["unit_mask"],
                p["glob"],
                p["tmask"],
                p["mmask"],
                self._stream(),
            ),
            "encode",
        )

    def act_step(self, uc, mc, raw=None, raw_mask=None, acts_out=None):
        """Execute the choice of both seats + step all unfinished games. uc [n,2,16], mc [n,2,24] (int32 or int64);
        raw int32 [n,2,170] + raw_mask [n,2] - ready actions for the marked seats; acts_out int32 [n,2,170]."""
        n = self.n
        if self.backend == "cpu" and not _is_torch(uc) and not isinstance(uc, np.ndarray):
            uc = np.asarray(uc, np.int64)
        if self.backend == "cpu" and not _is_torch(mc) and not isinstance(mc, np.ndarray):
            mc = np.asarray(mc, np.int64)
        pu, bu = self._buf(uc, (np.int32, np.int64), n * 2 * self.dims["MAXU"], "uc")
        pm, bm = self._buf(mc, (np.int32, np.int64), n * 2 * self.dims["NHEADS"], "mc")
        if bu != bm:
            raise ValueError("uc and mc must have the same dtype")
        r = self._host_in(raw, np.int32)
        rm = self._host_in(raw_mask, np.uint8)
        pr, _ = self._buf(r, (np.int32,), n * 2 * self.dims["ACT_I32"], "raw")
        prm, _ = self._buf(rm, (np.uint8, np.bool_), n * 2, "raw_mask")
        pa, _ = self._buf(acts_out, (np.int32,), n * 2 * self.dims["ACT_I32"], "acts_out", out=True)
        self._ck(self.lib.kagrl_act_step_dev(self.h, pu, pm, bu, pr, prm, pa, self._stream()), "act_step")

    def execute(self, uc, mc, acts_out):
        """Engine actions of both seats only (no step) -> acts_out int32 [n,2,170]; finished games - zeros."""
        n = self.n
        pu, bu = self._buf(uc, (np.int32, np.int64), n * 2 * self.dims["MAXU"], "uc")
        pm, bm = self._buf(mc, (np.int32, np.int64), n * 2 * self.dims["NHEADS"], "mc")
        if bu != bm:
            raise ValueError("uc and mc must have the same dtype")
        pa, _ = self._buf(acts_out, (np.int32,), n * 2 * self.dims["ACT_I32"], "acts_out", out=True)
        self._ck(self.lib.kagrl_execute_dev(self.h, pu, pm, bu, pa, self._stream()), "execute")

    def step_raw(self, acts):
        """Step with ready engine actions int32 [m,2,170] (game i - row i % m)."""
        a = self._host_in(acts, np.int32)
        m = a.shape[0]
        p, _ = self._buf(a, (np.int32,), m * 2 * self.dims["ACT_I32"], "acts")
        self._ck(self.lib.kagrl_step_raw_dev(self.h, p, int(m), self._stream()), "step_raw")

    def random_choices(self, obs, seed, salt, uc, mc):
        """Random legal choice from obs (2n rows) -> uc [n,2,16], mc [n,2,24] (int32 or int64)."""
        n = self.n
        pt, _ = self._buf(obs["tmask"], (np.uint8,), 2 * n * self.dims["MAXU"] * self.dims["TMASK_BYTES"], "tmask")
        pum, _ = self._buf(obs["unit_mask"], (np.bool_,), 2 * n * self.dims["MAXU"], "unit_mask")
        pu, bu = self._buf(uc, (np.int32, np.int64), n * 2 * self.dims["MAXU"], "uc", out=True)
        pm, bm = self._buf(mc, (np.int32, np.int64), n * 2 * self.dims["NHEADS"], "mc", out=True)
        if bu != bm:
            raise ValueError("uc and mc must have the same dtype")
        self._ck(
            self.lib.kagrl_random_choices_dev(
                self.h, pt, pum, int(seed) & (2**64 - 1), int(salt) & (2**64 - 1), pu, pm, bu, self._stream()
            ),
            "random_choices",
        )

    def _small_out(self, fn, out, np_dtype, shape, name):
        if out is None:
            if self.backend == "cpu":
                out = np.empty(shape, np_dtype)
            else:
                import torch

                t = torch.empty(shape, dtype=_torch_dtype(np_dtype), device=f"cuda:{self.device}")
                self._small_out(fn, t, np_dtype, shape, name)
                return t.cpu().numpy()
        p, _ = self._buf(
            out, (np_dtype,) if np_dtype != np.uint8 else (np.uint8, np.bool_), int(np.prod(shape)), name, out=True
        )
        self._ck(fn(self.h, p, self._stream()), name)
        return out

    def rewards(self, out=None):
        """Bank of both seats float64 [n,2]. out=None: result is numpy (on the GPU - with synchronization)."""
        return self._small_out(self.lib.kagrl_rewards_dev, out, np.float64, (self.n, 2), "rewards")

    def done(self, out=None):
        """uint8 [n]: 1 for finished games (out=None: numpy, as in rewards)."""
        return self._small_out(self.lib.kagrl_done_dev, out, np.uint8, (self.n,), "done")

    def steps(self, out=None):
        """int32 [n]: steps played by each game (out=None: numpy, as in rewards)."""
        return self._small_out(self.lib.kagrl_steps_dev, out, np.int32, (self.n,), "steps")

    def serialize(self, first=0, count=None):
        """Full state of games [first, first+count) - uint8 [count, SER_BYTES] (synchronous)."""
        count = self.n - first if count is None else count
        out = np.empty((count, self.dims["SER_BYTES"]), np.uint8)
        self._ck(
            self.lib.kagrl_serialize_host(
                self.h, int(first), int(count), ctypes.c_void_p(out.ctypes.data), self._stream()
            ),
            "serialize",
        )
        return out

    def sync(self):
        """Wait for the current stream (GPU); no-op wait in the CPU build."""
        self._ck(self.lib.kagrl_stream_sync(self._stream()), "sync")

    def close(self):
        if getattr(self, "h", None):
            self.lib.kagrl_destroy(self.h)
            self.h = None

    def __del__(self):
        try:  # noqa: SIM105  plain try/except: __del__ may run at interpreter shutdown, when modules are torn down
            self.close()
        except Exception:  # noqa: BLE001
            pass
