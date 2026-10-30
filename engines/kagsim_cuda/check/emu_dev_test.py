#!/usr/bin/env python3
"""Check of the *_dev C API functions and of the mask/pinned shops on the emulated build
(build/libkagsim_emu.so, where "device memory" = host memory, so numpy pointers
can be passed). Compared byte-for-byte against the CPU module kaghd_cpu.

    bash engines/kagsim_cuda/check/build_emu.sh && bash engines/kagsim_cuda/build_cpu.sh
    PYTHONHASHSEED=0 python engines/kagsim_cuda/record_streams.py   # data/streams.pkl.gz, not shipped
    python engines/kagsim_cuda/check/emu_dev_test.py

Plays the first 24 streams of data/streams.pkl.gz (a fixed path; full-length 719-step streams) in 50 games with
0..8 randomly pinned shops, resets every 3rd game by mask after step 300, and checks two error paths. Exit code 1 on
any mismatch."""

import ctypes
import gzip
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import kagsim_cuda as KC  # noqa: E402  (needs the sys.path insert above)

K = KC._cpu_module()
lib = KC.load_lib(os.path.join(HERE, "build", "libkagsim_emu.so"))
recs = pickle.load(gzip.open(os.path.join(HERE, "data", "streams.pkl.gz")))[:24]
n, m = 50, len(recs)  # m < n: game i plays stream i % m
T = 719
acts = np.zeros((T, m, 2, K.ACT_I32), np.int32)
for g, r in enumerate(recs):
    for p in (0, 1):
        acts[:, g, p] = K.encode_actions(r["acts"][p])
seeds = np.arange(1000, 1000 + n, dtype=np.uint64) * 7919
shops = np.random.RandomState(0).randint(0, 8, size=(n, 8)).astype(np.uint8)
nshops = np.random.RandomState(1).randint(0, 9, size=n).astype(np.int32)
P = lambda a: ctypes.c_void_p(a.ctypes.data)  # noqa: E731  one-line pointer helper
h = lib.kag_create(n, 720, 0)
assert h
assert lib.kag_reset_dev(h, P(seeds), None, P(shops), P(nshops), None) == 0
cpu = K.Batch(n)
cpu.reset(seeds, None, shops, nshops)
bad = 0
rew = np.empty((n, 2))
done = np.empty(n, np.uint8)
S = lib.kag_ser_bytes()
ser = np.empty((n, S), np.uint8)
for t in range(T):
    a = np.ascontiguousarray(acts[t])
    assert lib.kag_step_dev(h, P(a), m, None) == 0
    cpu.step(a)
    assert lib.kag_rewards_dev(h, P(rew), None) == 0 and lib.kag_done_dev(h, P(done), None) == 0
    assert lib.kag_serialize_host(h, 0, n, P(ser)) == 0
    bad += (
        int((ser != cpu.serialize()).any(axis=1).sum())
        + int((rew != cpu.rewards()).any(axis=1).sum())
        + int((done != cpu.done()).sum())
    )
    if t == 300:  # partial reset by mask in the middle of the game
        mask = (np.arange(n) % 3 == 0).astype(np.uint8)
        seeds2 = seeds + 17
        assert lib.kag_reset_dev(h, P(seeds2), P(mask), None, None, None) == 0
        cpu.reset(seeds2, mask)
# errors: m < 1 and an invalid serialization range
assert lib.kag_step_host(h, P(acts[0]), 0) != 0 and b"m < 1" in lib.kag_last_error()
assert lib.kag_serialize_host(h, n - 1, 5, P(ser)) != 0
lib.kag_destroy(h)
print("emu *_dev test: games", n, "steps", T, "mismatches", bad)
sys.exit(0 if bad == 0 else 1)
