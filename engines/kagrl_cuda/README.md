# kagrl_cuda — RL self-play entirely on the GPU: engine + encode + execute

> **Note added for the release (Oct 2026).** This README is the development log of Sep 24, kept as written except
> that machine-specific details, timings and a few inaccuracies were removed or corrected. Since then the library has
> been built with `build_cuda.sh`, checked on the GPU (GPU == CPU) and used for training:
> `rl/appo.py --gpu-games G --kagrl-lib engines/kagrl_cuda/build/libkagrl_cuda.so` (relative to the repository root)
> calls `kagrl.RLBatch` directly (the "integration plan" below and `gpu_rollout.py` describe an earlier design); the
> setting of the final run d7b is in `configs/appo_d7b.json`. The recorded streams
> `engines/kagsim_cuda/data/streams.pkl.gz` are not part of this release; create them with
> `PYTHONHASHSEED=0 python engines/kagsim_cuda/record_streams.py` (by default games of the bots shipped in this
> repository, caretaker and pass_agent; `--pairs competition` is the competition-time job list, which needs bots that
> are not released). `parity_cpu.py` likewise plays caretaker and pass_agent in modes A and L by default;
> `--competition-agents` selects the agents of the Sep 24 run below.

A batch of n games lives in device memory (`kaghd::Sim[n]`). One step, without going through the host:
observation encoding for both seats (exactly `rl/encode.py` / `kagcore::encode`) → policy choice (torch, outside
the library) → execution of the choice (`rl/act.py` / `kagcore::execute`) + engine step (`sim_hd.hpp`, bit-exact with
vendor kagsim 1.32.7). The single source `kagrl.cu` is built in two ways: nvcc (GPU) and a regular C++ compiler with an emulated
CUDA runtime (CPU, for parity checks without a GPU). C API for ctypes, conventions as in `engines/kagsim_cuda` (batch.cuh,
kagsim_cuda.py).

Status: the CPU build matches all references (0 mismatches, protocol below); the GPU build matches the CPU build byte
for byte (`gpu_parity.py`).

## Files

| file | what |
|---|---|
| `kagrl_common.hpp` | logic of one game (`KAG_HD`): row layout, encode, execute, step, random legal choice |
| `kagrl.cuh` | C API (`extern "C"`) |
| `kagrl.cu` | kernels (thread = game or row) + C API; `KAG_EMULATE` → kernel launch = loop |
| `emu/cuda_runtime.h` | runtime emulation for the CPU build (copy of `engines/kagsim_cuda/check/emu` + the required functions) |
| `build_cpu.sh`, `build_cuda.sh` | build `build/libkagrl_cpu.so` / `build/libkagrl_cuda.so` |
| `check/check_cuda_syntax.sh` | clang `-x cuda` (device and host passes) without the CUDA SDK: catches host-function calls from kernels |
| `kagrl.py` | ctypes wrapper: `RLBatch` (numpy for CPU, torch CUDA for GPU), `load_lib`, `kernel_info`, `set_block` |
| `parity_cpu.py` | parity check of the CPU build against kagcore, Python rl/encode+rl/act+kagsim, the vendor engine (CPU only) |
| `gpu_rollout.py` | `Rollout`: self-play with one policy (or against a past version) in the `rl/appo.py` storage format |
| `gpu_parity.py` | GPU == CPU + benchmarks on a GPU machine |

External dependencies (read-only): `engines/kagsim_cuda/{sim_hd.hpp,pyrandom_hd.hpp,batch_common.hpp,shim/}`,
`engines/kagcore/kagcore.hpp`, for the parity checks — `engines/kagsim_cuda/build/kaghd_cpu*`, `engines/kagcore/build/kagcore*`,
`engines/kagsim_cuda/data/streams.pkl.gz`, `rl/`, `agents/`. Include order: `-I. -I../kagsim_cuda/shim
-I../kagsim_cuda -I../kagcore` (shim substitutes `sim_hd.hpp` for `sim.hpp`, vendor paths are not needed).

## Build

```bash
bash engines/kagrl_cuda/build_cpu.sh                                   # CPU only: c++ -O2 -ffp-contract=off -DKAG_EMULATE
bash engines/kagrl_cuda/check/check_cuda_syntax.sh                     # no CUDA SDK: clang -x cuda, device+host
CXX="g++ -fuse-ld=gold" bash build_cpu.sh                          # Linux, if the default linker fails
LDFLAGS_EXTRA="-Xcompiler -fuse-ld=gold" bash build_cuda.sh <arch>  # machine with nvcc; <arch> e.g. native or sm_80
```

GPU: `nvcc -O3 -arch=<arch> -fmad=false --expt-relaxed-constexpr -Xcompiler -ffp-contract=off -Xptxas -v`
(CUDA 13.3). No `--use_fast_math`. `-fmad=false`
and `-ffp-contract=off` — each operation is rounded separately, as in CPython/NumPy.

## API

C (`kagrl.cuh`): all `*_dev` functions take pointers to device memory (host memory in the CPU build) and `stream`
(`cudaStream_t`, NULL = default stream); they return 0 on success, otherwise −1 or the CUDA error code (message —
`kagrl_last_error()`); they **do not
synchronize** — everything is enqueued on the stream. Only `*_host` (before copying they wait for the
given stream: torch streams are non-blocking) and `kagrl_stream_sync` synchronize.

| function | meaning |
|---|---|
| `kagrl_dims(int32 out[16])` | NT=100, FT=30, MAXU=16, FU=16, FG=77, TMASK_BYTES=700, MB=489, NHEADS=24, NOPS=56, ACT_I32=170, SER_BYTES=7645, SIM_BYTES, MAX_ORDERS=10, EMULATED, API_VERSION=1 |
| `kagrl_create(n, episode_steps, device)` / `kagrl_destroy` | batch; games are not valid until `reset` |
| `kagrl_reset_dev(b, seeds u64[n], mask u8[n]\|NULL, shops u8[n,8]\|NULL, nshops i32[n]\|NULL, s)` | reset of all or of the marked games; `shops` — pinned shop order (as in the recorded streams) |
| `kagrl_encode_dev(b, rows, games, seats, tiles, units, unit_pos, unit_mask, glob, tmask, mmask, s)` | observation rows; `games=seats=NULL` → rows=2n, row 2g+s; only `seats` → rows=n, row g, seat seats[g]; both — an arbitrary list of (g, s) |
| `kagrl_act_step_dev(b, uc, mc, idx_bytes, raw, raw_mask, acts_out, s)` | choice of both seats `uc[n,2,16]`, `mc[n,2,24]` (int32 or int64 — `idx_bytes` 4/8) → execute → step; `raw i32[n,2,170]` + `raw_mask u8[n,2]` — a ready engine action for the seat (code opponent/stream); `acts_out i32[n,2,170]` — submitted actions (NULL — do not write) |
| `kagrl_execute_dev(b, uc, mc, idx_bytes, acts_out, s)` | engine actions only, no step (checks/debugging) |
| `kagrl_step_raw_dev(b, acts i32[m,2,170], m, s)` | step with ready actions; game i takes `acts[i % m]` |
| `kagrl_random_choices_dev(b, tmask, unit_mask, seed, salt, uc, mc, idx_bytes, s)` | random legal choice over encode rows (layout rows=2n); counter-based splitmix64 → identical on CPU and GPU |
| `kagrl_rewards_dev` (f64 [n,2]) / `kagrl_done_dev` (u8 [n]) / `kagrl_steps_dev` (i32 [n]) | banks, flags, step counter |
| `kagrl_serialize_host(b, first, count, out u8[count,7645], s)` | full state (format of `kaghd_cpu.RefBatch.serialize`) |
| `kagrl_set_block(threads)`, `kagrl_kernel_info(i32 out[24])` | threads per block (default 128); registers/local/maxThreads/shared of the 6 kernels (zeros in the CPU build) |

Semantics: a game finishes after `episode_steps − 1` steps; `act_step`/`execute` on a finished game is a
no-op, `acts_out` = zeros. Loop of T=720 iterations as in the appo worker: the last iteration encodes the final
state, its step does nothing. `tmask` is packed (`np.packbits`, bit t·56+o, most significant bit first) —
unpacked with `rl.appo.unpack`. The `uc`/`mc` indices are not checked on the device (int64 is saturated to int32, beyond that —
the semantics of `kagcore::execute` = `rl/act.py`). Random choice `random_choices`: a unit — uniform over the
legal bits of its row (an absent unit → 0); market — heads 21–23 (ORDER, SELL_FIRST, SEED_FIRST) uniform, SELL p<0.12 "everything",
p<0.70 zero, otherwise any; HIRE p<0.95 zero, otherwise 1–2; LAND p<0.99 zero, otherwise 1; other purchases p<0.90 zero,
otherwise any (so that games do not go bankrupt instantly; they still go bankrupt — for "rich" states there are
recorded streams via `raw`/`step_raw`).

Python (`kagrl.py`):

```python
from kagrl import RLBatch

rb = RLBatch(n, 720, backend="gpu", device=0)  # "cpu" — libkagrl_cpu.so, numpy or torch CPU
obs = rb.alloc_obs(kind="torch")  # 7 arrays for 2n rows
rb.reset(torch.arange(n, device="cuda") + 1000)  # host arrays are copied automatically
for t in range(720):
    rb.encode(obs)  # stream = torch.cuda.current_stream()
    uc, mc = policy(obs)  # [2n,16] / [2n,24] or [n,2,16] / [n,2,24], int32|int64
    rb.act_step(uc, mc)  # optional: raw=, raw_mask=, acts_out=
bank = rb.rewards()  # out=None -> numpy (synchronizes); out=tensor -> no synchronization
```

`gpu_rollout.Rollout(n, T=720, backend, device, chunk=4096, pad_chunks=False).run(net, seeds, store=None,
opp_net=None, opp_games=None, me_seat=None, margin_w=0.3, margin_scale=30000)` → `{store, NA, R, rows, bank,
margin, stats, info}`: storage in the `make_store` format of `rl/appo.py` (tiles/units/glob bf16, uc/mc int64,
lpu/lpm/val), sampling and logp — verbatim appo `act_compute`/`past_compute` (autocast bf16, Gumbel
`rl.ppo.sample`), `R` = outcome ±1/0 + margin_w·clip(margin/margin_scale). Self-play: NA = 2n, row 2g+s. With a past
version: `opp_net` plays seat `1 − me_seat[g]` in the games `opp_games`, only the learner seats are stored.
Host synchronization — once, at the end of the game.

## Parity check of the CPU build

Build: `build_cpu.sh`, clang, `-O2 -ffp-contract=off -DKAG_EMULATE`; Python 3.11, numpy 2.4.6,
torch 2.13 (CPU); `PYTHONHASHSEED=0`. Sources as of Sep 24 (the files have changed since): `kagrl.cu` sha256
9010b106…, `kagrl_common.hpp` 260f83cc…, `kagrl.cuh` fe963f23….

**`parity_cpu.py --procs 4` → `data/parity_cpu_result.json`: 0 mismatches.** 66 games × 720
iterations (47 454 game steps; every game runs to the end, 719 steps):

| mode | games | seeds | who moves |
|---|---|---|---|
| R | 4 batches × 6 | 940000+101j+13k | random choice: Python (`rand`, `randH` — "greedy" market) and the `random_choices` kernel (`kern`); int32 and int64 interleaved |
| P | 3 × 6 | 950000+101j+13k | policy `rl.model.Net(d=64, layers=2, heads=4, dk=16, vtower=1)` with random initialization, `rl.ppo.sample`, fp32 CPU, torch.long indices |
| A | 3 × 4 | 960000+101j+13k | agent vs agent (c3bc, v4_look12, bcr_tdp_v4, pub0923_tetsutani_dp, pub0923_arsg_herdsafe3, soil_v7; now `--competition-agents`) via `raw_mask`; at every step additionally K=2 random choices via `execute` (no step) — check of execute on "rich" states |
| L | 3 × 4 | 970000+101j+13k | league: our seat rand/pol/kern, the other — agent code via `raw` (same agents as A) |
| API | — | — | 20 checks: encode row layouts, partial reset with a mask and pinned shops (+600 steps against vendor), done only for the games that were not reset, all done after 719, no-op on finished games, `step_raw` of 5 recorded streams (m<n) == vendor and == recorded banks, determinism and legality of `random_choices`, argument errors |

What was compared at every step: encode of both seats — 7 arrays byte for byte with `kagcore.GameBatch.encode_into`
(95 040 rows) and with Python `rl/encode.py` over `kagsim.Game` (95 040 rows); engine actions — with
`kagcore.GameBatch.execute_raw` (103 536) and with `conv_action_raw(rl.act.execute(...))` (129 420); full state
(serialization) — with the vendor engine `kaghd_cpu.RefBatch` stepped with our `acts_out` (47 520); banks of both seats with
kagcore, vendor and Python kagsim (95 040); done/steps (95 040); final banks to the dollar (132). Coverage: up to 16
units per seat, 308 states with >16 units (truncated to 16 as in rl/encode.py), 34 869 states with animals,
40 156 executions with 10 orders, all 18 unit operations and 6 market ones, 25 884 seat steps via raw. Final banks are
non-zero only in games with code agents (36 of 132; random play and the random policy go bankrupt).
Sensitivity verified: deliberately breaking the int64 path produced 1401 mismatches (the file was restored, md5 matched).
The default run of the released script plays caretaker and pass_agent in modes A and L instead (same seeds, same other
roles), so its coverage and final banks differ from the figures above.

**`gpu_rollout.py --selftest` → `data/rollout_selftest.json`: OK.** B=8 games, T=64, chunk=6 (the policy chunk does not
divide the number of rows), `Net(d=64, layers=2, heads=4, dk=16, vtower=1)`, seeds 880000+7919i; self-play (NA=16) and
the mode with a past version (half of the games, NA=12). Replay in `kagcore.GameBatch` with the same choices:
encodings of every step == storage (after bf16) — 7168 + 5376 rows, choices 1024 + 768, banks to the dollar;
logp and value recomputed with the same network from the storage differ by exactly 0.0 (CPU).

**`gpu_parity.py --dry-run` → `data/gpu_parity_dryrun.json`: OK** (both sides — CPU build, the "GPU" side via
torch tensors): P1 24 games from recorded streams × 720, P2 16 games × 720 (+ check against kagcore: 161 280 encode
rows, 23 040 banks), 0 mismatches; the first 4 games of P1 (recorded seeds and shops) reproduce the recorded
banks to the dollar. The dry-run benchmarks only test the code, not the speed.

## GPU check (`gpu_parity.py`; passed later, see the note at the top)

1. nvcc 13.3 build; registers, local memory (spill), stack frame of each kernel (`-Xptxas -v` in the log and
   `kernel_info` in the JSON).
2. GPU == CPU byte for byte: P1 — 256 games (220 recorded streams with recorded seeds/shops + 36 of the same
   streams on new seeds) × 720 iterations: encode of 512 rows (7 arrays), `random_choices`, `execute`, `act_step`
   with `raw_mask=1` (`acts_out`), full state, banks, done, steps at every step. P2 — 256 new seeds: even
   steps the `random_choices` kernel (int64), odd steps — the `Net(d=64)` policy on the GPU (bf16 autocast, sampling on the GPU, the same
   indices are copied to the CPU build), in odd games seat 1 plays a recorded stream via `raw`; plus the CPU build
   against `kagcore.GameBatch` (encode and banks), if the module was built.
3. Benchmarks (CUDA events), JSON lines:
   (i) `engine` B ∈ {4096, 16384, 65536}: 719 steps of encode 2B rows + random_choices + act_step, games/s and ms/step per
   kernel; block-size sweep 64/128/256 at B=65536; (i-b) `streams`: step_raw of recorded actions + encode + random +
   execute (real states); (ii) `policy_eager` / `policy_compiled`: encode + `Net(d=384, layers=8, vtower=2)`
   bf16 autocast over 2B rows in chunks of 8192 (= appo act_compute) + act_step; warm-up of 300 steps, window of 8 steps, estimate
   games/s = B / (sec/step × 719); (iii) `rollout`: full self-play `Rollout` of 1024 games × 720 with storage
   (storage size, peak memory).

The cost of self-play is set by the network's forward pass, not by the engine; the gain of the GPU engine is removing
the CPU workers, the wait for Python opponents and the copies through the host.

## Running the check

On a machine with a GPU: build `libkagrl_cpu.so` (`build_cpu.sh`), `libkagrl_cuda.so` (`build_cuda.sh`) and, optionally,
the `kagcore` module (`scripts/build_engines.sh`), then run `python engines/kagrl_cuda/gpu_parity.py --out res.json`. The
recorded streams it reads (`engines/kagsim_cuda/data/streams.pkl.gz`, a fixed path) are not part of this release; create
them first with `PYTHONHASHSEED=0 python engines/kagsim_cuda/record_streams.py` (needs kagsim and kaggle-environments;
the same file feeds the `step_raw` check of `parity_cpu.py`). Exit code 1 — GPU/CPU mismatches found. If time/memory is
not enough: `--policy-B 4096,16384 --rollout-B 512` or `--skip-parity`.

## Limitations

- Code opponents (c3bc, look12 and other Python agents) do not play on the GPU. A path exists only for ready actions:
  `raw`/`raw_mask` (int32[170] to the device every step), but the library does not build the observation dict for a Python agent
  (kagsim builds it on the CPU). So the league with code opponents stays on the CPU workers (hybrid).
- `Rollout` supports one opponent policy per call (`opp_net`); "past" and "best" at the same time — two calls
  on different games, or further work.
- The CPU build is single-threaded (parity, not speed). In `Rollout` all games of a batch start together and run T iterations
  (partial reset by mask exists in the API, `Rollout` does not use it).
- `mmask` — as in `rl/encode.py`. Choice indices are checked only for host arrays, in the wrapper.
- Numerics: in encode, `glob` takes `kag_log1p(money)` (kagcore.hpp) — libdevice on the device, libm on the host;
  they can differ by 1 ulp of a double; after `/12` and the cast to float they almost always match, but this is not
  guaranteed. The engine prices (log/log10/sqrt → rounding to an integer) have already been checked on the GPU in `engines/kagsim_cuda`. Both seats are checked by P1/P2 in `gpu_parity.py`.

## Plan for integration into rl/appo.py (not done, the file was not changed)

1. Flag `--gpu-rollout N` (games per rank) and, with it, skipping the start of CPU workers for the "self/past/best" share.
2. On the rank: `ro = Rollout(N, T, backend="gpu", device=LOCAL_RANK, chunk=…, pad_chunks=True)`; storages —
   `make_store(T, NS=2N, dev)` (or the current appo `make_store` with `NS = 2N`).
3. In `actor()`, instead of the loop over workers: inside `with torch.cuda.stream(astream)` —
   `out = ro.run(act_net, seeds, store=st, opp_net=snap_net, opp_games=…, me_seat=…, act_fn=act_c, past_fn=past_c)`;
   seeds `x.seed0 + (it·world + rank)·N + arange(N)`; `NA = out["NA"]`, `R = out["R"]`, `stats = out["stats"]`,
   `info = out["info"]` → `ready.put((it, NA, R, stats, ev, info, pw))` in the same format.
4. Games with code opponents — leave them to the CPU workers (smaller W), put their rows into the same storage after
   the GPU rows (hybrid) — or, at the first stage, train only self/past/best on the GPU.
5. Recompute `--minibatches` for the new NA; `torch.compile` of the actor —
   with a fixed `chunk` and `pad_chunks=True`.

## Risks

- Registers and local memory: one thread holds a whole game (`kaghd::Sim` ~8 KB in global memory) and the local
  encode/execute arrays (kilobytes) → spill to local memory, low occupancy. Check `-Xptxas -v` and
  `kernel_info`; if needed — a smaller block (`kagrl_set_block`) or splitting encode into a "warp per row" kernel.
- "Thread per row" writes: neighbouring threads write addresses 12 KB apart (tiles) — the writes are not coalesced; if encode
  turns out to be a bottleneck — warp per row or staging through shared memory.
- Storage memory grows with games × steps (tmask and the bf16 tiles dominate each row-step); appo keeps two storages
  (double buffering).
- Learner: forward+backward on the same rows × epochs is more expensive than sampling; once the actor is faster, the bottleneck
  will be training — measure them together.
