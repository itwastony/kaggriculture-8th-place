# Code map

## `rl/`: the method

| File | Role | Used by |
|---|---|---|
| `spec.py` | action space: 56 operations, 24 market heads and their bins, order templates | everything below |
| `encode.py` | observation → arrays: global token (77), 100 tile tokens (30), 16 unit tokens (16), legality masks | data, training, submission |
| `act.py` | executor (target + operation → engine action, x-first walking, order assembly and the 10-order cap) and hindsight labels from recorded actions | BC data, training, submission |
| `model.py` | the transformer (`Net`), unit pointer head, market heads, value tower, privileged critic; log-prob helpers | BC, APPO |
| `np_net.py` | NumPy forward pass of the actor (fp16 storage, fp32 compute) | submission |
| `gen_bc_tapes.py` | BC demonstrations from tape worlds: replay, select teacher seats, label | `make_bc_data.py` |
| `data.py` | BC shards → tensor batches (encoding in data-loader workers) | `train_bc.py` |
| `bc_train.py` | BC training (cross-entropy on unit pairs and market heads + value MSE), single GPU or DDP | `train_bc.py` |
| `appo.py` | APPO: GPU self-play through `kagrl`, CPU workers on kagcore/kagsim against code bots and tapes, league and PFSP, reward, GAE + UPGO, per-component PPO, KL to the anchor, anchor gate, EMA, the 4th-quadrant rule, checkpoints | `train_appo.py` |
| `ppo.py` | the earlier synchronous PPO (lines r1/r2); `appo.py` imports its shared pieces (`ENC`, `LEAGUE`, `sample`, `unit_terms`, `market_terms`) | `appo.py` (import) |
| `tape_check.py` | replays a tape pool through kagcore with recorded actions; banks must match to the dollar | `prepare_data.py --repro` |
| `tape_pool.py` | builds a pool manifest from a world index | data preparation |
| `record.py` | plays two `main.py` agents on kagsim and records actions | `verify_submission.py --games`, `appo.py` (code bots) |
| `export_agent.py` | checkpoint → `main.py` + `weights.npz` + `rl_<sha8>/` package; inference-time options (`--land-inject`, `--dtype`) | `export_submission.py` |
| `latency.py` | per-turn latency of an exported agent and torch/NumPy action agreement (gate G2) | optional |
| `policy.py` | torch policy wrapper (checkpoint → engine action) | `latency.py` |
| `gen_bc.py` | BC demonstrations from a code teacher against a pool (lines r1/r2) | not used by the final line |

## `entry_points/`, `harness/`, `tools/`

| File | Role |
|---|---|
| `entry_points/prepare_data.py` | data integrity and replay check of the pools |
| `entry_points/make_bc_data.py` | step 2: BC demonstrations (`configs/bcgen_d7.json`) |
| `entry_points/train_bc.py` | step 3: BC (`configs/bc_d7.json`) |
| `entry_points/train_appo.py` | step 4: APPO (`configs/appo_d7b.json`, `configs/league_d7.json`); `--preset smoke`, `--no-pool-src`, `--league-available-only` |
| `entry_points/export_submission.py` | step 5: checkpoint → submission archive (`configs/export_d7b.json`) |
| `entry_points/verify_submission.py` | the released archives follow from the released checkpoints and this code |
| `entry_points/predict.py` | play an archive in the official engine (`kaggle_environments`) |
| `entry_points/evaluate.py` | paired head-to-head on kagsim, SE per seed |
| `harness/kagsim_play.py` | play two Python agents on kagsim, loading and calling them as kaggle-environments does |
| `harness/tape_play.py` | replay recorded actions against an agent (or another tape) on kagsim with the recorded shop schedule |
| `harness/tape_worlds.py` | recorded episodes (daily-dataset zips or compact tapes) → world files with the replay check |
| `tools/fetch_episodes.py` | download replays through the Kaggle API as compact tapes |
| `tools/rebuild_worlds.py` | rebuild every world referenced by `data/pools/*.json` |
| `tools/fetch_league_bots.py` | download the public league bots listed in `agents/sources.json`, install those whose `main.py` matches our record |
| `tools/compare_worlds.py` | compare two sets of world files field by field |
| `tools/strip_init.py` | keep only the actor part of an RL checkpoint (BC initialisation) |

## `engines/`: our engine ports

| Folder | What | Built by |
|---|---|---|
| `engines/kagcore/` | C++ observation encoder and action executor (`kagcore.hpp`), bit-exact with `rl/encode.py` / `rl/act.py`, plus a pybind11 module `kagcore` with a batched game API on top of the kagsim engine (`GameBatch`: tapes, code-bot games, our policy's choices). `parity.py` checks it against the Python code; `bench.py` measures speed | `build.sh` |
| `engines/kagsim_cuda/` | `sim_hd.hpp`: a copy of kagsim's engine core that compiles both for the host and for CUDA (one GPU thread = one game), with CPython-compatible random numbers (`pyrandom_hd.hpp`) and a canonical state serialisation (`state_visit.hpp`); a CPU module and a GPU batch library for parity checks and benchmarks | `build_cpu.sh`, `build_cuda.sh` |
| `engines/kagrl_cuda/` | `kagrl`: engine + encoder + executor for thousands of games in GPU memory (`kagrl.cu`, C API `kagrl.cuh`, ctypes wrapper `kagrl.py`). `rl/appo.py --gpu-games` uses `kagrl.RLBatch`. The same source builds for the CPU with an emulated CUDA runtime (`build_cpu.sh`) for parity checks. Its README is the development log of the library | `build_cuda.sh` (`scripts/build_engines.sh --gpu`) |

`vendor/kagsim/` is third-party (Apache-2.0): a C++ port of the Kaggriculture engine with a Python API, bit-exact with
`kaggle-environments` 1.32.7, with its own tests. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Engine parity

The chain is kaggle-environments 1.32.7 → kagsim → kagcore (+ Python `rl/encode.py`, `rl/act.py`) → kagrl CPU build →
kagrl GPU build. Every link is checked against the previous one.

| Check | Protocol | Result |
|---|---|---|
| kagsim vs kaggle-environments | `vendor/kagsim/tests`: golden traces of recorded 1.32.7 episodes; observations compared every turn | pass (run by `build_engines.sh --check`) |
| kagsim vs kaggle-environments, our agents | games of live code agents on both engines, `PYTHONHASHSEED=0`, final banks compared | 14,336 / 14,336 identical |
| tape worlds | both recorded action streams replayed in kagsim on the recorded seed; final banks vs the record | all 5,539 worlds of the pools match to the dollar; in the 10,454-seat leaderboard-field corpus 5,258 of 5,286 records passed (the rest were not used) |
| kagcore vs Python `rl/encode.py`, `rl/act.py` | `engines/kagcore/parity.py`: encodings and executed actions compared on kagsim games | pass with the competition-time game list (`--competition-agents`). `build_engines.sh --check` runs `--quick` with the default agents, caretaker and pass_agent |
| kagrl CPU build vs kagcore, the Python reference and the vendor engine | `engines/kagrl_cuda/parity_cpu.py --competition-agents` (Sep 24): 66 games × 720 steps; random legal choices, policy choices, and live code agents through the raw-action path. The default plays caretaker and pass_agent instead | 0 mismatches in encodings, executed actions, states and banks |
| kagrl GPU vs kagrl CPU | `engines/kagrl_cuda/gpu_parity.py` (CUDA 13.3, `-fmad=false`): P1 = 256 games replaying recorded action streams; P2 = 256 games on new seeds, random choices on even steps, a small random policy on odd steps, and in odd games seat 1 replays a recorded stream | 512 × 720 steps: 0 mismatches in 5,160,960 encoded rows, 368,896 state comparisons, 737,792 bank comparisons, executed actions and choices; P1 final banks equal the recorded ones |
| kagcore inside P2 (extra check) | the same P2 games stepped in kagcore | mismatches in 12 of the 128 games where a recorded stream is replayed **on a different seed** (raw invalid actions take two different conversion paths); 0 in the 128 policy-only games. Training never uses this mode |
| torch vs NumPy export | greedy actions of the exported agent vs the torch policy on control games (gate G2) | ≥ 2,860 of 2,876 steps identical; differences come from fp16 rounding |

`gpu_parity.py` and `parity_cpu.py` need recorded action streams; create them first with
`PYTHONHASHSEED=0 python engines/kagsim_cuda/record_streams.py` (default: games of caretaker and pass_agent;
`--pairs competition` is the competition-time list and needs league bots that are not in this repository).
`parity_cpu.py` also needs two CPU builds that `scripts/build_engines.sh` does not make:
`bash engines/kagsim_cuda/build_cpu.sh` (module `kaghd_cpu`) and `bash engines/kagrl_cuda/build_cpu.sh`
(`libkagrl_cpu.so`); it stops before any game and names the command if one is missing.

## Differences from the competition-time code

- **Comments, docstrings and log messages were translated from Russian to English** after the competition. For every
  changed Python file the abstract syntax tree was identical to the original except string constants that contained
  Russian; for C++/CUDA/shell files the token stream without comments was identical. Russian (Cyrillic) text remains
  only in these places: inside the two submitted archives (file contents kept byte for byte); in
  `configs/export_d7b.json`, which quotes the first line of the submitted `main.py`; in a test fixture
  (`tests/test_entry_verify.py`) and the Cyrillic character range in `entry_points/verify_submission.py`, which check
  that Russian string constants are ignored when programs are compared; and in some other teams' names in
  `data/pools/`. Because of the translation a re-exported archive is not byte-identical to the submitted one (the
  package name `rl_<sha8>` is a hash of the module sources; line 1 of `main.py` names the checkpoint), so
  `verify_submission.py` compares programs, not bytes.
- **Lint and format clean-up** (settings in `pyproject.toml`): imports split one per line in their original order,
  unused imports removed, unused or one-letter loop and comprehension variables renamed, `d.get(k, None)` →
  `d.get(k)`, one `%` format → f-string, docstrings, argparse help texts and one error message rewritten. Each changed
  Python file was compared with the version before the clean-up by abstract syntax tree; every difference is one of
  these or one of the repository changes listed below. The four modules of the submitted package (`rl/spec.py`,
  `rl/encode.py`, `rl/act.py`, `rl/np_net.py`) and the `main.py` template in `rl/export_agent.py` changed only in
  comments and formatting, which `verify_submission.py` confirms. C++/CUDA files changed only in comments.
  `agents/caretaker/main.py` is kept byte for byte (`agents/sources.json`).
- **Engine checks default to the bots in this repository.** `engines/kagcore/parity.py`,
  `engines/kagsim_cuda/record_streams.py` and `engines/kagrl_cuda/parity_cpu.py` play caretaker and pass_agent by
  default; the competition-time game lists, which need unreleased league bots, are behind `--competition-agents`
  (`--pairs competition` for `record_streams.py`). `engines/kagcore/bench.py` plays caretaker against itself. The
  kagcore and kagrl parity results above were obtained with the competition-time lists.
- **For this repository:**
  - the engine folders were moved from `tmp/` to `engines/` (the paths in `rl/appo.py`, the build scripts and
    `SETTINGS.json` were changed accordingly), and CUDA builds default to the build machine's GPU architecture;
  - `entry_points/train_appo.py` got `--league-available-only`, and its `--set` option can now be repeated
    (`--set=--gpu-games=64 --set=--minibatches=72`). Before, `--set` kept only the last value, and the form given in
    its help text did not parse. Each override still replaces one argument of `rl/appo.py`;
  - the directory names of the snapshot pools in `configs/appo_d7b.json` (`--pool-src`) were shortened to `d1`, `d4`
    and `d6`; the pools are not published;
  - `entry_points/prepare_data.py` checks rebuilt worlds by replay instead of by hash, and `scripts/slurm/verify.sbatch`
    skips the replay check with a note when no tape-world pool is complete;
  - `tools/rebuild_worlds.py`, `tools/fetch_league_bots.py` and `agents/sources.json` were added;
  - `scripts/slurm/appo_d7b.sbatch` passes extra arguments on to `train_appo.py`;
  - `engines/kagrl_cuda/parity_cpu.py` checks that its compiled modules are built before any game, and
    `engines/kagsim_cuda/build_cpu.sh` uses `python3` on PATH by default (before: `../../.venv/bin/python`);
  - examples use `agents/caretaker` instead of league bots that are not released.
- **Launch scripts** with internal paths were replaced by `configs/*.json` + `SETTINGS.json`; the flags are those of the
  original run (only the snapshot-pool directory names were shortened, see above).

## Known inaccuracies in code comments

None affects behaviour.

| Where | The comment says | The code does |
|---|---|---|
| `engines/kagrl_cuda/README.md` | an "integration plan" through `gpu_rollout.Rollout` | outdated: training uses `kagrl.RLBatch` (note at the top of the README) |
