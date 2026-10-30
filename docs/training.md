# Training: from data to the submission archive

Commands run from the repository root. Directories come from `SETTINGS.json`; every script writes under
`OUTPUT_DIR/<run>/` (default `runs/`) and prints the commands it runs.

| Step | Command | Runs on | Deterministic? |
|---|---|---|---|
| 0. Environment and engines | `pip install …`, `scripts/build_engines.sh` | CPU (+ nvcc for the GPU engine) | — |
| 1. Tape worlds | `tools/rebuild_worlds.py`, `entry_points/prepare_data.py` | CPU, Kaggle API (read-only) | yes (replay check to the dollar) |
| 2. BC demonstrations | `entry_points/make_bc_data.py --run d7` | CPU processes | yes |
| 3. BC | `entry_points/train_bc.py --run d7 --nproc G` | G GPUs on one node | yes with the same GPU type, GPU count and software; otherwise small differences |
| 4. APPO d7b to `ema_00110.pt` | `entry_points/train_appo.py --run d7 --nnodes N --nproc G` | N nodes × G GPUs, 26 CPU worker processes per GPU | no (asynchronous collection) |
| 5. Export | `entry_points/export_submission.py --run d7 --ckpt …` | CPU | yes |
| 6. Inference | `entry_points/predict.py --archive …` | CPU (official engine) | yes, except games against the engine's `random` agent |

SLURM templates for steps 2–4 and for the verification are in `scripts/slurm/` (`mkdir -p runs/logs` first: SLURM
opens the log file before the script starts).

## 0. Environment

```bash
python3.11 -m venv .venv && . .venv/bin/activate
# torch from the CUDA 13.0 index first: after a plain torch 2.13.0 from PyPI (requirements.txt pins it), pip counts
# the +cu130 wheel as already installed and skips it
pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0
pip install -r requirements-train.txt kaggle==2.2.4   # kaggle: tools/rebuild_worlds.py, tools/fetch_league_bots.py
bash scripts/build_engines.sh --gpu --check           # kagsim + kagcore + the CUDA engine, with self-tests
```

- **Verification and inference only** need `pip install -r requirements.txt` (no GPU; README, Quick start). For
  training, start from a fresh environment as above. The engines need a C++17 compiler; the CUDA engine also needs
  `nvcc`.
- **Software of the original run:** Linux x86-64 (glibc 2.39), gcc 13.3, CUDA toolkit 13.3, Python 3.11.16,
  torch 2.13.0+cu130 (triton 3.7.1, cuDNN 9.20.0.48, NCCL 2.29.7), numpy 2.4.6, kaggle-environments 1.32.7, kagsim 0.5.0.
- **Expected end of `build_engines.sh --check`:** the kagsim tests pass (golden traces of recorded 1.32.7 episodes
  reproduce their banks; observations equal kaggle-environments' every turn), `engines/kagcore/parity.py --quick`
  reports no mismatches, then `engines OK`.
- **GPU architecture.** The CUDA library is built for the GPU of the build machine by default
  (`-arch=native`); pass another one with `GPU_ARCH=sm_80 bash scripts/build_engines.sh --gpu`. It is compiled with
  `-fmad=false`, so results do not depend on fused multiply-add. Its GPU-vs-CPU check is
  `engines/kagrl_cuda/gpu_parity.py` ([code_map.md](code_map.md#engine-parity)).
- **Game engine version.** Everything assumes `kaggle-environments` 1.32.7, the server version during the
  competition. On another version, re-run the engine checks first.

## 1. Tape worlds

```bash
python tools/rebuild_worlds.py --list            # 11 world folders, 5,539 world files (5,465 distinct episodes)
python tools/rebuild_worlds.py                   # download + replay check + copy to data/worlds/ (Kaggle API token)
python entry_points/prepare_data.py --repro 50   # data integrity + replay of 50 seats per pool on the training path
```

Kaggle rate-limits replay downloads; expect hours for all episodes. `prepare_data.py --repro` replays only the pools
whose world files are all present and stops if no pool is complete; training with incomplete pools needs
`train_appo.py --league-available-only` (step 4). Details, pool contents and alternatives (the daily episode
datasets): [data/README.md](../data/README.md).

## 2. BC demonstrations (exact)

```bash
python entry_points/make_bc_data.py --run d7
```

Replays the 2,049 worlds of `data/worlds/d0930b` in 224 shards (`configs/bcgen_d7.json`; shard i takes the worlds whose
index in the sorted file list is i mod 224; the shard count fixes the train/validation split). Expected, exactly:
809 trajectories — DECEM 211, DSM 200, M & M & P & Q 127, Majkel1337 112, Victor @ Tufa Labs 97, CDE 41,
Vadim Vasilenko 16, Unknown Mother-Goose 5 — and the script prints `matches the original run ...: True`. Totals of the
original run: [logs/bcgen_d7_totals.json](../logs/bcgen_d7_totals.json).

If some worlds could not be rebuilt (a replay no longer downloadable), the counts will be lower and the check fails;
the rest of the pipeline still runs.

## 3. Behaviour cloning

```bash
python entry_points/train_bc.py --run d7 --nproc G    # G GPUs on one node; G must divide the global batch of 2,048 states
```

- Initial weights: `artifacts/checkpoints/final/bc_d7_init_d6_140_pol.pt` (the actor of `d6` EMA 140; release asset).
- All hyper-parameters: `configs/bc_d7.json`. Output: `runs/d7/bc/ckpt_00{1,2,3}000.pt` and `train.jsonl`.

Compare `runs/d7/bc/train.jsonl` with [logs/bc_d7_train.jsonl](../logs/bc_d7_train.jsonl). The published log is
aggregated: training metrics are means over blocks of 1,000 steps, validation accuracies are those at the end of each
block, all rounded to 0.01. Aggregate your log the same way before comparing
([logs/README.md](../logs/README.md)).

| Steps | val_acc_unit | val_acc_op | val_acc_mkt | val_acc_mkt_nz | train acc_unit (block mean) |
|---|---:|---:|---:|---:|---:|
| 1–1,000 | 0.90 | 0.93 | 0.96 | 0.73 | 0.90 |
| 1,001–2,000 | 0.90 | 0.93 | 0.97 | 0.75 | 0.92 |
| 2,001–3,000 | **0.90** | **0.93** | **0.97** | **0.76** | 0.94 |

Validation: shards with index divisible by 50 (16 of 809 trajectories). The first 20 per-GPU batches of the
validation stream are scored, so the validation numbers depend slightly on the number of GPUs.

**Determinism.** On Oct 12 we reran steps 2 and 3 with the pinned software on the same machine type and the same
number of GPUs as the original run. All 224 demonstration shards were identical, the whole `train.jsonl` was
identical, and `ckpt_003000.pt` was byte-identical to the released `bc_d7_ckpt_003000.pt` (sha256 `40d53ea9…19ab`,
122 of 122 tensors equal). On another GPU type, another number of GPUs or other library versions the reduction order
changes; expect small differences (not measured).

To skip this step, `train_appo.py` falls back to the released `artifacts/checkpoints/final/bc_d7_ckpt_003000.pt`.

## 4. APPO

```bash
# N nodes with G GPUs each, one rank per GPU (scripts/slurm/appo_d7b.sbatch does this under SLURM)
python entry_points/train_appo.py --run d7 --nnodes N --nproc G --node-rank $NODE_RANK --rdzv $MASTER_HOST:$PORT \
    --no-pool-src
# one node
python entry_points/train_appo.py --run d7 --nnodes 1 --nproc G --no-pool-src
```

**Inputs.**
- The BC checkpoint from step 3, or the released one.
- Five fixed past checkpoints: the BC checkpoint and the four in `artifacts/checkpoints/past/` (release assets).
- The league `configs/league_d7.json`: self/past/best, 101 code bots in `agents/`, 6 tape pools in `data/pools/`.
  Only 2 of the 101 bots are in this repository; `python tools/fetch_league_bots.py` installs the public ones
  ([agents/README.md](../agents/README.md)). The script stops if a bot or a tape world is missing;
  `--league-available-only` drops the missing bots, keeps only the tape-pool seats whose world files exist, and trains
  with the rest. This changes the opponent mix; we did not measure its effect.
- Snapshot pools of three earlier lines (`SNAPSHOT_POOL_DIR/{d1,d4,d6}`) are not published. Pass
  `--no-pool-src`: past opponents are then the run's own snapshots plus the 5 fixed checkpoints. This changes the
  past-opponent mix; we did not measure its effect.
- The GPU engine library (`KAGRL_LIB` in `SETTINGS.json`).

**Smoke test first** (1 GPU, 2 iterations; checks the plumbing, not the training):

```bash
python entry_points/train_appo.py --run smoke --preset smoke --no-pool-src --league-available-only
python entry_points/export_submission.py --run smoke --ckpt runs/smoke/appo/ema_00002.pt --name smoke_agent
python entry_points/predict.py --archive runs/smoke/smoke_agent.tar.gz --opp starter --seeds 1
```

On Oct 12 the smoke preset with the full league ran end to end on one GPU: 2 iterations, `ema_00001.pt` and
`ema_00002.pt` written, the exported agent won 2 of 2 games against `starter` at 88.5 ms per call. A 2-iteration
agent is essentially the BC policy. A reduced league from `--league-available-only` has not been run end to end yet.

**Reference values of the original run** ([logs/appo_d7b_curves.csv](../logs/appo_d7b_curves.csv); training games;
means over blocks of 10 iterations, win rates rounded to 0.05):

| Iterations | P_field | P vs Sep 30 tapes | P vs past versions | entropy | KL coef. | % learner seats ending with 4 quadrants (rank 0) |
|---|---:|---:|---:|---:|---:|---:|
| 0–9 | 0.75 | 0.65 | 0.30 | 0.13 | 0.0197 | 99.9 |
| 10–19 | 0.85 | 0.80 | 0.40 | 0.11 | 0.0189 | 100 |
| 30–39 | 0.90 | 0.85 | 0.50 | 0.11 | 0.0174 | 100 |
| 60–69 | 0.95 | 0.90 | 0.60 | 0.11 | 0.0152 | 100 |
| **100–109** | **0.95** | **0.95** | 0.65 | 0.12 | 0.0122 | 100 |

- `P_field` = win rate against the 101 code bots and the 6 tape pools (self/past/best excluded).
- `ema_00110.pt` is written after 110 iterations (0–109); it is the submitted checkpoint. The original run continued
  to iteration 117 and was then stopped; nothing after iteration 109 was used.
- Learner win rates against the fixed past checkpoints (blocks of 20 iterations):
  [logs/appo_d7b_vs_fixed_past.csv](../logs/appo_d7b_vs_fixed_past.csv). The parent `d6` EMA 140 is overtaken around
  iteration 60.

**Scaling.** The config fixes only per-rank settings; d7b was trained in one layout, and other layouts were not run
end to end, so their learning curves are unmeasured.
- **Ranks.** Every rank collects its own games and the gradients are averaged (DDP). Fewer ranks give fewer games per
  iteration with the same number of optimizer steps: expect noisier updates and more iterations to reach the same
  `P_field`. Watch `P_field` and the win rate against the Sep 30 tapes rather than the iteration number.
- **GPU memory.** If 128 GPU games per rank do not fit in GPU memory, lower `--gpu-games`; if a learner minibatch does
  not fit, raise `--minibatches` (smaller minibatches).
- **CPU workers.** Each worker is one single-threaded process that plays CPU games; lower `--workers` if the CPU is
  oversubscribed (fewer workers give fewer CPU games per iteration).
- **Overrides.** `--set` replaces one argument of `rl/appo.py` from the config. Repeat it for several, always in the
  `=` form (`--set --gpu-games=64` does not parse):

```bash
python entry_points/train_appo.py --run d7 --nnodes 1 --nproc G --no-pool-src \
    --set=--gpu-games=64 --set=--minibatches=72 --set=--workers=12
```

## 5. Export, play and compare

```bash
python entry_points/export_submission.py --run d7 --ckpt runs/d7/appo/ema_00110.pt --name my_d7b_110   # step 5
python entry_points/predict.py --archive runs/d7/my_d7b_110.tar.gz --opp starter pass                 # step 6
python entry_points/evaluate.py --a runs/d7/my_d7b_110.tar.gz \
    --b submission/rl_d7b_110.tar.gz submission/rl_d6_250.tar.gz agents/caretaker/main.py pass --seeds 256 --seed0 98300000
```

The export applies the 4th-quadrant rule (`configs/export_d7b.json`: `--land-inject 248:72`) and writes fp16 weights.
The `evaluate.py` command plays 1,024 games between two networks, at least ~2 core-minutes each
([README.md](../README.md#reproduce-the-selection-number)), plus 1,024 cheaper games against the code bot and the
empty farm; lower `--seeds` for a first look.

What a successful reproduction looks like:

- **Against the released `rl_d7b_110`:** P near 0.5. We did not run the recipe twice from the same BC checkpoint, so
  we cannot give a measured spread.
- **Against `rl_d6_250`:** about 0.6 (d7b at EMA 110 scored 0.621 over 512 seeds × 2 seats from 98,300,000).
- **Against the code bot and the empty farm:** wins in every game, and a bank against `pass` not far below the
  released agent's on the same seeds.
- `predict.py` with the new archive: every game `DONE`, about 90 ms per call on a server CPU core.

## Known differences between a rerun and the original

| Item | Reason | Effect |
|---|---|---|
| BC weights | GPU reduction order depends on GPU type, GPU count and library versions | byte-identical in our rerun with the pinned software on the same machine type; small differences elsewhere (not measured) |
| APPO weights | asynchronous collection, multi-process game workers | a statistically similar, not identical, policy |
| League | 99 bots and the snapshot pools are not in this repository; 22 bots are not public, and some notebooks have changed since | a different opponent mix (agents/README.md) |
| Tape worlds | rebuilt from Kaggle; a replay that is no longer available is skipped | smaller pools |
| Re-exported `main.py` and package | translated comments; the package name `rl_<sha8>` is a hash of the sources | same program, different bytes; `verify_submission.py` compares programs |
| Code bots | some iterate over sets of strings, so their actions can depend on the hash seed | every game process runs with `PYTHONHASHSEED=0` |
