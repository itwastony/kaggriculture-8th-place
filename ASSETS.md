# Release assets: checkpoints and submission archives

The weights and the two submitted archives are published as one release asset, `kaggriculture-public-assets.tar.gz`
(666 MB; 714 MB unpacked), not in git. It uses the repository layout (`submission/`, `artifacts/checkpoints/`; see
`SETTINGS.json`). Unpack it into the repository root and check the files:

```bash
tar -xzf kaggriculture-public-assets.tar.gz
shasum -a 256 -c ASSETS.sha256      # or: sha256sum -c ASSETS.sha256
```

Release: <https://github.com/itwastony/kaggriculture-8th-place/releases/tag/v1.0>

## Submission archives → `submission/`

| File | Size | Kaggle submission | Submitted | sha256 |
|---|---:|---|---|---|
| `rl_d7b_110.tar.gz` | 30.5 MB | 56718362 | 2026-09-30 | `bef80bb13c1d93b0279cc4e41b0c08b4f8cdc517eba0d2ea50c0a1820ed34b0e` |
| `rl_d6_250.tar.gz` | 30.5 MB | 56713726 | 2026-09-30 | `d89d96eb42cee43eee12dbf161a3284d3f4be78210c08e1a858041c8b00108bc` |

These are the archives uploaded to Kaggle, repacked for publication: a neutral owner (`itwastony`) and a fixed
timestamp (2026-10-01 00:00 UTC) in the tar headers, gzip timestamp 0. Member names, order, permissions and file
contents are unchanged. The uploaded files had sha256
`29d5f12557846ec005fa70f90f96bebb3895a34eb216bc306fe4f4d37497f0ca` (`rl_d7b_110.tar.gz`) and
`73dbc480c6084ea8ba62b6f5ead863b59ed9b15f86051dfb2cdfb990dcb0a195` (`rl_d6_250.tar.gz`). Each contains:

- `main.py`: `agent(observation, configuration)`, NumPy only. Line 1 is a comment naming the checkpoint. It applies
  the 4th-quadrant rule (`--land-inject 248:72`, [docs/method.md](docs/method.md)); its other inference options are
  switched off in both archives.
- `weights.npz`: 124 arrays: 122 float16 (16,320,409 actor parameters plus a value-head MLP that inference does not
  use) and 2 int64 metadata arrays.
- `rl_0c5fe4ba/`: `__init__.py`, `spec.py`, `encode.py`, `act.py`, `np_net.py`, copied from `rl/` at export time;
  the folder name is a hash of their source.
- Comments and docstrings of `main.py` and `rl_0c5fe4ba/` are in Russian, as submitted. `rl_0c5fe4ba/` is the code
  of `rl/{spec,encode,act,np_net}.py` in this repository (identical apart from comments and docstrings), which has
  English comments.

| Archive | `main.py` sha256 (first 8) | `weights.npz` sha256 (first 8) |
|---|---|---|
| `rl_d7b_110.tar.gz` | `8b0d8f57` | `5b154ec0` |
| `rl_d6_250.tar.gz` | `dc1b8bbb` | `e360529d` |

`entry_points/verify_submission.py` shows that both archives follow from the checkpoints below and the code in this
repository. Re-exports made by `entry_points/export_submission.py` are not byte-identical (see
[docs/code_map.md](docs/code_map.md#differences-from-the-competition-time-code)).

## Checkpoints → `artifacts/checkpoints/`

PyTorch files (`torch.save` of a dict). `model` is the state dict; `d` / `layers` give the trunk size (384 / 8). RL
checkpoints also contain the value tower (`vtower`, 2 blocks) and the privileged critic (`pcritic`, 2 blocks); the
submission uses only the actor.

| File | Size | What it is | Role in the final run | sha256 |
|---|---:|---|---|---|
| `final/d7b_ema_00110.pt` | 87 MB | APPO run d7b, EMA (τ 0.999) after 110 iterations (0–109) | **exported into `rl_d7b_110.tar.gz`** | `4b9888e0405f0e07570dbd81c8f75547e85e2d8501e4ae96ca9521ae88dd3af5` |
| `final/bc_d7_ckpt_003000.pt` | 67 MB | BC of line d7 after 3,000 steps | initial policy, first KL anchor and a past opponent of d7b | `40d53ea9ba8c836d22c85649fd721d38ddd3bf4fb3d78f951caedeadeddf19ab` |
| `final/bc_d7_init_d6_140_pol.pt` | 67 MB | actor of `past/d6_ema_00140.pt` without the value towers | initial weights of BC d7 | `5cff7682aeebd2687c3c57020b9370101b51dc9db245ff0cc641172e98254283` |
| `second/d6_ema_00250.pt` | 87 MB | APPO line d6, EMA after 250 iterations | **exported into `rl_d6_250.tar.gz`** | `1794d476b9fe51be83ecf161aab55e520b47812fa5716ebd54239e5b29469288` |
| `past/d6_ema_00140.pt` | 87 MB | line d6, EMA 140 | past opponent of d7b; source of the BC d7 initialisation | `a1072056680a0acffb38d362ee1663d5a34843568c53ecf3c1390f14c4d8ded5` |
| `past/d4a_ema_00370.pt` | 87 MB | line d4, run d4a, EMA 370 | past opponent of d7b | `cea020d71f937126c5aff9ca0886f7adbbda06fc020c9d03ae9ae5dac105c4cd` |
| `past/s100_180.pt` | 87 MB | equal-weight average of 9 EMA snapshots 100…180 of run d1b | past opponent of d7b | `fd1a737bc0f6e931adb80e8519e32f64a1d391e77cb904fd8c600a4c81b3ee68` |
| `past/d1b_ckpt_00150.pt` | 87 MB | run d1b after 150 iterations (model only) | past opponent of d7b | `6410ce5e722ffd2cd3d7ce4d14062b56eaa15897dbed39738423fe25293826af` |

Changes against the files used in the run (tensors untouched, compared tensor by tensor):

- `past/d1b_ckpt_00150.pt`: the optimizer state was removed (the training code reads only `model` from past
  opponents).
- `final/bc_d7_init_d6_140_pol.pt` (`src`) and `past/s100_180.pt` (`soup_of`): absolute paths of the training machine
  in the metadata were replaced by file names.

The lineage of these models (lines r1 … d7) is in [docs/method.md](docs/method.md).

## Not published

- **Snapshot pools of three earlier lines** (`SNAPSHOT_POOL_DIR/{d1,d4,d6}/v*.pt`): the d7b league
  samples part of its "past" opponents from them. `entry_points/train_appo.py --no-pool-src` trains without them; the
  past opponents are then the run's own snapshots plus the five fixed checkpoints (the BC checkpoint and the four in
  `artifacts/checkpoints/past/`; a different opponent mix).
- **Tape worlds and most league bots**: see [data/README.md](data/README.md) and [agents/README.md](agents/README.md).
