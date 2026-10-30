# Third-party notices

Our own code, weights and documentation are licensed under CC BY 4.0 ([LICENSE](LICENSE)). The items below are
third-party work, or derived from it, and keep their own licenses.

## 1. kagsim (`vendor/kagsim/`): Apache License 2.0

- **What:** a bit-exact C++ implementation of the Kaggriculture engine (`kaggle-environments` 1.32.7) with a Python
  API, package `kagsim` 0.5.0.
- **Source:** https://github.com/destbreso/kaggriculture-cppsim (also published by its author as a Kaggle dataset).
  Its engine core is nikital7's C++ port from the Kaggle notebook "4000x environment speedup — Kaggriculture"
  (https://www.kaggle.com/code/nikital7/4000x-environment-speedup-kaggriculture), patched to engine 1.32.7 by the
  repository author. See `vendor/kagsim/README.md` ("Credit first").
- **License:** Apache License 2.0, full text in `vendor/kagsim/LICENSE`.
- **Changes:** none. The folder is the 0.5.0 source as obtained (compared file by file; build outputs removed).

## 2. Files derived from kagsim: Apache License 2.0

These files are our modifications of kagsim's source and are distributed under the Apache License 2.0
(`vendor/kagsim/LICENSE`). Each one states in its header what was changed relative to the original:

| File | Derived from | Changes (summary; full list in the file header) |
|---|---|---|
| `engines/kagsim_cuda/sim_hd.hpp` | `vendor/kagsim/sim/sim.hpp` | made compilable as CUDA device code (one GPU thread = one game); static tables turned into functions; FMA-free arithmetic helpers; behaviour unchanged (bit-exact, checked against the original) |
| `engines/kagsim_cuda/pyrandom_hd.hpp` | `vendor/kagsim/sim/pyrandom.hpp` | CPython-compatible Mersenne Twister made usable in device code; behaviour unchanged |
| `engines/kagsim_cuda/shim/sim.hpp` | — | a 3-line shim that maps the name `sim.hpp` to `sim_hd.hpp` |
| `engines/kagsim_cuda/cpu_batch.cpp` | `vendor/kagsim/python/kagsim.cpp` | the action converters (`conv_unit`, `conv_order`, `conv_action`, as `encode_action(s)`) and the observation/telemetry builders (`ser_tile`, `ser_farm`, `observe`, `telemetry`) ported to the `kaghd` namespace and the flat int32[170] action layout; the rest of the file (batch API, `RefBatch`) is our code |

`engines/kagcore/`, `engines/kagsim_cuda/` (other files) and `engines/kagrl_cuda/` are our code; they include or link
the files above. They also include kagsim files unmodified at build time: `engines/kagcore/py_kagcore.cpp` includes
`vendor/kagsim/python/kagsim.cpp` (with its module entry point renamed by a macro), and `engines/kagcore/kagcore.hpp`
(kagcore build) and the `RefBatch` reference in `engines/kagsim_cuda/cpu_batch.cpp` include
`vendor/kagsim/sim/sim.hpp`.

## 3. League bots (`agents/`)

Code bots used only as training opponents; none is part of the submission. Only our own `caretaker` and `pass_agent`
are in this repository (`caretaker` contains a market-flow table measured from two public Kaggle replays; see
agents/README.md). [agents/README.md](agents/README.md) lists every other bot of the league with its source
(public Kaggle notebook, or the public episode and seat of an embedded recording), its authors and the sha256 of its
`main.py`; `tools/fetch_league_bots.py` downloads the public ones from Kaggle.

- Public Kaggle notebooks: Kaggle shows public notebooks as released under the Apache 2.0 open-source license;
  license and notice files in a notebook's output are kept in the installed folder.
- Two bots (`pub0923_koshinm_0920`, `pub0923_koshinm_0921`) are taken from the public Kaggle dataset
  `destbreso/kaggriculture-donor-agents`, which records them as public-notebook code under Kaggle's default Apache 2.0
  license and asks for attribution on reuse; their original author is listed in agents/README.md.

## 4. Data (`data/`)

The pool manifests in `data/pools/` refer to recorded Kaggriculture episodes ([data/README.md](data/README.md)):

- Kaggle daily episode datasets `kaggle/kaggriculture-episodes-YYYY-MM-DD`: CC0-1.0.
- Episodes downloaded through Kaggle's public replay API: public competition replays. The manifests contain episode
  ids, seeds, shop orders, team names and leaderboard ranks; the recorded actions are not redistributed here.

## 5. Python packages

Installed from PyPI by `requirements*.txt`, not shipped: NumPy (BSD-3-Clause), PyTorch (BSD-3-Clause),
kaggle-environments (Apache-2.0), pybind11 (BSD-3-Clause), setuptools (MIT), matplotlib (PSF-based, for
`docs/make_figures.py`), kaggle CLI (Apache-2.0, optional).
