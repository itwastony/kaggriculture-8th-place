#!/usr/bin/env bash
# Build the game engines used by this repository. Run from the repository root with the target Python environment active.
#
#   bash scripts/build_engines.sh                 # kagsim (pip install) + kagcore (C++ module)          - CPU only
#   bash scripts/build_engines.sh --gpu           # + kagrl GPU library engines/kagrl_cuda/build/libkagrl_cuda.so (needs nvcc)
#   bash scripts/build_engines.sh --check         # + self-tests: kagsim golden traces / observation lockstep, kagcore parity
#   PY=/path/to/python GPU_ARCH=sm_80 bash scripts/build_engines.sh --gpu --check
#
# What gets built and why:
#   vendor/kagsim        kagsim 0.5.0 (Apache-2.0, third party): bit-exact C++ port of the Kaggriculture engine 1.32.7
#                        with a Python API. Used for every CPU game: BC demonstrations, tape checks, evaluation, league.
#   engines/kagcore      our C++ module: kagsim's engine + the network's observation encoder and action executor
#                        (bit-exact with rl/encode.py and rl/act.py). Used by rl/appo.py (--kagcore 1), rl/tape_check.py.
#   engines/kagrl_cuda   our CUDA library: engine + encoder + executor for thousands of network-vs-network games on one
#                        GPU (rl/appo.py --gpu-games). Training only; never needed for verification or inference.
#
# Side effect: kagcore is made importable by writing one file, <site-packages>/kaggriculture_kagcore.pth, that adds
# engines/kagcore/build to sys.path. Remove that file to undo. Nothing else outside this repository is changed
# (apart from the pip install of kagsim into the active environment).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-$(command -v python3)}"
GPU=0
CHECK=0
for a in "$@"; do
  case "$a" in
    --gpu) GPU=1 ;;
    --check) CHECK=1 ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done
echo "python: $PY ($("$PY" -c 'import sys; print(sys.version.split()[0])'))"

echo "== 1/3 kagsim (vendor/kagsim) =="
# If the build fails at link time with a broken system ld, try: LDFLAGS=-fuse-ld=gold bash scripts/build_engines.sh
"$PY" -m pip install --no-deps --no-build-isolation ./vendor/kagsim   # setuptools + pybind11 come from requirements.txt
"$PY" -c 'import kagsim; print("kagsim OK", getattr(kagsim, "__file__", ""))'

echo "== 2/3 kagcore (engines/kagcore) =="
PY="$PY" SKIP_CUDA_CHECK="${SKIP_CUDA_CHECK:-1}" bash engines/kagcore/build.sh
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$ROOT/engines/kagcore/build" > "$SITE/kaggriculture_kagcore.pth"
"$PY" -c 'import kagcore; print("kagcore OK", kagcore.__file__)'

if [ "$GPU" = 1 ]; then
  echo "== 3/3 kagrl GPU library (engines/kagrl_cuda) =="
  command -v nvcc >/dev/null || { echo "nvcc not found: add the CUDA toolkit bin directory to PATH" >&2; exit 1; }
  (cd engines/kagrl_cuda && bash build_cuda.sh "${GPU_ARCH:-native}")
  ls -l engines/kagrl_cuda/build/libkagrl_cuda.so
else
  echo "== 3/3 kagrl GPU library: skipped (pass --gpu; needed only for training) =="
fi

if [ "$CHECK" = 1 ]; then
  echo "== self-tests =="
  "$PY" vendor/kagsim/tests/test_golden.py          # recorded 1.32.7 episodes replay to the same banks
  "$PY" vendor/kagsim/tests/test_l1.py              # kagsim observations == kaggle-environments observations, every turn
  "$PY" engines/kagcore/parity.py --quick           # kagcore encode/execute == rl/encode.py, rl/act.py on kagsim games (agents: caretaker, pass_agent)
fi
echo "engines OK"
