#!/usr/bin/env bash
# Builds the kagcore module into engines/kagcore/build/ (nothing is installed into .venv) + checks the device code with clang.
#   bash engines/kagcore/build.sh            # module
#   PY=/path/to/python bash build.sh     # another interpreter (default <root>/.venv/bin/python)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
CXX="${CXX:-c++}"
mkdir -p "$HERE/build"
EXT="$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
INC="$("$PY" -m pybind11 --includes)"
LDFLAGS=""
[ "$(uname)" = "Darwin" ] && LDFLAGS="-undefined dynamic_lookup"
# -ffp-contract=off: no FMA contraction (bit-exact identity with NumPy)
"$CXX" -O3 -std=c++17 -ffp-contract=off -fvisibility=hidden -fPIC -shared $LDFLAGS $INC \
    -I"$ROOT/vendor/kagsim/sim" "$HERE/py_kagcore.cpp" -o "$HERE/build/kagcore$EXT"
echo "OK: $HERE/build/kagcore$EXT"

# Device check: clang in CUDA front-end mode (without nvcc/CUDA SDK) - catches calls to host functions from the core.
if [ "${SKIP_CUDA_CHECK:-0}" != "1" ] && clang++ -x cuda --cuda-device-only -nocudainc -nocudalib \
        --cuda-gpu-arch=sm_80 -fsyntax-only -std=c++17 /dev/null >/dev/null 2>&1; then
    clang++ -x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=sm_80 -fsyntax-only -std=c++17 \
        -Wall -I"$ROOT/vendor/kagsim/sim" "$HERE/cuda_check.cu"
    echo "OK: device check cuda_check.cu (clang, sm_80)"
fi
