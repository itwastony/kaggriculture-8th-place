#!/usr/bin/env bash
# Build the CPU module kaghd_cpu (sim_hd.hpp + the vendor sim.hpp reference) into build/.
# Usage: bash build_cpu.sh [path_to_python]   (default: python3 on PATH, as in scripts/build_engines.sh)
set -euo pipefail
cd "$(dirname "$0")"
PY="${1:-$(command -v python3)}"
mkdir -p build
INC="$("$PY" -m pybind11 --includes)"
SUF="$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
EXTRA=""
if [ "$(uname)" = "Darwin" ]; then EXTRA="-undefined dynamic_lookup"; fi
CXX="${CXX:-c++}"
set -x
$CXX -O3 -std=c++17 -ffp-contract=off -Wall -Wextra -Wno-unused-parameter -fPIC -shared $EXTRA $INC \
    cpu_batch.cpp -o "build/kaghd_cpu$SUF"
