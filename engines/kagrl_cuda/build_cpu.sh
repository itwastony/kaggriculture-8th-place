#!/usr/bin/env bash
# CPU build of kagrl.cu (the same source as for the GPU) with a regular C++ compiler: emulated CUDA runtime
# (emu/cuda_runtime.h), kernels = loops over threads. Output: build/libkagrl_cpu.so with the same C API (ctypes, kagrl.py).
# Usage: bash build_cpu.sh            (CXX=... - a different compiler, e.g. "g++ -fuse-ld=gold" if the default linker fails)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
INC="-I. -I../kagsim_cuda/shim -I../kagsim_cuda -I../kagcore"
set -x
${CXX:-c++} -x c++ -std=c++17 -O2 -ffp-contract=off -DKAG_EMULATE -Iemu $INC -Wall -Wextra -Wno-unused-parameter \
    -fPIC -shared kagrl.cu -o build/libkagrl_cpu.so
