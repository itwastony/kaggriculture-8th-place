#!/usr/bin/env bash
# Check build of batch.cu WITHOUT CUDA: a regular C++ compiler, runtime emulation
# (check/emu/cuda_runtime.h), kernels are loops on the CPU. Output: build/libkagsim_emu.so
# with the same C API, for running kagsim_cuda.GpuBatch / gpu_parity.py --lib on a machine without a GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
set -x
${CXX:-c++} -x c++ -std=c++17 -O2 -ffp-contract=off -DKAG_EMULATE -Icheck/emu -Wall -Wextra -Wno-unused-parameter \
    -fPIC -shared batch.cu -o build/libkagsim_emu.so
