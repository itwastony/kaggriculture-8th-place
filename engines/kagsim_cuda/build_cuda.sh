#!/usr/bin/env bash
# Build the GPU library build/libkagsim_cuda.so (C API from batch.cuh, called
# from Python via ctypes: kagsim_cuda.py). Requires nvcc from CUDA 12.x.
# Usage: bash build_cuda.sh [arch]   (default native = the GPU of this machine; or pass an architecture, e.g. sm_80)
set -euo pipefail
cd "$(dirname "$0")"
ARCH="${1:-native}"
mkdir -p build
set -x
nvcc -O3 -std=c++17 -arch="$ARCH" --expt-relaxed-constexpr -lineinfo -Xptxas -v \
     -Xcompiler -fPIC,-O3,-ffp-contract=off \
     -shared batch.cu -o build/libkagsim_cuda.so
