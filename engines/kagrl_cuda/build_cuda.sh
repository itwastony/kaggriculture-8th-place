#!/usr/bin/env bash
# GPU build of build/libkagrl_cuda.so (C API kagrl.cuh, ctypes: kagrl.py). Requires nvcc (we used CUDA 13.3).
# -fmad=false: no FMA contraction on the device (kagcore encode
# and the engine compute like CPython/NumPy - each operation is rounded separately); no --use_fast_math.
# Usage: bash build_cuda.sh [arch]   (default native = the GPU of this machine; or pass an architecture, e.g. sm_80); LDFLAGS_EXTRA="-Xcompiler -fuse-ld=gold"
set -euo pipefail
cd "$(dirname "$0")"
ARCH="${1:-native}"
mkdir -p build
INC="-I. -I../kagsim_cuda/shim -I../kagsim_cuda -I../kagcore"
set -x
nvcc -O3 -std=c++17 -arch="$ARCH" -fmad=false --expt-relaxed-constexpr -lineinfo -Xptxas -v $INC \
     -Xcompiler -fPIC,-O3,-ffp-contract=off ${LDFLAGS_EXTRA:-} \
     -shared kagrl.cu -o build/libkagrl_cuda.so
