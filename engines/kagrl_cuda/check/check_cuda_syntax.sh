#!/usr/bin/env bash
# Static check of kagrl.cu WITHOUT the CUDA SDK: clang in CUDA mode, -fsyntax-only, separate device pass (sm_80)
# and host pass, with the stub check/cuda_stub/cuda_runtime.h. Catches host-function calls from device code,
# unannotated functions, exceptions/RTTI in kernels. Not a replacement for nvcc (different frontend), but a check before a GPU build.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMON="-x cuda -std=c++17 -nocudainc -nocudalib --cuda-gpu-arch=sm_80 -D__CUDACC__ -Icheck/cuda_stub -I. \
 -I../kagsim_cuda/shim -I../kagsim_cuda -I../kagcore -Wall -Wextra -Wno-unused-parameter -fsyntax-only"
set -x
clang++ $COMMON --cuda-device-only kagrl.cu
clang++ $COMMON --cuda-host-only kagrl.cu
