#!/usr/bin/env bash
# Static check of batch.cu WITHOUT the CUDA SDK: clang in CUDA mode, -fsyntax-only,
# a separate device pass (sm_80) and host pass, with a cuda_runtime.h stub.
# Catches the main class of device-code errors: calling host functions/variables from
# __device__ code, unannotated functions, exceptions/RTTI in device code.
# This does not replace nvcc (different frontend), but it is a strong check before a GPU build.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMON="-x cuda -std=c++17 -nocudainc -nocudalib --cuda-gpu-arch=sm_80 -D__CUDACC__ -Icheck/cuda_stub -Wall -Wextra -Wno-unused-parameter -fsyntax-only"
set -x
clang++ $COMMON --cuda-device-only batch.cu
clang++ $COMMON --cuda-host-only batch.cu
