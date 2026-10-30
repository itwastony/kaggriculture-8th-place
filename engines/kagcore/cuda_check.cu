// cuda_check.cu - check that encode/execute compile as device code.
// Two modes:
//   clang (no CUDA SDK):   clang++ -x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=sm_80
//                          -fsyntax-only -std=c++17 -I<root>/vendor/kagsim/sim cuda_check.cu
//   nvcc (on a machine with CUDA): nvcc -std=c++17 -arch=sm_80 -fmad=false -I<root>/vendor/kagsim/sim -c cuda_check.cu
// Kernel with one thread per (state, seat) - a template for batched encode/execute on GPU.
#if !defined(__NVCC__)
// clang without CUDA headers: attributes and a device declaration of log1p (with nvcc they come from cuda_runtime/math_functions)
#  define __host__ __attribute__((host))
#  define __device__ __attribute__((device))
#  define __global__ __attribute__((global))
__device__ double log1p(double);
#endif
#include "../kagcore/kagcore.hpp"

__global__ void kag_encode_kernel(const kag::State* states, const int* seats, int n, int tid,
                                  float* tiles, float* units, int64_t* unit_pos, bool* unit_mask, float* glob,
                                  uint8_t* tmask, bool* mmask) {
    // tid - in a real kernel blockIdx.x * blockDim.x + threadIdx.x (a parameter here: clang without CUDA headers)
    using namespace kagcore;
    const int i = tid;
    if (i >= n) return;
    encode(states[i], seats[i], tiles + i * NT * FT, units + i * MAXU * FU, unit_pos + i * MAXU,
           unit_mask + i * MAXU, glob + i * FG, tmask + i * MAXU * TMASK_BYTES, mmask + i * MB);
}

__global__ void kag_execute_kernel(const kag::State* states, const int* seats, int n, int tid,
                                   const int32_t* unit_choice, const int32_t* market_choice, kag::Action* out) {
    const int i = tid;
    if (i >= n) return;
    kagcore::execute(states[i], seats[i], unit_choice + i * kagcore::MAXU, market_choice + i * kagcore::NHEADS,
                     out[i]);
}
