// STUB cuda_runtime.h - ONLY for static checking of kagrl.cu with clang in CUDA mode (-fsyntax-only)
// on a machine without the CUDA SDK (check/check_cuda_syntax.sh). Copy of engines/kagsim_cuda/check/cuda_stub/cuda_runtime.h,
// extended with what kagrl.cu uses (log1p on the device, streams, async copies). Not used by the nvcc build.
#pragma once
#include <stddef.h>
#ifndef __CUDACC__
#define __CUDACC__ 1
#endif
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __constant__ __attribute__((constant))
#define __shared__ __attribute__((shared))
#include "__clang_cuda_builtin_vars.h"

struct dim3 {
    unsigned x, y, z;
    __host__ __device__ dim3(unsigned vx = 1, unsigned vy = 1, unsigned vz = 1) : x(vx), y(vy), z(vz) {}
};
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
enum cudaError { cudaSuccess = 0, cudaErrorUnknown = 999 };
typedef enum cudaError cudaError_t;
enum cudaMemcpyKind { cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2,
                      cudaMemcpyDeviceToDevice = 3, cudaMemcpyDefault = 4 };

struct cudaFuncAttributes {
    size_t sharedSizeBytes, constSizeBytes, localSizeBytes;
    int maxThreadsPerBlock, numRegs, ptxVersion, binaryVersion;
};
extern "C" {
cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes* attr, const void* func);
cudaError_t cudaMalloc(void** p, size_t n);
cudaError_t cudaFree(void* p);
cudaError_t cudaMemset(void* p, int v, size_t n);
cudaError_t cudaMemcpy(void* d, const void* s, size_t n, enum cudaMemcpyKind k);
cudaError_t cudaSetDevice(int d);
cudaError_t cudaGetLastError(void);
const char* cudaGetErrorName(cudaError_t e);
const char* cudaGetErrorString(cudaError_t e);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaStreamSynchronize(cudaStream_t s);
unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, void* stream = 0);
cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = 0);
cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
}
template <class T> inline cudaError_t cudaMalloc(T** p, size_t n) { return cudaMalloc((void**)p, n); }

// Math: as in the clang wrapper __clang_cuda_math.h - device overloads with the same signature are added
// to the libm host functions. Bodies are stubs (the check is only syntactic/semantic).
#include <math.h>
static __device__ inline double sqrt(double x) { return x; }
static __device__ inline double log(double x) { return x; }
static __device__ inline double log10(double x) { return x; }
static __device__ inline double log1p(double x) { return x; }
static __device__ inline double nearbyint(double x) { return x; }

// Rounding intrinsics (in CUDA - device_functions.h).
__device__ double __dmul_rn(double, double);
__device__ double __dadd_rn(double, double);
__device__ double __dsub_rn(double, double);
