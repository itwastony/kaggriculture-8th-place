// EMULATION of cuda_runtime.h on the CPU - for the CPU build of kagrl.cu with a regular C++ compiler (build_cpu.sh,
// -DKAG_EMULATE): cudaMalloc = malloc, cudaMemcpy* = memcpy, kernel launch = loop over blocks and threads
// (KAG_LAUNCH in kagrl.cu), streams/events are stubs. Copy of engines/kagsim_cuda/check/emu/cuda_runtime.h,
// extended with the functions that kagrl.cu uses. Not used by the real nvcc build.
#pragma once
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>

#define __host__
#define __device__
#define __global__
#define __constant__

struct uint3 { unsigned x, y, z; };
struct dim3 {
    unsigned x, y, z;
    dim3(unsigned vx = 1, unsigned vy = 1, unsigned vz = 1) : x(vx), y(vy), z(vz) {}
};
namespace kag_emu {
inline uint3 g_thread{0, 0, 0}, g_block{0, 0, 0}, g_bdim{1, 1, 1}, g_gdim{1, 1, 1};
inline void launch(dim3 grid, dim3 block, const std::function<void()>& body) {
    g_gdim = {grid.x, 1, 1};
    g_bdim = {block.x, 1, 1};
    for (unsigned b = 0; b < grid.x; ++b)
        for (unsigned t = 0; t < block.x; ++t) {
            g_block = {b, 0, 0};
            g_thread = {t, 0, 0};
            body();
        }
}
}  // namespace kag_emu
#define threadIdx (kag_emu::g_thread)
#define blockIdx (kag_emu::g_block)
#define blockDim (kag_emu::g_bdim)
#define gridDim (kag_emu::g_gdim)

typedef struct CUstream_st* cudaStream_t;
typedef std::chrono::steady_clock::time_point* cudaEvent_t;
enum cudaError { cudaSuccess = 0, cudaErrorMemoryAllocation = 2 };
typedef enum cudaError cudaError_t;
enum cudaMemcpyKind { cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2,
                      cudaMemcpyDeviceToDevice = 3, cudaMemcpyDefault = 4 };

inline cudaError_t cudaMalloc(void** p, size_t n) { *p = std::malloc(n ? n : 1); return *p ? cudaSuccess : cudaErrorMemoryAllocation; }
template <class T> inline cudaError_t cudaMalloc(T** p, size_t n) { return cudaMalloc((void**)p, n); }
inline cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaMemset(void* p, int v, size_t n) { std::memset(p, v, n); return cudaSuccess; }
inline cudaError_t cudaMemsetAsync(void* p, int v, size_t n, cudaStream_t = nullptr) { std::memset(p, v, n); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) { std::memcpy(d, s, n); return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind, cudaStream_t = nullptr) {
    std::memcpy(d, s, n); return cudaSuccess;
}
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaGetDevice(int* d) { *d = 0; return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline const char* cudaGetErrorName(cudaError_t) { return "emuError"; }
inline const char* cudaGetErrorString(cudaError_t) { return "emulated error"; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaEventCreate(cudaEvent_t* e) { *e = new std::chrono::steady_clock::time_point(); return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t = nullptr) { *e = std::chrono::steady_clock::now(); return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b) {
    *ms = std::chrono::duration<float, std::milli>(*b - *a).count(); return cudaSuccess;
}
inline cudaError_t cudaEventDestroy(cudaEvent_t e) { delete e; return cudaSuccess; }
