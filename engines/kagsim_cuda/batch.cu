// GPU implementation of the batch API (batch.cuh): an array of states of N games in
// device memory, one thread = one game. The game core is sim_hd.hpp, the logic
// of a single game (reset_one/step_one/decode_action) is batch_common.hpp; exactly
// the same code is built for the CPU in cpu_batch.cpp and checked against the original kagsim.
//
// Build (CUDA 12.x or later), see build_cuda.sh:
//   nvcc -O3 -std=c++17 -arch=<arch> --expt-relaxed-constexpr \
//        -Xcompiler -fPIC,-O3,-ffp-contract=off -shared batch.cu -o build/libkagsim_cuda.so
#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

#include "sim_hd.hpp"
#include "batch_common.hpp"
#include "state_visit.hpp"
#include "batch.cuh"

using kaghd::Sim;
using kaghd::Action;
using kaghd::Config;
using kaghd::ACT_I32;

static_assert(ACT_I32 == KAG_ACT_I32, "ACT_I32 mismatch");

// Kernel launch. Under nvcc: the usual <<<>>>; under KAG_EMULATE (check build
// check/build_emu.sh with a regular C++ compiler): a loop over blocks/threads on the CPU.
#if defined(KAG_EMULATE)
#define KAG_LAUNCH(kernel, grid, block, stream, ...) \
    kag_emu::launch((grid), (block), [&]() { kernel(__VA_ARGS__); })
#else
#define KAG_LAUNCH(kernel, grid, block, stream, ...) \
    kernel<<<(grid), (block), 0, (cudaStream_t)(stream)>>>(__VA_ARGS__)
#endif

namespace {

constexpr int kBlock = 128;

thread_local std::string g_err;

int fail(const char* where, cudaError_t e) {
    g_err = std::string(where) + ": " + cudaGetErrorName(e) + " — " + cudaGetErrorString(e);
    return e == cudaSuccess ? -1 : static_cast<int>(e);
}
int fail_msg(const std::string& msg) { g_err = msg; return -1; }

#define KAG_CK(call, where)                                   \
    do {                                                      \
        cudaError_t e__ = (call);                             \
        if (e__ != cudaSuccess) return fail((where), e__);    \
    } while (0)

inline int grid_for(int n) { return (n + kBlock - 1) / kBlock; }

// ------------------------------------------------------------------ kernels
__global__ void k_reset(Sim* sims, int n, Config base, const uint64_t* seeds,
                        const uint8_t* mask, const uint8_t* shops, const int32_t* nshops) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (mask && !mask[i]) return;
    int ns = nshops ? nshops[i] : 0;
    kaghd::reset_one(sims[i], base, seeds[i], shops ? shops + (size_t)i * 8 : nullptr, ns);
}

__global__ void k_step(Sim* sims, int n, const int32_t* acts, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int32_t* row = acts + (size_t)(i % m) * 2 * ACT_I32;
    kaghd::step_one<Sim, Action>(sims[i], row, row + ACT_I32);
}

__global__ void k_rewards(const Sim* sims, int n, double* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[2 * i] = sims[i].reward(0);
    out[2 * i + 1] = sims[i].reward(1);
}

__global__ void k_done(const Sim* sims, int n, uint8_t* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = sims[i].st.done ? 1 : 0;
}

__global__ void k_prices(int lo, int count, int32_t* out) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)kaghd::N_PRODUCTS * count;
    if (idx >= total) return;
    int item = (int)(idx / count);
    int v = lo + (int)(idx % count);
    out[idx] = kaghd::market_price(item, v);
}

}  // namespace

// ------------------------------------------------------------------ batch
struct KagBatch {
    int n = 0;
    int device = 0;
    Config base;
    Sim* d_sims = nullptr;
    // temporary buffers for the host variants
    int32_t* d_acts = nullptr;   size_t acts_cap = 0;   // in int32
    uint64_t* d_seeds = nullptr;
    uint8_t* d_mask = nullptr;
    uint8_t* d_shops = nullptr;
    int32_t* d_nshops = nullptr;
    double* d_rew = nullptr;
    uint8_t* d_done = nullptr;
    // uploaded action streams [T][m][2][ACT_I32]
    int32_t* d_up = nullptr; int up_T = 0; int up_m = 0;
};

static int ensure_acts(KagBatch* b, size_t count) {
    if (count <= b->acts_cap) return 0;
    if (b->d_acts) cudaFree(b->d_acts);
    b->d_acts = nullptr; b->acts_cap = 0;
    KAG_CK(cudaMalloc(&b->d_acts, count * sizeof(int32_t)), "cudaMalloc(acts)");
    b->acts_cap = count;
    return 0;
}

extern "C" {

const char* kag_last_error(void) { return g_err.c_str(); }
int kag_act_i32(void) { return ACT_I32; }
int kag_ser_bytes(void) { return (int)kaghd::state_ser_bytes<kaghd::State>(); }
int kag_sim_bytes(void) { return (int)sizeof(Sim); }
int kag_amps_selfcheck(void) { return kaghd::amps_selfcheck(); }

KagBatch* kag_create(int n, int episode_steps, int device) {
    if (n <= 0) { fail_msg("kag_create: n must be > 0"); return nullptr; }
    cudaError_t e = cudaSetDevice(device);
    if (e != cudaSuccess) { fail("cudaSetDevice", e); return nullptr; }
    KagBatch* b = new KagBatch();
    b->n = n;
    b->device = device;
    b->base = Config{};
    b->base.episode_steps = episode_steps;
    size_t sz = (size_t)n * sizeof(Sim);
    if ((e = cudaMalloc(&b->d_sims, sz)) != cudaSuccess ||
        (e = cudaMemset(b->d_sims, 0, sz)) != cudaSuccess ||
        (e = cudaMalloc(&b->d_seeds, (size_t)n * sizeof(uint64_t))) != cudaSuccess ||
        (e = cudaMalloc(&b->d_mask, (size_t)n)) != cudaSuccess ||
        (e = cudaMalloc(&b->d_shops, (size_t)n * 8)) != cudaSuccess ||
        (e = cudaMalloc(&b->d_nshops, (size_t)n * sizeof(int32_t))) != cudaSuccess ||
        (e = cudaMalloc(&b->d_rew, (size_t)n * 2 * sizeof(double))) != cudaSuccess ||
        (e = cudaMalloc(&b->d_done, (size_t)n)) != cudaSuccess) {
        fail("kag_create: cudaMalloc", e);
        kag_destroy(b);
        return nullptr;
    }
    return b;
}

void kag_destroy(KagBatch* b) {
    if (!b) return;
    cudaFree(b->d_sims); cudaFree(b->d_acts); cudaFree(b->d_seeds); cudaFree(b->d_mask);
    cudaFree(b->d_shops); cudaFree(b->d_nshops); cudaFree(b->d_rew); cudaFree(b->d_done);
    cudaFree(b->d_up);
    delete b;
}

int kag_n(const KagBatch* b) { return b ? b->n : 0; }
void* kag_sims_dev_ptr(KagBatch* b) { return b ? (void*)b->d_sims : nullptr; }

int kag_reset_dev(KagBatch* b, const uint64_t* seeds, const uint8_t* mask,
                  const uint8_t* shops, const int32_t* nshops, void* stream) {
    if (!b || !seeds) return fail_msg("kag_reset_dev: null batch/seeds");
    if ((shops == nullptr) != (nshops == nullptr)) return fail_msg("kag_reset_dev: shops and nshops go together");
    KAG_LAUNCH(k_reset, grid_for(b->n), kBlock, stream, b->d_sims, b->n, b->base, seeds, mask, shops, nshops);
    KAG_CK(cudaGetLastError(), "k_reset launch");
    return 0;
}

int kag_reset_host(KagBatch* b, const uint64_t* seeds, const uint8_t* mask,
                   const uint8_t* shops, const int32_t* nshops) {
    if (!b || !seeds) return fail_msg("kag_reset_host: null batch/seeds");
    if ((shops == nullptr) != (nshops == nullptr)) return fail_msg("kag_reset_host: shops and nshops go together");
    const size_t n = (size_t)b->n;
    if (shops) {   // same check as in the CPU module: id 0..7, count 0..8
        for (size_t i = 0; i < n; ++i) {
            if (nshops[i] < 0 || nshops[i] > 8) return fail_msg("kag_reset_host: nshops out of range 0..8");
            for (int k = 0; k < nshops[i]; ++k)
                if (shops[i * 8 + k] >= 8) return fail_msg("kag_reset_host: shop id out of range 0..7");
        }
    }
    KAG_CK(cudaMemcpy(b->d_seeds, seeds, n * sizeof(uint64_t), cudaMemcpyHostToDevice), "memcpy seeds");
    if (mask) KAG_CK(cudaMemcpy(b->d_mask, mask, n, cudaMemcpyHostToDevice), "memcpy mask");
    if (shops) {
        KAG_CK(cudaMemcpy(b->d_shops, shops, n * 8, cudaMemcpyHostToDevice), "memcpy shops");
        KAG_CK(cudaMemcpy(b->d_nshops, nshops, n * sizeof(int32_t), cudaMemcpyHostToDevice), "memcpy nshops");
    }
    KAG_LAUNCH(k_reset, grid_for(b->n), kBlock, nullptr, b->d_sims, b->n, b->base, b->d_seeds,
               mask ? b->d_mask : nullptr,
               shops ? b->d_shops : nullptr,
               shops ? b->d_nshops : nullptr);
    KAG_CK(cudaGetLastError(), "k_reset launch");
    KAG_CK(cudaDeviceSynchronize(), "k_reset");
    return 0;
}

int kag_step_dev(KagBatch* b, const int32_t* acts, int m, void* stream) {
    if (!b || !acts || m < 1) return fail_msg("kag_step_dev: null batch/acts or m < 1");
    KAG_LAUNCH(k_step, grid_for(b->n), kBlock, stream, b->d_sims, b->n, acts, m);
    KAG_CK(cudaGetLastError(), "k_step launch");
    return 0;
}

int kag_step_host(KagBatch* b, const int32_t* acts, int m) {
    if (!b || !acts || m < 1) return fail_msg("kag_step_host: null batch/acts or m < 1");
    size_t cnt = (size_t)m * 2 * ACT_I32;
    int rc = ensure_acts(b, cnt);
    if (rc) return rc;
    KAG_CK(cudaMemcpy(b->d_acts, acts, cnt * sizeof(int32_t), cudaMemcpyHostToDevice), "memcpy acts");
    KAG_LAUNCH(k_step, grid_for(b->n), kBlock, nullptr, b->d_sims, b->n, b->d_acts, m);
    KAG_CK(cudaGetLastError(), "k_step launch");
    KAG_CK(cudaDeviceSynchronize(), "k_step");
    return 0;
}

int kag_rewards_dev(KagBatch* b, double* out, void* stream) {
    if (!b || !out) return fail_msg("kag_rewards_dev: null");
    KAG_LAUNCH(k_rewards, grid_for(b->n), kBlock, stream, b->d_sims, b->n, out);
    KAG_CK(cudaGetLastError(), "k_rewards launch");
    return 0;
}

int kag_rewards_host(KagBatch* b, double* out) {
    if (!b || !out) return fail_msg("kag_rewards_host: null");
    KAG_LAUNCH(k_rewards, grid_for(b->n), kBlock, nullptr, b->d_sims, b->n, b->d_rew);
    KAG_CK(cudaGetLastError(), "k_rewards launch");
    KAG_CK(cudaMemcpy(out, b->d_rew, (size_t)b->n * 2 * sizeof(double), cudaMemcpyDeviceToHost), "memcpy rewards");
    return 0;
}

int kag_done_dev(KagBatch* b, uint8_t* out, void* stream) {
    if (!b || !out) return fail_msg("kag_done_dev: null");
    KAG_LAUNCH(k_done, grid_for(b->n), kBlock, stream, b->d_sims, b->n, out);
    KAG_CK(cudaGetLastError(), "k_done launch");
    return 0;
}

int kag_done_host(KagBatch* b, uint8_t* out) {
    if (!b || !out) return fail_msg("kag_done_host: null");
    KAG_LAUNCH(k_done, grid_for(b->n), kBlock, nullptr, b->d_sims, b->n, b->d_done);
    KAG_CK(cudaGetLastError(), "k_done launch");
    KAG_CK(cudaMemcpy(out, b->d_done, (size_t)b->n, cudaMemcpyDeviceToHost), "memcpy done");
    return 0;
}

int kag_copy_sims_host(KagBatch* b, int first, int count, void* out) {
    if (!b || !out || first < 0 || count < 0 || first + count > b->n) return fail_msg("kag_copy_sims_host: bad range");
    KAG_CK(cudaMemcpy(out, b->d_sims + first, (size_t)count * sizeof(Sim), cudaMemcpyDeviceToHost), "memcpy sims");
    return 0;
}

int kag_serialize_host(KagBatch* b, int first, int count, uint8_t* out) {
    if (!b || !out || first < 0 || count < 0 || first + count > b->n) return fail_msg("kag_serialize_host: bad range");
    // The Sim layout on the host and the device is the same (the same POD structs),
    // so the raw device bytes are interpreted as host objects.
    std::vector<Sim> tmp((size_t)count);
    KAG_CK(cudaMemcpy(tmp.data(), b->d_sims + first, (size_t)count * sizeof(Sim), cudaMemcpyDeviceToHost), "memcpy sims");
    const size_t S = kaghd::state_ser_bytes<kaghd::State>();
    for (int i = 0; i < count; ++i) kaghd::serialize_state(tmp[(size_t)i].st, out + (size_t)i * S);
    return 0;
}

int kag_upload_actions(KagBatch* b, const int32_t* acts_host, int T, int m) {
    if (!b || !acts_host || T < 1 || m < 1) return fail_msg("kag_upload_actions: bad args");
    if (b->d_up) { cudaFree(b->d_up); b->d_up = nullptr; b->up_T = b->up_m = 0; }
    size_t cnt = (size_t)T * m * 2 * ACT_I32;
    KAG_CK(cudaMalloc(&b->d_up, cnt * sizeof(int32_t)), "cudaMalloc(uploaded)");
    KAG_CK(cudaMemcpy(b->d_up, acts_host, cnt * sizeof(int32_t), cudaMemcpyHostToDevice), "memcpy uploaded");
    b->up_T = T; b->up_m = m;
    return 0;
}

int kag_run_uploaded(KagBatch* b, int t_begin, int t_end, float* ms) {
    if (!b || !b->d_up) return fail_msg("kag_run_uploaded: nothing uploaded");
    if (t_begin < 0 || t_end > b->up_T || t_begin > t_end) return fail_msg("kag_run_uploaded: bad step range");
    cudaEvent_t e0, e1;
    KAG_CK(cudaEventCreate(&e0), "cudaEventCreate");
    KAG_CK(cudaEventCreate(&e1), "cudaEventCreate");
    KAG_CK(cudaEventRecord(e0), "cudaEventRecord");
    const size_t per_t = (size_t)b->up_m * 2 * ACT_I32;
    for (int t = t_begin; t < t_end; ++t) {
        KAG_LAUNCH(k_step, grid_for(b->n), kBlock, nullptr, b->d_sims, b->n, b->d_up + (size_t)t * per_t, b->up_m);
        KAG_CK(cudaGetLastError(), "k_step launch");
    }
    KAG_CK(cudaEventRecord(e1), "cudaEventRecord");
    KAG_CK(cudaEventSynchronize(e1), "k_step (uploaded)");
    float t_ms = 0.f;
    KAG_CK(cudaEventElapsedTime(&t_ms, e0, e1), "cudaEventElapsedTime");
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    if (ms) *ms = t_ms;
    return 0;
}

int kag_price_check(int lo, int hi, long long* mismatches, int* first_item, int* first_inv) {
    if (hi < lo) return fail_msg("kag_price_check: hi < lo");
    const int count = hi - lo + 1;
    const size_t total = (size_t)kaghd::N_PRODUCTS * count;
    int32_t* d = nullptr;
    KAG_CK(cudaMalloc(&d, total * sizeof(int32_t)), "cudaMalloc(prices)");
    int grid = (int)((total + kBlock - 1) / kBlock);
    KAG_LAUNCH(k_prices, grid, kBlock, nullptr, lo, count, d);
    cudaError_t e = cudaGetLastError();
    std::vector<int32_t> h(total);
    if (e == cudaSuccess) e = cudaMemcpy(h.data(), d, total * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(d);
    if (e != cudaSuccess) return fail("kag_price_check", e);
    long long bad = 0; int fi = -1, fv = -1;
    for (int it = 0; it < kaghd::N_PRODUCTS; ++it)
        for (int v = lo; v <= hi; ++v) {
            int host = kaghd::market_price(it, v);   // host libm: the same path as in the CPU module
            if (host != h[(size_t)it * count + (v - lo)]) {
                if (bad == 0) { fi = it; fv = v; }
                ++bad;
            }
        }
    if (mismatches) *mismatches = bad;
    if (first_item) *first_item = fi;
    if (first_inv) *first_inv = fv;
    return 0;
}

int kag_sync(void) {
    KAG_CK(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    return 0;
}

}  // extern "C"
