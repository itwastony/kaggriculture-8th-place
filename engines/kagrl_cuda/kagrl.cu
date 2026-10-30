// kagrl.cu - implementation of the C API kagrl.cuh: array of n game states in device memory, kernels with one thread
// per encoding row (encode, random_choices) or per game (act_step, step_raw, rewards/done/steps).
// Logic of a single game/row - kagrl_common.hpp (engine sim_hd.hpp, encode/execute kagcore.hpp).
//
// One source - two builds:
//   GPU (build_cuda.sh):  nvcc -arch=<arch> -fmad=false ... -shared kagrl.cu -> build/libkagrl_cuda.so
//   CPU (build_cpu.sh):   c++ -x c++ -DKAG_EMULATE -Iemu ... -shared kagrl.cu -> build/libkagrl_cpu.so
//                         (emu/cuda_runtime.h: cudaMalloc = malloc, kernel launch = loop over threads)
// All *_dev functions only enqueue kernels on the stream (no host synchronization); launch errors - cudaGetLastError.
#include <cuda_runtime.h>

#include <stdint.h>
#include <string.h>
#include <string>
#include <vector>

#include "kagrl_common.hpp"
#include "state_visit.hpp"
#include "kagrl.cuh"

using kaghd::Sim;
using kaghd::Config;
using kaghd::ACT_I32;

#define KAGRL_API_VERSION 1

#if defined(KAG_EMULATE)
#define KAG_LAUNCH(kernel, grid, block, stream, ...) \
    kag_emu::launch((grid), (block), [&]() { kernel(__VA_ARGS__); })
#else
#define KAG_LAUNCH(kernel, grid, block, stream, ...) \
    kernel<<<(grid), (block), 0, (cudaStream_t)(stream)>>>(__VA_ARGS__)
#endif

namespace {

int kBlock = 128;                 // threads per block (kagrl_set_block): one thread = row/game

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
__global__ void k_reset(Sim* sims, int n, Config base, const uint64_t* seeds, const uint8_t* mask,
                        const uint8_t* shops, const int32_t* nshops) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (mask && !mask[i]) return;
    int ns = nshops ? nshops[i] : 0;
    kaghd::reset_one(sims[i], base, seeds[i], shops ? shops + (size_t)i * 8 : nullptr, ns);
}

__global__ void k_encode(const Sim* sims, int n, int rows, const int32_t* games, const int32_t* seats,
                         kagrl::EncOut o) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    int g, s;
    kagrl::row_game_seat(r, games, seats, g, s);
    if (g < 0 || g >= n || (s != 0 && s != 1)) return;      // an invalid row is not written
    kagrl::encode_row(sims[g].st, s, (size_t)r, o);
}

__global__ void k_act_step(Sim* sims, int n, const void* uc, const void* mc, int idx_bytes, const int32_t* raw,
                           const uint8_t* raw_mask, int32_t* acts_out) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    kagrl::act_step_game(sims[g], (size_t)g, uc, mc, idx_bytes, raw, raw_mask, acts_out);
}

__global__ void k_execute(const Sim* sims, int n, const void* uc, const void* mc, int idx_bytes, int32_t* acts_out) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    kagrl::execute_game(sims[g], (size_t)g, uc, mc, idx_bytes, acts_out);
}

__global__ void k_step_raw(Sim* sims, int n, const int32_t* acts, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int32_t* row = acts + (size_t)(i % m) * 2 * ACT_I32;
    kaghd::step_one<Sim, kaghd::Action>(sims[i], row, row + ACT_I32);
}

__global__ void k_random(int rows, const uint8_t* tmask, const bool* unit_mask, uint64_t seed, uint64_t salt,
                         void* uc, void* mc, int idx_bytes) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    kagrl::random_choice_row((size_t)r, tmask, unit_mask, seed, salt, uc, mc, idx_bytes);
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

__global__ void k_steps(const Sim* sims, int n, int32_t* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = sims[i].st.step;
}

}  // namespace

struct KagRL {
    int n = 0;
    int device = 0;
    Config base;
    Sim* d_sims = nullptr;
};

extern "C" {

int kagrl_dims(int32_t* out) {
    if (!out) return fail_msg("kagrl_dims: null");
    const int32_t v[16] = {kagrl::NT, kagrl::FT, kagrl::MAXU, kagrl::FU, kagrl::FG, kagrl::TMB, kagrl::MB,
                           kagrl::NH, kagrl::NOPS, ACT_I32,
                           (int32_t)kaghd::state_ser_bytes<kaghd::State>(), (int32_t)sizeof(Sim),
                           kagcore::MAX_ORDERS,
#if defined(KAG_EMULATE)
                           1,
#else
                           0,
#endif
                           KAGRL_API_VERSION, 0};
    for (int k = 0; k < 16; ++k) out[k] = v[k];
    return 16;
}

const char* kagrl_last_error(void) { return g_err.c_str(); }

KagRL* kagrl_create(int n, int episode_steps, int device) {
    if (n <= 0) { fail_msg("kagrl_create: n must be > 0"); return nullptr; }
    cudaError_t e = cudaSetDevice(device);
    if (e != cudaSuccess) { fail("cudaSetDevice", e); return nullptr; }
    KagRL* b = new KagRL();
    b->n = n;
    b->device = device;
    b->base = Config{};
    b->base.episode_steps = episode_steps;
    const size_t sz = (size_t)n * sizeof(Sim);
    if ((e = cudaMalloc(&b->d_sims, sz)) != cudaSuccess || (e = cudaMemset(b->d_sims, 0, sz)) != cudaSuccess) {
        fail("kagrl_create: cudaMalloc/cudaMemset", e);
        kagrl_destroy(b);
        return nullptr;
    }
    return b;
}

void kagrl_destroy(KagRL* b) {
    if (!b) return;
    cudaFree(b->d_sims);
    delete b;
}

int kagrl_n(const KagRL* b) { return b ? b->n : 0; }
void* kagrl_sims_ptr(KagRL* b) { return b ? (void*)b->d_sims : nullptr; }

int kagrl_reset_dev(KagRL* b, const uint64_t* seeds, const uint8_t* mask, const uint8_t* shops,
                    const int32_t* nshops, void* stream) {
    if (!b || !seeds) return fail_msg("kagrl_reset_dev: null batch/seeds");
    if ((shops == nullptr) != (nshops == nullptr)) return fail_msg("kagrl_reset_dev: shops and nshops go together");
    KAG_LAUNCH(k_reset, grid_for(b->n), kBlock, stream, b->d_sims, b->n, b->base, seeds, mask, shops, nshops);
    KAG_CK(cudaGetLastError(), "k_reset launch");
    return 0;
}

int kagrl_encode_dev(KagRL* b, int rows, const int32_t* games, const int32_t* seats, float* tiles, float* units,
                     int64_t* unit_pos, uint8_t* unit_mask, float* glob, uint8_t* tmask, uint8_t* mmask,
                     void* stream) {
    if (!b || rows <= 0) return fail_msg("kagrl_encode_dev: null batch or rows <= 0");
    if (games && !seats) return fail_msg("kagrl_encode_dev: games requires seats");
    if (!tiles || !units || !unit_pos || !unit_mask || !glob || !tmask || !mmask)
        return fail_msg("kagrl_encode_dev: null output buffer");
    if (!games && !seats && rows != 2 * b->n) return fail_msg("kagrl_encode_dev: both-seats layout needs rows == 2n");
    if (!games && seats && rows != b->n) return fail_msg("kagrl_encode_dev: seats-only layout needs rows == n");
    kagrl::EncOut o{tiles, units, unit_pos, reinterpret_cast<bool*>(unit_mask), glob, tmask,
                    reinterpret_cast<bool*>(mmask)};
    KAG_LAUNCH(k_encode, grid_for(rows), kBlock, stream, b->d_sims, b->n, rows, games, seats, o);
    KAG_CK(cudaGetLastError(), "k_encode launch");
    return 0;
}

int kagrl_act_step_dev(KagRL* b, const void* uc, const void* mc, int idx_bytes, const int32_t* raw,
                       const uint8_t* raw_mask, int32_t* acts_out, void* stream) {
    if (!b || !uc || !mc) return fail_msg("kagrl_act_step_dev: null batch/uc/mc");
    if (idx_bytes != 4 && idx_bytes != 8) return fail_msg("kagrl_act_step_dev: idx_bytes must be 4 or 8");
    if ((raw == nullptr) != (raw_mask == nullptr)) return fail_msg("kagrl_act_step_dev: raw and raw_mask go together");
    KAG_LAUNCH(k_act_step, grid_for(b->n), kBlock, stream, b->d_sims, b->n, uc, mc, idx_bytes, raw, raw_mask,
               acts_out);
    KAG_CK(cudaGetLastError(), "k_act_step launch");
    return 0;
}

int kagrl_execute_dev(KagRL* b, const void* uc, const void* mc, int idx_bytes, int32_t* acts_out, void* stream) {
    if (!b || !uc || !mc || !acts_out) return fail_msg("kagrl_execute_dev: null batch/uc/mc/acts_out");
    if (idx_bytes != 4 && idx_bytes != 8) return fail_msg("kagrl_execute_dev: idx_bytes must be 4 or 8");
    KAG_LAUNCH(k_execute, grid_for(b->n), kBlock, stream, b->d_sims, b->n, uc, mc, idx_bytes, acts_out);
    KAG_CK(cudaGetLastError(), "k_execute launch");
    return 0;
}

int kagrl_step_raw_dev(KagRL* b, const int32_t* acts, int m, void* stream) {
    if (!b || !acts || m < 1) return fail_msg("kagrl_step_raw_dev: null batch/acts or m < 1");
    KAG_LAUNCH(k_step_raw, grid_for(b->n), kBlock, stream, b->d_sims, b->n, acts, m);
    KAG_CK(cudaGetLastError(), "k_step_raw launch");
    return 0;
}

int kagrl_random_choices_dev(KagRL* b, const uint8_t* tmask, const uint8_t* unit_mask, uint64_t seed,
                             uint64_t salt, void* uc, void* mc, int idx_bytes, void* stream) {
    if (!b || !tmask || !unit_mask || !uc || !mc) return fail_msg("kagrl_random_choices_dev: null");
    if (idx_bytes != 4 && idx_bytes != 8) return fail_msg("kagrl_random_choices_dev: idx_bytes must be 4 or 8");
    const int rows = 2 * b->n;
    KAG_LAUNCH(k_random, grid_for(rows), kBlock, stream, rows, tmask, reinterpret_cast<const bool*>(unit_mask),
               seed, salt, uc, mc, idx_bytes);
    KAG_CK(cudaGetLastError(), "k_random launch");
    return 0;
}

int kagrl_rewards_dev(KagRL* b, double* out, void* stream) {
    if (!b || !out) return fail_msg("kagrl_rewards_dev: null");
    KAG_LAUNCH(k_rewards, grid_for(b->n), kBlock, stream, b->d_sims, b->n, out);
    KAG_CK(cudaGetLastError(), "k_rewards launch");
    return 0;
}

int kagrl_done_dev(KagRL* b, uint8_t* out, void* stream) {
    if (!b || !out) return fail_msg("kagrl_done_dev: null");
    KAG_LAUNCH(k_done, grid_for(b->n), kBlock, stream, b->d_sims, b->n, out);
    KAG_CK(cudaGetLastError(), "k_done launch");
    return 0;
}

int kagrl_steps_dev(KagRL* b, int32_t* out, void* stream) {
    if (!b || !out) return fail_msg("kagrl_steps_dev: null");
    KAG_LAUNCH(k_steps, grid_for(b->n), kBlock, stream, b->d_sims, b->n, out);
    KAG_CK(cudaGetLastError(), "k_steps launch");
    return 0;
}

int kagrl_copy_sims_host(KagRL* b, int first, int count, void* out, void* stream) {
    if (!b || !out || first < 0 || count < 0 || first + count > b->n) return fail_msg("kagrl_copy_sims_host: bad range");
    KAG_CK(cudaStreamSynchronize((cudaStream_t)stream), "cudaStreamSynchronize");
    KAG_CK(cudaMemcpy(out, b->d_sims + first, (size_t)count * sizeof(Sim), cudaMemcpyDeviceToHost), "memcpy sims");
    return 0;
}

int kagrl_serialize_host(KagRL* b, int first, int count, uint8_t* out, void* stream) {
    if (!b || !out || first < 0 || count < 0 || first + count > b->n) return fail_msg("kagrl_serialize_host: bad range");
    // The Sim layout is the same on host and device (the same POD structs).
    std::vector<Sim> tmp((size_t)count);
    int rc = kagrl_copy_sims_host(b, first, count, tmp.data(), stream);
    if (rc) return rc;
    const size_t S = kaghd::state_ser_bytes<kaghd::State>();
    for (int i = 0; i < count; ++i) kaghd::serialize_state(tmp[(size_t)i].st, out + (size_t)i * S);
    return 0;
}

int kagrl_set_block(int threads) {
    if (threads < 32 || threads > 1024 || threads % 32) return fail_msg("kagrl_set_block: 32..1024, multiple of 32");
    kBlock = threads;
    return 0;
}

int kagrl_kernel_info(int32_t* out) {
    if (!out) return fail_msg("kagrl_kernel_info: null");
    for (int k = 0; k < 24; ++k) out[k] = 0;
#if !defined(KAG_EMULATE)
    const void* ks[6] = {(const void*)k_encode, (const void*)k_act_step, (const void*)k_execute,
                         (const void*)k_random, (const void*)k_step_raw, (const void*)k_reset};
    for (int k = 0; k < 6; ++k) {
        cudaFuncAttributes a;
        KAG_CK(cudaFuncGetAttributes(&a, ks[k]), "cudaFuncGetAttributes");
        out[4 * k + 0] = a.numRegs;
        out[4 * k + 1] = (int32_t)a.localSizeBytes;
        out[4 * k + 2] = a.maxThreadsPerBlock;
        out[4 * k + 3] = (int32_t)a.sharedSizeBytes;
    }
#endif
    return 6;
}

int kagrl_stream_sync(void* stream) {
    KAG_CK(cudaStreamSynchronize((cudaStream_t)stream), "cudaStreamSynchronize");
    return 0;
}

}  // extern "C"
