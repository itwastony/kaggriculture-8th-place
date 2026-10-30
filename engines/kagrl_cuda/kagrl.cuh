// kagrl - batched Kaggriculture RL library entirely on the GPU: engine + observation encoding + execution
// of the policy's choice. Simple C API (extern "C") for ctypes (kagrl.py). All *_dev functions work with pointers
// to device memory (torch CUDA tensors) asynchronously on the given stream (cudaStream_t, NULL = default
// stream) and do NOT synchronize the host. In the CPU build (build_cpu.sh, the same kagrl.cu with an emulated runtime)
// "device memory" = host memory: the same functions accept numpy / torch CPU tensors.
//
// Buffer formats (all C-contiguous), n = number of games, R = number of encoding rows:
//   seeds     uint64 [n]; mask uint8 [n] (reset only mask != 0); shops uint8 [n][8] + nshops int32 [n]
//   encode:   row r -> (game, seat): games == seats == NULL: r = 2*game + seat (R = 2n);
//             games == NULL, seats != NULL: game r, seat seats[r] (R = n); otherwise games[r], seats[r].
//             tiles f32 [R][100][30], units f32 [R][16][16], unit_pos int64 [R][16], unit_mask bool [R][16],
//             glob f32 [R][77], tmask uint8 [R][16][700] (np.packbits over 5600 bits (target, op)),
//             mmask bool [R][489]  - bit-exact with rl/encode.py (+ packbits) and kagcore.GameBatch.encode_into.
//   act_step: uc [n][2][16] = target*56 + op, mc [n][2][24] = market head bins; int32 (idx_bytes=4)
//             or int64 (idx_bytes=8, torch.long straight from policy sampling). Row [g][s] = seat s of game g.
//             raw int32 [n][2][170] + raw_mask uint8 [n][2] (optional): ready engine actions for seats with
//             raw_mask != 0 (code opponent). acts_out int32 [n][2][170] (optional): submitted actions.
//   rewards float64 [n][2] (bank), done uint8 [n], steps int32 [n] (step index of the game, 0..719).
// Returns: 0 - success, otherwise a CUDA error code / -1; message - kagrl_last_error().
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct KagRL KagRL;

// Layout constants: out[0..15] = NT, FT, MAXU, FU, FG, TMASK_BYTES, MB, NHEADS, NOPS, ACT_I32,
// SER_BYTES, SIM_BYTES, MAX_ORDERS, EMULATED (1 = CPU build), API_VERSION, 0. Returns 16.
int kagrl_dims(int32_t* out);
const char* kagrl_last_error(void);

KagRL* kagrl_create(int n, int episode_steps, int device);     // NULL on error
void kagrl_destroy(KagRL* b);
int kagrl_n(const KagRL* b);
void* kagrl_sims_ptr(KagRL* b);                                  // kaghd::Sim[n] in device memory

int kagrl_reset_dev(KagRL* b, const uint64_t* seeds, const uint8_t* mask, const uint8_t* shops,
                    const int32_t* nshops, void* stream);
int kagrl_encode_dev(KagRL* b, int rows, const int32_t* games, const int32_t* seats, float* tiles, float* units,
                     int64_t* unit_pos, uint8_t* unit_mask, float* glob, uint8_t* tmask, uint8_t* mmask,
                     void* stream);
int kagrl_act_step_dev(KagRL* b, const void* uc, const void* mc, int idx_bytes, const int32_t* raw,
                       const uint8_t* raw_mask, int32_t* acts_out, void* stream);
// Engine actions of both seats only (no step) -> acts_out int32 [n][2][170]; finished games - zeros.
int kagrl_execute_dev(KagRL* b, const void* uc, const void* mc, int idx_bytes, int32_t* acts_out, void* stream);
// Step with ready actions int32 [m][2][170] (game i - row i % m), like kag_step_dev in kagsim_cuda.
int kagrl_step_raw_dev(KagRL* b, const int32_t* acts, int m, void* stream);
// Random legal choice over encoded rows in the "both seats" layout (tmask [2n][16][700],
// unit_mask [2n][16]) -> uc [n][2][16], mc [n][2][24] (int32/int64). Deterministic in (seed, salt, row),
// identical on CPU and GPU. For benchmarks and the random opponent.
int kagrl_random_choices_dev(KagRL* b, const uint8_t* tmask, const uint8_t* unit_mask, uint64_t seed,
                             uint64_t salt, void* uc, void* mc, int idx_bytes, void* stream);

int kagrl_rewards_dev(KagRL* b, double* out, void* stream);
int kagrl_done_dev(KagRL* b, uint8_t* out, void* stream);
int kagrl_steps_dev(KagRL* b, int32_t* out, void* stream);

// Synchronous: full state of games [first, first+count) -> host (canonical serialization state_visit.hpp,
// count * SER_BYTES bytes) or raw Sim objects (count * SIM_BYTES bytes). First waits for the stream `stream`
// (torch streams are non-blocking: cudaMemcpy itself does not wait for them).
int kagrl_serialize_host(KagRL* b, int first, int count, uint8_t* out, void* stream);
int kagrl_copy_sims_host(KagRL* b, int first, int count, void* out, void* stream);

int kagrl_stream_sync(void* stream);

// Diagnostics: threads per block for all kernels (default 128; 32..1024, multiple of 32) and kernel attributes
// (cudaFuncGetAttributes) for k_encode, k_act_step, k_execute, k_random, k_step_raw, k_reset:
// out[4*k + 0..3] = registers per thread, local memory per thread (bytes), max threads per block, shared (bytes).
// In the CPU build - zeros. Returns 6.
int kagrl_set_block(int threads);
int kagrl_kernel_info(int32_t* out);

#ifdef __cplusplus
}
#endif
