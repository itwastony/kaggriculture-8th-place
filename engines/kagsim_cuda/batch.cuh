// Batch API of the Kaggriculture engine on the GPU (one thread = one game).
// A simple C API (extern "C"), called from Python via ctypes
// (kagsim_cuda.py) with numpy buffers (host) or torch tensor pointers
// (device). The kernel code is sim_hd.hpp, the same as in the CPU module kaghd_cpu.
//
// Buffer formats (all C-contiguous):
//   seeds   uint64 [n]
//   mask    uint8  [n]      (optional: reset only games with mask != 0)
//   shops   uint8  [n][8]   (optional, together with nshops: pinned shops 0..7)
//   nshops  int32  [n]      (0..8)
//   acts    int32  [m][2][KAG_ACT_I32]  - game i takes row i % m
//                                         (layout: batch_common.hpp)
//   rewards float64 [n][2]
//   done    uint8  [n]
// All functions return 0 on success, otherwise a CUDA error code / -1; the message is in
// kag_last_error(). *_dev - pointers in device memory, asynchronous on the
// given stream (stream = cudaStream_t or NULL). *_host - pointers on the
// host, synchronous.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KAG_ACT_I32 170

typedef struct KagBatch KagBatch;

const char* kag_last_error(void);
int kag_act_i32(void);            // = 170
int kag_ser_bytes(void);          // size of the canonical State serialization
int kag_sim_bytes(void);          // sizeof(kaghd::Sim) in device memory
int kag_amps_selfcheck(void);     // 0 = amplitude literals matched the formula on the host

KagBatch* kag_create(int n, int episode_steps, int device);   // NULL on error
void kag_destroy(KagBatch* b);
int kag_n(const KagBatch* b);
void* kag_sims_dev_ptr(KagBatch* b);                           // kaghd::Sim[n] on the device

int kag_reset_dev(KagBatch* b, const uint64_t* seeds, const uint8_t* mask,
                  const uint8_t* shops, const int32_t* nshops, void* stream);
int kag_reset_host(KagBatch* b, const uint64_t* seeds, const uint8_t* mask,
                   const uint8_t* shops, const int32_t* nshops);

int kag_step_dev(KagBatch* b, const int32_t* acts, int m, void* stream);
int kag_step_host(KagBatch* b, const int32_t* acts, int m);

int kag_rewards_dev(KagBatch* b, double* out, void* stream);
int kag_rewards_host(KagBatch* b, double* out);
int kag_done_dev(KagBatch* b, uint8_t* out, void* stream);
int kag_done_host(KagBatch* b, uint8_t* out);

// Full state of games [first, first+count) -> host: canonical
// serialization (count * kag_ser_bytes() bytes) or raw Sim objects.
int kag_serialize_host(KagBatch* b, int first, int count, uint8_t* out);
int kag_copy_sims_host(KagBatch* b, int first, int count, void* out);

// Action streams uploaded to the device in advance: int32 [T][m][2][170].
// kag_run_uploaded runs steps t = t_begin..t_end-1 (game i uses row i % m)
// and synchronizes; *ms - GPU time (cudaEvent), may be NULL.
int kag_upload_actions(KagBatch* b, const int32_t* acts_host, int T, int m);
int kag_run_uploaded(KagBatch* b, int t_begin, int t_end, float* ms);

// Check of the price on the device against the host: market_price(item, inv) for all 9
// items and inv in [lo, hi]. *mismatches - number of mismatches; *first_item,
// *first_inv - the first mismatch (or -1).
int kag_price_check(int lo, int hi, long long* mismatches, int* first_item, int* first_inv);

int kag_sync(void);

#ifdef __cplusplus
}
#endif
