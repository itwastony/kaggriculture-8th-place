// kagrl_common.hpp - logic of a SINGLE game for the batched RL library (kagrl.cu): observation encoding,
// execution of the policy's choice, engine step, random legal choice. Everything is KAG_HD: the same code
// is compiled by nvcc for the GPU (one thread = one row/game) and by a regular C++ compiler for the CPU (build_cpu.sh, emulation).
//
// Sources (not copied, included as is - single source of truth, parity already proven):
//   ../kagsim_cuda/sim_hd.hpp        engine (bit-exact = vendor kagsim 1.32.7, GPU verified on 475 200 states)
//   ../kagsim_cuda/batch_common.hpp  flat action int32[170], decode_action, reset_one
//   ../kagcore/kagcore.hpp           encode/execute (bit-exact = rl/encode.py, rl/act.py; parity_result.json)
// kagcore.hpp is written against vendor sim.hpp (namespace kag); the shim ../kagsim_cuda/shim/sim.hpp substitutes
// sim_hd.hpp and `namespace kag = kaghd`. -I order: shim and kagsim_cuda; do NOT add the path vendor/kagsim/sim.
#pragma once
#include <stdint.h>
#include <stddef.h>

#include "sim_hd.hpp"
#include "batch_common.hpp"
#include "kagcore.hpp"

namespace kagrl {

using kaghd::Sim;
using kaghd::Action;
using kaghd::ACT_I32;

constexpr int NT = kagcore::NT;                 // 100 tiles
constexpr int FT = kagcore::FT;                 // 30
constexpr int MAXU = kagcore::MAXU;             // 16
constexpr int FU = kagcore::FU;                 // 16
constexpr int FG = kagcore::FG;                 // 77
constexpr int TMB = kagcore::TMASK_BYTES;       // 700 mask bytes per unit
constexpr int MB = kagcore::MB;                 // 489 market bins
constexpr int NH = kagcore::NHEADS;             // 24 market heads
constexpr int NOPS = kagcore::NOPS;             // 56

// encode output pointers (rows contiguous, C-contiguous): tiles f32[R,100,30], units f32[R,16,16],
// unit_pos i64[R,16], unit_mask bool[R,16], glob f32[R,77], tmask u8[R,16,700] (np.packbits), mmask bool[R,489].
struct EncOut {
    float* tiles;
    float* units;
    int64_t* unit_pos;
    bool* unit_mask;
    float* glob;
    uint8_t* tmask;
    bool* mmask;
};

// Row r -> (game, seat). games == seats == nullptr: r = 2*game + seat (both seats, R = 2n);
// games == nullptr, seats != nullptr: game = r, seat = seats[r] (R = n); otherwise games[r], seats[r].
KAG_HD inline void row_game_seat(int r, const int32_t* games, const int32_t* seats, int& g, int& s) {
    if (games) { g = games[r]; s = seats[r]; }
    else if (seats) { g = r; s = seats[r]; }
    else { g = r >> 1; s = r & 1; }
}

KAG_HD inline void encode_row(const kaghd::State& st, int seat, size_t r, const EncOut& o) {
    kagcore::encode(st, seat, o.tiles + r * (size_t)(NT * FT), o.units + r * (size_t)(MAXU * FU),
                    o.unit_pos + r * (size_t)MAXU, o.unit_mask + r * (size_t)MAXU, o.glob + r * (size_t)FG,
                    o.tmask + r * (size_t)(MAXU * TMB), o.mmask + r * (size_t)MB);
}

// Choice index from an int32 or int64 buffer (torch.long). int64 is saturated to int32: for legal values
// (target*56+op < 5600, bin < 32) nothing changes; out-of-range behaviour - as in kagcore::execute.
KAG_HD inline int32_t load_idx(const void* p, size_t i, int idx_bytes) {
    if (idx_bytes == 8) {
        const int64_t v = static_cast<const int64_t*>(p)[i];
        if (v > 2147483647LL) return 2147483647;
        if (v < -2147483647LL - 1) return -2147483647 - 1;
        return (int32_t)v;
    }
    return static_cast<const int32_t*>(p)[i];
}

// Action -> flat int32[170] (layout of batch_common.hpp; unset slots - zeros). Host and device.
KAG_HD inline void action_to_i32(const Action& a, int32_t* b) {
    for (int k = 0; k < ACT_I32; ++k) b[k] = 0;
    b[0] = a.n_units;
    for (int i = 0; i < a.n_units && i < kaghd::MAX_UNITS; ++i) {
        int32_t* u = b + kaghd::ACT_UNITS + 3 * i;
        u[0] = a.units[i].op; u[1] = a.units[i].arg; u[2] = a.units[i].n;
    }
    b[kaghd::ACT_NORD] = a.n_orders;
    for (int i = 0; i < a.n_orders && i < kaghd::MAX_ORDER_SLOTS; ++i) {
        int32_t* o = b + kaghd::ACT_ORDERS + 3 * i;
        o[0] = a.orders[i].op; o[1] = a.orders[i].item; o[2] = a.orders[i].n;
    }
}

// Actions of both seats of game g from the policy's choice (or from ready raw actions).
//   uc: [n][2][16] (int32 or int64), mc: [n][2][24]; idx_bytes = 4 | 8.
//   raw/raw_mask (optional): raw int32[n][2][170], raw_mask u8[n][2] - for seats with raw_mask != 0 the action
//     is taken as is (e.g. a code opponent on the CPU, dict -> kaghd_cpu.encode_action), the choice is ignored.
// Action has default member initializers; on the device (nvcc 13.3) aggregate initialization of nested arrays is unreliable
// (see sim_hd.hpp Sim::reset) - so all fields that are read are set explicitly: execute writes all 40 units and 16 orders,
// decode_action - n_units/n_orders and the first n_units/n_orders slots (the engine reads only those), action_to_i32
// reads only those as well.
KAG_HD inline void choose_actions(const Sim& s, size_t g, const void* uc, const void* mc, int idx_bytes,
                                  const int32_t* raw, const uint8_t* raw_mask, Action* a) {
    for (int p = 0; p < 2; ++p) {
        const size_t row = 2 * g + p;
        if (raw_mask && raw_mask[row]) {
            kaghd::decode_action(raw + row * ACT_I32, a[p]);
        } else {
            int32_t u[MAXU], m[NH];
            for (int k = 0; k < MAXU; ++k) u[k] = load_idx(uc, row * MAXU + k, idx_bytes);
            for (int k = 0; k < NH; ++k) m[k] = load_idx(mc, row * NH + k, idx_bytes);
            kagcore::execute(s.st, p, u, m, a[p]);
        }
    }
}

// Execution of the choice of both seats + step of game g. acts_out (optional): int32[n][2][170] - engine actions
// submitted to the step (for checking). Finished game: acts_out - zeros, no step is made (it stays finished).
KAG_HD inline void act_step_game(Sim& s, size_t g, const void* uc, const void* mc, int idx_bytes,
                                 const int32_t* raw, const uint8_t* raw_mask, int32_t* acts_out) {
    if (s.st.done) {
        if (acts_out)
            for (int k = 0; k < 2 * ACT_I32; ++k) acts_out[g * 2 * ACT_I32 + k] = 0;
        return;
    }
    Action a[2];
    choose_actions(s, g, uc, mc, idx_bytes, raw, raw_mask, a);
    if (acts_out)
        for (int p = 0; p < 2; ++p) action_to_i32(a[p], acts_out + (2 * g + p) * ACT_I32);
    s.step(a[0], a[1]);
}

// Actions only (no step): for checking execute on arbitrary states and for benchmarks. Finished game - zeros.
KAG_HD inline void execute_game(const Sim& s, size_t g, const void* uc, const void* mc, int idx_bytes,
                                int32_t* acts_out) {
    if (s.st.done) {
        for (int k = 0; k < 2 * ACT_I32; ++k) acts_out[g * 2 * ACT_I32 + k] = 0;
        return;
    }
    Action a[2];
    choose_actions(s, g, uc, mc, idx_bytes, nullptr, nullptr, a);
    for (int p = 0; p < 2; ++p) action_to_i32(a[p], acts_out + (2 * g + p) * ACT_I32);
}

// ---------------------------------------------------------------- random legal choice
// Counter-based RNG (splitmix64): identical result on host and device, stateless.
KAG_HD inline uint64_t mix64(uint64_t z) {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
KAG_HD inline int popc8(uint32_t v) {
    v = v - ((v >> 1) & 0x55u);
    v = (v & 0x33u) + ((v >> 2) & 0x33u);
    return (int)((v + (v >> 4)) & 0x0fu);
}

// Row r (game r/2, seat r%2): unit u in unit_mask - uniform over the legal (target, op) pairs of the packed
// tmask (bit i*56+o, most significant bit of a byte first - as np.packbits); an absent unit - 0. Market (mmask in
// rl/encode.py is always full): ORDER/SELL_FIRST/SEED_FIRST - uniform; SELL - 12% "the whole stock", 58% zero,
// otherwise a random bin; HIRE - 95% zero, otherwise 1-2; BUY_LAND - 99% zero; other purchases - 90% zero (stingier
// than rand_choices from engines/kagcore/parity.py, so that the game goes bankrupt later). Output - int32 or int64
// (idx_bytes) in the layout [n][2][16] / [n][2][24].
KAG_HD inline void random_choice_row(size_t r, const uint8_t* tmask, const bool* unit_mask, uint64_t seed,
                                     uint64_t salt, void* uc, void* mc, int idx_bytes) {
    const uint64_t key = mix64(seed ^ mix64(salt * 0x632be59bd9b4e019ull + (uint64_t)r));
    int32_t outu[MAXU], outm[NH];
    for (int u = 0; u < MAXU; ++u) {
        outu[u] = 0;
        if (!unit_mask[r * MAXU + u]) continue;
        const uint8_t* m = tmask + (r * MAXU + u) * (size_t)TMB;
        int cnt = 0;
        for (int b = 0; b < TMB; ++b) cnt += popc8(m[b]);
        if (cnt == 0) continue;
        int k = (int)(mix64(key + 0x1000u + (uint64_t)u) % (uint64_t)cnt);
        for (int b = 0; b < TMB; ++b) {
            const int c = popc8(m[b]);
            if (k >= c) { k -= c; continue; }
            for (int j = 0; j < 8; ++j) {
                if ((m[b] >> (7 - j)) & 1u) {
                    if (k == 0) { outu[u] = b * 8 + j; break; }
                    --k;
                }
            }
            break;
        }
    }
    for (int h = 0; h < NH; ++h) {
        const uint64_t z = mix64(key + 0x2000u + (uint64_t)h);
        const int nb = kagcore::head_nbins(h);
        const int p = (int)(z % 1000u);
        const int any = 1 + (int)((z >> 20) % (uint64_t)(nb - 1));
        int v;
        if (h >= 21) v = (int)((z >> 20) % (uint64_t)nb);
        else if (h < 9) v = p < 120 ? nb - 1 : (p < 700 ? 0 : any);
        else if (h == 19) v = p < 950 ? 0 : 1 + (int)((z >> 20) % 2u);
        else if (h == 20) v = p < 990 ? 0 : 1;
        else v = p < 900 ? 0 : any;
        outm[h] = v;
    }
    if (idx_bytes == 8) {
        int64_t* pu = static_cast<int64_t*>(uc) + r * MAXU;
        int64_t* pm = static_cast<int64_t*>(mc) + r * NH;
        for (int u = 0; u < MAXU; ++u) pu[u] = outu[u];
        for (int h = 0; h < NH; ++h) pm[h] = outm[h];
    } else {
        int32_t* pu = static_cast<int32_t*>(uc) + r * MAXU;
        int32_t* pm = static_cast<int32_t*>(mc) + r * NH;
        for (int u = 0; u < MAXU; ++u) pu[u] = outu[u];
        for (int h = 0; h < NH; ++h) pm[h] = outm[h];
    }
}

}  // namespace kagrl
