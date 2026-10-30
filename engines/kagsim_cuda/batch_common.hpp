// Shared part of the batch API (the CPU module and the CUDA module use the SAME code).
//
// The engine's internal action (kaghd::Action) is stored in the API buffers as a
// flat int32 array of fixed length ACT_I32 = 170 per seat:
//
//   [0]                         n_units  - how many unit slots are set (0..40),
//                                          slot 0 = farmer, 1.. = hired workers
//   [1 + 3*u + 0/1/2]           op, arg, n of unit u (u < 40)
//   [ACT_NORD = 121]            n_orders - how many order slots (0..16;
//                                          the engine reads the first max_orders=10)
//   [122 + 3*o + 0/1/2]         op, item, n of order o (o < 16)
//
// Values are cast to the Action fields exactly as in the engine: op/arg/item ->
// uint8_t, unit n -> int16_t, order n -> int32_t; n_units/n_orders
// are clamped to the valid range. Slots beyond n_units/n_orders are not
// read. The op/item codes are the Op/MOp/Item enums from sim_hd.hpp (as in
// vendor/kagsim/python/kagsim.cpp: OPI/ITI/MOPI). Action batch: int32
// [m][2][ACT_I32]; game i takes row i % m (m = n: each game has its own).
#pragma once
#include <stdint.h>
#include "sim_hd.hpp"

namespace kaghd {

constexpr int ACT_UNITS = 1;                              // start of the unit slots
constexpr int ACT_NORD = 1 + 3 * MAX_UNITS;               // = 121
constexpr int ACT_ORDERS = ACT_NORD + 1;                  // = 122
constexpr int ACT_I32 = ACT_ORDERS + 3 * MAX_ORDER_SLOTS; // = 170

// Decode a flat int32 action into the engine's internal Action.
// A template, so that the same code can also feed the reference kag::Action (vendor) during
// comparison (they have the same field names and array sizes).
template <class A>
KAG_HD inline void decode_action(const int32_t* b, A& a) {
    int nu = b[0];
    if (nu < 0) nu = 0;
    if (nu > MAX_UNITS) nu = MAX_UNITS;
    a.n_units = nu;
    for (int i = 0; i < nu; ++i) {
        const int32_t* u = b + ACT_UNITS + 3 * i;
        a.units[i].op = static_cast<uint8_t>(u[0]);
        a.units[i].arg = static_cast<uint8_t>(u[1]);
        a.units[i].n = static_cast<int16_t>(u[2]);
    }
    int no = b[ACT_NORD];
    if (no < 0) no = 0;
    if (no > MAX_ORDER_SLOTS) no = MAX_ORDER_SLOTS;
    a.n_orders = no;
    for (int i = 0; i < no; ++i) {
        const int32_t* o = b + ACT_ORDERS + 3 * i;
        a.orders[i].op = static_cast<uint8_t>(o[0]);
        a.orders[i].item = static_cast<uint8_t>(o[1]);
        a.orders[i].n = o[2];
    }
}

// Inverse: Action -> flat int32 (unset slots are zeros). Host.
inline void encode_action_i32(const Action& a, int32_t* b) {
    for (int k = 0; k < ACT_I32; ++k) b[k] = 0;
    b[0] = a.n_units;
    for (int i = 0; i < a.n_units && i < MAX_UNITS; ++i) {
        int32_t* u = b + ACT_UNITS + 3 * i;
        u[0] = a.units[i].op; u[1] = a.units[i].arg; u[2] = a.units[i].n;
    }
    b[ACT_NORD] = a.n_orders;
    for (int i = 0; i < a.n_orders && i < MAX_ORDER_SLOTS; ++i) {
        int32_t* o = b + ACT_ORDERS + 3 * i;
        o[0] = a.orders[i].op; o[1] = a.orders[i].item; o[2] = a.orders[i].n;
    }
}

// One game of the batch: reset with a seed and (optionally) pinned
// shops. shops8 - 8 shop ids (0..7), nshops - how many of them to
// pin (0..8); nullptr/0 - do not pin.
template <class SimT, class ConfigT>
KAG_HD inline void reset_one(SimT& s, const ConfigT& base, uint64_t seed,
                             const uint8_t* shops8, int nshops) {
    ConfigT c = base;
    c.seed = seed;
    if (shops8 && nshops > 0) {
        if (nshops > MAX_SHOP_INSTANCES) nshops = MAX_SHOP_INSTANCES;
        for (int k = 0; k < nshops; ++k) c.forced_shops[k] = shops8[k];
        c.n_forced_shops = nshops;
    }
    s.cfg = c;
    s.reset();
}

// One step of one game from two flat actions (seats 0 and 1).
template <class SimT, class ActionT>
KAG_HD inline void step_one(SimT& s, const int32_t* act0, const int32_t* act1) {
    if (s.st.done) return;
    ActionT a0, a1;
    decode_action(act0, a0);
    decode_action(act1, a1);
    s.step(a0, a1);
}

}  // namespace kaghd
