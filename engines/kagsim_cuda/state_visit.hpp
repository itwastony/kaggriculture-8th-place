// Canonical State serialization: ALL fields (including telemetry and the
// "stale" tails of the pos/inv/inv_keys arrays) in a fixed order,
// without padding bytes. Works the same for kaghd::State and for the
// original kag::State (same field names), so comparing
// serializations = byte-for-byte comparison of the full state without false
// mismatches in padding. Host only (the GPU state is copied to the host as
// is - the layout on host and device is the same - and serialized right here).
//
// Completeness of the field list is checked in parity.py (parsing the struct declarations
// in sim_hd.hpp and comparing with the names below).
#pragma once
#include <stdint.h>
#include <string.h>

namespace kaghd {

template <class St, class Put>
inline void visit_state(const St& s, Put&& put) {
    for (int p = 0; p < 2; ++p) {
        const auto& f = s.farms[p];
        put(f.money);
        for (int y = 0; y < 10; ++y)
            for (int x = 0; x < 10; ++x) {
                const auto& t = f.tiles[y][x];
                put(t.kind); put(t.what); put(t.has_animal); put(t.watered_today);
                put(t.fed_today); put(t.cared_today); put(t.fertilizer_available);
                put(t.consecutive_dry); put(t.yield_units); put(t.pending_care_bonus);
                put(t.planted_day); put(t.max_lifespan_step); put(t.fertilized_until_day);
            }
        put(f.pos_x); put(f.pos_y);
        put(f.n_units); put(f.n_quadrants); put(f.hires_today);
        put(f.shed); put(f.shed_total); put(f.seeds);
        put(f.inv); put(f.inv_keys); put(f.inv_nkeys);
        put(f.discarded); put(f.produced); put(f.sold_units);
        put(f.sell_revenue); put(f.total_spend);
        put(f.tel_sell_dead); put(f.tel_refused_product); put(f.tel_refused_seed);
        put(f.tel_refused_animal); put(f.tel_refused_hire); put(f.tel_refused_land);
        put(f.tel_hire_paid); put(f.tel_sell_zero_fill); put(f.tel_buy_zero_fill);
        put(f.tel_tile_cap_units); put(f.tel_fertilizer_forgone); put(f.tel_care_unfed);
        put(f.tel_animal_shed_days); put(f.tel_idle_tile_days); put(f.tel_weed_tile_days);
        put(f.tel_hand_pass_turns); put(f.tel_farmer_pass_turns); put(f.tel_phantom_hand_actions);
        put(f.tel_move_dead); put(f.tel_locked_dead); put(f.tel_pickup_dead);
        put(f.tel_drop_dead); put(f.tel_place_dead); put(f.tel_build_dead);
        put(f.tel_dig_dead); put(f.tel_fertilize_dead); put(f.tel_collect_dead);
        put(f.tel_care_dead); put(f.tel_sell_floor_units); put(f.tel_hire_premium);
        put(f.tel_feed_no_wheat); put(f.tel_feed_no_animal); put(f.tel_feed_redundant);
        put(f.tel_water_dead); put(f.tel_harvest_dead); put(f.tel_plant_dead);
        put(f.tel_animals_escaped); put(f.tel_escape_capital); put(f.tel_escape_yield_units);
        put(f.tel_escape_first_day); put(f.tel_animal_unfed_days); put(f.tel_plants_dry);
        put(f.tel_dry_seed_cost); put(f.tel_dry_yield_units); put(f.tel_dry_first_day);
        put(f.tel_plant_unwatered_days); put(f.tel_decay_units); put(f.tel_plants_decayed);
    }
    put(s.market.inventory); put(s.market.prices);
    put(s.shops); put(s.n_shops); put(s.step); put(s.day); put(s.hour); put(s.done);
}

// Serialization size (the same for all States with these fields).
template <class St>
inline size_t state_ser_bytes() {
    size_t n = 0;
    St* tmp = new St();   // only for walking the fields; values do not matter
    visit_state(*tmp, [&](const auto& x) { n += sizeof(x); });
    delete tmp;
    return n;
}

// Serialize into out (must have room for state_ser_bytes<St>() bytes).
template <class St>
inline size_t serialize_state(const St& s, uint8_t* out) {
    size_t pos = 0;
    visit_state(s, [&](const auto& x) { memcpy(out + pos, &x, sizeof(x)); pos += sizeof(x); });
    return pos;
}

}  // namespace kaghd
