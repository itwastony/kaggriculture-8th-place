// kaghd_cpu: CPU build of the CUDA-capable core sim_hd.hpp with the same batch API
// as the GPU module (batch.cu): a loop over games instead of threads.
//
// The action converters and the observe/telemetry builders below are ports of code from
// vendor/kagsim/python/kagsim.cpp (kagsim 0.5.0, Apache License 2.0, see vendor/kagsim/LICENSE
// and THIRD_PARTY_NOTICES.md), adapted to the kaghd namespace and the flat int32[170] layout.
//
// Plus everything needed for comparison:
//   * encode_action(s): action dict -> flat int32[170] (verbatim port of
//     conv_unit/conv_order/conv_action from vendor/kagsim/python/kagsim.cpp);
//   * Batch.observe/telemetry: the same dicts that kagsim.Game returns (port of
//     ser_tile/ser_farm/observe/telemetry), for comparison with the installed kagsim;
//   * RefBatch: the same batch API on top of the ORIGINAL vendor/kagsim/sim/sim.hpp
//     (the file is only included, not modified): the reference for byte-for-byte comparison
//     of the full state (canonical serialization from state_visit.hpp).
//
// Build: see build_cpu.sh (-O3 -std=c++17 -ffp-contract=off).
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "sim_hd.hpp"
#include "batch_common.hpp"
#include "state_visit.hpp"
#include "../../vendor/kagsim/sim/sim.hpp"   // original (namespace kag): reference only

namespace py = pybind11;
using kaghd::ACT_I32;

// The kaghd::State layout must match the original one (copy of the structs).
static_assert(sizeof(kaghd::State) == sizeof(kag::State), "State layout differs from vendor");
static_assert(sizeof(kaghd::Farm) == sizeof(kag::Farm), "Farm layout differs from vendor");
static_assert(sizeof(kaghd::Tile) == sizeof(kag::Tile), "Tile layout differs from vendor");
static_assert(sizeof(kaghd::Action) == sizeof(kag::Action), "Action layout differs from vendor");
static_assert(sizeof(kaghd::Config) == sizeof(kag::Config), "Config layout differs from vendor");
static_assert(offsetof(kaghd::Farm, tel_plants_decayed) == offsetof(kag::Farm, tel_plants_decayed), "Farm tail offset");
static_assert(offsetof(kaghd::State, done) == offsetof(kag::State, done), "State tail offset");
static_assert(ACT_I32 == 170, "ACT_I32");

// ------------------------------------------------------------------ encoder
// Verbatim as in kagsim.cpp (OPI/ITI/MOPI, conv_unit, conv_order, conv_action).
static const std::unordered_map<std::string, uint8_t> OPI = {
    {"PASS", 0}, {"NORTH", 1}, {"SOUTH", 2}, {"EAST", 3}, {"WEST", 4},
    {"PICKUP", 5}, {"DROP", 6}, {"PLACE", 7}, {"PLANT", 8}, {"WATER", 9},
    {"HARVEST", 10}, {"FERTILIZE", 11}, {"DIG", 12}, {"BUILD_COOP", 13},
    {"BUILD_PASTURE", 14}, {"FEED", 15}, {"COLLECT_FERTILIZER", 16},
    {"CARE", 17}};
static const std::unordered_map<std::string, uint8_t> ITI = {
    {"WHEAT", 0}, {"CARROT", 1}, {"TOMATO", 2}, {"STRAWBERRY", 3},
    {"MELON", 4}, {"EGG", 5}, {"MILK", 6}, {"WOOL", 7}, {"FERTILIZER", 8},
    {"GOOSE", 9}, {"COW", 10}, {"SHEEP", 11}};
static const std::unordered_map<std::string, uint8_t> MOPI = {
    {"HIRE", 1}, {"BUY_LAND", 2}, {"BUY_SEED", 3}, {"BUY_PRODUCT", 4},
    {"BUY_ANIMAL", 5}, {"SELL", 6}};

static kaghd::UnitAction conv_unit(const py::handle& h) {
    kaghd::UnitAction u;                // PASS by default
    if (!py::isinstance<py::list>(h)) return u;
    auto v = h.cast<py::list>();
    if (v.size() == 0) return u;
    auto it = OPI.find(py::str(v[0]).cast<std::string>());
    u.op = (it == OPI.end()) ? kaghd::OP_INVALID : it->second;
    if (v.size() >= 2 && py::isinstance<py::str>(v[1])) {
        auto ii = ITI.find(v[1].cast<std::string>());
        u.arg = (ii == ITI.end()) ? 255 : ii->second;
    }
    if (v.size() >= 3) {
        try { u.n = static_cast<int16_t>(v[2].cast<long>()); }
        catch (...) { u.n = 1; }
    }
    return u;
}

static kaghd::Order conv_order(const py::handle& h) {
    kaghd::Order o;                     // M_NONE by default
    if (!py::isinstance<py::list>(h)) return o;
    auto v = h.cast<py::list>();
    if (v.size() == 0) return o;
    auto it = MOPI.find(py::str(v[0]).cast<std::string>());
    if (it == MOPI.end()) return o;
    if (it->second == 1 || it->second == 2) {    // HIRE / BUY_LAND
        o.op = it->second; o.item = 0; o.n = 1;
        return o;
    }
    if (v.size() < 3) return o;
    o.op = it->second;
    auto ii = ITI.find(py::str(v[1]).cast<std::string>());
    o.item = (ii == ITI.end()) ? 255 : ii->second;
    try { o.n = static_cast<int32_t>(v[2].cast<long>()); }
    catch (...) { o.n = 0; if (o.n == 0) o.op = 0; }
    return o;
}

static kaghd::Action conv_action(const py::handle& t) {
    kaghd::Action a;
    a.clear();
    if (py::isinstance<py::dict>(t)) {
        auto d = t.cast<py::dict>();
        if (d.contains("farmer"))
            a.units[0] = conv_unit(d["farmer"]);
        int nu = 1;
        if (d.contains("hands")) {
            for (const auto& hh : d["hands"].cast<py::list>()) {
                if (nu >= kaghd::MAX_UNITS) break;
                a.units[nu++] = conv_unit(hh);
            }
        }
        a.n_units = nu;
        if (d.contains("market")) {
            for (const auto& oo : d["market"].cast<py::list>()) {
                if (a.n_orders >= kaghd::MAX_ORDER_SLOTS) break;
                // An invalid order stays as a PLACEHOLDER rather than being dropped
                // (slots i of both players are paired in lockstep).
                a.orders[a.n_orders++] = conv_order(oo);
            }
        }
    }
    return a;
}

static py::array_t<int32_t> encode_action(const py::handle& t) {
    py::array_t<int32_t> out(ACT_I32);
    kaghd::encode_action_i32(conv_action(t), out.mutable_data());
    return out;
}

static py::array_t<int32_t> encode_actions(const py::list& lst) {
    py::array_t<int32_t> out({(py::ssize_t)py::len(lst), (py::ssize_t)ACT_I32});
    int32_t* p = out.mutable_data();
    size_t k = 0;
    for (const auto& t : lst) kaghd::encode_action_i32(conv_action(t), p + (k++) * ACT_I32);
    return out;
}

// ------------------------------------------------------------------ batch
template <class SimT, class ActionT, class ConfigT>
struct BatchT {
    std::vector<SimT> sims;
    ConfigT base;
    int n;

    BatchT(int n_, int steps) : sims(n_), n(n_) {
        if (n_ <= 0) throw std::invalid_argument("n must be > 0");
        base.episode_steps = steps;
    }

    void reset(py::array_t<uint64_t, py::array::c_style | py::array::forcecast> seeds,
               py::object mask_o, py::object shops_o, py::object nshops_o) {
        if (seeds.ndim() != 1 || seeds.shape(0) != n) throw std::invalid_argument("seeds: shape (n,)");
        const uint64_t* sd = seeds.data();
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask, shops;
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> nshops;
        const uint8_t* mk = nullptr; const uint8_t* sh = nullptr; const int32_t* ns = nullptr;
        if (!mask_o.is_none()) {
            mask = mask_o.cast<decltype(mask)>();
            if (mask.ndim() != 1 || mask.shape(0) != n) throw std::invalid_argument("mask: shape (n,)");
            mk = mask.data();
        }
        if (!shops_o.is_none() || !nshops_o.is_none()) {
            if (shops_o.is_none() || nshops_o.is_none()) throw std::invalid_argument("shops and nshops go together");
            shops = shops_o.cast<decltype(shops)>();
            nshops = nshops_o.cast<decltype(nshops)>();
            if (shops.ndim() != 2 || shops.shape(0) != n || shops.shape(1) != 8) throw std::invalid_argument("shops: shape (n, 8)");
            if (nshops.ndim() != 1 || nshops.shape(0) != n) throw std::invalid_argument("nshops: shape (n,)");
            sh = shops.data(); ns = nshops.data();
            for (int i = 0; i < n; ++i) {
                if (ns[i] < 0 || ns[i] > 8) throw std::invalid_argument("nshops out of range 0..8");
                for (int k = 0; k < ns[i]; ++k)
                    if (sh[i * 8 + k] >= 8) throw std::invalid_argument("shop id out of range 0..7");
            }
        }
        for (int i = 0; i < n; ++i) {
            if (mk && !mk[i]) continue;
            kaghd::reset_one(sims[i], base, sd[i], sh ? sh + i * 8 : nullptr, ns ? ns[i] : 0);
        }
    }

    void step_raw(const int32_t* a, int m) {
        for (int i = 0; i < n; ++i) {
            const int32_t* row = a + (size_t)(i % m) * 2 * ACT_I32;
            kaghd::step_one<SimT, ActionT>(sims[i], row, row + ACT_I32);
        }
    }

    // acts: int32 (m, 2, 170); game i takes row i % m.
    void step(py::array_t<int32_t, py::array::c_style | py::array::forcecast> acts) {
        if (acts.ndim() != 3 || acts.shape(1) != 2 || acts.shape(2) != ACT_I32 || acts.shape(0) < 1)
            throw std::invalid_argument("acts: shape (m, 2, 170), m >= 1");
        int m = (int)acts.shape(0);
        const int32_t* a = acts.data();
        py::gil_scoped_release rel;
        step_raw(a, m);
    }

    // acts: int32 (T, m, 2, 170): T steps in a row (for speed measurements).
    // Returns seconds of pure engine time.
    double run(py::array_t<int32_t, py::array::c_style | py::array::forcecast> acts) {
        if (acts.ndim() != 4 || acts.shape(2) != 2 || acts.shape(3) != ACT_I32 || acts.shape(1) < 1)
            throw std::invalid_argument("acts: shape (T, m, 2, 170)");
        int T = (int)acts.shape(0), m = (int)acts.shape(1);
        const int32_t* a = acts.data();
        py::gil_scoped_release rel;
        auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < T; ++t) step_raw(a + (size_t)t * m * 2 * ACT_I32, m);
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(t1 - t0).count();
    }

    py::array_t<double> rewards() const {
        py::array_t<double> out({(py::ssize_t)n, (py::ssize_t)2});
        double* p = out.mutable_data();
        for (int i = 0; i < n; ++i) { p[2 * i] = sims[i].reward(0); p[2 * i + 1] = sims[i].reward(1); }
        return out;
    }
    py::array_t<uint8_t> done() const {
        py::array_t<uint8_t> out((py::ssize_t)n);
        uint8_t* p = out.mutable_data();
        for (int i = 0; i < n; ++i) p[i] = sims[i].st.done ? 1 : 0;
        return out;
    }
    py::array_t<int32_t> step_counts() const {
        py::array_t<int32_t> out((py::ssize_t)n);
        int32_t* p = out.mutable_data();
        for (int i = 0; i < n; ++i) p[i] = sims[i].st.step;
        return out;
    }
    py::array_t<uint8_t> serialize() const {
        const size_t S = kaghd::state_ser_bytes<decltype(sims[0].st)>();
        py::array_t<uint8_t> out({(py::ssize_t)n, (py::ssize_t)S});
        uint8_t* p = out.mutable_data();
        for (int i = 0; i < n; ++i) kaghd::serialize_state(sims[i].st, p + (size_t)i * S);
        return out;
    }
};

using HDBatch = BatchT<kaghd::Sim, kaghd::Action, kaghd::Config>;
using RefBatch = BatchT<kag::Sim, kag::Action, kag::Config>;

// ------------------------------------------------------------------ observe / telemetry (port of kagsim.cpp)
static const char* ITEM_NAMES[12] = {
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK",
    "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP"};
static const char* SHOP_NAMES[8] = {
    "BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
    "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"};
static const char* QUAD_NAMES[4] = {"NW", "NE", "SW", "SE"};

static py::object ser_tile(const kaghd::Tile& t) {
    using namespace kaghd;
    switch (t.kind) {
    case T_EMPTY: return py::none();
    case T_LOCKED: return py::str("LOCKED");
    case T_WEED: { py::dict d; d["kind"] = "WEED"; return d; }
    case T_PLANT: {
        py::dict d;
        d["kind"] = "PLANT";
        d["crop"] = ITEM_NAMES[t.what];
        d["planted_day"] = static_cast<int>(t.planted_day);
        d["watered_today"] = t.watered_today;
        d["consecutive_unwatered"] = static_cast<int>(t.consecutive_dry);
        d["yield_units"] = static_cast<int>(t.yield_units);
        d["max_lifespan_step"] = static_cast<int>(t.max_lifespan_step);
        d["fertilized_until_day"] = static_cast<int>(t.fertilized_until_day);
        return d;
    }
    case T_COOP:
    case T_PASTURE: {
        py::dict d;
        d["kind"] = (t.kind == T_COOP) ? "COOP" : "PASTURE";
        if (t.has_animal) {
            d["animal"] = ITEM_NAMES[t.what];
            d["placed_day"] = static_cast<int>(t.planted_day);
            d["yield_units"] = static_cast<int>(t.yield_units);
            d["consecutive_unfed"] = static_cast<int>(t.consecutive_dry);
            d["fed_today"] = t.fed_today;
            d["cared_today"] = t.cared_today;
            d["fertilizer_available"] = t.fertilizer_available;
            d["pending_care_bonus"] = static_cast<int>(t.pending_care_bonus);
        }
        return d;
    }
    }
    return py::none();
}

static py::dict ser_farm(const kaghd::Farm& f, int bs) {
    py::dict d;
    d["money"] = f.money;
    py::list tiles;
    for (int y = 0; y < bs; ++y) {
        py::list row;
        for (int x = 0; x < bs; ++x) row.append(ser_tile(f.tiles[y][x]));
        tiles.append(row);
    }
    d["tiles"] = tiles;
    py::list farmer;
    farmer.append(static_cast<int>(f.pos_x[0]));
    farmer.append(static_cast<int>(f.pos_y[0]));
    d["farmer"] = farmer;
    py::list hands;
    for (int u = 1; u < f.n_units; ++u) {
        py::list p;
        p.append(static_cast<int>(f.pos_x[u]));
        p.append(static_cast<int>(f.pos_y[u]));
        hands.append(p);
    }
    d["hands"] = hands;
    py::list quads;
    for (int q = 0; q < f.n_quadrants; ++q) quads.append(QUAD_NAMES[q]);
    d["unlocked_quadrants"] = quads;
    d["hires_today"] = f.hires_today;
    return d;
}

static py::dict observe(const HDBatch& b, int i, int player) {
    if (i < 0 || i >= b.n || player < 0 || player > 1) throw std::out_of_range("game/player");
    const kaghd::Sim& sim = b.sims[i];
    const auto& st = sim.st;
    py::dict o;
    o["remainingOverageTime"] = 60;
    o["step"] = st.step;
    o["player"] = player;
    py::list farms;
    farms.append(ser_farm(st.farms[0], sim.cfg.board_size));
    farms.append(ser_farm(st.farms[1], sim.cfg.board_size));
    o["farms"] = farms;
    py::dict inv, prices;
    for (int k = 0; k < kaghd::N_PRODUCTS; ++k) {
        inv[ITEM_NAMES[k]] = st.market.inventory[k];
        prices[ITEM_NAMES[k]] = st.market.prices[k];
    }
    py::dict market;
    market["inventory"] = inv;
    market["prices"] = prices;
    o["market"] = market;
    py::list shops;
    for (int k = 0; k < st.n_shops; ++k) shops.append(SHOP_NAMES[st.shops[k]]);
    py::dict town;
    town["unlocked_shops"] = shops;
    o["town"] = town;
    o["day"] = st.day;
    o["hour"] = st.hour;
    const kaghd::Farm& f = st.farms[player];
    py::dict shed;
    for (int k = 0; k < kaghd::N_ITEMS; ++k) shed[ITEM_NAMES[k]] = static_cast<int>(f.shed[k]);
    py::dict seeds;
    for (int k = 0; k < kaghd::N_CROPS; ++k) seeds[ITEM_NAMES[k]] = static_cast<int>(f.seeds[k]);
    py::list invs;
    for (int u = 0; u < f.n_units; ++u) {
        py::dict iu;
        for (int k = 0; k < f.inv_nkeys[u]; ++k) {
            int item = f.inv_keys[u][k];
            iu[ITEM_NAMES[item]] = static_cast<int>(f.inv[u][item]);
        }
        invs.append(iu);
    }
    py::dict priv;
    priv["shed"] = shed;
    priv["seeds"] = seeds;
    priv["inventories"] = invs;
    o["private"] = priv;
    return o;
}

static py::dict telemetry(const HDBatch& b, int i, int player) {
    using namespace kaghd;
    if (i < 0 || i >= b.n || player < 0 || player > 1) throw std::out_of_range("game/player");
    const Sim& sim = b.sims[i];
    const Farm& f = sim.st.farms[player];
    py::dict t;
    t["sell_dead_units"] = f.tel_sell_dead;
    t["refused_buy_product"] = f.tel_refused_product;
    t["refused_buy_seed"] = f.tel_refused_seed;
    t["refused_buy_animal"] = f.tel_refused_animal;
    t["refused_hire"] = f.tel_refused_hire;
    t["refused_buy_land"] = f.tel_refused_land;
    t["hire_paid"] = f.tel_hire_paid;
    t["sell_zero_fill"] = f.tel_sell_zero_fill;
    t["buy_zero_fill"] = f.tel_buy_zero_fill;
    t["sell_revenue"] = f.sell_revenue;
    t["total_spend"] = f.total_spend;
    int32_t disc = 0, sold = 0;
    for (int k = 0; k < N_ITEMS; ++k) { disc += f.discarded[k]; sold += f.sold_units[k]; }
    t["shed_discarded_units"] = disc;
    t["sold_units"] = sold;
    t["feed_no_wheat"] = f.tel_feed_no_wheat;
    t["feed_no_animal"] = f.tel_feed_no_animal;
    t["feed_redundant"] = f.tel_feed_redundant;
    t["water_dead"] = f.tel_water_dead;
    t["harvest_dead"] = f.tel_harvest_dead;
    t["plant_dead"] = f.tel_plant_dead;
    t["animals_escaped"] = f.tel_animals_escaped;
    t["escape_capital"] = f.tel_escape_capital;
    t["escape_yield_units"] = f.tel_escape_yield_units;
    t["escape_first_day"] = static_cast<int>(f.tel_escape_first_day);
    t["animal_unfed_days"] = f.tel_animal_unfed_days;
    t["plants_dry"] = f.tel_plants_dry;
    t["dry_seed_cost"] = f.tel_dry_seed_cost;
    t["dry_yield_units"] = f.tel_dry_yield_units;
    t["dry_first_day"] = static_cast<int>(f.tel_dry_first_day);
    t["plant_unwatered_days"] = f.tel_plant_unwatered_days;
    t["decay_units"] = f.tel_decay_units;
    t["plants_decayed"] = f.tel_plants_decayed;
    t["silent_loss_coins"] = f.tel_escape_capital + f.tel_dry_seed_cost;
    t["silent_loss_units"] = f.tel_escape_yield_units + f.tel_dry_yield_units
                             + f.tel_decay_units + disc;
    int32_t more_dead = f.tel_move_dead + f.tel_locked_dead + f.tel_pickup_dead + f.tel_drop_dead
                        + f.tel_place_dead + f.tel_build_dead + f.tel_dig_dead + f.tel_fertilize_dead
                        + f.tel_collect_dead + f.tel_care_dead;
    t["dead_actions"] = f.tel_feed_no_wheat + f.tel_feed_no_animal + f.tel_feed_redundant
                        + f.tel_water_dead + f.tel_harvest_dead + f.tel_plant_dead + more_dead;
    t["move_dead"] = f.tel_move_dead;
    t["locked_dead"] = f.tel_locked_dead;
    t["pickup_dead"] = f.tel_pickup_dead;
    t["drop_dead"] = f.tel_drop_dead;
    t["place_dead"] = f.tel_place_dead;
    t["build_dead"] = f.tel_build_dead;
    t["dig_dead"] = f.tel_dig_dead;
    t["fertilize_dead"] = f.tel_fertilize_dead;
    t["collect_dead"] = f.tel_collect_dead;
    t["care_dead"] = f.tel_care_dead;
    t["tile_cap_units"] = f.tel_tile_cap_units;
    t["fertilizer_forgone"] = f.tel_fertilizer_forgone;
    t["care_unfed"] = f.tel_care_unfed;
    t["animal_shed_days"] = f.tel_animal_shed_days;
    t["idle_tile_days"] = f.tel_idle_tile_days;
    t["weed_tile_days"] = f.tel_weed_tile_days;
    t["hand_pass_turns"] = f.tel_hand_pass_turns;
    t["farmer_pass_turns"] = f.tel_farmer_pass_turns;
    t["phantom_hand_actions"] = f.tel_phantom_hand_actions;
    t["sell_floor_units"] = f.tel_sell_floor_units;
    t["hire_premium"] = f.tel_hire_premium;
    int32_t shed_units = 0; double shed_value = 0; int32_t seeds = 0; double seed_cost = 0;
    for (int k = 0; k < N_PRODUCTS; ++k) { shed_units += f.shed[k]; shed_value += f.shed[k] * (double)sim.st.market.prices[k]; }
    for (int c = 0; c < N_CROPS; ++c) { seeds += f.seeds[c]; seed_cost += f.seeds[c] * (double)crop_def(c).seed; }
    t["shed_units_now"] = shed_units;
    t["shed_value_now"] = shed_value;
    t["animals_in_shed_now"] = (int)(f.shed[GOOSE] + f.shed[COW] + f.shed[SHEEP]);
    t["seeds_unplanted_now"] = seeds;
    t["seeds_unplanted_cost_now"] = seed_cost;
    return t;
}

// ------------------------------------------------------------------ prices
static py::array_t<int32_t> prices_hd(int lo, int hi) {
    if (hi < lo) throw std::invalid_argument("hi < lo");
    py::array_t<int32_t> out({(py::ssize_t)kaghd::N_PRODUCTS, (py::ssize_t)(hi - lo + 1)});
    int32_t* p = out.mutable_data();
    for (int it = 0; it < kaghd::N_PRODUCTS; ++it)
        for (int v = lo; v <= hi; ++v) p[(size_t)it * (hi - lo + 1) + (v - lo)] = kaghd::market_price(it, v);
    return out;
}
static py::array_t<int32_t> prices_ref(int lo, int hi) {
    if (hi < lo) throw std::invalid_argument("hi < lo");
    py::array_t<int32_t> out({(py::ssize_t)kag::N_PRODUCTS, (py::ssize_t)(hi - lo + 1)});
    int32_t* p = out.mutable_data();
    for (int it = 0; it < kag::N_PRODUCTS; ++it)
        for (int v = lo; v <= hi; ++v) p[(size_t)it * (hi - lo + 1) + (v - lo)] = kag::market_price(it, v);
    return out;
}

template <class B>
static void bind_batch(py::module_& m, const char* name, const char* doc) {
    py::class_<B>(m, name, doc)
        .def(py::init<int, int>(), py::arg("n"), py::arg("steps") = 720)
        .def("reset", &B::reset, py::arg("seeds"), py::arg("mask") = py::none(),
             py::arg("shops") = py::none(), py::arg("nshops") = py::none(),
             "Reset games: seeds uint64 (n,); mask uint8 (n,) - reset only mask!=0; "
             "shops uint8 (n,8) + nshops int32 (n,) - pin the first nshops shops.")
        .def("step", &B::step, py::arg("acts"),
             "One step of all games. acts int32 (m,2,170); game i takes row i % m.")
        .def("run", &B::run, py::arg("acts"),
             "T steps in a row: acts int32 (T,m,2,170). Returns seconds of engine work.")
        .def("rewards", &B::rewards, "float64 (n,2): bank of both seats.")
        .def("done", &B::done, "uint8 (n,)")
        .def("step_counts", &B::step_counts, "int32 (n,)")
        .def("serialize", &B::serialize, "uint8 (n, SER_BYTES): canonical serialization of the full State.")
        .def_readonly("n", &B::n);
}

PYBIND11_MODULE(kaghd_cpu, m) {
    m.doc() = "CPU build of the CUDA-capable Kaggriculture core (sim_hd.hpp) + the vendor sim.hpp reference for comparison.";
    bind_batch<HDBatch>(m, "Batch", "Batch of games on the sim_hd.hpp core (the same code as on the GPU).");
    bind_batch<RefBatch>(m, "RefBatch", "The same API on top of the ORIGINAL vendor/kagsim/sim/sim.hpp (reference).");
    m.def("observe", &observe, py::arg("batch"), py::arg("game"), py::arg("player"),
          "Observation of game `game` for seat `player`: the same dict as kagsim.Game.observe.");
    m.def("telemetry", &telemetry, py::arg("batch"), py::arg("game"), py::arg("player"),
          "Telemetry: the same dict as kagsim.Game.telemetry.");
    m.def("encode_action", &encode_action, py::arg("action"),
          "action dict {farmer,hands,market} -> int32[170] (as conv_action in kagsim).");
    m.def("encode_actions", &encode_actions, py::arg("actions"), "list of dicts -> int32 (len,170).");
    m.def("market_prices", &prices_hd, py::arg("lo"), py::arg("hi"), "int32 (9, hi-lo+1): market_price sim_hd.");
    m.def("ref_market_prices", &prices_ref, py::arg("lo"), py::arg("hi"), "the same for vendor sim.hpp.");
    m.def("amps_selfcheck", &kaghd::amps_selfcheck, "Number of mismatches of the amplitude hex literals with the formula (must be 0).");
    m.attr("ACT_I32") = kaghd::ACT_I32;
    m.attr("ACT_UNITS") = kaghd::ACT_UNITS;
    m.attr("ACT_NORD") = kaghd::ACT_NORD;
    m.attr("ACT_ORDERS") = kaghd::ACT_ORDERS;
    m.attr("MAX_UNITS") = kaghd::MAX_UNITS;
    m.attr("MAX_ORDER_SLOTS") = kaghd::MAX_ORDER_SLOTS;
    m.attr("SER_BYTES") = (int)kaghd::state_ser_bytes<kaghd::State>();
    m.attr("REF_SER_BYTES") = (int)kaghd::state_ser_bytes<kag::State>();
    m.attr("SIM_BYTES") = (int)sizeof(kaghd::Sim);
    m.attr("STATE_BYTES") = (int)sizeof(kaghd::State);
}
