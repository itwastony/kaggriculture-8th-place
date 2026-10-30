// kagsim: Python bindings for the bit-exact Kaggriculture C++ engine.
//
// L0 batch API: pre-convert action streams once, then play
// fixed-vs-fixed episodes at C++ speed (~1,000+ eps/sec/core):
//
//     import kagsim
//     s = kagsim.Stream(actions)              # list of {farmer,hands,market}
//     bank_a, bank_b = kagsim.run_episode(s, s2, seed)
//     results = kagsim.run_many([(s, s2, seed), ...])   # GIL released
//
// Engine core: sim.hpp, based on nikital7's bit-exact 1.32.6 port,
// patched to 1.32.7 (hinge scarcity branch) and re-validated trace-exact.
// Action encoding mirrors tools/export_trace.py exactly.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <atomic>
#include <thread>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "../sim/sim.hpp"

namespace py = pybind11;
using namespace kag;

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

static UnitAction conv_unit(const py::handle& h) {
    UnitAction u;                       // defaults to PASS
    if (!py::isinstance<py::list>(h)) return u;
    auto v = h.cast<py::list>();
    if (v.size() == 0) return u;
    auto it = OPI.find(py::str(v[0]).cast<std::string>());
    u.op = (it == OPI.end()) ? OP_INVALID : it->second;
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

static Order conv_order(const py::handle& h) {
    Order o;                            // defaults to M_NONE
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

static Action conv_action(const py::handle& t) {
    Action a;
    a.clear();
    if (py::isinstance<py::dict>(t)) {
        auto d = t.cast<py::dict>();
        if (d.contains("farmer"))
            a.units[0] = conv_unit(d["farmer"]);
        int nu = 1;
        if (d.contains("hands")) {
            for (const auto& hh : d["hands"].cast<py::list>()) {
                if (nu >= MAX_UNITS) break;
                a.units[nu++] = conv_unit(hh);
            }
        }
        a.n_units = nu;
        if (d.contains("market")) {
            for (const auto& oo : d["market"].cast<py::list>()) {
                if (a.n_orders >= 16) break;
                // Keep a malformed order as a PLACEHOLDER rather than dropping
                // it. The environment slices the raw list to max_orders and
                // parses slot by slot, so an empty or invalid entry still holds
                // its slot and the orders after it keep theirs; the per-unit
                // lockstep then pairs each player's slot i with the other's
                // slot i, and compacting the list shifts that pairing. Measured
                // on ladder episode 109033849 (2026-09-14): one `[]` in a
                // rival's list at step 362 moved two coins, and the final banks
                // missed the recording by 6,783 and 6,541.
                a.orders[a.n_orders++] = conv_order(oo);
            }
        }
    }
    return a;
}

struct Stream {
    std::vector<Action> turns;
    explicit Stream(const py::list& acts) {
        turns.reserve(acts.size());
        for (const auto& t : acts)
            turns.push_back(conv_action(t));
    }
    size_t size() const { return turns.size(); }
};

// ---------------------------------------------------------------- L1
// Step-mode: the simulator exposed turn by turn, emitting the SAME
// observation dicts the real interpreter hands each seat, so a Python
// agent (an adaptive rival, a decision layer, an RL loop) behaves
// identically inside the simulator. Validated two ways in
// tests/test_l1.py: a lockstep observation diff against the real
// environment, and a real adaptive agent reproducing its real-env
// banks to the dollar.

static const char* ITEM_NAMES[12] = {
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK",
    "WOOL", "FERTILIZER", "GOOSE", "COW", "SHEEP"};
static const char* SHOP_NAMES[8] = {
    "BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
    "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"};
static const char* QUAD_NAMES[4] = {"NW", "NE", "SW", "SE"};

static py::object ser_tile(const Tile& t) {
    switch (t.kind) {
    case T_EMPTY:
        return py::none();
    case T_LOCKED:
        return py::str("LOCKED");
    case T_WEED: {
        py::dict d;
        d["kind"] = "WEED";
        return d;
    }
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

static py::dict ser_farm(const Farm& f, int bs) {
    py::dict d;
    d["money"] = f.money;
    py::list tiles;
    for (int y = 0; y < bs; ++y) {
        py::list row;
        for (int x = 0; x < bs; ++x)
            row.append(ser_tile(f.tiles[y][x]));
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
    for (int q = 0; q < f.n_quadrants; ++q)
        quads.append(QUAD_NAMES[q]);
    d["unlocked_quadrants"] = quads;
    d["hires_today"] = f.hires_today;
    return d;
}

// A shop sequence given from Python: names in sorted(SHOPS) order, or indices.
// Anything unrecognised is refused loudly rather than silently ignored, because
// a pin that quietly does nothing is the worst possible control.
static void apply_forced_shops(Config& c, const py::object& shops) {
    if (shops.is_none()) return;
    static const char* NAMES[N_SHOPS] = {
        "BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
        "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"};
    py::list items = py::cast<py::list>(shops);
    if (py::len(items) > MAX_SHOP_INSTANCES)
        throw std::invalid_argument("at most 8 shops can be pinned");
    int n = 0;
    for (auto h : items) {
        int idx = -1;
        if (py::isinstance<py::str>(h)) {
            std::string s = py::cast<std::string>(h);
            for (int i = 0; i < N_SHOPS; ++i)
                if (s == NAMES[i]) { idx = i; break; }
            if (idx < 0) throw std::invalid_argument("unknown shop name: " + s);
        } else {
            idx = py::cast<int>(h);
            if (idx < 0 || idx >= N_SHOPS)
                throw std::invalid_argument("shop index out of range");
        }
        c.forced_shops[n++] = static_cast<uint8_t>(idx);
    }
    c.n_forced_shops = n;
}

struct Game {
    Sim sim;
    py::dict telemetry(int player) const {
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
        for (int i = 0; i < N_ITEMS; ++i) { disc += f.discarded[i]; sold += f.sold_units[i]; }
        t["shed_discarded_units"] = disc;
        t["sold_units"] = sold;
        // Silent leaks: actions the engine ignored, and value it destroyed with
        // no message. See the comment on the counters in sim.hpp.
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
        // One number an audit can watch: every coin and unit lost in silence.
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
        // Efficiency: value the farm had and did not use.
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
        // Stock and seed on hand NOW, which at the last step is what the season
        // stranded: the bank counts money only, so every unit still in the shed
        // and every seed still in the bag is worth exactly nothing.
        int32_t shed_units = 0; double shed_value = 0; int32_t seeds = 0; double seed_cost = 0;
        for (int i = 0; i < N_PRODUCTS; ++i) { shed_units += f.shed[i]; shed_value += f.shed[i] * (double)sim.st.market.prices[i]; }
        for (int c = 0; c < N_CROPS; ++c) { seeds += f.seeds[c]; seed_cost += f.seeds[c] * (double)CROPS[c].seed; }
        t["shed_units_now"] = shed_units;
        t["shed_value_now"] = shed_value;
        t["animals_in_shed_now"] = (int)(f.shed[GOOSE] + f.shed[COW] + f.shed[SHEEP]);
        t["seeds_unplanted_now"] = seeds;
        t["seeds_unplanted_cost_now"] = seed_cost;
        return t;
    }
    explicit Game(uint64_t seed, int steps = 720,
                  const py::object& shops = py::none()) {
        Config c;
        c.seed = seed;
        c.episode_steps = steps;
        apply_forced_shops(c, shops);
        sim = Sim(c);
    }
    py::dict observe(int player) const {
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
        for (int i = 0; i < N_PRODUCTS; ++i) {
            inv[ITEM_NAMES[i]] = st.market.inventory[i];
            prices[ITEM_NAMES[i]] = st.market.prices[i];
        }
        py::dict market;
        market["inventory"] = inv;
        market["prices"] = prices;
        o["market"] = market;
        py::list shops;
        for (int i = 0; i < st.n_shops; ++i)
            shops.append(SHOP_NAMES[st.shops[i]]);
        py::dict town;
        town["unlocked_shops"] = shops;
        o["town"] = town;
        o["day"] = st.day;
        o["hour"] = st.hour;
        const Farm& f = st.farms[player];
        py::dict shed;
        for (int i = 0; i < N_ITEMS; ++i)
            shed[ITEM_NAMES[i]] = static_cast<int>(f.shed[i]);
        py::dict seeds;
        for (int i = 0; i < N_CROPS; ++i)
            seeds[ITEM_NAMES[i]] = static_cast<int>(f.seeds[i]);
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
    void step(const py::handle& a, const py::handle& b) {
        sim.step(conv_action(a), conv_action(b));
    }
    bool done() const { return sim.st.done; }
    double reward(int p) const { return sim.reward(p); }
    int step_count() const { return sim.st.step; }
};


static std::pair<double, double> run_episode_raw_pinned(const Stream& sa,
                                                        const Stream& sb,
                                                        uint64_t seed, int steps,
                                                        const Config& pinned) {
    Config c = pinned;
    c.seed = seed;
    c.episode_steps = steps;
    Sim sim(c);
    Action empty;
    empty.clear();
    for (int t = 0; !sim.st.done; ++t) {
        const Action& a = (t < static_cast<int>(sa.turns.size()))
                              ? sa.turns[t] : empty;
        const Action& b = (t < static_cast<int>(sb.turns.size()))
                              ? sb.turns[t] : empty;
        sim.step(a, b);
        if (t > steps + 8) break;
    }
    return {sim.reward(0), sim.reward(1)};
}

static std::pair<double, double> run_episode_raw(const Stream& sa,
                                                 const Stream& sb,
                                                 uint64_t seed, int steps,
                                                 const py::object& shops = py::none()) {
    Config c;
    c.seed = seed;
    c.episode_steps = steps;
    apply_forced_shops(c, shops);
    Sim sim(c);
    Action empty;
    empty.clear();
    for (int t = 0; !sim.st.done; ++t) {
        const Action& a = (t < static_cast<int>(sa.turns.size()))
                              ? sa.turns[t] : empty;
        const Action& b = (t < static_cast<int>(sb.turns.size()))
                              ? sb.turns[t] : empty;
        sim.step(a, b);
    }
    return {sim.reward(0), sim.reward(1)};
}

PYBIND11_MODULE(kagsim, m) {
    m.doc() = "Bit-exact Kaggriculture engine (1.32.7), batch API. "
              "Based on nikital7's C++ port, hinge-patched and "
              "trace-validated.";
    py::class_<Stream>(m, "Stream")
        .def(py::init<const py::list&>())
        .def("__len__", &Stream::size);
    m.def("run_episode",
          [](const Stream& a, const Stream& b, uint64_t seed, int steps,
             const py::object& shops) {
              Config probe;
              apply_forced_shops(probe, shops);      // validate before releasing
              py::gil_scoped_release rel;
              return run_episode_raw_pinned(a, b, seed, steps, probe);
          },
          py::arg("stream_a"), py::arg("stream_b"), py::arg("seed"),
          py::arg("steps") = 720, py::arg("shops") = py::none());
    m.def("run_many",
          [](const std::vector<std::tuple<const Stream*, const Stream*,
                                          uint64_t>>& jobs, int steps,
             int threads) {
              // Episodes are independent and seeded independently, so this
              // parallelises with no effect on any result: job i writes only
              // out[i], the Streams are const and shared read-only, and no
              // RNG state crosses episodes. Verified by the golden tests,
              // which are bit-exact.
              std::vector<std::pair<double, double>> out(jobs.size());
              py::gil_scoped_release rel;
              unsigned hw = std::thread::hardware_concurrency();
              int n = threads > 0 ? threads
                                  : static_cast<int>(hw ? hw : 1u);
              n = std::max(1, std::min<int>(n, static_cast<int>(jobs.size())));
              if (n == 1) {
                  for (size_t i = 0; i < jobs.size(); ++i) {
                      const auto& [a, b, seed] = jobs[i];
                      out[i] = run_episode_raw(*a, *b, seed, steps);
                  }
                  return out;
              }
              std::atomic<size_t> next{0};
              std::vector<std::thread> pool;
              pool.reserve(n);
              for (int t = 0; t < n; ++t) {
                  pool.emplace_back([&] {
                      for (;;) {
                          size_t i = next.fetch_add(1,
                                                    std::memory_order_relaxed);
                          if (i >= jobs.size()) return;
                          const auto& [a, b, seed] = jobs[i];
                          out[i] = run_episode_raw(*a, *b, seed, steps);
                      }
                  });
              }
              for (auto& th : pool) th.join();
              return out;
          },
          py::arg("jobs"), py::arg("steps") = 720, py::arg("threads") = 0,
          "Play many episodes. threads=0 uses every core; 1 forces the "
          "sequential path. Results are identical either way.");
    py::class_<Game>(m, "Game")
        .def(py::init<uint64_t, int, const py::object&>(), py::arg("seed"),
             py::arg("steps") = 720, py::arg("shops") = py::none())
        .def("observe", &Game::observe, py::arg("player"),
             "The observation dict the real interpreter hands this seat "
             "at the current step.")
        .def("step", &Game::step, py::arg("action_a"), py::arg("action_b"),
             "Advance one turn with raw action dicts "
             "({farmer, hands, market}).")
        .def("reward", &Game::reward, py::arg("player"))
        .def("telemetry", &Game::telemetry, py::arg("player"),
             "Settle telemetry for one player: counters of what the engine "
             "did SILENTLY (refused purchases by op, dead sell units, shed-"
             "cap destruction, real hire cost). Instrumentation only; "
             "behaviour and parity are unaffected.")
        .def_property_readonly("done", &Game::done)
        .def_property_readonly("step_count", &Game::step_count);
    m.attr("__version__") = "0.4.0";
    m.attr("ENGINE_VERSION") = "1.32.7";
}
