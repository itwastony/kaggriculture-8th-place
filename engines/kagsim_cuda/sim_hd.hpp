// Derived from vendor/kagsim/sim/sim.hpp (kagsim 0.5.0, Apache License 2.0, see vendor/kagsim/LICENSE and
// THIRD_PARTY_NOTICES.md); modified as listed below.
//
// Kaggriculture: a copy of the vendor/kagsim/sim/sim.hpp core (bit-exact port of
// kaggriculture.py 1.32.7), usable in CUDA: one GPU thread = one game.
//
// Host and device build THE SAME code (no branches on __CUDA_ARCH__,
// except for the three arithmetic helpers dmul/dadd/dsub, where the device uses
// intrinsics with the same IEEE rounding). Therefore the CPU build of this file,
// checked against the original kagsim, also proves the logic of the GPU build.
//
// What is changed relative to the original (same behavior):
//   * all functions/methods needed by reset/step/reward are marked KAG_HD;
//   * std::min/std::max -> kmin/kmax (same semantics: min = b<a?b:a,
//     max = a<b?b:a); <algorithm>, <array>, <cstring> are not used;
//   * static tables (CROPS, ANIMALS, MARKET, SHOP_MASK, SHOP_MULT,
//     LAND_ORDER, LAND_PRICES) -> functions with a switch: host-constexpr arrays
//     cannot be indexed with a runtime index in device code;
//   * market_amps() (static std::array with lazy initialization) -> the same 18
//     amplitudes written as hex literals (bit-identical to the values that
//     the original and CPython compute; checked by amps_selfcheck() on the host);
//   * price: base ± amp*shape and the hinge tail are computed via dmul/dadd/dsub,
//     which forbid FMA contraction (on host: volatile, on device:
//     __dmul_rn/__dadd_rn/__dsub_rn). These are CPython semantics (each operation
//     is rounded separately) regardless of compiler flags (-ffp-contract,
//     --fmad);
//   * sqrt/log/log10/nearbyint are the global :: functions (available both on host and on
//     device). sqrt is IEEE-exact on both; log on device may differ from
//     host libm by 1 ulp; the margin to the rounding boundary (.5) for the branches with log over
//     all inventories [-20000, 80000] is >= 8.8e-8 relative (8 orders of magnitude
//     larger than a 1 ulp error); on the GPU it is checked by kag_price_check() (gpu_parity.py).
//     Exact "halves" occur only in the rational branches (e.g. CARROT at
//     inventory = 10450: 35 - amp*sqrt(450) = 10.5 exactly; reached in
//     real games); there the absence of FMA is what matters, see dmul/dsub;
//   * the KAG_ABLATE block (a profiling seam with an extern global) is removed;
//   * Sim::init(cfg) is added to initialize an object in device memory.
// Telemetry (tel_*) is fully kept: the State layout is the same as in the
// original, and the state is compared in full.
#pragma once
#include <stdint.h>
#include <math.h>
#include "pyrandom_hd.hpp"

namespace kaghd {

// ---------------------------------------------------------------- helpers
template <class T> KAG_HD inline T kmin(T a, T b) { return (b < a) ? b : a; }
template <class T> KAG_HD inline T kmax(T a, T b) { return (a < b) ? b : a; }

// Multiplication/addition/subtraction with separate rounding of each operation
// (no FMA contraction), as in CPython.
KAG_HD inline double dmul(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dmul_rn(a, b);
#else
    volatile double r = a * b;   // volatile: the compiler cannot contract into an FMA
    return r;
#endif
}
KAG_HD inline double dadd(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dadd_rn(a, b);
#else
    volatile double r = a + b;
    return r;
#endif
}
KAG_HD inline double dsub(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dsub_rn(a, b);
#else
    volatile double r = a - b;
    return r;
#endif
}

// ---------------------------------------------------------------- item space
// The PRODUCTS order matches kaggriculture.py. The first five products are
// also the five crops in CROPS order, so crop id = product id.
enum Item : uint8_t {
    WHEAT = 0, CARROT, TOMATO, STRAWBERRY, MELON, EGG, MILK, WOOL, FERTILIZER,
    GOOSE, COW, SHEEP, N_ITEMS
};
constexpr int N_PRODUCTS = 9;
constexpr int N_CROPS = 5;
constexpr int N_ANIMALS = 3;
KAG_HD inline bool is_crop(uint8_t i) { return i < N_CROPS; }
KAG_HD inline bool is_product(uint8_t i) { return i < N_PRODUCTS; }
KAG_HD inline bool is_animal(uint8_t i) { return i >= GOOSE && i < N_ITEMS; }

// ---------------------------------------------------------------- unit ops
enum Op : uint8_t {
    OP_PASS = 0, OP_NORTH, OP_SOUTH, OP_EAST, OP_WEST,
    OP_PICKUP, OP_DROP, OP_PLACE,
    OP_PLANT, OP_WATER, OP_HARVEST, OP_FERTILIZE, OP_DIG,
    OP_BUILD_COOP, OP_BUILD_PASTURE,
    OP_FEED, OP_COLLECT_FERTILIZER, OP_CARE, OP_INVALID
};

enum MOp : uint8_t {
    M_NONE = 0, M_HIRE, M_BUY_LAND, M_BUY_SEED, M_BUY_PRODUCT, M_BUY_ANIMAL, M_SELL
};

// ---------------------------------------------------------------- static data
// Functions with a switch instead of arrays (the same on host and device).
// Indices are always valid (checked by the calling code, as in the original);
// the default branch is unreachable and returns the last entry.
struct CropDef { int seed, first_yield_day, max_yield_day, interval, max_yield; bool ongoing; };
KAG_HD inline CropDef crop_def(int c) {
    switch (c) {
        case WHEAT:      return CropDef{  10, 2,  4, 0, 6, false };
        case CARROT:     return CropDef{  20, 2,  3, 0, 4, false };
        case TOMATO:     return CropDef{  50, 8,  8, 1, 4, true  };
        case STRAWBERRY: return CropDef{ 100, 10, 10, 2, 4, true };
        case MELON:
        default:         return CropDef{  80, 10, 12, 0, 6, false };
    }
}

enum Structure : uint8_t { ST_COOP = 0, ST_PASTURE = 1 };
struct AnimalDef { int cost; Structure structure; int first_yield_day, interval, max_held; Item product; };
// The argument is the animal index (item - GOOSE), as in ANIMALS[...] of the original.
KAG_HD inline AnimalDef animal_def(int a) {
    switch (a) {
        case 0:  return AnimalDef{ 300, ST_COOP,    4, 1, 4, EGG  };  // GOOSE
        case 1:  return AnimalDef{ 400, ST_PASTURE, 8, 2, 6, MILK };  // COW
        case 2:
        default: return AnimalDef{ 500, ST_PASTURE, 6, 3, 6, WOOL };  // SHEEP
    }
}

enum Shape : uint8_t { F_LINEAR, F_SQ, F_SQRT, F_LOG, F_LOG10, F_HINGE };
// amp_below/amp_above = target * base / shape(f, T, T) are derived
// constants; in the original they were computed lazily in market_amps(). Here they are
// hex literals, bit-identical to the result of the original and CPython (checked:
// amps_selfcheck() below and a /tmp script during development).
struct MarketDef {
    double base; int I0; double T; Shape below_f; double below_t; Shape above_f; double above_t;
    double amp_below, amp_above;
};
KAG_HD inline MarketDef market_def(int item) {
    switch (item) {
        case WHEAT:      return MarketDef{  25, 10000, 400, F_SQRT,   0.80, F_LOG,    0.20,
                                            0x1p+0, 0x1.ab18b4ce6d282p-1 };
        case CARROT:     return MarketDef{  35, 10000, 450, F_HINGE,  1.00, F_SQRT,   0.70,
                                            0x1.18p+5, 0x1.27aa37e5f5fc5p+0 };   // 1.32.7
        case TOMATO:     return MarketDef{  60, 10000, 200, F_HINGE,  0.40, F_SQRT,   0.60,
                                            0x1.8p+4, 0x1.45d5b5c3f4f6bp+1 };    // 1.32.7
        case STRAWBERRY: return MarketDef{ 120, 10000, 100, F_SQRT,   0.70, F_LINEAR, 1.60,
                                            0x1.0cccccccccccdp+3, 0x1.eb851eb851eb8p+0 };
        case MELON:      return MarketDef{ 250, 10000, 300, F_LOG,    0.20, F_SQ,     3.60,
                                            0x1.185a1f3bc103ap+3, 0x1.47ae147ae147bp-7 };
        case EGG:        return MarketDef{  50, 10000, 332, F_HINGE,  0.40, F_LOG,    0.20,
                                            0x1.4p+4, 0x1.b8c2b285463dep+0 };    // 1.32.7
        case MILK:       return MarketDef{ 160, 10000, 122, F_SQRT,   0.60, F_LINEAR, 1.60,
                                            0x1.16203534757d4p+3, 0x1.0c9714fbcda3bp+1 };
        case WOOL:       return MarketDef{ 200, 10000, 105, F_LOG,    0.20, F_SQ,     3.20,
                                            0x1.1279bd80336b2p+3, 0x1.db8b6f9266dd5p-5 };
        case FERTILIZER:
        default:         return MarketDef{ 100, 10000, 200, F_LINEAR, 0.40, F_LINEAR, 0.40,
                                            0x1.999999999999ap-3, 0x1.999999999999ap-3 };
    }
}

KAG_HD inline double shape(Shape f, double x, double T = 0.0) {
    if (x < 0.0) x = 0.0;
    switch (f) {
        case F_LINEAR: return x;
        case F_SQ:     return x * x;
        case F_SQRT:   return ::sqrt(x);
        case F_LOG:    return ::log(1.0 + x);
        case F_LOG10:  return ::log10(1.0 + x);
        case F_HINGE: {                 // 1.32.7: u + 8*max(0, u-1)^2
            if (T <= 0.0) return x;
            double u = x / T;
            double e = u - 1.0;
            if (e < 0.0) e = 0.0;
            // (8*e)*e == 8*(e*e) bit-for-bit (multiplication by a power of two);
            // the addition is without FMA contraction.
            return dadd(u, dmul(8.0 * e, e));
        }
    }
    return x;
}

// Host check: amplitude literals == the original formula on this host.
// Returns the number of mismatches (must be 0). Host only (uses
// host libm), not called from device code.
inline int amps_selfcheck() {
    int bad = 0;
    for (int i = 0; i < N_PRODUCTS; ++i) {
        MarketDef p = market_def(i);
        double below = p.below_t * p.base / shape(p.below_f, p.T, p.T);
        double above = p.above_t * p.base / shape(p.above_f, p.T, p.T);
        if (below != p.amp_below) ++bad;
        if (above != p.amp_above) ++bad;
    }
    return bad;
}

// Python: max(PRICE_FLOOR, int(round(price))). Python's round() on a float rounds
// to nearest even, exactly the default nearbyint mode (on the GPU the rounding
// mode is always "to nearest even").
KAG_HD inline int market_price(int item, int inv) {
    const MarketDef p = market_def(item);
    double price;
    if (inv < p.I0) price = dadd(p.base, dmul(p.amp_below, shape(p.below_f, static_cast<double>(p.I0 - inv), p.T)));
    else            price = dsub(p.base, dmul(p.amp_above, shape(p.above_f, static_cast<double>(inv - p.I0), p.T)));
    int v = static_cast<int>(::nearbyint(price));
    return v < 1 ? 1 : v;
}

// ---------------------------------------------------------------- shops / town
enum ShopId : uint8_t {
    SHOP_BAKERY = 0, SHOP_BRUNCH_SPOT, SHOP_FARMERS_MARKET, SHOP_ICE_CREAM_SHOP,
    SHOP_PET_CAFE, SHOP_PIZZA_SHOP, SHOP_SMOOTHIE_SHOP, SHOP_YARN_STORE, N_SHOPS
};
// The order is sorted(SHOPS), because the environment opens a shop via
// `rng.choice(sorted(SHOPS))`. An invalid id (>= 8) consumes nothing.
KAG_HD inline uint16_t shop_mask(int s) {
    switch (s) {
        case SHOP_BAKERY:         return (1u << EGG) | (1u << WHEAT);
        case SHOP_BRUNCH_SPOT:    return (1u << EGG) | (1u << WHEAT) | (1u << STRAWBERRY);
        case SHOP_FARMERS_MARKET: return (1u << WHEAT) | (1u << CARROT) | (1u << TOMATO) | (1u << STRAWBERRY);
        case SHOP_ICE_CREAM_SHOP: return (1u << STRAWBERRY) | (1u << MILK) | (1u << WHEAT);
        case SHOP_PET_CAFE:       return (1u << CARROT);
        case SHOP_PIZZA_SHOP:     return (1u << MILK) | (1u << TOMATO) | (1u << WHEAT);
        case SHOP_SMOOTHIE_SHOP:  return (1u << STRAWBERRY) | (1u << MILK);
        case SHOP_YARN_STORE:     return (1u << WOOL);
        default:                  return 0;
    }
}
// Mono-shops pull 2x.
KAG_HD inline int shop_mult(int s) {
    return (s == SHOP_PET_CAFE || s == SHOP_YARN_STORE) ? 2 : 1;
}
constexpr int MAX_SHOP_INSTANCES = 8;

// ---------------------------------------------------------------- config
struct Config {
    int episode_steps = 720;
    int board_size = 10;
    int starting_money = 3000;
    int max_orders = 10;
    int turns_per_day = 24;
    int shed_capacity = 100;
    double weed_chance = 0.005;
    int shop_unlock_interval = 3;
    int shop_sell_interval = 4;
    int center_sell_interval = 24;
    int hire_mult = 1;
    uint64_t seed = 0;

// Forced shop sequence (see the comment in the
// original sim.hpp): pins 8 shops; everything else, including
// weeds, reads the board exactly like the environment. Ids must be 0..7.
    uint8_t forced_shops[MAX_SHOP_INSTANCES] = {};
    int n_forced_shops = 0;
};

// ---------------------------------------------------------------- state
enum TileKind : uint8_t { T_EMPTY = 0, T_LOCKED, T_WEED, T_COOP, T_PASTURE, T_PLANT };

struct Tile {
    TileKind kind = T_EMPTY;
    uint8_t  what = 0;          // crop id (PLANT) or animal id (COOP/PASTURE with an animal)
    bool     has_animal = false;
    bool     watered_today = false;
    bool     fed_today = false;
    bool     cared_today = false;
    bool     fertilizer_available = false;
    int8_t   consecutive_dry = 0;    // not watered (plant) / not fed (animal)
    int8_t   yield_units = 0;
    int8_t   pending_care_bonus = 0;
    int16_t  planted_day = 0;        // or placed_day
    int32_t  max_lifespan_step = -1;
    int16_t  fertilized_until_day = -1;
};

constexpr int MAX_UNITS = 40;   // farmer + hired workers; hiring is Fibonacci-priced, this is ample
constexpr int BOARD = 10;

struct Farm {
    double money = 0;
    Tile tiles[BOARD][BOARD];
    int8_t pos_x[MAX_UNITS], pos_y[MAX_UNITS];
    int n_units = 1;                 // index 0 is the main farmer
    int n_quadrants = 1;             // NW is always unlocked
    int hires_today = 0;

    int16_t shed[N_ITEMS] = {0};
    int shed_total = 0;
    int16_t seeds[N_CROPS] = {0};
    int16_t inv[MAX_UNITS][N_ITEMS] = {{0}};
    // Unit inventories in Python are dicts, and several branches iterate them in
    // INSERTION order (this decides which items fit into the shed when it hits 100);
    // the order is modeled explicitly.
    uint8_t inv_keys[MAX_UNITS][N_ITEMS] = {{0}};
    uint8_t inv_nkeys[MAX_UNITS] = {0};

    // Instrumentation (the interpreter does not read it; layout and values
    // are kept as in the original, the state is compared in full).
    int32_t discarded[N_ITEMS] = {0};
    int32_t produced[N_ITEMS] = {0};
    int32_t sold_units[N_ITEMS] = {0};
    double  sell_revenue = 0;
    double  total_spend = 0;
    int32_t tel_sell_dead = 0;
    int32_t tel_refused_product = 0;
    int32_t tel_refused_seed = 0;
    int32_t tel_refused_animal = 0;
    int32_t tel_refused_hire = 0;
    int32_t tel_refused_land = 0;
    int32_t tel_hire_paid = 0;
    int32_t tel_sell_zero_fill = 0;
    int32_t tel_buy_zero_fill = 0;

    int32_t tel_tile_cap_units = 0;
    int32_t tel_fertilizer_forgone = 0;
    int32_t tel_care_unfed = 0;
    int32_t tel_animal_shed_days = 0;
    int32_t tel_idle_tile_days = 0;
    int32_t tel_weed_tile_days = 0;
    int32_t tel_hand_pass_turns = 0;
    int32_t tel_farmer_pass_turns = 0;
    int32_t tel_phantom_hand_actions = 0;
    int32_t tel_move_dead = 0;
    int32_t tel_locked_dead = 0;
    int32_t tel_pickup_dead = 0;
    int32_t tel_drop_dead = 0;
    int32_t tel_place_dead = 0;
    int32_t tel_build_dead = 0;
    int32_t tel_dig_dead = 0;
    int32_t tel_fertilize_dead = 0;
    int32_t tel_collect_dead = 0;
    int32_t tel_care_dead = 0;
    int32_t tel_sell_floor_units = 0;
    int32_t tel_hire_premium = 0;

    int32_t tel_feed_no_wheat = 0;
    int32_t tel_feed_no_animal = 0;
    int32_t tel_feed_redundant = 0;
    int32_t tel_water_dead = 0;
    int32_t tel_harvest_dead = 0;
    int32_t tel_plant_dead = 0;
    int32_t tel_animals_escaped = 0;
    double  tel_escape_capital = 0;
    int32_t tel_escape_yield_units = 0;
    int16_t tel_escape_first_day = -1;
    int32_t tel_animal_unfed_days = 0;
    int32_t tel_plants_dry = 0;
    double  tel_dry_seed_cost = 0;
    int32_t tel_dry_yield_units = 0;
    int16_t tel_dry_first_day = -1;
    int32_t tel_plant_unwatered_days = 0;
    int32_t tel_decay_units = 0;
    int32_t tel_plants_decayed = 0;

    KAG_HD void inv_add(int u, int item, int n) {
        if (n <= 0) return;
        if (inv[u][item] == 0) inv_keys[u][inv_nkeys[u]++] = (uint8_t)item;
        inv[u][item] += (int16_t)n;
    }
    KAG_HD void inv_erase(int u, int item) {
        inv[u][item] = 0;
        int k = 0;
        while (k < inv_nkeys[u] && inv_keys[u][k] != item) ++k;
        if (k == inv_nkeys[u]) return;
        for (int j = k; j + 1 < inv_nkeys[u]; ++j) inv_keys[u][j] = inv_keys[u][j + 1];
        inv_nkeys[u]--;
    }
    KAG_HD bool inv_take(int u, int item, int n) {
        if (inv[u][item] < n) return false;
        inv[u][item] -= (int16_t)n;
        if (inv[u][item] == 0) inv_erase(u, item);
        return true;
    }
    KAG_HD void inv_clear(int u) { for (int i = 0; i < N_ITEMS; ++i) inv[u][i] = 0; inv_nkeys[u] = 0; }
};

struct Market {
    int32_t inventory[N_PRODUCTS];
    int32_t prices[N_PRODUCTS];
};

struct State {
    Farm farms[2];
    Market market;
    uint8_t shops[MAX_SHOP_INSTANCES];
    int n_shops = 0;
    int step = 0;
    int day = 0;
    int hour = 0;
    bool done = false;
};

// ---------------------------------------------------------------- actions
struct UnitAction { uint8_t op = OP_PASS; uint8_t arg = 0; int16_t n = 1; };
struct Order { uint8_t op = M_NONE; uint8_t item = 0; int32_t n = 0; };

constexpr int MAX_ORDER_SLOTS = 16;   // order slots in the internal action (the engine reads max_orders=10)

struct Action {
    UnitAction units[MAX_UNITS];     // units[0] = farmer
    int n_units = 1;
    Order orders[MAX_ORDER_SLOTS];
    int n_orders = 0;
    KAG_HD void clear() { n_units = 1; units[0] = UnitAction{}; n_orders = 0; }
};

// ---------------------------------------------------------------- helpers
KAG_HD inline int quadrant_of(int x, int y, int bs) {
    // 0=NW 1=NE 2=SW 3=SE
    int half = bs / 2;
    return (y < half ? 0 : 2) + (x < half ? 0 : 1);
}
// Land unlocks in the order NE, SW, SE; prices 1000/2000/4000.
KAG_HD inline int land_order(int extra) { return extra + 1; }   // LAND_ORDER = {1, 2, 3}
KAG_HD inline int land_price(int extra) {                        // LAND_PRICES = {1000, 2000, 4000}
    switch (extra) {
        case 0:  return 1000;
        case 1:  return 2000;
        default: return 4000;
    }
}

KAG_HD inline void shed_access_tiles(int bs, int out[4][2]) {
    int h = bs / 2;
    out[0][0] = h - 1; out[0][1] = h - 1;
    out[1][0] = h;     out[1][1] = h - 1;
    out[2][0] = h - 1; out[2][1] = h;
    out[3][0] = h;     out[3][1] = h;
}
KAG_HD inline bool is_shed_adjacent(int x, int y, int bs) {
    int h = bs / 2;
    return (x == h - 1 || x == h) && (y == h - 1 || y == h);
}

KAG_HD inline int fib(int n) { int a = 1, b = 1; for (int i = 0; i < n; ++i) { int t = b; b = a + b; a = t; } return a; }

class Sim {
public:
    Config cfg;
    State st;

    KAG_HD explicit Sim(const Config& c = Config{}) : cfg(c) { reset(); }

    // For objects in raw memory (cudaMalloc / array): set the config and reset.
    KAG_HD void init(const Config& c) { cfg = c; reset(); }

    KAG_HD void reset() {
        st = State{};
        int h = cfg.board_size / 2;
        for (int p = 0; p < 2; ++p) {
            Farm& f = st.farms[p];
            f.money = cfg.starting_money;
            // nvcc 13.3: State{} in device code sets the Tile defaults only in row tiles[0][*];
            // in rows 1..9 max_lifespan_step and fertilized_until_day stay 0 instead of -1.
            // Set explicitly for all tiles; on the CPU the values are the same.
            for (int y = 0; y < BOARD; ++y)
                for (int x = 0; x < BOARD; ++x) {
                    f.tiles[y][x].max_lifespan_step = -1;
                    f.tiles[y][x].fertilized_until_day = -1;
                }
            for (int y = 0; y < cfg.board_size; ++y)
                for (int x = 0; x < cfg.board_size; ++x)
                    f.tiles[y][x].kind = (quadrant_of(x, y, cfg.board_size) == 0) ? T_EMPTY : T_LOCKED;
            f.n_units = 1;
            f.pos_x[0] = static_cast<int8_t>(h - 1);
            f.pos_y[0] = static_cast<int8_t>(h - 1);
        }
        for (int i = 0; i < N_PRODUCTS; ++i) {
            const MarketDef md = market_def(i);
            st.market.inventory[i] = md.I0;
            st.market.prices[i] = static_cast<int>(md.base);
        }
    }

    KAG_HD double reward(int p) const { return st.farms[p].money; }

    // One environment step from the actions of both players.
    KAG_HD void step(const Action& a0, const Action& a1) {
        if (st.done) return;
        const Action* acts[2] = { &a0, &a1 };
        int step_i = st.step;
        int day = step_i / cfg.turns_per_day;

        for (int p = 0; p < 2; ++p) apply_unit_actions(p, *acts[p], day);
        process_market(*acts[0], *acts[1]);
        town_consume(step_i);
        for (int p = 0; p < 2; ++p) decay_plants(st.farms[p], step_i);
        if ((step_i + 1) % cfg.turns_per_day == 0) end_of_day(day);

        st.step = step_i + 1;
        st.day = st.step / cfg.turns_per_day;
        st.hour = st.step % cfg.turns_per_day;
        if (step_i >= cfg.episode_steps - 2) st.done = true;
    }

private:
    // ------------------------------------------------------------ unit actions
    KAG_HD void apply_unit_actions(int p, const Action& a, int day) {
        Farm& f = st.farms[p];
        // Atomic PLANT check: if there are more requests for a crop than
        // seeds, ALL plantings of that crop for the turn are dropped. Demand
        // is counted over all submitted actions, including those addressed to
        // non-existent hired workers (as in Python).
        int demand[N_CROPS] = {0};
        for (int i = 0; i < a.n_units; ++i)
            if (a.units[i].op == OP_PLANT && a.units[i].arg < N_CROPS) demand[a.units[i].arg]++;
        int n = kmin(a.n_units, f.n_units);
        if (a.n_units > f.n_units) f.tel_phantom_hand_actions += a.n_units - f.n_units;
        for (int i = 1; i < f.n_units; ++i)          // a hired worker without an order = a hired worker that passed
            if (i >= a.n_units || a.units[i].op == OP_PASS) f.tel_hand_pass_turns += 1;
        if (a.n_units == 0 || a.units[0].op == OP_PASS) f.tel_farmer_pass_turns += 1;
        bool blocked[N_CROPS];
        for (int c = 0; c < N_CROPS; ++c) blocked[c] = demand[c] > f.seeds[c];

        for (int i = 0; i < n; ++i) {
            UnitAction u = a.units[i];
            if (u.op == OP_PLANT && u.arg < N_CROPS && blocked[u.arg]) {
                f.tel_plant_dead += 1;
                continue;
            }
            apply_unit(f, i, u, day);
        }
    }

    KAG_HD void apply_unit(Farm& f, int idx, const UnitAction& u, int day) {
        const int bs = cfg.board_size;
        int fx = f.pos_x[idx], fy = f.pos_y[idx];
        int16_t* inv = f.inv[idx];

        switch (u.op) {
            case OP_PASS: return;
            case OP_NORTH: case OP_SOUTH: case OP_EAST: case OP_WEST: {
                int nx = fx + (u.op == OP_EAST) - (u.op == OP_WEST);
                int ny = fy + (u.op == OP_SOUTH) - (u.op == OP_NORTH);
                // Moving onto LOCKED tiles is allowed (hired workers can spawn there).
                if (nx < 0 || nx >= bs || ny < 0 || ny >= bs) { f.tel_move_dead += 1; return; }
                f.pos_x[idx] = static_cast<int8_t>(nx);
                f.pos_y[idx] = static_cast<int8_t>(ny);
                return;
            }
            default: break;
        }

        Tile& tile = f.tiles[fy][fx];

        // Shed operations come before the LOCKED check: three of the four shed
        // access tiles are locked initially, but the shed always belongs to the player.
        if (u.op == OP_DROP) {
            if (!is_shed_adjacent(fx, fy, bs)) { f.tel_drop_dead += 1; return; }
            // Insertion order, as in `for item, n in list(inv.items())`.
            uint8_t keys[N_ITEMS];
            int nk = f.inv_nkeys[idx];
            for (int k = 0; k < nk; ++k) keys[k] = f.inv_keys[idx][k];
            for (int k = 0; k < nk; ++k) {
                int it = keys[k];
                if (inv[it] <= 0) { f.inv_erase(idx, it); continue; }
                int room = kmax(0, cfg.shed_capacity - f.shed_total);
                int take = kmin<int>(inv[it], room);
                if (take > 0) { f.shed[it] += take; f.shed_total += take; }
                f.discarded[it] += inv[it] - take;
                f.inv_erase(idx, it);
            }
            return;
        }
        if (u.op == OP_PICKUP) {
            if (!is_shed_adjacent(fx, fy, bs)) { f.tel_pickup_dead += 1; return; }
            if (u.n <= 0 || u.arg >= N_ITEMS) { f.tel_pickup_dead += 1; return; }
            int take = kmin<int>(u.n, f.shed[u.arg]);
            if (take <= 0) { f.tel_pickup_dead += 1; return; }
            f.shed[u.arg] -= take; f.shed_total -= take;
            f.inv_add(idx, u.arg, take);
            return;
        }
        if (u.op == OP_PLACE) {
            if (u.arg >= N_ITEMS) { f.tel_place_dead += 1; return; }
            if (is_animal(u.arg)) {
                const AnimalDef ad = animal_def(u.arg - GOOSE);
                TileKind want = (ad.structure == ST_COOP) ? T_COOP : T_PASTURE;
                if (tile.kind == want && !tile.has_animal) {
                    if (f.inv_take(idx, u.arg, 1)) {
                        Tile t{};
                        t.kind = want; t.what = u.arg; t.has_animal = true;
                        t.planted_day = static_cast<int16_t>(day);
                        tile = t;
                    } else {
                        f.tel_place_dead += 1;         // the right structure, but nothing to place
                    }
                    return;
                }
            }
            // Not an animal on its structure: PLACE next to the shed puts the item into the
            // shed; anywhere else it does nothing.
            if (is_shed_adjacent(fx, fy, bs)) {
                int n = kmin<int>(u.n, inv[u.arg]);
                if (n <= 0) { f.tel_place_dead += 1; return; }
                n = kmin(n, kmax(0, cfg.shed_capacity - f.shed_total));
                if (n <= 0) { f.tel_place_dead += 1; return; }
                f.inv_take(idx, u.arg, n);
                f.shed[u.arg] += n; f.shed_total += n;
            } else {
                f.tel_place_dead += 1;
            }
            return;
        }

        if (tile.kind == T_LOCKED) { f.tel_locked_dead += 1; return; }

        switch (u.op) {
            case OP_PLANT: {
                if (u.arg >= N_CROPS || tile.kind != T_EMPTY || f.seeds[u.arg] <= 0) {
                    f.tel_plant_dead += 1;
                    return;
                }
                f.seeds[u.arg] -= 1;
                const CropDef cd = crop_def(u.arg);
                Tile t{};
                t.kind = T_PLANT; t.what = u.arg;
                t.planted_day = static_cast<int16_t>(day);
                t.consecutive_dry = 1;                 // the planting day counts as unwatered
                t.yield_units = cd.ongoing ? 0 : 1;
                t.max_lifespan_step = cd.ongoing ? -1
                    : (day + cd.max_yield_day + 1) * cfg.turns_per_day;
                tile = t;
                return;
            }
            case OP_WATER: {
                if (tile.kind != T_PLANT || tile.watered_today) { f.tel_water_dead += 1; return; }
                tile.watered_today = true;
                const CropDef cd = crop_def(tile.what);
                if (!cd.ongoing) {
                    int age = day - tile.planted_day;
                    int w0 = (cd.max_yield_day + 1) / 2;
                    if (age >= w0 && age <= cd.max_yield_day) {
                        int bonus = (tile.fertilized_until_day >= day) ? 2 : 1;
                        int want = tile.yield_units + bonus;
                        int got = kmin(cd.max_yield, want);
                        f.tel_tile_cap_units += want - got;
                        tile.yield_units = static_cast<int8_t>(got);
                    }
                }
                return;
            }
            case OP_HARVEST: {
                if (tile.kind == T_EMPTY || tile.kind == T_WEED) { f.tel_harvest_dead += 1; return; }
                if (tile.yield_units <= 0) { f.tel_harvest_dead += 1; return; }
                if (tile.kind == T_PLANT) {
                    const CropDef cd = crop_def(tile.what);
                    if (day - tile.planted_day < cd.first_yield_day) { f.tel_harvest_dead += 1; return; }
                    f.inv_add(idx, tile.what, tile.yield_units);
                    f.produced[tile.what] += tile.yield_units;
                    tile.yield_units = 0;
                    if (!cd.ongoing) { Tile e{}; e.kind = T_EMPTY; tile = e; }
                } else if (tile.has_animal) {
                    const Item prod = animal_def(tile.what - GOOSE).product;
                    f.inv_add(idx, prod, tile.yield_units);
                    f.produced[prod] += tile.yield_units;
                    tile.yield_units = 0;
                }
                return;
            }
            case OP_FERTILIZE: {
                if (tile.kind != T_PLANT) { f.tel_fertilize_dead += 1; return; }
                if (!f.inv_take(idx, FERTILIZER, 1)) { f.tel_fertilize_dead += 1; return; }
                tile.fertilized_until_day = kmax<int16_t>(tile.fertilized_until_day,
                                                          static_cast<int16_t>(day + 2));
                return;
            }
            case OP_DIG: {
                if (tile.kind == T_EMPTY) { f.tel_dig_dead += 1; return; }
                if (tile.has_animal) { f.tel_dig_dead += 1; return; }   // animals cannot be removed
                Tile e{}; e.kind = T_EMPTY; tile = e;
                return;
            }
            case OP_BUILD_COOP:
                if (tile.kind != T_EMPTY) { f.tel_build_dead += 1; return; }
                { Tile t{}; t.kind = T_COOP; tile = t; }
                return;
            case OP_BUILD_PASTURE:
                if (tile.kind != T_EMPTY) { f.tel_build_dead += 1; return; }
                { Tile t{}; t.kind = T_PASTURE; tile = t; }
                return;
            case OP_FEED:
                if (!tile.has_animal) { f.tel_feed_no_animal += 1; return; }
                if (tile.fed_today) { f.tel_feed_redundant += 1; return; }
                if (!f.inv_take(idx, WHEAT, 1)) { f.tel_feed_no_wheat += 1; return; }
                tile.fed_today = true;
                return;
            case OP_COLLECT_FERTILIZER:
                if (!tile.has_animal || !tile.fertilizer_available) { f.tel_collect_dead += 1; return; }
                tile.fertilizer_available = false;
                f.inv_add(idx, FERTILIZER, 1);
                f.produced[FERTILIZER] += 1;
                return;
            case OP_CARE:
                if (!tile.has_animal || tile.cared_today) { f.tel_care_dead += 1; return; }
                tile.cared_today = true;
                return;
            default: return;
        }
    }

    // ------------------------------------------------------------ market
    struct OState { uint8_t type; uint8_t item; int32_t remaining; bool live; int32_t asked; };

    KAG_HD void process_market(const Action& a0, const Action& a1) {
        const Action* acts[2] = { &a0, &a1 };
        int nq[2];
        for (int p = 0; p < 2; ++p) nq[p] = kmin(acts[p]->n_orders, cfg.max_orders);
        int max_len = kmax(nq[0], nq[1]);

        for (int i = 0; i < max_len; ++i) {
            OState os[2];
            for (int p = 0; p < 2; ++p) {
                os[p].live = false;
                if (i < nq[p]) {
                    const Order& o = acts[p]->orders[i];
                    if (o.op == M_HIRE || o.op == M_BUY_LAND) {
                        os[p].type = o.op; os[p].live = true; os[p].remaining = 1; os[p].asked = 1;
                    } else if (o.op != M_NONE && o.n > 0) {
                        os[p].type = o.op; os[p].item = o.item; os[p].remaining = o.n;
                        os[p].asked = o.n; os[p].live = true;
                    }
                }
            }
            // Atomic orders are executed once, in player order.
            for (int p = 0; p < 2; ++p) {
                if (!os[p].live) continue;
                if (os[p].type == M_HIRE) { do_hire(st.farms[p]); os[p].live = false; }
                else if (os[p].type == M_BUY_LAND) { do_buy_land(st.farms[p]); os[p].live = false; }
            }

            // Per-unit lockstep: both players see the same inventory until commit.
            for (;;) {
                struct Q { bool ok = false; uint8_t type = 0; uint8_t item = 0; int price = 0; };
                Q q[2];
                for (int p = 0; p < 2; ++p) {
                    if (!os[p].live || os[p].remaining <= 0) continue;
                    uint8_t t = os[p].type, it = os[p].item;
                    if (t == M_SELL && is_product(it)) {
                        q[p] = { true, t, it, market_price(it, st.market.inventory[it]) };
                    } else if (t == M_BUY_PRODUCT && (it == WHEAT || it == FERTILIZER)) {
                        // Quote at the post-purchase inventory, so that a buy-sell round trip nets zero.
                        q[p] = { true, t, it, market_price(it, st.market.inventory[it] - 1) };
                    } else if (t == M_BUY_SEED && is_crop(it)) {
                        q[p] = { true, t, it, crop_def(it).seed };
                    } else if (t == M_BUY_ANIMAL && is_animal(it)) {
                        q[p] = { true, t, it, animal_def(it - GOOSE).cost };
                    } else {
                        os[p].live = false;
                    }
                }
                if (!q[0].ok && !q[1].ok) break;
                bool committed = false;
                for (int p = 0; p < 2; ++p) {
                    if (!q[p].ok) continue;
                    if (commit_unit(q[p].type, q[p].item, q[p].price, st.farms[p])) {
                        os[p].remaining -= 1; committed = true;
                    } else {
                        // the rest of the order silently dies; telemetry counts
                        // every undelivered unit
                        Farm& fp = st.farms[p];
                        switch (os[p].type) {
                            case M_SELL:        fp.tel_sell_dead += os[p].remaining;
                                                if (os[p].remaining == os[p].asked) fp.tel_sell_zero_fill += 1;
                                                break;
                            case M_BUY_PRODUCT: fp.tel_refused_product += os[p].remaining; break;
                            case M_BUY_SEED:    fp.tel_refused_seed += os[p].remaining; break;
                            case M_BUY_ANIMAL:  fp.tel_refused_animal += os[p].remaining; break;
                        }
                        if (os[p].type != M_SELL && os[p].remaining == os[p].asked) fp.tel_buy_zero_fill += 1;
                        os[p].live = false;
                    }
                }
                if (!committed) break;
            }
            refresh_prices();
        }
    }

    KAG_HD bool commit_unit(uint8_t op, uint8_t item, int price, Farm& f) {
        switch (op) {
            case M_SELL:
                if (f.shed[item] <= 0) return false;
                f.shed[item] -= 1; f.shed_total -= 1;
                f.money += price;
                if (price <= 1) f.tel_sell_floor_units += 1;
                f.sold_units[item] += 1;
                f.sell_revenue += price;
                if (price > 1) st.market.inventory[item] += 1;   // a sale at $1 adds no supply
                return true;
            case M_BUY_PRODUCT:
                if (f.money < price) return false;
                if (f.shed_total >= cfg.shed_capacity) return false;
                f.money -= price; f.total_spend += price;
                f.shed[item] += 1; f.shed_total += 1;
                st.market.inventory[item] -= 1;
                return true;
            case M_BUY_SEED:
                if (f.money < price) return false;
                f.money -= price; f.total_spend += price; f.seeds[item] += 1;
                return true;
            case M_BUY_ANIMAL:
                if (f.money < price) return false;
                if (f.shed_total >= cfg.shed_capacity) return false;
                f.money -= price; f.total_spend += price;
                f.shed[item] += 1; f.shed_total += 1;
                return true;
        }
        return false;
    }

    KAG_HD void refresh_prices() {
        for (int i = 0; i < N_PRODUCTS; ++i)
            st.market.prices[i] = market_price(i, st.market.inventory[i]);
    }

    KAG_HD void do_hire(Farm& f) {
        int cost = cfg.hire_mult * fib(f.hires_today);
        if (f.money < cost || f.n_units >= MAX_UNITS) { f.tel_refused_hire += 1; return; }
        f.money -= cost; f.total_spend += cost;
        f.tel_hire_paid += cost;
        f.tel_hire_premium += cost - cfg.hire_mult * fib(0);
        f.hires_today += 1;
        // Spawn on the least occupied shed access tile; ties go to the first
        // one in NWSE order (the tile does not have to be free).
        int acc[4][2]; shed_access_tiles(cfg.board_size, acc);
        int occ[4] = {0,0,0,0};
        for (int u = 0; u < f.n_units; ++u)
            for (int k = 0; k < 4; ++k)
                if (f.pos_x[u] == acc[k][0] && f.pos_y[u] == acc[k][1]) occ[k]++;
        int best = 0;
        for (int k = 1; k < 4; ++k) if (occ[k] < occ[best]) best = k;
        int idx = f.n_units++;
        f.pos_x[idx] = static_cast<int8_t>(acc[best][0]);
        f.pos_y[idx] = static_cast<int8_t>(acc[best][1]);
        f.inv_clear(idx);
    }

    KAG_HD void do_buy_land(Farm& f) {
        int extra = f.n_quadrants - 1;
        if (extra >= 3) { f.tel_refused_land += 1; return; }
        int cost = land_price(extra);
        if (f.money < cost) { f.tel_refused_land += 1; return; }
        f.money -= cost; f.total_spend += cost;
        int quad = land_order(extra);
        f.n_quadrants += 1;
        for (int y = 0; y < cfg.board_size; ++y)
            for (int x = 0; x < cfg.board_size; ++x)
                if (quadrant_of(x, y, cfg.board_size) == quad && f.tiles[y][x].kind == T_LOCKED)
                    f.tiles[y][x].kind = T_EMPTY;
    }

    // ------------------------------------------------------------ town / decay
    KAG_HD void town_consume(int step_i) {
        if (step_i % cfg.shop_sell_interval == 0) {
            for (int s = 0; s < st.n_shops; ++s) {
                uint16_t mask = shop_mask(st.shops[s]);
                int mult = shop_mult(st.shops[s]);
                for (int it = 0; it < N_PRODUCTS; ++it)
                    if (mask & (1u << it)) st.market.inventory[it] -= mult;
            }
        }
        if (step_i % cfg.center_sell_interval == 0) {
            for (int it = 0; it < N_PRODUCTS; ++it)
                if (it != FERTILIZER) st.market.inventory[it] -= 1;
        }
        refresh_prices();
    }

    KAG_HD void decay_plants(Farm& f, int step_i) {
        // Shortcut (provable, see the original): all max_lifespan_step values are multiples of
        // turns_per_day; with an even turns_per_day the scan on an odd step
        // is guaranteed to do nothing.
        if ((cfg.turns_per_day % 2) == 0 && (step_i % 2) != 0) return;
        for (int y = 0; y < cfg.board_size; ++y)
            for (int x = 0; x < cfg.board_size; ++x) {
                Tile& t = f.tiles[y][x];
                if (t.kind != T_PLANT) continue;
                if (t.max_lifespan_step < 0 || step_i < t.max_lifespan_step) continue;
                if ((step_i - t.max_lifespan_step) % 2 != 0) continue;
                t.yield_units -= 1;
                f.tel_decay_units += 1;
                if (t.yield_units <= 0) { Tile w{}; w.kind = T_WEED; t = w; f.tel_plants_decayed += 1; }
            }
    }

    KAG_HD void end_of_day(int day) {
        PyRandom rng((cfg.seed * 1000003ull) ^ static_cast<uint64_t>(day));
        for (int p = 0; p < 2; ++p) {
            Farm& f = st.farms[p];
            daily_refresh_plants(f, day);
            daily_refresh_animals(f, day);
            spawn_weeds(f, rng);
            drop_inventories(f);
            f.tel_animal_shed_days += f.shed[GOOSE] + f.shed[COW] + f.shed[SHEEP];
            for (int y = 0; y < cfg.board_size; ++y)
                for (int x = 0; x < cfg.board_size; ++x) {
                    TileKind k = f.tiles[y][x].kind;
                    if (k == T_EMPTY || k == T_WEED) f.tel_idle_tile_days += 1;
                    if (k == T_WEED) f.tel_weed_tile_days += 1;
                }
            int h = cfg.board_size / 2;
            f.n_units = 1;
            f.pos_x[0] = static_cast<int8_t>(h - 1);
            f.pos_y[0] = static_cast<int8_t>(h - 1);
            f.hires_today = 0;
            for (int u = 0; u < MAX_UNITS; ++u) f.inv_clear(u);
        }
        int next_day = day + 1;
        if (next_day > 0 && next_day % cfg.shop_unlock_interval == 0 && st.n_shops < MAX_SHOP_INSTANCES) {
            // The draw is consumed in any case, so pinning the
            // sequence changes the shop and nothing else.
            int drawn = rng.choice_index(N_SHOPS);
            st.shops[st.n_shops] = (st.n_shops < cfg.n_forced_shops)
                                       ? cfg.forced_shops[st.n_shops]
                                       : static_cast<uint8_t>(drawn);
            ++st.n_shops;
        }
    }

    KAG_HD void daily_refresh_plants(Farm& f, int day) {
        int next_day = day + 1;
        for (int y = 0; y < cfg.board_size; ++y)
            for (int x = 0; x < cfg.board_size; ++x) {
                Tile& t = f.tiles[y][x];
                if (t.kind != T_PLANT) continue;
                bool was_watered = t.watered_today;
                t.consecutive_dry = was_watered ? 0 : static_cast<int8_t>(t.consecutive_dry + 1);
                t.watered_today = false;
                if (!was_watered) f.tel_plant_unwatered_days += 1;
                if (t.consecutive_dry >= 2) {
                    f.tel_plants_dry += 1;
                    f.tel_dry_seed_cost += crop_def(t.what).seed;
                    f.tel_dry_yield_units += kmax<int>(0, t.yield_units);
                    if (f.tel_dry_first_day < 0) f.tel_dry_first_day = static_cast<int16_t>(day);
                    Tile w{}; w.kind = T_WEED; t = w; continue;
                }
                const CropDef cd = crop_def(t.what);
                if (!cd.ongoing) continue;
                int since = next_day - t.planted_day - cd.first_yield_day;
                if (since < 0) continue;
                if (since % cd.interval != 0) continue;
                int count = since / cd.interval + 1;
                if (count > cd.max_yield) continue;
                bool fert = was_watered && t.fertilized_until_day >= day;
                int want = t.yield_units + (fert ? 2 : 1);
                int got = kmin(cd.max_yield, want);
                f.tel_tile_cap_units += want - got;
                t.yield_units = static_cast<int8_t>(got);
                if (count == cd.max_yield) t.max_lifespan_step = (next_day + 1) * cfg.turns_per_day;
            }
    }

    KAG_HD void daily_refresh_animals(Farm& f, int day) {
        int next_day = day + 1;
        for (int y = 0; y < cfg.board_size; ++y)
            for (int x = 0; x < cfg.board_size; ++x) {
                Tile& t = f.tiles[y][x];
                if (!t.has_animal) continue;
                t.consecutive_dry = t.fed_today ? 0 : static_cast<int8_t>(t.consecutive_dry + 1);
                if (!t.fed_today) f.tel_animal_unfed_days += 1;
                if (t.consecutive_dry >= 2) {           // escapes; the structure remains
                    f.tel_animals_escaped += 1;
                    f.tel_escape_capital += animal_def(t.what - GOOSE).cost;
                    f.tel_escape_yield_units += kmax<int>(0, t.yield_units);
                    if (f.tel_escape_first_day < 0) f.tel_escape_first_day = static_cast<int16_t>(day);
                    TileKind k = t.kind;
                    Tile s{}; s.kind = k; t = s;
                    continue;
                }
                const AnimalDef ad = animal_def(t.what - GOOSE);
                int since = next_day - t.planted_day - ad.first_yield_day;
                if (since >= 0 && since % ad.interval == 0) {
                    int bonus = t.fed_today ? t.pending_care_bonus : 0;
                    int want = t.yield_units + 1 + bonus;
                    int got = kmin(ad.max_held, want);
                    f.tel_tile_cap_units += want - got;
                    t.yield_units = static_cast<int8_t>(got);
                    t.pending_care_bonus = 0;
                }
                if (t.cared_today && t.fed_today) t.pending_care_bonus += 1;
                if (t.cared_today && !t.fed_today) f.tel_care_unfed += 1;
                if (t.fertilizer_available) f.tel_fertilizer_forgone += 1;
                t.fertilizer_available = true;
                t.fed_today = false;
                t.cared_today = false;
            }
    }

    KAG_HD void spawn_weeds(Farm& f, PyRandom& rng) {
        for (int y = 0; y < cfg.board_size; ++y)
            for (int x = 0; x < cfg.board_size; ++x)
                if (f.tiles[y][x].kind == T_EMPTY && rng.random() < cfg.weed_chance)
                    f.tiles[y][x].kind = T_WEED;
    }

    KAG_HD void drop_inventories(Farm& f) {
        // Insertion order again: when the shed is almost full it decides which
        // items survive the day and which are discarded.
        for (int u = 0; u < f.n_units; ++u) {
            uint8_t keys[N_ITEMS];
            int nk = f.inv_nkeys[u];
            for (int k = 0; k < nk; ++k) keys[k] = f.inv_keys[u][k];
            for (int k = 0; k < nk; ++k) {
                int it = keys[k];
                if (f.inv[u][it] <= 0) { f.inv_erase(u, it); continue; }
                int room = kmax(0, cfg.shed_capacity - f.shed_total);
                int take = kmin<int>(f.inv[u][it], room);
                if (take > 0) { f.shed[it] += take; f.shed_total += take; }
                f.discarded[it] += f.inv[u][it] - take;
                f.inv_erase(u, it);
            }
        }
    }
};

}  // namespace kaghd
