// py_kagcore.cpp - pybind11 module kagcore: batched API on top of the kagsim engine for RL CPU workers.
//
//   import kagcore
//   gb = kagcore.GameBatch(seeds, 720)                  # B independent games
//   gb.encode_into(seats, tiles, units, unit_pos, unit_mask, glob, tmask_packed, mmask)   # without a dict observation
//   gb.step(unit_choices[B,2,16], market_choices[B,2,24], actions)   # actions[b] = None | (a0, a1), a_p = dict|None
//   gb.step_vs(seats, unit_choices[B,16], market_choices[B,24], opponent_action_dicts)
//   gb.observe(b, p)                                    # observation dict (for a Python opponent), as kagsim.Game
//
// Parsing of the opponent's action dict and building of the observation dict are taken FROM vendor/kagsim/python/
// kagsim.cpp ITSELF (included below as text; its PYBIND11_MODULE is turned into an unused function), so the
// conversion matches kagsim.Game.step/observe by construction. Encoding and the executor live in kagcore.hpp.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <chrono>
#include <string>
#include <vector>

// --- vendor kagsim.cpp without its module entry point
#pragma push_macro("PYBIND11_MODULE")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, variable, ...) \
    [[maybe_unused]] static void kagsim_vendor_bindings_unused(::pybind11::module_ & variable)
#include "../../vendor/kagsim/python/kagsim.cpp"
#undef PYBIND11_MODULE
#pragma pop_macro("PYBIND11_MODULE")

#include "kagcore.hpp"

namespace py = pybind11;

// Output buffer check: exactly the required dtype, C-contiguous, writable, required shape (no silent copies).
template <class T>
static T* out_buf(const py::array& a, const std::vector<py::ssize_t>& shape, const char* name) {
    if (!a.dtype().is(py::dtype::of<T>()))
        throw std::invalid_argument(std::string(name) + ": wrong dtype");
    if (!(a.flags() & py::array::c_style)) throw std::invalid_argument(std::string(name) + ": C-contiguous required");
    if (!a.writeable()) throw std::invalid_argument(std::string(name) + ": buffer is read-only");
    if ((size_t)a.ndim() != shape.size()) throw std::invalid_argument(std::string(name) + ": wrong number of dimensions");
    for (size_t k = 0; k < shape.size(); ++k)
        if (a.shape(k) != shape[k]) throw std::invalid_argument(std::string(name) + ": wrong shape");
    return static_cast<T*>(const_cast<void*>(a.data()));
}

using IArr = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

static const int32_t* in_buf(const IArr& a, const std::vector<py::ssize_t>& shape, const char* name) {
    if ((size_t)a.ndim() != shape.size()) throw std::invalid_argument(std::string(name) + ": wrong number of dimensions");
    for (size_t k = 0; k < shape.size(); ++k)
        if (a.shape(k) != shape[k]) throw std::invalid_argument(std::string(name) + ": wrong shape");
    return a.data();
}

// Engine action -> tuple (n_units, ((op,arg,n),...), n_orders, ((op,item,n),...)) for comparison in Python.
static py::tuple action_tuple(const kag::Action& a) {
    py::tuple us(a.n_units);
    for (int i = 0; i < a.n_units; ++i)
        us[i] = py::make_tuple((int)a.units[i].op, (int)a.units[i].arg, (int)a.units[i].n);
    py::tuple os(a.n_orders);
    for (int k = 0; k < a.n_orders; ++k)
        os[k] = py::make_tuple((int)a.orders[k].op, (int)a.orders[k].item, (int)a.orders[k].n);
    return py::make_tuple(a.n_units, us, a.n_orders, os);
}

struct GameBatch {
    std::vector<Game> games;           // vendor Game (kagsim.cpp): Sim + observe()
    int steps;
    py::object shops;

    GameBatch(const std::vector<uint64_t>& seeds, int steps_, const py::object& shops_)
        : steps(steps_), shops(shops_) {
        games.reserve(seeds.size());
        for (uint64_t s : seeds) games.emplace_back(s, steps, shops);
    }
    size_t size() const { return games.size(); }
    void check_b(int b) const {
        if (b < 0 || (size_t)b >= games.size()) throw std::out_of_range("game index out of range");
    }
    void reset(int b, uint64_t seed) { check_b(b); games[b] = Game(seed, steps, shops); }

    py::dict observe(int b, int p) const { check_b(b); return games[b].observe(p); }
    double reward(int b, int p) const { check_b(b); return games[b].sim.reward(p); }
    py::array_t<double> rewards() const {
        py::array_t<double> r({(py::ssize_t)games.size(), (py::ssize_t)2});
        auto m = r.mutable_unchecked<2>();
        for (size_t b = 0; b < games.size(); ++b) { m(b, 0) = games[b].sim.reward(0); m(b, 1) = games[b].sim.reward(1); }
        return r;
    }
    py::array_t<bool> done() const {
        py::array_t<bool> r((py::ssize_t)games.size());
        auto m = r.mutable_unchecked<1>();
        for (size_t b = 0; b < games.size(); ++b) m(b) = games[b].sim.st.done;
        return r;
    }
    py::array_t<int32_t> step_counts() const {
        py::array_t<int32_t> r((py::ssize_t)games.size());
        auto m = r.mutable_unchecked<1>();
        for (size_t b = 0; b < games.size(); ++b) m(b) = games[b].sim.st.step;
        return r;
    }

    // Buffer row i = game games_idx[i] (i by default), seat seats[i].
    void encode_into(const IArr& seats, const py::array& tiles, const py::array& units, const py::array& unit_pos,
                     const py::array& unit_mask, const py::array& glob, const py::array& tmask,
                     const py::array& mmask, const py::object& games_idx) {
        using namespace kagcore;
        if (seats.ndim() != 1) throw std::invalid_argument("seats: a one-dimensional array is required");
        const py::ssize_t n = seats.shape(0);
        std::vector<int> gi(n);
        if (games_idx.is_none()) {
            for (py::ssize_t i = 0; i < n; ++i) gi[i] = (int)i;
        } else {
            IArr g = py::cast<IArr>(games_idx);
            if (g.ndim() != 1 || g.shape(0) != n) throw std::invalid_argument("games: length does not match seats");
            for (py::ssize_t i = 0; i < n; ++i) gi[i] = g.data()[i];
        }
        const int32_t* sp = seats.data();
        for (py::ssize_t i = 0; i < n; ++i) {
            check_b(gi[i]);
            if (sp[i] != 0 && sp[i] != 1) throw std::invalid_argument("seats: seat must be 0 or 1");
        }
        float* t = out_buf<float>(tiles, {n, NT, FT}, "tiles");
        float* u = out_buf<float>(units, {n, MAXU, FU}, "units");
        int64_t* up = out_buf<int64_t>(unit_pos, {n, MAXU}, "unit_pos");
        bool* um = out_buf<bool>(unit_mask, {n, MAXU}, "unit_mask");
        float* g = out_buf<float>(glob, {n, FG}, "glob");
        uint8_t* tm = out_buf<uint8_t>(tmask, {n, MAXU, TMASK_BYTES}, "tmask_packed");
        bool* mm = out_buf<bool>(mmask, {n, MB}, "mmask");
        py::gil_scoped_release rel;
        for (py::ssize_t i = 0; i < n; ++i)
            encode(games[gi[i]].sim.st, sp[i], t + i * NT * FT, u + i * MAXU * FU, up + i * MAXU, um + i * MAXU,
                   g + i * FG, tm + i * MAXU * TMASK_BYTES, mm + i * MB);
    }

    // Engine action for seat p of game b from the choice (unit_choice[16], market_choice[24]) - for cross-checking.
    py::tuple execute_raw(int b, int p, const IArr& uc, const IArr& mc) const {
        check_b(b);
        const int32_t* u = in_buf(uc, {kagcore::MAXU}, "unit_choice");
        const int32_t* m = in_buf(mc, {kagcore::NHEADS}, "market_choice");
        kag::Action a;
        kagcore::execute(games[b].sim.st, p, u, m, a);
        return action_tuple(a);
    }

    // Common step: for each (b, p) the action comes either from a dict (from_dict = 1) or from the choice (0).
    void step_core(std::vector<kag::Action>& acts, const std::vector<uint8_t>& from_dict,
                   const int32_t* uc, const int32_t* mc, const std::vector<int>& seat_row) {
        py::gil_scoped_release rel;
        for (size_t b = 0; b < games.size(); ++b) {
            Sim& sim = games[b].sim;
            if (sim.st.done) continue;
            for (int p = 0; p < 2; ++p) {
                const size_t k = 2 * b + p;
                if (from_dict[k]) continue;
                const int row = seat_row[k];
                kagcore::execute(sim.st, p, uc + (size_t)row * kagcore::MAXU, mc + (size_t)row * kagcore::NHEADS, acts[k]);
            }
            sim.step(acts[2 * b], acts[2 * b + 1]);
        }
    }

    // unit_choices [B,2,16], market_choices [B,2,24]; actions: None or a list of length B of None | (a0, a1),
    // where a_p = action dict (as for kagsim.Game.step) or None (= take the choice from the arrays).
    void step(const IArr& uc, const IArr& mc, const py::object& actions) {
        const py::ssize_t B = (py::ssize_t)games.size();
        const int32_t* u = in_buf(uc, {B, 2, kagcore::MAXU}, "unit_choices");
        const int32_t* m = in_buf(mc, {B, 2, kagcore::NHEADS}, "market_choices");
        std::vector<kag::Action> acts(2 * B);
        std::vector<uint8_t> from_dict(2 * B, 0);
        std::vector<int> row(2 * B);
        for (py::ssize_t k = 0; k < 2 * B; ++k) row[k] = (int)k;
        if (!actions.is_none()) {
            py::sequence seq = py::cast<py::sequence>(actions);
            if ((py::ssize_t)py::len(seq) != B) throw std::invalid_argument("actions: length does not equal the number of games");
            for (py::ssize_t b = 0; b < B; ++b) {
                py::object e = seq[b];
                if (e.is_none()) continue;
                py::sequence pr = py::cast<py::sequence>(e);
                if (py::len(pr) != 2) throw std::invalid_argument("actions[b]: a pair (a0, a1) is required");
                for (int p = 0; p < 2; ++p) {
                    py::object a = pr[p];
                    if (a.is_none()) continue;
                    acts[2 * b + p] = conv_action(a);
                    from_dict[2 * b + p] = 1;
                }
            }
        }
        step_core(acts, from_dict, u, m, row);
    }

    // Our seat seats[b] plays from the choice (unit_choices [B,16], market_choices [B,24]), the other one plays the opponent dict
    // opponent_actions[b] (None = empty action, as in kagsim.Game.step).
    void step_vs(const IArr& seats, const IArr& uc, const IArr& mc, const py::object& opponent_actions) {
        const py::ssize_t B = (py::ssize_t)games.size();
        const int32_t* sp = in_buf(seats, {B}, "seats");
        const int32_t* u = in_buf(uc, {B, kagcore::MAXU}, "unit_choices");
        const int32_t* m = in_buf(mc, {B, kagcore::NHEADS}, "market_choices");
        py::sequence seq = py::cast<py::sequence>(opponent_actions);
        if ((py::ssize_t)py::len(seq) != B) throw std::invalid_argument("opponent_actions: length does not equal the number of games");
        std::vector<kag::Action> acts(2 * B);
        std::vector<uint8_t> from_dict(2 * B, 0);
        std::vector<int> row(2 * B, 0);
        for (py::ssize_t b = 0; b < B; ++b) {
            if (sp[b] != 0 && sp[b] != 1) throw std::invalid_argument("seats: seat must be 0 or 1");
            const int opp = 1 - sp[b];
            acts[2 * b + opp] = conv_action(seq[b]);
            from_dict[2 * b + opp] = 1;
            row[2 * b + sp[b]] = (int)b;
        }
        step_core(acts, from_dict, u, m, row);
    }
};

// Mean time of one encode in a pure C++ loop (without Python), ns.
static double bench_encode(const GameBatch& gb, int b, int p, int iters) {
    using namespace kagcore;
    gb.check_b(b);
    std::vector<float> t(NT * FT), u(MAXU * FU), g(FG);
    std::vector<int64_t> up(MAXU);
    std::vector<uint8_t> um(MAXU), tm(MAXU * TMASK_BYTES), mm(MB);
    const kag::State& st = gb.games[b].sim.st;
    py::gil_scoped_release rel;
    volatile uint32_t sink = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < iters; ++k) {
        encode(st, p, t.data(), u.data(), up.data(), reinterpret_cast<bool*>(um.data()), g.data(), tm.data(),
               reinterpret_cast<bool*>(mm.data()));
        sink = sink + tm[k % (MAXU * TMASK_BYTES)];
    }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
}

// Mean time of one execute in a pure C++ loop, ns.
static double bench_execute(const GameBatch& gb, int b, int p, const IArr& uc, const IArr& mc, int iters) {
    gb.check_b(b);
    const int32_t* u = in_buf(uc, {kagcore::MAXU}, "unit_choice");
    const int32_t* m = in_buf(mc, {kagcore::NHEADS}, "market_choice");
    const kag::State& st = gb.games[b].sim.st;
    kag::Action a;
    py::gil_scoped_release rel;
    volatile int sink = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < iters; ++k) {
        kagcore::execute(st, p, u, m, a);
        sink = sink + a.n_orders;
    }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
}

PYBIND11_MODULE(kagcore, m) {
    m.doc() = "C++ encode/execute of the Kaggriculture RL agent on top of the kagsim engine (bit-exact twin of rl/encode.py, rl/act.py)";
    m.attr("NT") = kagcore::NT;
    m.attr("NOPS") = kagcore::NOPS;
    m.attr("MAXU") = kagcore::MAXU;
    m.attr("FT") = kagcore::FT;
    m.attr("FU") = kagcore::FU;
    m.attr("FG") = kagcore::FG;
    m.attr("MB") = kagcore::MB;
    m.attr("NHEADS") = kagcore::NHEADS;
    m.attr("TMASK_BYTES") = kagcore::TMASK_BYTES;

    py::class_<GameBatch>(m, "GameBatch")
        .def(py::init<const std::vector<uint64_t>&, int, const py::object&>(), py::arg("seeds"),
             py::arg("steps") = 720, py::arg("shops") = py::none())
        .def("__len__", &GameBatch::size)
        .def("reset", &GameBatch::reset, py::arg("b"), py::arg("seed"), "New game in slot b.")
        .def("observe", &GameBatch::observe, py::arg("b"), py::arg("player"),
             "Observation dict of seat player in game b (as kagsim.Game.observe).")
        .def("reward", &GameBatch::reward, py::arg("b"), py::arg("player"))
        .def("rewards", &GameBatch::rewards, "float64 [B, 2]")
        .def_property_readonly("done", &GameBatch::done, "bool [B]")
        .def_property_readonly("step_counts", &GameBatch::step_counts, "int32 [B]")
        .def("encode_into", &GameBatch::encode_into, py::arg("seats"), py::arg("tiles"), py::arg("units"),
             py::arg("unit_pos"), py::arg("unit_mask"), py::arg("glob"), py::arg("tmask_packed"), py::arg("mmask"),
             py::arg("games") = py::none(),
             "Row i = game games[i] (i by default), seat seats[i]. Buffers: tiles f32[N,100,30], "
             "units f32[N,16,16], unit_pos i64[N,16], unit_mask bool[N,16], glob f32[N,77], "
             "tmask_packed u8[N,16,700], mmask bool[N,489]. The GIL is released.")
        .def("execute_raw", &GameBatch::execute_raw, py::arg("b"), py::arg("player"), py::arg("unit_choice"),
             py::arg("market_choice"), "Engine action as a tuple (n_units, units, n_orders, orders).")
        .def("step", &GameBatch::step, py::arg("unit_choices"), py::arg("market_choices"),
             py::arg("actions") = py::none(),
             "unit_choices i32[B,2,16] (target*56+operation), market_choices i32[B,2,24]; actions[b] = None | "
             "(a0, a1), a_p = action dict or None (= choice from the arrays). Finished games are skipped.")
        .def("step_vs", &GameBatch::step_vs, py::arg("seats"), py::arg("unit_choices"), py::arg("market_choices"),
             py::arg("opponent_actions"),
             "Our seat seats[b] from the choice (i32[B,16], i32[B,24]), the other one from the opponent dict (None = empty).");

    m.def("conv_action_raw", [](const py::object& d) { return action_tuple(conv_action(d)); }, py::arg("action"),
          "action dict -> engine action (kagsim.cpp converter) as a tuple, like execute_raw.");
    m.def("head_bins", [](int h) {
              std::vector<int> v;
              for (int i = 0; i < kagcore::head_nbins(h); ++i) v.push_back(kagcore::bin_value(h, i));
              return v;
          }, py::arg("h"), "Bins of market head h (for cross-checking against rl/spec.py).");
    m.def("op_engine", [](int op) {
              kag::UnitAction u;
              kagcore::op_to_engine(op, u);
              return py::make_tuple((int)u.op, (int)u.arg, (int)u.n);
          }, py::arg("op"));
    m.def("template_groups", [](int t) {
              std::string s;
              const char* L = "SPDAHL";
              for (int k = 0; k < 6; ++k) s += L[kagcore::tmpl_group(t, k)];
              return s;
          }, py::arg("t"));
    m.def("bench_encode", &bench_encode, py::arg("gb"), py::arg("b"), py::arg("player"), py::arg("iters"),
          "ns per encode in a C++ loop");
    m.def("bench_execute", &bench_execute, py::arg("gb"), py::arg("b"), py::arg("player"), py::arg("unit_choice"),
          py::arg("market_choice"), py::arg("iters"), "ns per execute in a C++ loop");
}
