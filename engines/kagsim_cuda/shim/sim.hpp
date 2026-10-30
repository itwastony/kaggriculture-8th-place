// Part of the kagsim-derived files (Apache License 2.0, see vendor/kagsim/LICENSE and THIRD_PARTY_NOTICES.md).
//
// Shim: substitutes the CUDA-capable core sim_hd.hpp under the name "sim.hpp" and the
// namespace kag, so that code written against the vendor sim.hpp
// (e.g. engines/kagcore/kagcore.hpp) builds against kaghd without changes.
// Usage: -I<...>/engines/kagsim_cuda/shim -I<...>/engines/kagsim_cuda BEFORE the vendor path.
#pragma once
#include "sim_hd.hpp"
namespace kag = kaghd;
