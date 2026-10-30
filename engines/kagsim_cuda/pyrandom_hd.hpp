// Derived from vendor/kagsim/sim/pyrandom.hpp (kagsim 0.5.0, Apache License 2.0, see vendor/kagsim/LICENSE and
// THIRD_PARTY_NOTICES.md); modified as listed below.
//
// CPython-compatible Mersenne Twister: a copy of vendor/kagsim/sim/pyrandom.hpp
// usable in CUDA device code (host and device build the SAME code).
//
// Differences from the original (bit-identical behavior):
//   * all methods are marked KAG_HD (= __host__ __device__ under nvcc);
//   * the genrand_base() table on a std::vector with static
//     initialization is removed (not allowed in device code): init_by_array again, as in
//     CPython, starts with init_genrand(19650218), which is the same recurrence
//     formula with the same constant, i.e. exactly the same 624 words;
//   * std::memcpy/<cstring>/<vector> are no longer needed.
#pragma once
#include <stdint.h>

#ifndef KAG_HD
#if defined(__CUDACC__)
#define KAG_HD __host__ __device__
#else
#define KAG_HD
#endif
#endif

namespace kaghd {

class PyRandom {
public:
    KAG_HD explicit PyRandom(uint64_t key) { seed(key); }

    KAG_HD void seed(uint64_t key) {
        // CPython: abs(n) -> array of 32-bit words, little endian. A zero
        // key still gives an array of one word [0].
        uint32_t words[3];
        int n = 0;
        if (key == 0) {
            words[0] = 0;
            n = 1;
        } else {
            while (key) {
                words[n++] = static_cast<uint32_t>(key & 0xffffffffu);
                key >>= 32;
            }
        }
        init_by_array(words, n);
    }

    KAG_HD uint32_t genrand_uint32() {
        uint32_t y;
        if (index_ >= N) {
            for (int k = 0; k < N - M; ++k) {
                y = (mt_[k] & UPPER) | (mt_[k + 1] & LOWER);
                mt_[k] = mt_[k + M] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
            }
            for (int k = N - M; k < N - 1; ++k) {
                y = (mt_[k] & UPPER) | (mt_[k + 1] & LOWER);
                mt_[k] = mt_[k + (M - N)] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
            }
            y = (mt_[N - 1] & UPPER) | (mt_[0] & LOWER);
            mt_[N - 1] = mt_[M - 1] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
            index_ = 0;
        }
        y = mt_[index_++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= (y >> 18);
        return y;
    }

    // random.random(): 53 bits from two 32-bit draws. a*2^26 is exact
    // (a power of two), the sum < 2^53 is exact, multiplying by 2^-53 is exact, so
    // the result does not depend on FMA contraction: identical on CPU and GPU.
    KAG_HD double random() {
        uint32_t a = genrand_uint32() >> 5;
        uint32_t b = genrand_uint32() >> 6;
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
    }

    // random.getrandbits(k) for 0 < k <= 32.
    KAG_HD uint32_t getrandbits(int k) { return genrand_uint32() >> (32 - k); }

    // Random._randbelow_with_getrandbits(n), n > 0.
    KAG_HD uint32_t randbelow(uint32_t n) {
        int k = 0;
        for (uint32_t v = n; v; v >>= 1) ++k;   // n.bit_length()
        uint32_t r = getrandbits(k);
        while (r >= n) r = getrandbits(k);
        return r;
    }

    // random.choice(seq) -> index
    KAG_HD uint32_t choice_index(uint32_t len) { return randbelow(len); }

private:
    static constexpr int N = 624;
    static constexpr int M = 397;
    static constexpr uint32_t MATRIX_A = 0x9908b0dfu;
    static constexpr uint32_t UPPER = 0x80000000u;
    static constexpr uint32_t LOWER = 0x7fffffffu;

    uint32_t mt_[N];
    int index_ = N;

    KAG_HD void init_genrand(uint32_t s) {
        mt_[0] = s;
        for (int i = 1; i < N; ++i)
            mt_[i] = 1812433253u * (mt_[i - 1] ^ (mt_[i - 1] >> 30)) + static_cast<uint32_t>(i);
        index_ = N;
    }

    KAG_HD void init_by_array(const uint32_t* key, int keylen) {
        // The original does a memcpy from the cached init_genrand(19650218) table here;
        // we compute it with the same recurrence in place (as CPython does): the same words.
        init_genrand(19650218u);
        // Then the same two mixing passes as in the original
        // (the operations, their order and operands are unchanged).
        int i = 1, j = 0;
        // = max(N, keylen); without a ternary over the lvalue N, so that the static
        // constexpr member is not "odr-used" in device code.
        int k = keylen;
        if (k < N) k = N;
        uint32_t prev = mt_[0];
        while (k > 0) {
            int run = N - i;                       // until wrap-around
            if (run > k) run = k;
            if (keylen == 1) {
                const uint32_t kv = key[0];
                for (int n = run; n; --n) {
                    prev = (mt_[i] ^ ((prev ^ (prev >> 30)) * 1664525u)) + kv;
                    mt_[i] = prev;
                    ++i;
                }
            } else if (keylen == 2) {
                for (int n = run; n; --n) {
                    prev = (mt_[i] ^ ((prev ^ (prev >> 30)) * 1664525u))
                           + key[j] + static_cast<uint32_t>(j);
                    mt_[i] = prev;
                    ++i;
                    j ^= 1;                        // 0,1,0,1,...
                }
            } else {
                for (int n = run; n; --n) {
                    prev = (mt_[i] ^ ((prev ^ (prev >> 30)) * 1664525u))
                           + key[j] + static_cast<uint32_t>(j);
                    mt_[i] = prev;
                    ++i; ++j;
                    if (j >= keylen) j = 0;
                }
            }
            k -= run;
            if (i >= N) { mt_[0] = mt_[N - 1]; prev = mt_[0]; i = 1; }
        }
        k = N - 1;
        while (k > 0) {
            int run = N - i;
            if (run > k) run = k;
            for (int n = run; n; --n) {
                prev = (mt_[i] ^ ((prev ^ (prev >> 30)) * 1566083941u))
                       - static_cast<uint32_t>(i);
                mt_[i] = prev;
                ++i;
            }
            k -= run;
            if (i >= N) { mt_[0] = mt_[N - 1]; prev = mt_[0]; i = 1; }
        }
        mt_[0] = 0x80000000u;
        index_ = N;
    }
};

}  // namespace kaghd
