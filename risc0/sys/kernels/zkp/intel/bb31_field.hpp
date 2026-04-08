#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

namespace esimd = sycl::ext::intel::esimd;
namespace esimd_exp = sycl::ext::intel::experimental::esimd;

// BabyBear field arithmetic via Intel ESIMD.
// All operations on simd<uint32_t, 16> -- 16 independent field elements per vector.
//
// Uses the sppark Montgomery convention:
//   P   = 0x78000001  (prime = 15 * 2^27 + 1)
//   M0  = 0x77ffffff  (-P^{-1} mod 2^32)
//   RR  = 0x45dddde3  (R^2 mod P, R = 2^32)
//   ONE = 0x0ffffffe  (R mod P = Montgomery form of 1)
//
// Montgomery multiplication: mont_mul(a, b) = a * b * R^{-1} mod P
// where a, b are in Montgomery form.

namespace bb31 {

static constexpr uint32_t MOD = 0x78000001;
static constexpr uint32_t M0  = 0x77ffffff;
static constexpr uint32_t RR  = 0x45dddde3;
static constexpr uint32_t ONE = 0x0ffffffe;

// NBETA for extension field: Montgomery form of -11 mod P (RISC Zero convention)
static constexpr uint32_t NBETA = 0x40000018;
// BETA for extension field: Montgomery form of 11 mod P
static constexpr uint32_t BETA = 0x37ffffe9;

using Vec16 = esimd::simd<uint32_t, 16>;
using Mask16 = esimd::simd_mask<16>;

// ============================================================================
// Core Montgomery Multiplication
// ============================================================================

// Montgomery multiplication: a * b * R^{-1} mod P
// Uses experimental::esimd::imul() for full 64-bit product.
// IGC compiles this to native MADW instructions at SIMD16.
ESIMD_INLINE Vec16 mont_mul(Vec16 a, Vec16 b) {
    const Vec16 P_vec(MOD);
    const Vec16 M0_vec(M0);

    // Step 1: Full 32x32 -> 64-bit product
    Vec16 lo;
    Vec16 hi = esimd_exp::imul(lo, a, b);

    // Step 2: Compute reduction factor
    Vec16 red = lo * M0_vec;

    // Step 3: red * P + (lo, hi)
    Vec16 prod_lo;
    Vec16 prod_hi = esimd_exp::imul(prod_lo, red, P_vec);

    // Montgomery property: lo + prod_lo = 0 (mod 2^32) or 2^32 exactly.
    // Proof: red = lo * M0, prod_lo = (red * P) mod 2^32.
    // Since M0 * P = -1 (mod 2^32), prod_lo = -lo (mod 2^32).
    // So lo + prod_lo = 0 when lo=0, or 2^32 when lo!=0.
    // The comparison lo != 0 depends only on imul#1 output (available early),
    // not on prod_lo from imul#2, breaking the dependency chain (~2 cyc win).
    Vec16 carry_val(0u);
    carry_val.merge(Vec16(1u), lo != Vec16(0u));

    // Result = hi + prod_hi + carry
    Vec16 result = hi + prod_hi + carry_val;

    // Conditional final subtraction (branchless)
    Vec16 reduced = result - P_vec;
    result.merge(reduced, result >= P_vec);

    return result;
}

// Montgomery multiplication WITHOUT final subtraction (lazy reduction).
// Result is in [0, 2*P). Caller must ensure no overflow accumulates.
// +20% speedup when used in butterfly sequences where add/sub follows.
ESIMD_INLINE Vec16 mont_mul_lazy(Vec16 a, Vec16 b) {
    const Vec16 P_vec(MOD);
    const Vec16 M0_vec(M0);

    Vec16 lo;
    Vec16 hi = esimd_exp::imul(lo, a, b);
    Vec16 red = lo * M0_vec;
    Vec16 prod_lo;
    Vec16 prod_hi = esimd_exp::imul(prod_lo, red, P_vec);
    // Montgomery carry elimination (same proof as mont_mul above).
    Vec16 carry_val(0u);
    carry_val.merge(Vec16(1u), lo != Vec16(0u));

    return hi + prod_hi + carry_val; // No final_sub -- result in [0, 2*P)
}

// Multiply by scalar broadcast
ESIMD_INLINE Vec16 mont_mul_scalar(Vec16 a, uint32_t b_scalar) {
    return mont_mul(a, Vec16(b_scalar));
}

// ============================================================================
// Field Addition / Subtraction
// ============================================================================

ESIMD_INLINE Vec16 field_add(Vec16 a, Vec16 b) {
    const Vec16 P_vec(MOD);
    Vec16 sum = a + b;
    Vec16 reduced = sum - P_vec;
    sum.merge(reduced, sum >= P_vec);
    return sum;
}

ESIMD_INLINE Vec16 field_sub(Vec16 a, Vec16 b) {
    const Vec16 P_vec(MOD);
    Vec16 diff = a - b;
    Vec16 corrected = diff + P_vec;
    diff.merge(corrected, a < b);
    return diff;
}

// Final subtraction only (for reducing lazy results back to [0, P))
ESIMD_INLINE Vec16 final_sub(Vec16 a) {
    const Vec16 P_vec(MOD);
    Vec16 reduced = a - P_vec;
    a.merge(reduced, a >= P_vec);
    return a;
}

// ============================================================================
// Conditional Operations
// ============================================================================

// mask ? a : b
ESIMD_INLINE Vec16 field_csel(Vec16 a, Vec16 b, Mask16 mask) {
    Vec16 result = b;
    result.merge(a, mask);
    return result;
}

// Conditional negation: if flag[lane], result = P - a; else result = a
// Handles a=0 correctly (P - 0 would give P, but we keep 0).
ESIMD_INLINE Vec16 field_cneg(Vec16 a, Mask16 flag) {
    Vec16 negated = Vec16(MOD) - a;
    // If a == 0, negated would be MOD (out of range), so keep 0
    Mask16 nonzero = (a != Vec16(0u));
    Mask16 do_negate = flag & nonzero;
    Vec16 result = a;
    result.merge(negated, do_negate);
    return result;
}

// ============================================================================
// Montgomery Squaring (optimized: a*a uses same operand twice)
// ============================================================================

ESIMD_INLINE Vec16 mont_sqr(Vec16 a) {
    return mont_mul(a, a);
}

// Squaring without final subtraction (for chains)
ESIMD_INLINE Vec16 mont_sqr_lazy(Vec16 a) {
    return mont_mul_lazy(a, a);
}

// ============================================================================
// Exponentiation Helpers
// ============================================================================

// n repeated squarings: a^(2^n)
// Uses full reduction on every iteration for safety.
// Agent review proved that alternating lazy/full can produce values > 2P
// when a lazy output (~2.96*10^9) is squared, yielding pre-subtraction
// results up to ~4.05*10^9 where a single final_sub is insufficient.
ESIMD_INLINE Vec16 sqr_n(Vec16 s, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        s = mont_sqr(s); // full reduction every iteration -> [0, P)
    }
    return s;
}

// n squarings followed by one multiplication: (a^(2^n)) * m
ESIMD_INLINE Vec16 sqr_n_mul(Vec16 s, uint32_t n, Vec16 m) {
    s = sqr_n(s, n);
    return mont_mul(s, m);
}

// ============================================================================
// Reciprocal (Fermat's Little Theorem: a^{-1} = a^{P-2} mod P)
// ============================================================================

// Optimized binary exponentiation chain for P-2 = 0x77FFFFFF
// Matching sppark's baby_bear.hpp reciprocal()
// Total: 31 squarings + 7 multiplications = 38 field ops
ESIMD_INLINE Vec16 reciprocal(Vec16 val) {
    Vec16 x11, xff, ret = val;

    x11 = sqr_n_mul(ret, 4, ret);      // x^(2^4+1) = x^17
    ret = sqr_n_mul(x11, 1, x11);      // x^(17*2+17) = x^51
    ret = sqr_n_mul(ret, 1, x11);      // x^(51*2+17) = x^119
    xff = sqr_n_mul(ret, 1, x11);      // x^(119*2+17) = x^255
    ret = sqr_n_mul(ret, 8, xff);      // x^(119*2^8+255)
    ret = sqr_n_mul(ret, 8, xff);      // extend by 8 more bits
    ret = sqr_n_mul(ret, 8, xff);      // extend by 8 more bits

    return ret;
}

// ============================================================================
// Heptaroot: a^{(P-1)/7}
// ============================================================================

// Optimized binary exponentiation chain for (P-1)/7
// Matching sppark's baby_bear.hpp heptaroot()
ESIMD_INLINE Vec16 heptaroot(Vec16 val) {
    Vec16 x03, x18, x1b, ret = val;

    x03 = sqr_n_mul(ret, 1, ret);      // x^3
    x18 = sqr_n(x03, 3);               // x^24
    x1b = mont_mul(x18, x03);          // x^27
    ret = mont_mul(x18, x1b);          // x^51
    ret = sqr_n_mul(ret, 6, x1b);      // extend
    ret = sqr_n_mul(ret, 6, x1b);      // extend
    ret = sqr_n_mul(ret, 6, x1b);      // extend
    ret = sqr_n_mul(ret, 6, x1b);      // extend
    ret = sqr_n_mul(ret, 1, val);       // final multiply by original

    return ret;
}

// ============================================================================
// Encode / Decode
// ============================================================================

ESIMD_INLINE Vec16 encode(Vec16 val) {
    return mont_mul(val, Vec16(RR));
}

ESIMD_INLINE Vec16 decode(Vec16 val) {
    return mont_mul(val, Vec16(1u));
}

} // namespace bb31
