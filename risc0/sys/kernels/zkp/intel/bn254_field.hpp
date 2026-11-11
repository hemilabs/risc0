#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cstdint>

namespace esimd = sycl::ext::intel::esimd;
namespace esimd_exp = sycl::ext::intel::experimental::esimd;

// BN254 scalar field (Fr) arithmetic via Intel ESIMD.
// Lane-per-hash SIMD16 layout: one field element = 8 × simd<uint32_t, 16>.
// Each SIMD16 lane holds one independent field element (16 parallel elements).
//
// Constants from sppark alt_bn128.hpp (ALT_BN128_r):
//   r (modulus) = 21888242871839275222246405745257275088548364400416034343698204186575808495617
//   M0 = -r^{-1} mod 2^32 = 0xefffffff
//   rRR = (1<<512) mod r  (for to-Montgomery conversion)
//   rONE = (1<<256) mod r (Montgomery form of 1)
//
// Limbs are little-endian: limb[0] is LSB.

namespace bn254 {

using Vec16 = esimd::simd<uint32_t, 16>;
using Mask16 = esimd::simd_mask<16>;

// Number of 32-bit limbs (254 bits rounds up to 8 × 32 = 256)
static constexpr int N_LIMBS = 8;

// Modulus r = 0x30644e72e131a029_b85045b68181585d_2833e84879b97091_43e1f593f0000001
static constexpr uint32_t R_MOD[N_LIMBS] = {
    0xf0000001u, 0x43e1f593u, 0x79b97091u, 0x2833e848u,
    0x8181585du, 0xb85045b6u, 0xe131a029u, 0x30644e72u
};

// -r^{-1} mod 2^32
static constexpr uint32_t R_M0 = 0xefffffffu;

// (1<<512) mod r -- used for to-Montgomery conversion
static constexpr uint32_t R_RR[N_LIMBS] = {
    0xae216da7u, 0x1bb8e645u, 0xe35c59e3u, 0x53fe3ab1u,
    0x53bb8085u, 0x8c49833du, 0x7f4e44a5u, 0x0216d0b1u
};

// (1<<256) mod r -- Montgomery form of 1
static constexpr uint32_t R_ONE[N_LIMBS] = {
    0x4ffffffbu, 0xac96341cu, 0x9f60cd29u, 0x36fc7695u,
    0x7879462eu, 0x666ea36fu, 0x9a07df2fu, 0x0e0a77c1u
};

// ============================================================================
// Fp: Field element as 8 limbs, each limb is Vec16 (one per SIMD lane)
// ============================================================================
struct Fp {
    Vec16 v[N_LIMBS];

    ESIMD_INLINE Fp() = default;

    // Construct zero
    ESIMD_INLINE static Fp zero() {
        Fp x;
        #pragma unroll
        for (int i = 0; i < N_LIMBS; ++i) x.v[i] = Vec16(0u);
        return x;
    }

    // Construct Montgomery-one (broadcast scalar constants into each lane)
    ESIMD_INLINE static Fp one() {
        Fp x;
        #pragma unroll
        for (int i = 0; i < N_LIMBS; ++i) x.v[i] = Vec16(R_ONE[i]);
        return x;
    }
};

// ============================================================================
// Core Montgomery Multiplication (CIOS)
// ============================================================================
//
// Classical CIOS (Koç et al.): Coarsely Integrated Operand Scanning.
// For each limb b[i], accumulate (a * b[i]) into t, then do one Montgomery
// reduction step (compute m = t[0]*M0, add m*r to t, right-shift by one limb).
//
// Invariant: carry fits in u32 (proven for u32*u32 + u32 + u32 since max
// imul high-half is 2^32 - 2, and carry_new = hi + c1 + c2 <= 2^32 - 1).
//
ESIMD_INLINE Fp mont_mul(const Fp& a, const Fp& b) {
    // Accumulator: N+1 words (extra word holds overflow from final add)
    Vec16 t[N_LIMBS + 1];
    #pragma unroll
    for (int j = 0; j <= N_LIMBS; ++j) t[j] = Vec16(0u);

    #pragma unroll
    for (int i = 0; i < N_LIMBS; ++i) {
        const Vec16 bi = b.v[i];

        // Phase 1: t += a * b[i]
        Vec16 carry = Vec16(0u);
        #pragma unroll
        for (int j = 0; j < N_LIMBS; ++j) {
            // (hi, lo) = a[j] * b[i]
            Vec16 lo;
            Vec16 hi = esimd_exp::imul(lo, a.v[j], bi);

            // t[j] += lo + carry (two addc steps, propagate into hi)
            Vec16 c1, c2;
            t[j] = esimd::addc(c1, t[j], lo);
            t[j] = esimd::addc(c2, t[j], carry);
            // new carry = hi + c1 + c2 (fits in u32, see invariant above)
            carry = hi + c1 + c2;
        }
        // Absorb final carry into the top word
        t[N_LIMBS] = t[N_LIMBS] + carry;

        // Phase 2: compute m = t[0] * M0 (low 32 bits only)
        const Vec16 m = t[0] * Vec16(R_M0);

        // Phase 3: t += m * r (this should zero t[0])
        carry = Vec16(0u);
        #pragma unroll
        for (int j = 0; j < N_LIMBS; ++j) {
            Vec16 lo;
            Vec16 hi = esimd_exp::imul(lo, m, Vec16(R_MOD[j]));

            Vec16 c1, c2;
            t[j] = esimd::addc(c1, t[j], lo);
            t[j] = esimd::addc(c2, t[j], carry);
            carry = hi + c1 + c2;
        }
        t[N_LIMBS] = t[N_LIMBS] + carry;

        // Phase 4: right-shift by one limb (t[0] is zero now)
        // Shift: t[j] = t[j+1] for j in 0..N_LIMBS, t[N_LIMBS] = 0
        #pragma unroll
        for (int j = 0; j < N_LIMBS; ++j) {
            t[j] = t[j + 1];
        }
        t[N_LIMBS] = Vec16(0u);
    }

    // Final conditional subtraction: if t >= r, compute t - r; else keep t.
    // Multi-limb compare via borrow chain: compute (t - r) with borrow.
    // If borrow == 0 at the top, t >= r, so use t-r; else use t.
    Vec16 borrow = Vec16(0u);
    Vec16 r_sub[N_LIMBS];
    #pragma unroll
    for (int j = 0; j < N_LIMBS; ++j) {
        // r_sub[j] = t[j] - R_MOD[j] - borrow
        Vec16 b1, b2;
        Vec16 tmp = esimd::subb(b1, t[j], Vec16(R_MOD[j]));
        r_sub[j] = esimd::subb(b2, tmp, borrow);
        borrow = b1 + b2;
    }

    // keep_sub lanes: where borrow == 0 (meaning t >= r, subtraction succeeded)
    Mask16 keep_sub = (borrow == Vec16(0u));

    Fp out;
    #pragma unroll
    for (int j = 0; j < N_LIMBS; ++j) {
        out.v[j] = t[j];
        out.v[j].merge(r_sub[j], keep_sub);
    }
    return out;
}

// ============================================================================
// Field Addition (with conditional subtraction)
// ============================================================================
ESIMD_INLINE Fp add_mod(const Fp& a, const Fp& b) {
    Fp sum;
    Vec16 carry = Vec16(0u);
    #pragma unroll
    for (int j = 0; j < N_LIMBS; ++j) {
        Vec16 c1, c2;
        Vec16 s = esimd::addc(c1, a.v[j], b.v[j]);
        sum.v[j] = esimd::addc(c2, s, carry);
        carry = c1 + c2;
    }
    // carry bit could be 1 (means sum >= 2^256)
    // Even if carry == 0, sum might be >= r. Conditional subtract r:
    Vec16 borrow = Vec16(0u);
    Vec16 r_sub[N_LIMBS];
    #pragma unroll
    for (int j = 0; j < N_LIMBS; ++j) {
        Vec16 b1, b2;
        Vec16 tmp = esimd::subb(b1, sum.v[j], Vec16(R_MOD[j]));
        r_sub[j] = esimd::subb(b2, tmp, borrow);
        borrow = b1 + b2;
    }
    // If we had carry-out from the add OR borrow == 0 (sum >= r), use r_sub.
    // sum >= r  iff  (carry_from_add == 1) OR (subtraction had no borrow-out).
    Mask16 keep_sub = (carry == Vec16(1u)) | (borrow == Vec16(0u));

    Fp out;
    #pragma unroll
    for (int j = 0; j < N_LIMBS; ++j) {
        out.v[j] = sum.v[j];
        out.v[j].merge(r_sub[j], keep_sub);
    }
    return out;
}

// ============================================================================
// Field Subtraction (a - b mod r)
// ============================================================================
ESIMD_INLINE Fp sub_mod(const Fp& a, const Fp& b) {
    Fp diff;
    Vec16 borrow = Vec16(0u);
    #pragma unroll
    for (int j = 0; j < N_LIMBS; ++j) {
        Vec16 b1, b2;
        Vec16 d = esimd::subb(b1, a.v[j], b.v[j]);
        diff.v[j] = esimd::subb(b2, d, borrow);
        borrow = b1 + b2;
    }
    // If borrow == 1, a < b, so add r
    Mask16 need_add = (borrow == Vec16(1u));

    Vec16 carry = Vec16(0u);
    Fp added;
    #pragma unroll
    for (int j = 0; j < N_LIMBS; ++j) {
        Vec16 c1, c2;
        Vec16 s = esimd::addc(c1, diff.v[j], Vec16(R_MOD[j]));
        added.v[j] = esimd::addc(c2, s, carry);
        carry = c1 + c2;
    }

    Fp out;
    #pragma unroll
    for (int j = 0; j < N_LIMBS; ++j) {
        out.v[j] = diff.v[j];
        out.v[j].merge(added.v[j], need_add);
    }
    return out;
}

// ============================================================================
// Convert TO Montgomery form: x_mont = x * R^2 * R^{-1} = x * R
// ============================================================================
ESIMD_INLINE Fp to_mont(const Fp& x) {
    Fp rr;
    #pragma unroll
    for (int j = 0; j < N_LIMBS; ++j) rr.v[j] = Vec16(R_RR[j]);
    return mont_mul(x, rr);
}

// ============================================================================
// Convert FROM Montgomery form: x = x_mont * 1 * R^{-1} = x_mont / R
// ============================================================================
ESIMD_INLINE Fp from_mont(const Fp& x_mont) {
    Fp one_norm;
    one_norm.v[0] = Vec16(1u);
    #pragma unroll
    for (int j = 1; j < N_LIMBS; ++j) one_norm.v[j] = Vec16(0u);
    return mont_mul(x_mont, one_norm);
}

// ============================================================================
// Montgomery Squaring
// ============================================================================
ESIMD_INLINE Fp mont_sqr(const Fp& a) {
    return mont_mul(a, a);
}

} // namespace bn254
