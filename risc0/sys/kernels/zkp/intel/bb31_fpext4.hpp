// BabyBear Extension Field (FpExt) for HAL kernels — "4 × Vec16" layout.
//
// Each FpExt value is 4 separate Vec16 registers, one per component.
// Each SIMD16 lane holds an independent FpExt value.
// This layout is used by fri_fold, mix_poly_coeffs, batch_evaluate_any.
//
// Extension: Fp[X] / (X^4 + 11), so X^4 = -11 = NBETA.
// NBETA = P - 11 = -11 mod P (in Montgomery form: bb31::NBETA = 0x40000018).
//
// Product formula:
//   ret[0] = a0*b0 + NBETA*(a1*b3 + a2*b2 + a3*b1)
//   ret[1] = a0*b1 + a1*b0 + NBETA*(a2*b3 + a3*b2)
//   ret[2] = a0*b2 + a1*b1 + a2*b0 + NBETA*(a3*b3)
//   ret[3] = a0*b3 + a1*b2 + a2*b1 + a3*b0

#pragma once
#include "bb31_field.hpp"

namespace fpext4 {

using Vec16 = bb31::Vec16;

struct FpExt4 {
    Vec16 c[4]; // c[0..3] = components, each lane independent

    FpExt4() : c{Vec16(0u), Vec16(0u), Vec16(0u), Vec16(0u)} {}
    explicit FpExt4(uint32_t scalar_mont) : c{Vec16(scalar_mont), Vec16(0u), Vec16(0u), Vec16(0u)} {}
    FpExt4(Vec16 c0, Vec16 c1, Vec16 c2, Vec16 c3) : c{c0, c1, c2, c3} {}
};

// Component-wise add/sub
ESIMD_INLINE FpExt4 fpext_add(const FpExt4& a, const FpExt4& b) {
    return {bb31::field_add(a.c[0], b.c[0]), bb31::field_add(a.c[1], b.c[1]),
            bb31::field_add(a.c[2], b.c[2]), bb31::field_add(a.c[3], b.c[3])};
}

ESIMD_INLINE FpExt4 fpext_sub(const FpExt4& a, const FpExt4& b) {
    return {bb31::field_sub(a.c[0], b.c[0]), bb31::field_sub(a.c[1], b.c[1]),
            bb31::field_sub(a.c[2], b.c[2]), bb31::field_sub(a.c[3], b.c[3])};
}

// FpExt * Fp (scalar multiply each component)
ESIMD_INLINE FpExt4 fpext_mul_fp(const FpExt4& a, Vec16 scalar) {
    return {bb31::mont_mul(a.c[0], scalar), bb31::mont_mul(a.c[1], scalar),
            bb31::mont_mul(a.c[2], scalar), bb31::mont_mul(a.c[3], scalar)};
}

// FpExt * FpExt (full extension field multiplication)
// 16 mont_muls + 12 field_adds per call
ESIMD_INLINE FpExt4 fpext_mul(const FpExt4& a, const FpExt4& b) {
    Vec16 nb(bb31::NBETA);

    // Direct products
    auto a0b0 = bb31::mont_mul(a.c[0], b.c[0]);
    auto a0b1 = bb31::mont_mul(a.c[0], b.c[1]);
    auto a0b2 = bb31::mont_mul(a.c[0], b.c[2]);
    auto a0b3 = bb31::mont_mul(a.c[0], b.c[3]);
    auto a1b0 = bb31::mont_mul(a.c[1], b.c[0]);
    auto a1b1 = bb31::mont_mul(a.c[1], b.c[1]);
    auto a1b2 = bb31::mont_mul(a.c[1], b.c[2]);
    auto a1b3 = bb31::mont_mul(a.c[1], b.c[3]);
    auto a2b0 = bb31::mont_mul(a.c[2], b.c[0]);
    auto a2b1 = bb31::mont_mul(a.c[2], b.c[1]);
    auto a2b2 = bb31::mont_mul(a.c[2], b.c[2]);
    auto a2b3 = bb31::mont_mul(a.c[2], b.c[3]);
    auto a3b0 = bb31::mont_mul(a.c[3], b.c[0]);
    auto a3b1 = bb31::mont_mul(a.c[3], b.c[1]);
    auto a3b2 = bb31::mont_mul(a.c[3], b.c[2]);
    auto a3b3 = bb31::mont_mul(a.c[3], b.c[3]);

    // Wrap-around terms multiplied by NBETA
    auto wrap0 = bb31::field_add(bb31::field_add(a1b3, a2b2), a3b1);
    auto wrap1 = bb31::field_add(a2b3, a3b2);
    auto wrap2 = a3b3;

    FpExt4 r;
    r.c[0] = bb31::field_add(a0b0, bb31::mont_mul(nb, wrap0));
    r.c[1] = bb31::field_add(bb31::field_add(a0b1, a1b0), bb31::mont_mul(nb, wrap1));
    r.c[2] = bb31::field_add(bb31::field_add(bb31::field_add(a0b2, a1b1), a2b0),
                              bb31::mont_mul(nb, wrap2));
    r.c[3] = bb31::field_add(bb31::field_add(bb31::field_add(a0b3, a1b2), a2b1), a3b0);
    return r;
}

// FpExt exponentiation: x^n via square-and-multiply
ESIMD_INLINE FpExt4 fpext_pow(const FpExt4& base, uint32_t n) {
    FpExt4 result(bb31::ONE); // 1 in extension field
    FpExt4 cur = base;
    while (n > 0) {
        if (n & 1) result = fpext_mul(result, cur);
        cur = fpext_mul(cur, cur);
        n >>= 1;
    }
    return result;
}

} // namespace fpext4
