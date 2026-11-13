#pragma once

#include "bb31_field.hpp"

// BabyBear Extension Field (FpExt = Fp^4) via Intel ESIMD.
//
// FpExt = Fp[X] / (X^4 + 11), i.e., X^4 = -11 = NBETA in Montgomery form.
// Each FpExt element has 4 components: [c0, c1, c2, c3] representing c0 + c1*X + c2*X^2 + c3*X^3.
//
// SIMD16 Layout (Option D from optimization review):
//   Process 4 independent FpExt elements per simd<uint32_t, 16> register:
//   Lanes [0:3]   = element 0: {c0, c1, c2, c3}
//   Lanes [4:7]   = element 1: {c0, c1, c2, c3}
//   Lanes [8:11]  = element 2: {c0, c1, c2, c3}
//   Lanes [12:15] = element 3: {c0, c1, c2, c3}
//
// This gives 100% SIMD utilization on all operations.
// Cross-component access uses compile-time iselect index vectors (zero cost on Xe2).

namespace bb31_ext {

using Vec16 = esimd::simd<uint32_t, 16>;
using Mask16 = esimd::simd_mask<16>;

// ============================================================================
// Compile-time index vectors for cross-component shuffle within 4-lane groups.
// Each group of 4 lanes holds one FpExt element [c0, c1, c2, c3].
// These patterns broadcast or permute components within each group.
// ============================================================================

// Broadcast component k to all 4 lanes within each group
static constexpr uint16_t BC0_IDX[16] = {0,0,0,0,  4,4,4,4,    8,8,8,8,    12,12,12,12};
static constexpr uint16_t BC1_IDX[16] = {1,1,1,1,  5,5,5,5,    9,9,9,9,    13,13,13,13};
static constexpr uint16_t BC2_IDX[16] = {2,2,2,2,  6,6,6,6,    10,10,10,10, 14,14,14,14};
static constexpr uint16_t BC3_IDX[16] = {3,3,3,3,  7,7,7,7,    11,11,11,11, 15,15,15,15};

// Rotate components within each 4-lane group for multiplication patterns:
// ROT1: {c3, c0, c1, c2} -- rotate right by 1
static constexpr uint16_t ROT1_IDX[16] = {3,0,1,2,  7,4,5,6,    11,8,9,10,  15,12,13,14};
// ROT2: {c2, c3, c0, c1} -- rotate right by 2
static constexpr uint16_t ROT2_IDX[16] = {2,3,0,1,  6,7,4,5,    10,11,8,9,  14,15,12,13};
// ROT3: {c1, c2, c3, c0} -- rotate right by 3 (= rotate left by 1)
static constexpr uint16_t ROT3_IDX[16] = {1,2,3,0,  5,6,7,4,    9,10,11,8,  13,14,15,12};

// ============================================================================
// FpExt Addition / Subtraction (component-wise)
// ============================================================================

ESIMD_INLINE Vec16 fpext_add(Vec16 a, Vec16 b) {
    return bb31::field_add(a, b);
}

ESIMD_INLINE Vec16 fpext_sub(Vec16 a, Vec16 b) {
    return bb31::field_sub(a, b);
}

// ============================================================================
// FpExt * Fp (mixed multiplication -- broadcast scalar to all 4 components)
// ============================================================================

// Multiply 4 FpExt elements by 4 Fp scalars.
// `scalars` has one Fp value per 4-lane group in lane 0 (c0 position).
// We broadcast it to all 4 component lanes, then multiply.
ESIMD_INLINE Vec16 fpext_mul_fp(Vec16 ext, Vec16 scalar_broadcast) {
    return bb31::mont_mul(ext, scalar_broadcast);
}

// Helper: broadcast lane 0 of each 4-lane group to all 4 lanes
ESIMD_INLINE Vec16 broadcast_c0(Vec16 v) {
    esimd::simd<uint16_t, 16> idx(BC0_IDX);
    return v.iselect(idx);
}

// ============================================================================
// FpExt * FpExt (full extension field multiplication)
// ============================================================================
//
// For a = [a0,a1,a2,a3] and b = [b0,b1,b2,b3], the product mod (X^4 - BETA) is:
//   ret[0] = a0*b0 + NBETA*(a1*b3 + a2*b2 + a3*b1)
//   ret[1] = a0*b1 + a1*b0 + NBETA*(a2*b3 + a3*b2)
//   ret[2] = a0*b2 + a1*b1 + a2*b0 + NBETA*(a3*b3)
//   ret[3] = a0*b3 + a1*b2 + a2*b1 + a3*b0
//
// Implementation: Use 4 broadcast-multiply-accumulate steps.
// Each step broadcasts one component of `a` and multiplies with a rotated `b`.

ESIMD_INLINE Vec16 fpext_mul(Vec16 a, Vec16 b) {
    esimd::simd<uint16_t, 16> bc0(BC0_IDX);
    esimd::simd<uint16_t, 16> bc1(BC1_IDX);
    esimd::simd<uint16_t, 16> bc2(BC2_IDX);
    esimd::simd<uint16_t, 16> bc3(BC3_IDX);
    esimd::simd<uint16_t, 16> rot1(ROT1_IDX);
    esimd::simd<uint16_t, 16> rot2(ROT2_IDX);
    esimd::simd<uint16_t, 16> rot3(ROT3_IDX);

    // Step 1: a0 * [b0, b1, b2, b3] = [a0*b0, a0*b1, a0*b2, a0*b3]
    // This contributes directly to ret[0..3]
    Vec16 a0_bc = a.iselect(bc0);   // broadcast a[0] to all 4 lanes per group
    Vec16 acc = bb31::mont_mul(a0_bc, b);

    // Step 2: a1 * [b3, b0, b1, b2] (b rotated right by 1)
    // Contributions: a1*b3 -> ret[0] (with NBETA), a1*b0 -> ret[1], a1*b1 -> ret[2], a1*b2 -> ret[3]
    Vec16 a1_bc = a.iselect(bc1);
    Vec16 b_rot1 = b.iselect(rot1);
    Vec16 term1 = bb31::mont_mul(a1_bc, b_rot1);

    // Step 3: a2 * [b2, b3, b0, b1] (b rotated right by 2)
    // Contributions: a2*b2 -> ret[0] (with NBETA), a2*b3 -> ret[1] (with NBETA), a2*b0 -> ret[2], a2*b1 -> ret[3]
    Vec16 a2_bc = a.iselect(bc2);
    Vec16 b_rot2 = b.iselect(rot2);
    Vec16 term2 = bb31::mont_mul(a2_bc, b_rot2);

    // Step 4: a3 * [b1, b2, b3, b0] (b rotated right by 3)
    // Contributions: a3*b1 -> ret[0] (with NBETA), a3*b2 -> ret[1], a3*b3 -> ret[2] (with NBETA), a3*b0 -> ret[3]
    Vec16 a3_bc = a.iselect(bc3);
    Vec16 b_rot3 = b.iselect(rot3);
    Vec16 term3 = bb31::mont_mul(a3_bc, b_rot3);

    // Now accumulate with NBETA multiplication on the "wrapped" terms.
    // Terms that wrapped past X^4 need multiplication by NBETA.
    //
    // For each output component, the wrap pattern is:
    //   ret[0]: a0*b0 (direct) + NBETA*(a1*b3 + a2*b2 + a3*b1) -- terms 1,2,3 all wrap
    //   ret[1]: a0*b1 + a1*b0 (direct) + NBETA*(a2*b3 + a3*b2) -- terms 2,3 wrap
    //   ret[2]: a0*b2 + a1*b1 + a2*b0 (direct) + NBETA*(a3*b3) -- term 3 wraps
    //   ret[3]: a0*b3 + a1*b2 + a2*b1 + a3*b0 (direct) -- no wraps
    //
    // Within each 4-lane group:
    //   Lane 0 (c0): term1[0], term2[0], term3[0] all need NBETA
    //   Lane 1 (c1): term2[1], term3[1] need NBETA; term1[1] is direct
    //   Lane 2 (c2): term3[2] needs NBETA; term1[2], term2[2] are direct
    //   Lane 3 (c3): all direct (no wraps)
    //
    // Approach: multiply the "wrapping" parts by NBETA, then add the "direct" parts.

    // Create NBETA mask: lanes where the term wrapped past X^4
    // term1 wraps in lane 0 only (a1*b3 contributes to c0 via X^4)
    // term2 wraps in lanes 0,1
    // term3 wraps in lanes 0,1,2

    // Simpler approach: separate each term into "direct" and "beta" parts
    // using per-lane masks, multiply beta parts by NBETA, then sum everything.

    // For term1: lane 0 needs NBETA, lanes 1,2,3 are direct
    // Pattern per group: [NBETA, 1, 1, 1]
    Vec16 nbeta_1_1_1(bb31::ONE); // fill with ONE
    for (int g = 0; g < 4; g++) {
        nbeta_1_1_1.select<1, 1>(g * 4) = bb31::NBETA; // lane 0 of each group
    }
    Vec16 scaled_term1 = bb31::mont_mul(term1, nbeta_1_1_1);

    // For term2: lanes 0,1 need NBETA, lanes 2,3 are direct
    Vec16 nbeta_nb_1_1(bb31::ONE);
    for (int g = 0; g < 4; g++) {
        nbeta_nb_1_1.select<1, 1>(g * 4 + 0) = bb31::NBETA;
        nbeta_nb_1_1.select<1, 1>(g * 4 + 1) = bb31::NBETA;
    }
    Vec16 scaled_term2 = bb31::mont_mul(term2, nbeta_nb_1_1);

    // For term3: lanes 0,1,2 need NBETA, lane 3 is direct
    Vec16 nbeta_nb_nb_1(bb31::NBETA); // fill with NBETA
    for (int g = 0; g < 4; g++) {
        nbeta_nb_nb_1.select<1, 1>(g * 4 + 3) = bb31::ONE; // lane 3 = ONE
    }
    Vec16 scaled_term3 = bb31::mont_mul(term3, nbeta_nb_nb_1);

    // Final accumulation: acc (from step 1) + scaled terms
    Vec16 result = acc;
    result = bb31::field_add(result, scaled_term1);
    result = bb31::field_add(result, scaled_term2);
    result = bb31::field_add(result, scaled_term3);

    return result;
}

// ============================================================================
// FpExt Reciprocal (composite field inversion)
// ============================================================================
//
// Uses the 2-level composite field decomposition:
//   1. Compute conjugate a' (negate odd components)
//   2. b = a * a' (result has zeros in odd positions: b1=b3=0)
//   3. Compute c = b0^2 - BETA * b2^2
//   4. Invert c using base field reciprocal
//   5. Result = a' * [c_inv, 0, -BETA*b2*c_inv, 0] (simplified)

// FpExt reciprocal via composite field inversion.
// The extension field is Fp[X]/(X^4 + 11), so X^4 = -11 = NBETA.
// The decomposition into separate mont_mul/field_sub IS mathematically correct
// because R^{-1} distributes over addition/subtraction in a field.
ESIMD_INLINE Vec16 fpext_reciprocal(Vec16 val) {
    esimd::simd<uint16_t, 16> bc0(BC0_IDX);
    esimd::simd<uint16_t, 16> bc1(BC1_IDX);
    esimd::simd<uint16_t, 16> bc2(BC2_IDX);
    esimd::simd<uint16_t, 16> bc3(BC3_IDX);

    Vec16 a0 = val.iselect(bc0);
    Vec16 a1 = val.iselect(bc1);
    Vec16 a2 = val.iselect(bc2);
    Vec16 a3 = val.iselect(bc3);

    // NBETA = Montgomery(-11). The extension field has X^4 = -11,
    // so all "beta" terms in the composite field inversion use NBETA.
    Vec16 beta_vec(bb31::NBETA);

    // recip_b0 (from sppark line 506):
    // b0 = a0^2 - beta*(a1*2*a3 - a2^2)
    //    = a0^2 - beta*a1*2*a3 + beta*a2^2
    Vec16 a3x2 = bb31::field_add(a3, a3); // 2*a3 in field
    Vec16 t1 = bb31::field_sub(bb31::mont_mul(a1, a3x2), bb31::mont_mul(a2, a2));
    Vec16 b0 = bb31::field_sub(bb31::mont_mul(a0, a0), bb31::mont_mul(beta_vec, t1));

    // recip_b2 (from sppark line 522):
    // b2 = a0*2*a2 - a1^2 - beta*a3^2
    Vec16 a2x2 = bb31::field_add(a2, a2); // 2*a2 in field
    Vec16 b2 = bb31::field_sub(
        bb31::field_sub(bb31::mont_mul(a0, a2x2), bb31::mont_mul(a1, a1)),
        bb31::mont_mul(beta_vec, bb31::mont_mul(a3, a3))
    );

    // c = b0^2 - beta*b2^2 (from sppark line 580)
    Vec16 c_val = bb31::field_sub(
        bb31::mont_mul(b0, b0),
        bb31::mont_mul(beta_vec, bb31::mont_mul(b2, b2))
    );

    // Invert c (base field reciprocal)
    Vec16 c_inv = bb31::reciprocal(c_val);

    // ib0 = b0 * c_inv, ib2 = b2 * c_inv
    Vec16 ib0 = bb31::mont_mul(b0, c_inv);
    Vec16 ib2 = bb31::mont_mul(b2, c_inv);

    // beta_b2 = beta * ib2 (from sppark line 537)
    Vec16 beta_b2 = bb31::mont_mul(beta_vec, ib2);

    // recip_ret (from sppark lines 542-564):
    //   ret[0] = a0*ib0 - a2*beta_b2
    //   ret[1] = a3*beta_b2 - a1*ib0
    //   ret[2] = a2*ib0 - a0*ib2
    //   ret[3] = a1*ib2 - a3*ib0
    Vec16 r0 = bb31::field_sub(bb31::mont_mul(a0, ib0), bb31::mont_mul(a2, beta_b2));
    Vec16 r1 = bb31::field_sub(bb31::mont_mul(a3, beta_b2), bb31::mont_mul(a1, ib0));
    Vec16 r2 = bb31::field_sub(bb31::mont_mul(a2, ib0), bb31::mont_mul(a0, ib2));
    Vec16 r3 = bb31::field_sub(bb31::mont_mul(a1, ib2), bb31::mont_mul(a3, ib0));

    // Reassemble: pack r0..r3 back into 4-lane-group layout
    Vec16 result(0u); // Initialize all lanes
    for (int g = 0; g < 4; g++) {
        result.select<1, 1>(g * 4 + 0) = r0.select<1, 1>(g * 4);
        result.select<1, 1>(g * 4 + 1) = r1.select<1, 1>(g * 4);
        result.select<1, 1>(g * 4 + 2) = r2.select<1, 1>(g * 4);
        result.select<1, 1>(g * 4 + 3) = r3.select<1, 1>(g * 4);
    }

    return result;
}

} // namespace bb31_ext
