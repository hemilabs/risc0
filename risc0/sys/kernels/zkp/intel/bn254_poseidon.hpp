#pragma once

#include "bn254_field.hpp"
#include "bn254_poseidon_constants.hpp"

namespace bn254 {

// Poseidon-254 parameters (matching risc0 Rust consts.rs)
static constexpr int P254_CELLS = 3;
static constexpr int P254_ROUNDS_HALF_FULL = 4;
static constexpr int P254_ROUNDS_PARTIAL = 42;
static constexpr int P254_ROUNDS_TOT = 2 * P254_ROUNDS_HALF_FULL + P254_ROUNDS_PARTIAL; // 50

// S-box: x^8 via 3 squarings
ESIMD_INLINE Fp sbox(const Fp& x) {
    Fp x2 = mont_mul(x, x);
    Fp x4 = mont_mul(x2, x2);
    Fp x8 = mont_mul(x4, x4);
    return x8;
}

// Broadcast a constant Fp from AoS uint32_t[8] to all 16 lanes.
ESIMD_INLINE Fp broadcast_const(const uint32_t limbs[8]) {
    Fp x;
    #pragma unroll
    for (int j = 0; j < 8; ++j) x.v[j] = Vec16(limbs[j]);
    return x;
}

// Add round constants: cells[i] += RC[round*3 + i]
ESIMD_INLINE void add_round_constants(Fp cells[3], int round) {
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        Fp rc = broadcast_const(POSEIDON254_RC[round * 3 + i]);
        cells[i] = add_mod(cells[i], rc);
    }
}

// MDS matrix multiplication: new_cells[i] = sum_j MDS[i][j] * old_cells[j]
ESIMD_INLINE void multiply_by_mds(Fp cells[3]) {
    Fp old0 = cells[0], old1 = cells[1], old2 = cells[2];
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        Fp m0 = broadcast_const(POSEIDON254_MDS[i * 3 + 0]);
        Fp m1 = broadcast_const(POSEIDON254_MDS[i * 3 + 1]);
        Fp m2 = broadcast_const(POSEIDON254_MDS[i * 3 + 2]);
        Fp t0 = mont_mul(m0, old0);
        Fp t1 = mont_mul(m1, old1);
        Fp t2 = mont_mul(m2, old2);
        cells[i] = add_mod(add_mod(t0, t1), t2);
    }
}

// Full round: ARC -> all 3 S-boxes -> MDS
ESIMD_INLINE void full_round(Fp cells[3], int round) {
    add_round_constants(cells, round);
    #pragma unroll
    for (int i = 0; i < 3; ++i) cells[i] = sbox(cells[i]);
    multiply_by_mds(cells);
}

// Partial round: ARC -> sbox(cells[0] only) -> MDS
ESIMD_INLINE void partial_round(Fp cells[3], int round) {
    add_round_constants(cells, round);
    cells[0] = sbox(cells[0]);
    multiply_by_mds(cells);
}

// Main permutation using compile-time constants (for standalone tests).
ESIMD_INLINE void poseidon_mix(Fp cells[3]) {
    int round = 0;
    for (int i = 0; i < P254_ROUNDS_HALF_FULL; ++i) {
        full_round(cells, round); round++;
    }
    for (int i = 0; i < P254_ROUNDS_PARTIAL; ++i) {
        partial_round(cells, round); round++;
    }
    for (int i = 0; i < P254_ROUNDS_HALF_FULL; ++i) {
        full_round(cells, round); round++;
    }
}

// ============================================================================
// Device-memory variant: loads constants from device pointers on-the-fly.
// This avoids embedding 5KB of constexpr data into the kernel IR, which
// causes IGC's GenXPromoteArray to spill to 15KB stack.
// Pattern matches Poseidon2's poseidon2_mix() approach.
// ============================================================================

// Load one Fp (8 u32) from a device pointer via scalar broadcast.
ESIMD_INLINE Fp load_fp_dev(const uint32_t* ptr) {
    Fp x;
    #pragma unroll
    for (int j = 0; j < 8; ++j) x.v[j] = Vec16(*(ptr + j));
    return x;
}

// Add round constants from device memory
ESIMD_INLINE void add_round_constants_dev(Fp cells[3], const uint32_t* prc, int round) {
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        Fp rc = load_fp_dev(prc + (round * 3 + i) * 8);
        cells[i] = add_mod(cells[i], rc);
    }
}

// MDS from device memory
ESIMD_INLINE void multiply_by_mds_dev(Fp cells[3], const uint32_t* pmds) {
    Fp old0 = cells[0], old1 = cells[1], old2 = cells[2];
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        Fp m0 = load_fp_dev(pmds + (i * 3 + 0) * 8);
        Fp m1 = load_fp_dev(pmds + (i * 3 + 1) * 8);
        Fp m2 = load_fp_dev(pmds + (i * 3 + 2) * 8);
        Fp t0 = mont_mul(m0, old0);
        Fp t1 = mont_mul(m1, old1);
        Fp t2 = mont_mul(m2, old2);
        cells[i] = add_mod(add_mod(t0, t1), t2);
    }
}

ESIMD_INLINE void full_round_dev(Fp cells[3], const uint32_t* prc, const uint32_t* pmds, int round) {
    add_round_constants_dev(cells, prc, round);
    #pragma unroll
    for (int i = 0; i < 3; ++i) cells[i] = sbox(cells[i]);
    multiply_by_mds_dev(cells, pmds);
}

ESIMD_INLINE void partial_round_dev(Fp cells[3], const uint32_t* prc, const uint32_t* pmds, int round) {
    add_round_constants_dev(cells, prc, round);
    cells[0] = sbox(cells[0]);
    multiply_by_mds_dev(cells, pmds);
}

// Main permutation using device-memory constants.
ESIMD_INLINE void poseidon_mix_dev(Fp cells[3], const uint32_t* prc, const uint32_t* pmds) {
    int round = 0;
    for (int i = 0; i < P254_ROUNDS_HALF_FULL; ++i) {
        full_round_dev(cells, prc, pmds, round); round++;
    }
    for (int i = 0; i < P254_ROUNDS_PARTIAL; ++i) {
        partial_round_dev(cells, prc, pmds, round); round++;
    }
    for (int i = 0; i < P254_ROUNDS_HALF_FULL; ++i) {
        full_round_dev(cells, prc, pmds, round); round++;
    }
}

// ============================================================================
// SPARSE-MATRIX VARIANT (CUDA-optimized, 474 muls vs 648 for plain MDS)
//
// Uses pre_sparse_matrix at the boundary between full and partial rounds,
// per-round sparse matrices for partial rounds, and shared m_0_0 diagonal.
// Constants layout: 68 RC + 9 MDS + 9 pre_sparse + 42×4 sparse + 1 m_0_0 = 255.
// ============================================================================

// Sparse partial round: only state[0] gets sbox, then sparse MDS.
// sparse_matrix[4] = [w0, w1, v0, v1] for CELLS=3.
// state[0] = p0 * m_0_0 + w0 * state[1] + w1 * state[2]
// state[j] += p0 * v[j-1]   for j=1..2
ESIMD_INLINE void partial_round_sparse_dev(Fp cells[3],
                                            const uint32_t* prc, int rc_idx,
                                            const uint32_t* p_sparse,  // 4 Fr entries
                                            const uint32_t* p_m00) {
    cells[0] = sbox(cells[0]);

    // p0 = state[0] + round_constant
    Fp rc = load_fp_dev(prc + rc_idx * 8);
    Fp p0 = add_mod(cells[0], rc);

    // state[0] = p0 * m_0_0
    Fp m00 = load_fp_dev(p_m00);
    cells[0] = mont_mul(p0, m00);

    // state[0] += dot_product(state[1..], sparse_w[0..1], 2)
    // w0 = sparse_matrix[0], w1 = sparse_matrix[1]
    Fp w0 = load_fp_dev(p_sparse + 0 * 8);
    Fp w1 = load_fp_dev(p_sparse + 1 * 8);
    Fp d0 = mont_mul(w0, cells[1]);
    Fp d1 = mont_mul(w1, cells[2]);
    cells[0] = add_mod(cells[0], add_mod(d0, d1));

    // state[j] += p0 * sparse_v[j-1]  for j=1,2
    // v0 = sparse_matrix[2], v1 = sparse_matrix[3]
    Fp v0 = load_fp_dev(p_sparse + 2 * 8);
    Fp v1 = load_fp_dev(p_sparse + 3 * 8);
    cells[1] = add_mod(cells[1], mont_mul(p0, v0));
    cells[2] = add_mod(cells[2], mont_mul(p0, v1));
}

// Full round with device-memory MDS matrix (sparse variant uses same full_round)
ESIMD_INLINE void full_round_sparse_dev(Fp cells[3], const uint32_t* prc, int rc_idx,
                                         const uint32_t* pmds) {
    // ARC
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        Fp rc = load_fp_dev(prc + (rc_idx + i) * 8);
        cells[i] = add_mod(cells[i], rc);
    }
    // sbox all 3
    #pragma unroll
    for (int i = 0; i < 3; ++i) cells[i] = sbox(cells[i]);
    // MDS
    multiply_by_mds_dev(cells, pmds);
}

// Main sparse permutation using device pointers for ALL constants:
//   prc: 68 round constants (Fr[68])
//   pmds: full MDS (Fr[9])
//   p_pre_sparse: pre-sparse matrix (Fr[9])
//   p_sparse: sparse matrices (Fr[42][4])
//   p_m00: m_0_0 (Fr[1])
ESIMD_INLINE void poseidon_mix_sparse_dev(Fp cells[3],
                                           const uint32_t* prc,
                                           const uint32_t* pmds,
                                           const uint32_t* p_pre_sparse,
                                           const uint32_t* p_sparse,
                                           const uint32_t* p_m00) {
    int rc_idx = 0;

    // First half of full rounds (rounds 0..3)
    // Round 3 uses pre_sparse_matrix instead of mds
    for (int r = 0; r < P254_ROUNDS_HALF_FULL; ++r) {
        const uint32_t* mat = (r == P254_ROUNDS_HALF_FULL - 1) ? p_pre_sparse : pmds;

        // ARC
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            Fp rc = load_fp_dev(prc + (rc_idx + i) * 8);
            cells[i] = add_mod(cells[i], rc);
        }
        rc_idx += 3;

        // sbox all 3
        #pragma unroll
        for (int i = 0; i < 3; ++i) cells[i] = sbox(cells[i]);

        // MDS with either full MDS or pre_sparse_matrix
        Fp old0 = cells[0], old1 = cells[1], old2 = cells[2];
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            Fp m0 = load_fp_dev(mat + (i * 3 + 0) * 8);
            Fp m1 = load_fp_dev(mat + (i * 3 + 1) * 8);
            Fp m2 = load_fp_dev(mat + (i * 3 + 2) * 8);
            cells[i] = add_mod(add_mod(mont_mul(m0, old0), mont_mul(m1, old1)), mont_mul(m2, old2));
        }
    }

    // Add round constants for the first partial round
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        Fp rc = load_fp_dev(prc + (rc_idx + i) * 8);
        cells[i] = add_mod(cells[i], rc);
    }
    rc_idx += 3;

    // 42 partial rounds with sparse matrices
    for (int pr = 0; pr < P254_ROUNDS_PARTIAL; ++pr) {
        // RC index: after the (4*3 + 3) = 15 full-round RCs, partial RCs are 1 per round
        // But the last partial round has rc = 0 (absorbed). In CUDA, it uses czero on the last.
        int rc_partial = (pr < P254_ROUNDS_PARTIAL - 1) ? rc_idx : -1;
        // sparse matrix for this round: p_sparse[pr * 4 * 8]
        const uint32_t* sp = p_sparse + pr * 4 * 8;

        cells[0] = sbox(cells[0]);

        // p0 = state[0] + rc (or just state[0] if last partial)
        Fp p0;
        if (pr < P254_ROUNDS_PARTIAL - 1) {
            Fp rc = load_fp_dev(prc + rc_idx * 8);
            p0 = add_mod(cells[0], rc);
            rc_idx += 1;
        } else {
            p0 = cells[0];
        }

        // state[0] = p0 * m_0_0 + w0 * state[1] + w1 * state[2]
        Fp m00 = load_fp_dev(p_m00);
        cells[0] = mont_mul(p0, m00);
        Fp w0 = load_fp_dev(sp + 0 * 8);
        Fp w1 = load_fp_dev(sp + 1 * 8);
        cells[0] = add_mod(cells[0], add_mod(mont_mul(w0, cells[1]), mont_mul(w1, cells[2])));

        // state[j] += p0 * v[j-1]
        Fp v0 = load_fp_dev(sp + 2 * 8);
        Fp v1 = load_fp_dev(sp + 3 * 8);
        cells[1] = add_mod(cells[1], mont_mul(p0, v0));
        cells[2] = add_mod(cells[2], mont_mul(p0, v1));
    }

    // Second half of full rounds (rounds 4..7)
    // Last round (round 7) only computes state[0] output (CELLS_OUT=1 optimization)
    for (int r = 0; r < P254_ROUNDS_HALF_FULL; ++r) {
        // ARC
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            Fp rc = load_fp_dev(prc + (rc_idx + i) * 8);
            cells[i] = add_mod(cells[i], rc);
        }
        rc_idx += 3;

        // sbox all 3
        #pragma unroll
        for (int i = 0; i < 3; ++i) cells[i] = sbox(cells[i]);

        // MDS — last round only computes row 0 (output optimization)
        if (r < P254_ROUNDS_HALF_FULL - 1) {
            Fp old0 = cells[0], old1 = cells[1], old2 = cells[2];
            #pragma unroll
            for (int i = 0; i < 3; ++i) {
                Fp m0 = load_fp_dev(pmds + (i * 3 + 0) * 8);
                Fp m1 = load_fp_dev(pmds + (i * 3 + 1) * 8);
                Fp m2 = load_fp_dev(pmds + (i * 3 + 2) * 8);
                cells[i] = add_mod(add_mod(mont_mul(m0, old0), mont_mul(m1, old1)), mont_mul(m2, old2));
            }
        } else {
            // Last round: only compute cells[0]
            Fp old0 = cells[0], old1 = cells[1], old2 = cells[2];
            Fp m0 = load_fp_dev(pmds + 0 * 8);
            Fp m1 = load_fp_dev(pmds + 1 * 8);
            Fp m2 = load_fp_dev(pmds + 2 * 8);
            cells[0] = add_mod(add_mod(mont_mul(m0, old0), mont_mul(m1, old1)), mont_mul(m2, old2));
        }
    }
}

} // namespace bn254
