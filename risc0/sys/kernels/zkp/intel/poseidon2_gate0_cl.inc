// Auto-generated from poseidon2_gate0.cl. DO NOT EDIT.
// Embeds the gate0 OpenCL source as a C++ raw string literal so the
// SYCL OpenCL-source kernel-bundle path can JIT it at runtime.
static const char POSEIDON2_GATE0_CL_SRC[] = R"OPENCL(
/*
 * RISC Zero Poseidon2 over BabyBear (T=24, R_F=8, R_P=21, S-box x^7).
 *
 * Mirrors risc0/zkp/src/core/hash/poseidon2/{mod,consts}.rs exactly.
 *
 * Each work-item processes one permutation (24 BabyBear elements).
 * Layout: input/output as (n_perms × 24) row-major u32 array, with each row a
 *         permutation state (canonical, NOT Montgomery).
 *
 * Hot inner loops:
 *   - sbox: 4 mul-mod (x^7 = x*x^2 ; x^2 ; x^4=x^2*x^2 ; x^6=x^4*x^2 ; x^7=x^6*x)
 *   - multiply_by_m_ext: 6 calls to 4x4 circulant + cross-block sums
 *   - multiply_by_m_int: 24 mul-mod + 1 reduction
 *
 * 8 full rounds × (24 sboxes + m_ext) + 21 partial × (1 sbox + m_int) per perm.
 * Each work-item: ~200 muls + ~600 adds. Per-work-item-private SLM is ideal but
 * we keep state in registers (24 × 4 bytes = 96 B/lane = OK in GRF).
 */

#define BABY_BEAR_P 0x78000001u
#define CELLS 24u
#define ROUNDS_HALF_FULL 4u
#define ROUNDS_PARTIAL 21u
#define ROUND_CONSTANTS_LEN ((2u * ROUNDS_HALF_FULL + ROUNDS_PARTIAL) * CELLS)

inline uint bb_add(uint a, uint b) {
    uint s = a + b;
    return min(s, s - BABY_BEAR_P);
}

inline uint bb_mul(uint a, uint b) {
    return (uint)(((ulong)a * (ulong)b) % (ulong)BABY_BEAR_P);
}

/* Montgomery multiplication: with R = 2^32 and MU = -p^-1 mod R = 0x77FFFFFF.
 * bb_mont_redc(a*b) computes (a*b)*R^-1 mod p in [0, p).
 *
 * Implementation note: the IGC i64 mul lowering (Mul64) emits FOUR cross-product
 * mults plus two adds — even when high halves are zero. For our case where
 * `m * BABY_BEAR_P` has both operands < 2^31, two of those cross-products are
 * literal `mul * 0` (dead code that survives DCE). We avoid this by explicitly
 * splitting the 64-bit add into 32-bit lo/hi pieces with carry, using
 * `mul_hi(m, P)` for the high half — that's one i32 `mul` + one i32 `mulh`
 * vs. four mults in the i64 path.
 *
 * VISA dump confirmed: this saves ~3 dead instructions per REDC.
 */
#define BABY_BEAR_MU 0x77FFFFFFu

/* Mont REDC operating on (hi, lo) pair directly — avoids the i64 intermediate
 * entirely. Eliminates the dead `mul * 0` cross-products from FullMul32.
 *
 * KEY OPTIMIZATION: by Mont's defining property MU = -p^-1 mod R, we have
 *   m * p ≡ -t_lo (mod 2^32)
 * so mp_lo + t_lo ≡ 0 (mod 2^32), i.e. sum_lo is ALWAYS 0 and the carry-out
 * is simply (t_lo != 0). We don't need to compute mp_lo at all → saves one
 * full 32x32 multiply (macl+mach pair) per REDC.
 *
 *   Mont mul cost: 3 mul-pairs (a*b, m*MU lo-only, m*p hi-only) vs naive 4-pair.
 */
inline uint bb_mont_redc_split(uint t_lo, uint t_hi) {
    const uint m = t_lo * BABY_BEAR_MU;
    const uint mp_hi = mul_hi(m, BABY_BEAR_P);
    const uint carry = (t_lo != 0u) ? 1u : 0u;
    const uint r = t_hi + mp_hi + carry;
    return min(r, r - BABY_BEAR_P);
}

inline uint bb_mont_mul(uint a, uint b) {
    /* Use 32-bit mul + mul_hi, no i64 intermediates.
     * Tested i64 Mul (to fire IGC's FullMul32 pattern); regressed because the
     * 64-bit value pack/unpack inserted ~9500 extra movs (88K → 70K instr cut). */
    const uint t_lo = a * b;
    const uint t_hi = mul_hi(a, b);
    return bb_mont_redc_split(t_lo, t_hi);
}

/* Inline-vISA-asm `madw`-based mul-pair. SIMD16 form (kernel must require
 * sub_group_size(16)). Produces lo and hi of (a*b+0) using one mul/mach/mov
 * triplet — saves one 32x32 mul-pair per Mont multiplication.
 *
 * SIMD16 layout: `lohi` is uint2 = 2 GRFs in OpenCL semantics (s0 in GRF0,
 * s1 in GRF1). madw writes lo to dst GRF, hi to dst+1 GRF — exact match. */
inline uint2 bb_mul_pair_madw_simd16(uint a, uint b) {
    uint2 lohi;
    __asm__ volatile(
        "madw (M1, 16) %0(0,0)<1> %1(0,0)<1;1,0> %2(0,0)<1;1,0> 0:ud\n"
        : "=rw"(lohi)
        : "rw"(a), "rw"(b)
    );
    return lohi;
}

inline uint bb_mont_mul_madw(uint a, uint b) {
    uint2 t = bb_mul_pair_madw_simd16(a, b);
    return bb_mont_redc_split(t.s0, t.s1);
}

/* SIMD32 inline-asm madw was investigated — would require 4-GRF scratch
 * + 4 SIMD16 movs to reassemble into SIMD32 SoA `lo` and `hi`. Total cost
 * (10 ops vs 8 for naive baseline) makes this net-worse than just running
 * SIMD16-with-madw. The right SIMD32 path is the IGC `MulHiAndMulFusion`
 * patch (Commits A+B+C in `igc-bug-report/mulpair-fusion/`) which lets
 * IGC's native register allocator place the mul.pair output directly in
 * the SIMD32 layout without explicit reassembly. */

inline uint bb_sbox_mont_madw(uint x) {
    uint x2 = bb_mont_mul_madw(x, x);
    uint x4 = bb_mont_mul_madw(x2, x2);
    uint x6 = bb_mont_mul_madw(x4, x2);
    return bb_mont_mul_madw(x6, x);
}

/* Legacy interface for callers that have an i64 product handy. */
inline uint bb_mont_redc(ulong t) {
    return bb_mont_redc_split((uint)t, (uint)(t >> 32));
}

/* Sbox in Mont form: x_mont -> x^7_mont via 4 bb_mont_muls. */
inline uint bb_sbox_mont(uint x) {
    uint x2 = bb_mont_mul(x, x);
    uint x4 = bb_mont_mul(x2, x2);
    uint x6 = bb_mont_mul(x4, x2);
    return bb_mont_mul(x6, x);
}

inline uint bb_sbox(uint x) {
    /* x^7 = x * x^2 * x^4 = x * x^2 * (x^2)^2 */
    uint x2 = bb_mul(x, x);
    uint x4 = bb_mul(x2, x2);
    uint x6 = bb_mul(x4, x2);
    return bb_mul(x6, x);
}

/* multiply-by-2/4 mod p — unsigned shifts with modular reduction. */
inline uint bb_mul2(uint x) { return bb_add(x, x); }
inline uint bb_mul4(uint x) { uint y = bb_add(x, x); return bb_add(y, y); }

/* 4x4 circulant block-multiply (Poseidon2 paper App. B, 7-add formula).
 * Modifies x in place: x = M4 * x. */
inline void m4(uint *x) {
    uint t0 = bb_add(x[0], x[1]);
    uint t1 = bb_add(x[2], x[3]);
    uint t2 = bb_add(bb_mul2(x[1]), t1);
    uint t3 = bb_add(bb_mul2(x[3]), t0);
    uint t4 = bb_add(bb_mul4(t1), t3);
    uint t5 = bb_add(bb_mul4(t0), t2);
    uint t6 = bb_add(t3, t5);
    uint t7 = bb_add(t2, t4);
    x[0] = t6; x[1] = t5; x[2] = t7; x[3] = t4;
}

/* Externally-bound matrix M_ext: per-block m4 + cross-block sums.
 * Tree-reduces the cross-block sum (6 blocks per column) at depth 3 instead
 * of serial depth 6, shortening the m_ext critical path by ~3 add-latencies.
 * cells is read-modify-write of size CELLS=24. */
inline void multiply_by_m_ext(uint *cells) {
    uint blk[6][4];
    /* Apply m4 to each 4-block. */
    #pragma unroll
    for (uint i = 0u; i < CELLS / 4u; ++i) {
        blk[i][0] = cells[i * 4u + 0u];
        blk[i][1] = cells[i * 4u + 1u];
        blk[i][2] = cells[i * 4u + 2u];
        blk[i][3] = cells[i * 4u + 3u];
        m4(blk[i]);
    }
    /* Tree-reduce per-column sums: 6 → 3 → 2 → 1 (depth 3). */
    uint sums[4];
    #pragma unroll
    for (uint j = 0u; j < 4u; ++j) {
        const uint a = bb_add(blk[0][j], blk[1][j]);
        const uint b = bb_add(blk[2][j], blk[3][j]);
        const uint c = bb_add(blk[4][j], blk[5][j]);
        sums[j] = bb_add(bb_add(a, b), c);
    }
    /* Each cell = m4(block)[j] + sums[j]. */
    #pragma unroll
    for (uint i = 0u; i < CELLS / 4u; ++i) {
        cells[i * 4u + 0u] = bb_add(blk[i][0], sums[0]);
        cells[i * 4u + 1u] = bb_add(blk[i][1], sums[1]);
        cells[i * 4u + 2u] = bb_add(blk[i][2], sums[2]);
        cells[i * 4u + 3u] = bb_add(blk[i][3], sums[3]);
    }
}

/* Internal matrix M_int = J + diag(M_INT_DIAG_HZN). cells[i] = sum + diag[i]*cells[i]. */
inline void multiply_by_m_int(uint *cells, __global const uint * restrict diag) {
    uint s = 0u;
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) s = bb_add(s, cells[i]);
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) {
        cells[i] = bb_add(s, bb_mul(diag[i], cells[i]));
    }
}

/* ---------------------------------------------------------------------------
 * Cooperative permutation kernel: ONE permutation per sub-group of 32 lanes.
 *
 * Layout:
 *   - Each sub-group of 32 lanes runs one permutation.
 *   - Lane sg_lid (0..23) holds cell sg_lid; lanes 24..31 are idle (cell=0).
 *   - WG of 256 = 8 sub-groups = 8 permutations per WG.
 *
 * Cross-cell operations (m_ext, m_int, full-state add-constant):
 *   - Modular sum reduction: tree of 5 stages of `bb_add(x, shuffle_xor(x, K))`.
 *   - Within-block 4x4 circulant: shuffle_xor with masks 1/2/3 within each
 *     4-lane block.
 *   - Cross-block accumulation: shuffle_xor with masks 4/8/16 to sum across
 *     blocks within a column.
 *
 * Eliminates register pressure entirely: each lane carries 1 cell of state
 * (~4-8 bytes including intermediates) vs ~96 bytes per lane in the naive
 * single-perm-per-work-item kernel.
 * ------------------------------------------------------------------------- */

inline uint sg_modular_sum(uint x) {
  /* Tree-reduce sum across 32 lanes with modular reduction at each step. */
  x = bb_add(x, sub_group_shuffle_xor(x, 1));
  x = bb_add(x, sub_group_shuffle_xor(x, 2));
  x = bb_add(x, sub_group_shuffle_xor(x, 4));
  x = bb_add(x, sub_group_shuffle_xor(x, 8));
  x = bb_add(x, sub_group_shuffle_xor(x, 16));
  return x;
}

/* Cooperative m_ext: one cell per lane, sub-group shuffles for cross-cell.
 *
 * The 4x4 circulant per block (4 lanes), expanded:
 *   out[0] = t6 = t3 + 4*t0 + t2 = 5*t0 + 2*x[3] + 2*x[1] + t1
 *   out[1] = t5 = 4*t0 + t2      = 4*t0 + 2*x[1] + t1
 *   out[2] = t7 = t2 + 4*t1 + t3 = 5*t1 + 2*x[1] + 2*x[3] + t0
 *   out[3] = t4 = 4*t1 + t3      = 4*t1 + 2*x[3] + t0
 *
 * In sub-group form, where pair_sum = x + shuffle_xor(x, 1) (= my pair's t)
 * and cross_pair = shuffle_xor(pair_sum, 2) (= other pair's t):
 *
 *   Even lane (0, 2): pair_sum is its own t (t0 for 0, t1 for 2). 5*pair_sum
 *     + 2*x_xor3 (= other pair's odd x, e.g., x[3] for lane 0) + 2*x_xor1
 *     (= partner's x in same pair) + cross_pair.
 *   Odd lane (1, 3): 4*pair_sum + 2*x (its own x, which is x[1] or x[3])
 *     + cross_pair.
 *
 * Cross-block: sum across 6 blocks at same column (sg_lid % 4) via 3-level
 * tree reduce on shuffle masks 4, 8, 16. Idle lanes (24..31) carry 0.
 */
inline uint sg_multiply_by_m_ext(uint x, const uint sg_lid) {
  /* Lane position within block (0..3). */
  const uint i_in_blk = sg_lid & 3u;
  const bool is_even = (i_in_blk & 1u) == 0u;

  /* Gather lane partners within block via XOR-1, XOR-3. */
  const uint x_xor1 = sub_group_shuffle_xor(x, 1);  /* partner in same pair */
  const uint x_xor3 = sub_group_shuffle_xor(x, 3);  /* odd x in OTHER pair */

  /* pair_sum = my pair's t. cross_pair = other pair's t (within block). */
  const uint pair_sum = bb_add(x, x_xor1);
  const uint cross_pair = sub_group_shuffle_xor(pair_sum, 2);

  /* Compute m4 output per lane using the simplified formulas above.
   *   Even (0, 2): 5*pair_sum + 2*x_xor3 + 2*x_xor1 + cross_pair
   *   Odd  (1, 3): 4*pair_sum + 2*x + cross_pair
   */
  const uint two_pair  = bb_add(pair_sum, pair_sum);
  const uint four_pair = bb_add(two_pair, two_pair);
  const uint five_pair = bb_add(four_pair, pair_sum);
  uint m4_out;
  if (is_even) {
    const uint two_x_xor3 = bb_add(x_xor3, x_xor3);
    const uint two_x_xor1 = bb_add(x_xor1, x_xor1);
    m4_out = bb_add(bb_add(five_pair, two_x_xor3), bb_add(two_x_xor1, cross_pair));
  } else {
    const uint two_x = bb_add(x, x);
    m4_out = bb_add(bb_add(four_pair, two_x), cross_pair);
  }

  /* Cross-block accumulation: sum m4_out across the 6 blocks at the same column.
   * Tree reduce on masks 4, 8, 16 within the sub-group (idle blocks contribute 0). */
  const uint m4_out_orig = m4_out;
  uint col_sum = m4_out;
  col_sum = bb_add(col_sum, sub_group_shuffle_xor(col_sum, 4));
  col_sum = bb_add(col_sum, sub_group_shuffle_xor(col_sum, 8));
  col_sum = bb_add(col_sum, sub_group_shuffle_xor(col_sum, 16));
  /* IMPORTANT: idle lanes (24..31) MUST return 0 to keep the next m_ext's
   * cross-block reduce correct. Tree reduce above propagates non-zero
   * col_sum into idle lanes; explicitly zero them. */
  const uint result = bb_add(m4_out_orig, col_sum);
  return (sg_lid < CELLS) ? result : 0u;
}

/* DEAD CODE — kept for reference, will be removed.
 * Original convoluted derivation. Replaced by simpler formula above. */

/* Cooperative m_int: cells[i] = sum + diag[i]*cells[i]. Idle lanes return 0.
 * Note: sg_modular_sum propagates non-zero sums into idle lanes; bb_mul of
 * 0 * x is 0 for idle (since diag_i = 0 for idle); but the bb_add(s, 0) = s
 * which is non-zero. So we must mask idle lanes to 0 at end. */
inline uint sg_multiply_by_m_int(uint x, uint diag_i, const uint sg_lid) {
  const uint s = sg_modular_sum(x);  /* sum across 32 lanes (idle = 0) */
  const uint result = bb_add(s, bb_mul(diag_i, x));
  return (sg_lid < CELLS) ? result : 0u;
}

/* DEBUG: run init m_ext + 1 full round. */
__kernel
__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(32)))
void poseidon2_only_one_full(__global uint * restrict data,
                             __global const uint * restrict round_cs,
                             __global const uint * restrict m_int_diag,
                             const uint n_perms) {
    const uint sg_id_global = get_global_id(0) / 32u;
    if (sg_id_global >= n_perms) return;
    const uint sg_lid = get_sub_group_local_id();
    const bool active = (sg_lid < CELLS);
    uint cell = 0u;
    if (active) cell = data[sg_id_global * CELLS + sg_lid];
    cell = sg_multiply_by_m_ext(cell, sg_lid);
    /* one full round */
    if (active) cell = bb_add(cell, round_cs[sg_lid]);
    cell = bb_sbox(cell);
    cell = sg_multiply_by_m_ext(cell, sg_lid);
    if (active) data[sg_id_global * CELLS + sg_lid] = cell;
}

/* DEBUG: run init m_ext + sbox + m_ext. */
__kernel
__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(32)))
void poseidon2_only_sbox_mext(__global uint * restrict data,
                              __global const uint * restrict round_cs,
                              __global const uint * restrict m_int_diag,
                              const uint n_perms) {
    const uint sg_id_global = get_global_id(0) / 32u;
    if (sg_id_global >= n_perms) return;
    const uint sg_lid = get_sub_group_local_id();
    const bool active = (sg_lid < CELLS);
    uint cell = 0u;
    if (active) cell = data[sg_id_global * CELLS + sg_lid];
    cell = sg_multiply_by_m_ext(cell, sg_lid);
    cell = bb_sbox(cell);
    cell = sg_multiply_by_m_ext(cell, sg_lid);
    if (active) data[sg_id_global * CELLS + sg_lid] = cell;
}

/* DEBUG: run init m_ext + sbox only (no constants, no m_ext after). */
__kernel
__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(32)))
void poseidon2_only_sbox(__global uint * restrict data,
                         __global const uint * restrict round_cs,
                         __global const uint * restrict m_int_diag,
                         const uint n_perms) {
    const uint sg_id_global = get_global_id(0) / 32u;
    if (sg_id_global >= n_perms) return;
    const uint sg_lid = get_sub_group_local_id();
    const bool active = (sg_lid < CELLS);
    uint cell = 0u;
    if (active) cell = data[sg_id_global * CELLS + sg_lid];
    cell = sg_multiply_by_m_ext(cell, sg_lid);
    cell = bb_sbox(cell);
    if (active) data[sg_id_global * CELLS + sg_lid] = cell;
}

/* DEBUG: run init m_ext + 4 full rounds, no partials. */
__kernel
__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(32)))
void poseidon2_only_full(__global uint * restrict data,
                         __global const uint * restrict round_cs,
                         __global const uint * restrict m_int_diag,
                         const uint n_perms) {
    const uint sg_id_global = get_global_id(0) / 32u;
    if (sg_id_global >= n_perms) return;
    const uint sg_lid = get_sub_group_local_id();
    const bool active = (sg_lid < CELLS);
    uint cell = 0u;
    if (active) cell = data[sg_id_global * CELLS + sg_lid];
    cell = sg_multiply_by_m_ext(cell, sg_lid);
    uint rd = 0u;
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        const uint base = rd * CELLS;
        if (active) cell = bb_add(cell, round_cs[base + sg_lid]);
        cell = bb_sbox(cell);
        cell = sg_multiply_by_m_ext(cell, sg_lid);
        ++rd;
    }
    if (active) data[sg_id_global * CELLS + sg_lid] = cell;
}

/* DEBUG: run only the initial m_ext, no rounds. For diagnosing m_ext correctness. */
__kernel
__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(32)))
void poseidon2_only_mext(__global uint * restrict data,
                         __global const uint * restrict round_cs,
                         __global const uint * restrict m_int_diag,
                         const uint n_perms) {
    const uint sg_id_global = get_global_id(0) / 32u;
    if (sg_id_global >= n_perms) return;
    const uint sg_lid = get_sub_group_local_id();
    const bool active = (sg_lid < CELLS);
    uint cell = 0u;
    if (active) cell = data[sg_id_global * CELLS + sg_lid];
    cell = sg_multiply_by_m_ext(cell, sg_lid);
    if (active) data[sg_id_global * CELLS + sg_lid] = cell;
}

__kernel
__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(32)))
void poseidon2_mix_coop(__global uint * restrict data,
                        __global const uint * restrict round_cs,
                        __global const uint * restrict m_int_diag,
                        const uint n_perms)
{
    const uint sg_id_global = get_global_id(0) / 32u;  /* one perm per sub-group, globally */
    if (sg_id_global >= n_perms) return;

    const uint sg_lid = get_sub_group_local_id();      /* 0..31 */
    const uint cell_idx = sg_lid;                       /* lanes 0..23 are active */
    const bool active = (sg_lid < CELLS);

    /* Load my cell (or 0 if idle). */
    uint cell = 0u;
    if (active) {
        cell = data[sg_id_global * CELLS + cell_idx];
    }

    /* Load my diagonal entry. */
    const uint my_diag = active ? m_int_diag[cell_idx] : 0u;

    /* Initial linear layer: m_ext. */
    cell = sg_multiply_by_m_ext(cell, sg_lid);

    uint rd = 0u;

    /* First half-full rounds (0..3). */
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        const uint base = rd * CELLS;
        /* Add round constant to all 24 active cells. */
        if (active) {
            cell = bb_add(cell, round_cs[base + cell_idx]);
        }
        /* Sbox to all. */
        cell = bb_sbox(cell);
        /* m_ext. */
        cell = sg_multiply_by_m_ext(cell, sg_lid);
        ++rd;
    }

    /* Partial rounds (4..24). Only cell 0 gets constant + sbox. */
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_PARTIAL; ++r) {
        const uint base = rd * CELLS;
        /* Lane 0 adds constant; others get 0 added (== identity, but use sub_group_broadcast for uniform). */
        const uint rc0 = round_cs[base];
        if (sg_lid == 0u) {
            cell = bb_add(cell, rc0);
        }
        /* Sbox lane 0 only. */
        if (sg_lid == 0u) {
            cell = bb_sbox(cell);
        }
        /* m_int. */
        cell = sg_multiply_by_m_int(cell, my_diag, sg_lid);
        ++rd;
    }

    /* Last half-full rounds. */
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        const uint base = rd * CELLS;
        if (active) {
            cell = bb_add(cell, round_cs[base + cell_idx]);
        }
        cell = bb_sbox(cell);
        cell = sg_multiply_by_m_ext(cell, sg_lid);
        ++rd;
    }

    /* Store. */
    if (active) {
        data[sg_id_global * CELLS + cell_idx] = cell;
    }
}

/* Main permutation kernel: each work-item processes one permutation.
 * Args:
 *   data:       [n_perms × 24] in/out, canonical u32
 *   round_cs:   ROUND_CONSTANTS, length 696 (8+21 rounds × 24 cells), padded
 *   m_int_diag: M_INT_DIAG_HZN, length 24
 *   n_perms:    number of independent permutations
 */
/* Mont-form helper: m_ext on Mont-form cells. Same mathematical structure as
 * non-Mont m_ext (only additions and multiplies-by-small-constant), except the
 * "multiply by 2 / 4" remains Mont-correct because Mont is linear:
 * R*(a+b) = R*a + R*b. So bb_add and bb_mul2/4 work unchanged on Mont elems. */

/* Mont-form variant of multiply_by_m_int — batched ILP + tree-reduced sum.
 * Sum is depth-5 add-tree instead of depth-24 chain; on partial rounds the sum
 * is on the critical path (cells[0] sbox→sum→mul→write), so flattening the tree
 * shortens latency by ~19 add-cycles per partial round (× 21 partials per perm). */
inline void multiply_by_m_int_mont(uint *cells, __global const uint * restrict diag_mont) {
    /* Tree-reduce 24 cells: 24→12→6→3→2→1 (depth 5). */
    uint a01 = bb_add(cells[0],  cells[1]);   uint a23 = bb_add(cells[2],  cells[3]);
    uint a45 = bb_add(cells[4],  cells[5]);   uint a67 = bb_add(cells[6],  cells[7]);
    uint a89 = bb_add(cells[8],  cells[9]);   uint a1011 = bb_add(cells[10], cells[11]);
    uint a1213 = bb_add(cells[12], cells[13]); uint a1415 = bb_add(cells[14], cells[15]);
    uint a1617 = bb_add(cells[16], cells[17]); uint a1819 = bb_add(cells[18], cells[19]);
    uint a2021 = bb_add(cells[20], cells[21]); uint a2223 = bb_add(cells[22], cells[23]);
    /* 12 → 6 */
    uint b03   = bb_add(a01, a23);     uint b47   = bb_add(a45, a67);
    uint b811  = bb_add(a89, a1011);   uint b1215 = bb_add(a1213, a1415);
    uint b1619 = bb_add(a1617, a1819); uint b2023 = bb_add(a2021, a2223);
    /* 6 → 3 */
    uint c07   = bb_add(b03, b47);
    uint c815  = bb_add(b811, b1215);
    uint c1623 = bb_add(b1619, b2023);
    /* 3 → 1 */
    uint d015  = bb_add(c07, c815);
    uint s     = bb_add(d015, c1623);

    uint dx[CELLS];
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) dx[i] = bb_mont_mul(diag_mont[i], cells[i]);
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) cells[i] = bb_add(s, dx[i]);
}

/* Same multiply_by_m_ext (no muls — works in Mont form unchanged). */

/* Two-perms-per-work-item Mont kernel. Each WI processes 2 independent perms
 * with state interleaved in registers. The compiler can pipeline mul-deps
 * between the two perms (perm A's sbox computing while perm B's mul-mod is
 * in flight). Doubles register state per lane (24 cells × 2 = 48 cells).
 */
inline void mont_mix_one(uint *cells,
                         __global const uint * restrict round_cs_mont,
                         __global const uint * restrict m_int_diag_mont) {
    multiply_by_m_ext(cells);
    uint rd = 0u;
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        const uint base = rd * CELLS;
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) cells[i] = bb_add(cells[i], round_cs_mont[base + i]);
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) cells[i] = bb_sbox_mont(cells[i]);
        multiply_by_m_ext(cells);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_PARTIAL; ++r) {
        const uint base = rd * CELLS;
        cells[0] = bb_add(cells[0], round_cs_mont[base]);
        cells[0] = bb_sbox_mont(cells[0]);
        multiply_by_m_int_mont(cells, m_int_diag_mont);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        const uint base = rd * CELLS;
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) cells[i] = bb_add(cells[i], round_cs_mont[base + i]);
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) cells[i] = bb_sbox_mont(cells[i]);
        multiply_by_m_ext(cells);
        ++rd;
    }
}

__kernel
void poseidon2_mix_mont_x2(__global uint * restrict data,
                           __global const uint * restrict round_cs_mont,
                           __global const uint * restrict m_int_diag_mont,
                           const uint n_perms)
{
    /* Each WI processes perm 2*gid and perm 2*gid+1. */
    const uint gid = get_global_id(0);
    if (2u * gid + 1u >= n_perms) return;

    uint cellsA[CELLS], cellsB[CELLS];
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) {
        cellsA[i] = data[(2u * gid + 0u) * CELLS + i];
        cellsB[i] = data[(2u * gid + 1u) * CELLS + i];
    }

    /* Run both perms in parallel — interleaved in source so IGC can pipeline. */
    multiply_by_m_ext(cellsA);
    multiply_by_m_ext(cellsB);

    uint rd = 0u;
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        const uint base = rd * CELLS;
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) {
            cellsA[i] = bb_add(cellsA[i], round_cs_mont[base + i]);
            cellsB[i] = bb_add(cellsB[i], round_cs_mont[base + i]);
        }
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) {
            cellsA[i] = bb_sbox_mont(cellsA[i]);
            cellsB[i] = bb_sbox_mont(cellsB[i]);
        }
        multiply_by_m_ext(cellsA);
        multiply_by_m_ext(cellsB);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_PARTIAL; ++r) {
        const uint base = rd * CELLS;
        cellsA[0] = bb_add(cellsA[0], round_cs_mont[base]);
        cellsB[0] = bb_add(cellsB[0], round_cs_mont[base]);
        cellsA[0] = bb_sbox_mont(cellsA[0]);
        cellsB[0] = bb_sbox_mont(cellsB[0]);
        multiply_by_m_int_mont(cellsA, m_int_diag_mont);
        multiply_by_m_int_mont(cellsB, m_int_diag_mont);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        const uint base = rd * CELLS;
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) {
            cellsA[i] = bb_add(cellsA[i], round_cs_mont[base + i]);
            cellsB[i] = bb_add(cellsB[i], round_cs_mont[base + i]);
        }
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) {
            cellsA[i] = bb_sbox_mont(cellsA[i]);
            cellsB[i] = bb_sbox_mont(cellsB[i]);
        }
        multiply_by_m_ext(cellsA);
        multiply_by_m_ext(cellsB);
        ++rd;
    }

    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) {
        data[(2u * gid + 0u) * CELLS + i] = cellsA[i];
        data[(2u * gid + 1u) * CELLS + i] = cellsB[i];
    }
}

/* Permutation kernel using Montgomery form throughout.
 * Caller (host) must:
 *   - Pre-convert ROUND_CONSTANTS to Mont form: rc_mont[i] = (rc[i] * R) mod p
 *   - Pre-convert M_INT_DIAG_HZN to Mont form: diag_mont[i] = (diag[i] * R) mod p
 *   - At kernel entry, inputs are converted to Mont form via bb_mont_mul(x, R^2 mod p)
 *     (or skip — host can pre-convert input.bin to Mont form)
 *   - At kernel exit, outputs are converted back via bb_mont_mul(x_mont, 1)
 *     (which = x_mont * R^-1 = x_canonical)
 *
 * For benchmarking, we let the host upload data already in Mont form, and
 * convert back at exit. This avoids paying the entry-conversion cost in the
 * timing loop.
 */
/* Batched-sbox layer: instead of `for i in 0..24: sbox(cells[i])` (which keeps
 * each cell's chain x→x²→x⁴→x⁶→x⁷ contiguous), interleave the chains explicitly
 * so the compiler sees 24 independent length-1 dependencies at each phase.
 *
 * Empirically: `#pragma unroll` already lets IGC re-schedule across cells, but
 * presenting the dataflow this way gives the scheduler less work to do and
 * can reduce register pressure (intermediates have shorter live ranges). */
inline void sbox_layer_mont(uint *cells) {
    uint x2[CELLS], x4[CELLS], x6[CELLS];
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) x2[i] = bb_mont_mul(cells[i], cells[i]);
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) x4[i] = bb_mont_mul(x2[i], x2[i]);
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) x6[i] = bb_mont_mul(x4[i], x2[i]);
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) cells[i] = bb_mont_mul(x6[i], cells[i]);
}

/* madw variant of sbox layer — same as sbox_layer_mont but uses
 * bb_mont_mul_madw, dropping each mul-pair from 4 to 3 multiplies. */
inline void sbox_layer_mont_madw(uint *cells) {
    uint x2[CELLS], x4[CELLS], x6[CELLS];
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) x2[i] = bb_mont_mul_madw(cells[i], cells[i]);
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) x4[i] = bb_mont_mul_madw(x2[i], x2[i]);
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) x6[i] = bb_mont_mul_madw(x4[i], x2[i]);
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) cells[i] = bb_mont_mul_madw(x6[i], cells[i]);
}

inline void add_rc_sbox_layer_mont_madw(uint *cells, __global const uint * restrict rc) {
    uint x[CELLS], x2[CELLS], x4[CELLS], x6[CELLS];
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) x[i] = bb_add(cells[i], rc[i]);
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) x2[i] = bb_mont_mul_madw(x[i], x[i]);
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) x4[i] = bb_mont_mul_madw(x2[i], x2[i]);
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) x6[i] = bb_mont_mul_madw(x4[i], x2[i]);
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) cells[i] = bb_mont_mul_madw(x6[i], x[i]);
}

inline void multiply_by_m_int_mont_madw(uint *cells, __global const uint * restrict diag_mont) {
    uint a01 = bb_add(cells[0],  cells[1]);   uint a23 = bb_add(cells[2],  cells[3]);
    uint a45 = bb_add(cells[4],  cells[5]);   uint a67 = bb_add(cells[6],  cells[7]);
    uint a89 = bb_add(cells[8],  cells[9]);   uint a1011 = bb_add(cells[10], cells[11]);
    uint a1213 = bb_add(cells[12], cells[13]); uint a1415 = bb_add(cells[14], cells[15]);
    uint a1617 = bb_add(cells[16], cells[17]); uint a1819 = bb_add(cells[18], cells[19]);
    uint a2021 = bb_add(cells[20], cells[21]); uint a2223 = bb_add(cells[22], cells[23]);
    uint b03   = bb_add(a01, a23);     uint b47   = bb_add(a45, a67);
    uint b811  = bb_add(a89, a1011);   uint b1215 = bb_add(a1213, a1415);
    uint b1619 = bb_add(a1617, a1819); uint b2023 = bb_add(a2021, a2223);
    uint c07   = bb_add(b03, b47);
    uint c815  = bb_add(b811, b1215);
    uint c1623 = bb_add(b1619, b2023);
    uint d015  = bb_add(c07, c815);
    uint s     = bb_add(d015, c1623);
    uint dx[CELLS];
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) dx[i] = bb_mont_mul_madw(diag_mont[i], cells[i]);
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) cells[i] = bb_add(s, dx[i]);
}

/* Production madw kernel — same shape as poseidon2_mix_mont_ilp but uses
 * inline-asm bb_mont_mul_madw throughout. Requires SIMD16 sub-group. */
__kernel
__attribute__((intel_reqd_sub_group_size(16)))
void poseidon2_mix_mont_madw(__global uint * restrict data,
                             __global const uint * restrict round_cs_mont,
                             __global const uint * restrict m_int_diag_mont,
                             const uint n_perms)
{
    const uint gid = get_global_id(0);
    if (gid >= n_perms) return;

    uint cells[CELLS];
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) cells[i] = data[gid * CELLS + i];

    multiply_by_m_ext(cells);
    uint rd = 0u;
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        const uint base = rd * CELLS;
        add_rc_sbox_layer_mont_madw(cells, round_cs_mont + base);
        multiply_by_m_ext(cells);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_PARTIAL; ++r) {
        const uint base = rd * CELLS;
        cells[0] = bb_add(cells[0], round_cs_mont[base]);
        cells[0] = bb_sbox_mont_madw(cells[0]);
        multiply_by_m_int_mont_madw(cells, m_int_diag_mont);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        const uint base = rd * CELLS;
        add_rc_sbox_layer_mont_madw(cells, round_cs_mont + base);
        multiply_by_m_ext(cells);
        ++rd;
    }
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) data[gid * CELLS + i] = cells[i];
}

/* Fused: add round constant + sbox in one batched layer. Saves a pass over
 * the cells array between add and first square. */
inline void add_rc_sbox_layer_mont(uint *cells, __global const uint * restrict rc) {
    uint x[CELLS], x2[CELLS], x4[CELLS], x6[CELLS];
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) x[i] = bb_add(cells[i], rc[i]);
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) x2[i] = bb_mont_mul(x[i], x[i]);
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) x4[i] = bb_mont_mul(x2[i], x2[i]);
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) x6[i] = bb_mont_mul(x4[i], x2[i]);
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) cells[i] = bb_mont_mul(x6[i], x[i]);
}

/* ILP variant — same as poseidon2_mix_mont but uses sbox_layer_mont to expose
 * cell-parallelism explicitly to the scheduler. */
__kernel
void poseidon2_mix_mont_ilp(__global uint * restrict data,
                            __global const uint * restrict round_cs_mont,
                            __global const uint * restrict m_int_diag_mont,
                            const uint n_perms)
{
    const uint gid = get_global_id(0);
    if (gid >= n_perms) return;

    uint cells[CELLS];
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) {
        cells[i] = data[gid * CELLS + i];
    }

    multiply_by_m_ext(cells);
    uint rd = 0u;
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        const uint base = rd * CELLS;
        add_rc_sbox_layer_mont(cells, round_cs_mont + base);
        multiply_by_m_ext(cells);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_PARTIAL; ++r) {
        const uint base = rd * CELLS;
        cells[0] = bb_add(cells[0], round_cs_mont[base]);
        cells[0] = bb_sbox_mont(cells[0]);
        multiply_by_m_int_mont(cells, m_int_diag_mont);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        const uint base = rd * CELLS;
        add_rc_sbox_layer_mont(cells, round_cs_mont + base);
        multiply_by_m_ext(cells);
        ++rd;
    }
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) {
        data[gid * CELLS + i] = cells[i];
    }
}

/* SIMD16 variant — forces 16-wide sub-group; sometimes helps if SIMD32 has
 * register-pressure / scheduling-pressure issues on long-dependency kernels. */
__kernel
__attribute__((intel_reqd_sub_group_size(16)))
void poseidon2_mix_mont_simd16(__global uint * restrict data,
                               __global const uint * restrict round_cs_mont,
                               __global const uint * restrict m_int_diag_mont,
                               const uint n_perms)
{
    const uint gid = get_global_id(0);
    if (gid >= n_perms) return;

    uint cells[CELLS];
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) {
        cells[i] = data[gid * CELLS + i];
    }

    multiply_by_m_ext(cells);
    uint rd = 0u;
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        const uint base = rd * CELLS;
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) {
            cells[i] = bb_add(cells[i], round_cs_mont[base + i]);
        }
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) {
            cells[i] = bb_sbox_mont(cells[i]);
        }
        multiply_by_m_ext(cells);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_PARTIAL; ++r) {
        const uint base = rd * CELLS;
        cells[0] = bb_add(cells[0], round_cs_mont[base]);
        cells[0] = bb_sbox_mont(cells[0]);
        multiply_by_m_int_mont(cells, m_int_diag_mont);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        const uint base = rd * CELLS;
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) {
            cells[i] = bb_add(cells[i], round_cs_mont[base + i]);
        }
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) {
            cells[i] = bb_sbox_mont(cells[i]);
        }
        multiply_by_m_ext(cells);
        ++rd;
    }
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) {
        data[gid * CELLS + i] = cells[i];
    }
}

__kernel
void poseidon2_mix_mont(__global uint * restrict data,
                        __global const uint * restrict round_cs_mont,
                        __global const uint * restrict m_int_diag_mont,
                        const uint n_perms)
{
    const uint gid = get_global_id(0);
    if (gid >= n_perms) return;

    uint cells[CELLS];
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) {
        cells[i] = data[gid * CELLS + i];   /* assumed already in Mont form */
    }

    multiply_by_m_ext(cells);

    uint rd = 0u;

    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        const uint base = rd * CELLS;
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) {
            cells[i] = bb_add(cells[i], round_cs_mont[base + i]);
        }
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) {
            cells[i] = bb_sbox_mont(cells[i]);
        }
        multiply_by_m_ext(cells);
        ++rd;
    }

    #pragma unroll
    for (uint r = 0u; r < ROUNDS_PARTIAL; ++r) {
        const uint base = rd * CELLS;
        cells[0] = bb_add(cells[0], round_cs_mont[base]);
        cells[0] = bb_sbox_mont(cells[0]);
        multiply_by_m_int_mont(cells, m_int_diag_mont);
        ++rd;
    }

    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        const uint base = rd * CELLS;
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) {
            cells[i] = bb_add(cells[i], round_cs_mont[base + i]);
        }
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) {
            cells[i] = bb_sbox_mont(cells[i]);
        }
        multiply_by_m_ext(cells);
        ++rd;
    }

    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) {
        data[gid * CELLS + i] = cells[i];   /* still Mont form; host converts */
    }
}

/* Note: tried `__constant` for round_cs / m_int_diag (small-table read-only
 * scalar-cache path). Measured: ~3% regression vs `__global const restrict`.
 * IGC promotes both already; explicit __constant didn't help. Reverted. */
__kernel
void poseidon2_mix(__global uint * restrict data,
                   __global const uint * restrict round_cs,
                   __global const uint * restrict m_int_diag,
                   const uint n_perms)
{
    const uint gid = get_global_id(0);
    if (gid >= n_perms) return;

    /* Load 24 cells into private state. */
    uint cells[CELLS];
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) {
        cells[i] = data[gid * CELLS + i];
    }

    /* Initial linear layer. */
    multiply_by_m_ext(cells);

    uint rd = 0u;

    /* First half-full rounds (0..3): + constants, x^7 to all, m_ext. */
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        const uint base = rd * CELLS;
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) {
            cells[i] = bb_add(cells[i], round_cs[base + i]);
        }
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) {
            cells[i] = bb_sbox(cells[i]);
        }
        multiply_by_m_ext(cells);
        ++rd;
    }

    /* Partial rounds (4..24): + cells[0]+=constant[base], x^7 to cells[0], m_int. */
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_PARTIAL; ++r) {
        const uint base = rd * CELLS;
        cells[0] = bb_add(cells[0], round_cs[base]);
        cells[0] = bb_sbox(cells[0]);
        multiply_by_m_int(cells, m_int_diag);
        ++rd;
    }

    /* Last half-full rounds (25..28). */
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        const uint base = rd * CELLS;
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) {
            cells[i] = bb_add(cells[i], round_cs[base + i]);
        }
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) {
            cells[i] = bb_sbox(cells[i]);
        }
        multiply_by_m_ext(cells);
        ++rd;
    }

    /* Store. */
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) {
        data[gid * CELLS + i] = cells[i];
    }
}

/* x2: process TWO permutations per WI for ILP. Each WI holds 2*CELLS = 48 cells.
 * Constants (round_cs, m_int_diag) loaded once, used by both permutations.
 * The two perm chains are independent → compiler should interleave.
 */
__kernel
void poseidon2_mix_mont_ilp_x2(__global uint * restrict data,
                               __global const uint * restrict round_cs_mont,
                               __global const uint * restrict m_int_diag_mont,
                               const uint n_perms)
{
    const uint gid = get_global_id(0);
    const uint base = gid * 2u;
    if (base + 1u >= n_perms) {
        /* Tail: just do one perm if odd. */
        if (base >= n_perms) return;
        uint cells[CELLS];
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) cells[i] = data[base * CELLS + i];
        multiply_by_m_ext(cells);
        uint rd = 0u;
        #pragma unroll
        for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
            const uint b = rd * CELLS;
            add_rc_sbox_layer_mont(cells, round_cs_mont + b);
            multiply_by_m_ext(cells);
            ++rd;
        }
        #pragma unroll
        for (uint r = 0u; r < ROUNDS_PARTIAL; ++r) {
            const uint b = rd * CELLS;
            cells[0] = bb_add(cells[0], round_cs_mont[b]);
            cells[0] = bb_sbox_mont(cells[0]);
            multiply_by_m_int_mont(cells, m_int_diag_mont);
            ++rd;
        }
        #pragma unroll
        for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
            const uint b = rd * CELLS;
            add_rc_sbox_layer_mont(cells, round_cs_mont + b);
            multiply_by_m_ext(cells);
            ++rd;
        }
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) data[base * CELLS + i] = cells[i];
        return;
    }

    uint cells_a[CELLS], cells_b[CELLS];
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) {
        cells_a[i] = data[(base + 0u) * CELLS + i];
        cells_b[i] = data[(base + 1u) * CELLS + i];
    }
    multiply_by_m_ext(cells_a);
    multiply_by_m_ext(cells_b);

    uint rd = 0u;
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        const uint b = rd * CELLS;
        add_rc_sbox_layer_mont(cells_a, round_cs_mont + b);
        add_rc_sbox_layer_mont(cells_b, round_cs_mont + b);
        multiply_by_m_ext(cells_a);
        multiply_by_m_ext(cells_b);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_PARTIAL; ++r) {
        const uint b = rd * CELLS;
        cells_a[0] = bb_add(cells_a[0], round_cs_mont[b]);
        cells_b[0] = bb_add(cells_b[0], round_cs_mont[b]);
        cells_a[0] = bb_sbox_mont(cells_a[0]);
        cells_b[0] = bb_sbox_mont(cells_b[0]);
        multiply_by_m_int_mont(cells_a, m_int_diag_mont);
        multiply_by_m_int_mont(cells_b, m_int_diag_mont);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        const uint b = rd * CELLS;
        add_rc_sbox_layer_mont(cells_a, round_cs_mont + b);
        add_rc_sbox_layer_mont(cells_b, round_cs_mont + b);
        multiply_by_m_ext(cells_a);
        multiply_by_m_ext(cells_b);
        ++rd;
    }
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) {
        data[(base + 0u) * CELLS + i] = cells_a[i];
        data[(base + 1u) * CELLS + i] = cells_b[i];
    }
}

/* ============================================================================
 * poseidon2_hash_fold — Merkle-tree internal-node hashing.
 *
 * Each work-item runs one permutation that consumes two 8-element BabyBear
 * digests (=16 input cells, the "rate" of the sponge) and produces one
 * 8-element output digest (= first 8 cells after the permutation). The
 * remaining 8 capacity cells start at zero, exactly like risc0's poseidon2
 * fold (cf. zkp/src/core/hash/poseidon2/mod.rs and risc0/sys/kernels/zkp/
 * intel/poseidon2.cpp::esimd_poseidon2_fold).
 *
 *   src layout  : [num_hashes * 16] BabyBear elements in normal (canonical) form
 *   dst layout  : [num_hashes *  8] BabyBear elements in normal form
 *
 * Mont-form conversion is folded into the load/store: rate cells are
 * multiplied by R^2 on load (normal -> Mont, one bb_mont_mul each) and the
 * output digest cells are passed through bb_mont_mul(c, 1) on store (Mont
 * -> normal via REDC). Capacity cells stay zero so no conversion needed.
 *
 * 16 to_mont + 8 from_mont per perm ≈ 24 extra muls vs the ~200 muls of the
 * permutation body, i.e. ~12% overhead for the conversion bookkeeping.
 * ============================================================================ */
__kernel
void poseidon2_hash_fold(__global uint * restrict dst,
                         __global const uint * restrict src,
                         __global const uint * restrict round_cs_mont,
                         __global const uint * restrict m_int_diag_mont,
                         const uint num_hashes,
                         const uint r_sq)
{
    const uint gid = get_global_id(0);
    if (gid >= num_hashes) return;

    uint cells[CELLS];
    // Rate: 16 input cells, normal -> Mont via mul by R^2.
    #pragma unroll
    for (uint i = 0u; i < 16u; ++i) {
        cells[i] = bb_mont_mul(src[gid * 16u + i], r_sq);
    }
    // Capacity: 8 zero cells (Mont(0) == 0).
    #pragma unroll
    for (uint i = 16u; i < CELLS; ++i) {
        cells[i] = 0u;
    }

    // Full Poseidon2 permutation (mirrors poseidon2_mix_mont_ilp).
    multiply_by_m_ext(cells);
    uint rd = 0u;
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        add_rc_sbox_layer_mont(cells, round_cs_mont + rd * CELLS);
        multiply_by_m_ext(cells);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_PARTIAL; ++r) {
        cells[0] = bb_add(cells[0], round_cs_mont[rd * CELLS]);
        cells[0] = bb_sbox_mont(cells[0]);
        multiply_by_m_int_mont(cells, m_int_diag_mont);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        add_rc_sbox_layer_mont(cells, round_cs_mont + rd * CELLS);
        multiply_by_m_ext(cells);
        ++rd;
    }

    // First 8 cells form the output digest; Mont -> normal via REDC (mul by 1).
    #pragma unroll
    for (uint c = 0u; c < 8u; ++c) {
        dst[gid * 8u + c] = bb_mont_mul(cells[c], 1u);
    }
}

/* ============================================================================
 * poseidon2_hash_fold_x2 — two hashes per work-item (ILP-2).
 *
 * Mirrors poseidon2_mix_mont_ilp_x2 layout: each WI carries two independent
 * 24-cell states `a` and `b` and interleaves their compute so the scheduler
 * can hide mul-mod latency between the two chains. Tail (odd count) falls
 * back to a single hash.
 *
 * Throughput-vs-pressure tradeoff: 2 * 24 = 48 cells in registers per WI
 * plus add_rc_sbox_layer_mont's intermediates (x, x2, x4, x6 → 4*24 = 96
 * scratch). Total ~144 GRF * 4 bytes = 576 B per lane — fits in 256-GRF
 * mode and lets IGC keep everything register-resident without scratch
 * spills, while doubling the in-flight work.
 * ============================================================================ */
__kernel
void poseidon2_hash_fold_x2(__global uint * restrict dst,
                            __global const uint * restrict src,
                            __global const uint * restrict round_cs_mont,
                            __global const uint * restrict m_int_diag_mont,
                            const uint num_hashes,
                            const uint r_sq)
{
    const uint gid = get_global_id(0);
    const uint base = gid * 2u;

    // Tail path: single hash if count is odd or we're the final WI.
    if (base + 1u >= num_hashes) {
        if (base >= num_hashes) return;
        uint cells[CELLS];
        #pragma unroll
        for (uint i = 0u; i < 16u; ++i) cells[i] = bb_mont_mul(src[base * 16u + i], r_sq);
        #pragma unroll
        for (uint i = 16u; i < CELLS; ++i) cells[i] = 0u;
        multiply_by_m_ext(cells);
        uint rd = 0u;
        #pragma unroll
        for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
            add_rc_sbox_layer_mont(cells, round_cs_mont + rd * CELLS);
            multiply_by_m_ext(cells);
            ++rd;
        }
        #pragma unroll
        for (uint r = 0u; r < ROUNDS_PARTIAL; ++r) {
            cells[0] = bb_add(cells[0], round_cs_mont[rd * CELLS]);
            cells[0] = bb_sbox_mont(cells[0]);
            multiply_by_m_int_mont(cells, m_int_diag_mont);
            ++rd;
        }
        #pragma unroll
        for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
            add_rc_sbox_layer_mont(cells, round_cs_mont + rd * CELLS);
            multiply_by_m_ext(cells);
            ++rd;
        }
        #pragma unroll
        for (uint c = 0u; c < 8u; ++c) dst[base * 8u + c] = bb_mont_mul(cells[c], 1u);
        return;
    }

    // x2 path: two independent perms in flight.
    uint a[CELLS], b[CELLS];
    #pragma unroll
    for (uint i = 0u; i < 16u; ++i) {
        a[i] = bb_mont_mul(src[(base + 0u) * 16u + i], r_sq);
        b[i] = bb_mont_mul(src[(base + 1u) * 16u + i], r_sq);
    }
    #pragma unroll
    for (uint i = 16u; i < CELLS; ++i) { a[i] = 0u; b[i] = 0u; }

    multiply_by_m_ext(a);
    multiply_by_m_ext(b);
    uint rd = 0u;
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        const uint bidx = rd * CELLS;
        add_rc_sbox_layer_mont(a, round_cs_mont + bidx);
        add_rc_sbox_layer_mont(b, round_cs_mont + bidx);
        multiply_by_m_ext(a);
        multiply_by_m_ext(b);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_PARTIAL; ++r) {
        const uint bidx = rd * CELLS;
        a[0] = bb_add(a[0], round_cs_mont[bidx]);
        b[0] = bb_add(b[0], round_cs_mont[bidx]);
        a[0] = bb_sbox_mont(a[0]);
        b[0] = bb_sbox_mont(b[0]);
        multiply_by_m_int_mont(a, m_int_diag_mont);
        multiply_by_m_int_mont(b, m_int_diag_mont);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        const uint bidx = rd * CELLS;
        add_rc_sbox_layer_mont(a, round_cs_mont + bidx);
        add_rc_sbox_layer_mont(b, round_cs_mont + bidx);
        multiply_by_m_ext(a);
        multiply_by_m_ext(b);
        ++rd;
    }

    #pragma unroll
    for (uint c = 0u; c < 8u; ++c) {
        dst[(base + 0u) * 8u + c] = bb_mont_mul(a[c], 1u);
        dst[(base + 1u) * 8u + c] = bb_mont_mul(b[c], 1u);
    }
}

/* ============================================================================
 * poseidon2_hash_fold_madw — SIMD16 forced + inline-asm madw mont mul.
 *
 * Uses bb_mont_mul_madw / bb_sbox_mont_madw which fold mul-pair into a single
 * `madw` macroinstruction (saves one 32x32 mul per mont mul). Requires the
 * sub-group to be exactly SIMD16 so the inline-asm encoding matches. ~12%
 * theoretical per-mont-mul win; in practice limited by scheduling.
 * ============================================================================ */

/* add_rc_sbox_layer_mont_madw + multiply_by_m_int_mont_madw are already
 * defined earlier (used by poseidon2_mix_mont_madw); reuse those. */

__kernel
__attribute__((intel_reqd_sub_group_size(16)))
void poseidon2_hash_fold_madw(__global uint * restrict dst,
                              __global const uint * restrict src,
                              __global const uint * restrict round_cs_mont,
                              __global const uint * restrict m_int_diag_mont,
                              const uint num_hashes,
                              const uint r_sq)
{
    const uint gid = get_global_id(0);
    if (gid >= num_hashes) return;

    uint cells[CELLS];
    #pragma unroll
    for (uint i = 0u; i < 16u; ++i) cells[i] = bb_mont_mul_madw(src[gid * 16u + i], r_sq);
    #pragma unroll
    for (uint i = 16u; i < CELLS; ++i) cells[i] = 0u;

    multiply_by_m_ext(cells);
    uint rd = 0u;
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        add_rc_sbox_layer_mont_madw(cells, round_cs_mont + rd * CELLS);
        multiply_by_m_ext(cells);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_PARTIAL; ++r) {
        cells[0] = bb_add(cells[0], round_cs_mont[rd * CELLS]);
        cells[0] = bb_sbox_mont_madw(cells[0]);
        multiply_by_m_int_mont_madw(cells, m_int_diag_mont);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        add_rc_sbox_layer_mont_madw(cells, round_cs_mont + rd * CELLS);
        multiply_by_m_ext(cells);
        ++rd;
    }

    #pragma unroll
    for (uint c = 0u; c < 8u; ++c) dst[gid * 8u + c] = bb_mont_mul_madw(cells[c], 1u);
}

/* ============================================================================
 * poseidon2_hash_rows — sponge over `col_size` BabyBear elements per column.
 *
 * Exact mirror of risc0/sys/kernels/zkp/intel/poseidon2.cpp::esimd_poseidon2_rows:
 *   src layout : matrix[row * count + col], rows=col_size, cols=count
 *   dst layout : out[col * 8 + c]   for c=0..7
 *
 * Each work-item runs ONE column's sponge: absorb CELLS_RATE elements at a
 * time, permute, repeat for ceil(col_size/CELLS_RATE) blocks. With col_size=24
 * (the standard FRI commit row width), each column = 2 perms.
 *
 * Conversion: input matrix is in normal form. On the first absorb block we
 * load directly into Mont form via to_mont. Subsequent blocks add the new
 * input (in Mont form) to the rate cells (which are in Mont form post-perm)
 * — Mont addition is just modular add. After the final perm we convert the
 * first 8 cells back to normal form for output.
 * ============================================================================ */
__kernel
void poseidon2_hash_rows(__global uint * restrict dst,
                         __global const uint * restrict src,
                         const uint count,
                         const uint col_size,
                         const uint r_sq,
                         __global const uint * restrict round_cs_mont,
                         __global const uint * restrict m_int_diag_mont)
{
    const uint col = get_global_id(0);
    if (col >= count) return;

    uint cells[CELLS];
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) cells[i] = 0u;

    const uint CELLS_RATE = 16u;
    // Sponge absorb loop. Inputs come row-major as src[row * count + col].
    for (uint base = 0u; base < col_size; base += CELLS_RATE) {
        const uint take = (col_size - base < CELLS_RATE) ? (col_size - base) : CELLS_RATE;
        // Absorb: rate cells get += to_mont(input).
        for (uint j = 0u; j < take; ++j) {
            const uint v = src[(base + j) * count + col];
            cells[j] = bb_add(cells[j], bb_mont_mul(v, r_sq));
        }
        // Permute.
        multiply_by_m_ext(cells);
        uint rd = 0u;
        #pragma unroll
        for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
            add_rc_sbox_layer_mont(cells, round_cs_mont + rd * CELLS);
            multiply_by_m_ext(cells);
            ++rd;
        }
        #pragma unroll
        for (uint r = 0u; r < ROUNDS_PARTIAL; ++r) {
            cells[0] = bb_add(cells[0], round_cs_mont[rd * CELLS]);
            cells[0] = bb_sbox_mont(cells[0]);
            multiply_by_m_int_mont(cells, m_int_diag_mont);
            ++rd;
        }
        #pragma unroll
        for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
            add_rc_sbox_layer_mont(cells, round_cs_mont + rd * CELLS);
            multiply_by_m_ext(cells);
            ++rd;
        }
    }

    // Output first 8 cells, Mont -> normal.
    #pragma unroll
    for (uint c = 0u; c < 8u; ++c) {
        dst[col * 8u + c] = bb_mont_mul(cells[c], 1u);
    }
}

/* ============================================================================
 * poseidon2_hash_fold_const — same as baseline but round_cs/diag use
 * __constant. Tests if the const-cache path is faster than the L1-cached
 * __global path for the per-round constant load.
 * ============================================================================ */
__kernel
void poseidon2_hash_fold_const(__global uint * restrict dst,
                               __global const uint * restrict src,
                               __constant const uint * round_cs_mont,
                               __constant const uint * m_int_diag_mont,
                               const uint num_hashes,
                               const uint r_sq)
{
    const uint gid = get_global_id(0);
    if (gid >= num_hashes) return;

    uint cells[CELLS];
    #pragma unroll
    for (uint i = 0u; i < 16u; ++i) cells[i] = bb_mont_mul(src[gid * 16u + i], r_sq);
    #pragma unroll
    for (uint i = 16u; i < CELLS; ++i) cells[i] = 0u;

    // Inline-copy the perm body (helpers take __global so we re-emit with
    // __constant ptrs here).
    {
        uint blk[6][4]; uint sums[4];
        #pragma unroll
        for (uint i = 0u; i < CELLS / 4u; ++i) {
            blk[i][0]=cells[i*4u+0u]; blk[i][1]=cells[i*4u+1u];
            blk[i][2]=cells[i*4u+2u]; blk[i][3]=cells[i*4u+3u]; m4(blk[i]);
        }
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) {
            uint a = bb_add(blk[0][j], blk[1][j]);
            uint b = bb_add(blk[2][j], blk[3][j]);
            uint c = bb_add(blk[4][j], blk[5][j]);
            sums[j] = bb_add(bb_add(a, b), c);
        }
        #pragma unroll
        for (uint i = 0u; i < CELLS / 4u; ++i) {
            cells[i*4u+0u]=bb_add(blk[i][0],sums[0]); cells[i*4u+1u]=bb_add(blk[i][1],sums[1]);
            cells[i*4u+2u]=bb_add(blk[i][2],sums[2]); cells[i*4u+3u]=bb_add(blk[i][3],sums[3]);
        }
    }
    uint rd = 0u;
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        uint x[CELLS], x2[CELLS], x4[CELLS], x6[CELLS];
        __constant const uint *rc = round_cs_mont + rd * CELLS;
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) x[i] = bb_add(cells[i], rc[i]);
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) x2[i] = bb_mont_mul(x[i], x[i]);
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) x4[i] = bb_mont_mul(x2[i], x2[i]);
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) x6[i] = bb_mont_mul(x4[i], x2[i]);
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) cells[i] = bb_mont_mul(x6[i], x[i]);
        // m_ext inline
        uint blk[6][4]; uint sums[4];
        #pragma unroll
        for (uint i = 0u; i < CELLS / 4u; ++i) {
            blk[i][0]=cells[i*4u+0u]; blk[i][1]=cells[i*4u+1u];
            blk[i][2]=cells[i*4u+2u]; blk[i][3]=cells[i*4u+3u]; m4(blk[i]);
        }
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) {
            uint a=bb_add(blk[0][j],blk[1][j]); uint b=bb_add(blk[2][j],blk[3][j]);
            uint c=bb_add(blk[4][j],blk[5][j]); sums[j]=bb_add(bb_add(a,b),c);
        }
        #pragma unroll
        for (uint i = 0u; i < CELLS / 4u; ++i) {
            cells[i*4u+0u]=bb_add(blk[i][0],sums[0]); cells[i*4u+1u]=bb_add(blk[i][1],sums[1]);
            cells[i*4u+2u]=bb_add(blk[i][2],sums[2]); cells[i*4u+3u]=bb_add(blk[i][3],sums[3]);
        }
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_PARTIAL; ++r) {
        cells[0] = bb_add(cells[0], round_cs_mont[rd * CELLS]);
        cells[0] = bb_sbox_mont(cells[0]);
        // m_int_mont inline w/ __constant diag
        uint s = 0u;
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) s = bb_add(s, cells[i]);
        uint dx[CELLS];
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) dx[i] = bb_mont_mul(m_int_diag_mont[i], cells[i]);
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) cells[i] = bb_add(s, dx[i]);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        uint x[CELLS], x2[CELLS], x4[CELLS], x6[CELLS];
        __constant const uint *rc = round_cs_mont + rd * CELLS;
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) x[i] = bb_add(cells[i], rc[i]);
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) x2[i] = bb_mont_mul(x[i], x[i]);
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) x4[i] = bb_mont_mul(x2[i], x2[i]);
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) x6[i] = bb_mont_mul(x4[i], x2[i]);
        #pragma unroll
        for (uint i = 0u; i < CELLS; ++i) cells[i] = bb_mont_mul(x6[i], x[i]);
        uint blk[6][4]; uint sums[4];
        #pragma unroll
        for (uint i = 0u; i < CELLS / 4u; ++i) {
            blk[i][0]=cells[i*4u+0u]; blk[i][1]=cells[i*4u+1u];
            blk[i][2]=cells[i*4u+2u]; blk[i][3]=cells[i*4u+3u]; m4(blk[i]);
        }
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j) {
            uint a=bb_add(blk[0][j],blk[1][j]); uint b=bb_add(blk[2][j],blk[3][j]);
            uint c=bb_add(blk[4][j],blk[5][j]); sums[j]=bb_add(bb_add(a,b),c);
        }
        #pragma unroll
        for (uint i = 0u; i < CELLS / 4u; ++i) {
            cells[i*4u+0u]=bb_add(blk[i][0],sums[0]); cells[i*4u+1u]=bb_add(blk[i][1],sums[1]);
            cells[i*4u+2u]=bb_add(blk[i][2],sums[2]); cells[i*4u+3u]=bb_add(blk[i][3],sums[3]);
        }
        ++rd;
    }

    #pragma unroll
    for (uint c = 0u; c < 8u; ++c) dst[gid * 8u + c] = bb_mont_mul(cells[c], 1u);
}

/* hash_rows specialized for col_size=24 — the common FRI-commit row width.
 * Two sequential perms: first absorbs 16 cells (rate-full), second absorbs
 * remaining 8 cells, output digest from final state. No loop overhead. */
__kernel
void poseidon2_hash_rows_24(__global uint * restrict dst,
                            __global const uint * restrict src,
                            const uint count,
                            const uint col_size_unused,
                            const uint r_sq,
                            __global const uint * restrict round_cs_mont,
                            __global const uint * restrict m_int_diag_mont)
{
    const uint col = get_global_id(0);
    if (col >= count) return;

    uint cells[CELLS];
    // First absorb: rate-full (16 elements).
    #pragma unroll
    for (uint j = 0u; j < 16u; ++j) cells[j] = bb_mont_mul(src[j * count + col], r_sq);
    #pragma unroll
    for (uint i = 16u; i < CELLS; ++i) cells[i] = 0u;

    // Perm 1.
    multiply_by_m_ext(cells);
    uint rd = 0u;
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        add_rc_sbox_layer_mont(cells, round_cs_mont + rd * CELLS);
        multiply_by_m_ext(cells);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_PARTIAL; ++r) {
        cells[0] = bb_add(cells[0], round_cs_mont[rd * CELLS]);
        cells[0] = bb_sbox_mont(cells[0]);
        multiply_by_m_int_mont(cells, m_int_diag_mont);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        add_rc_sbox_layer_mont(cells, round_cs_mont + rd * CELLS);
        multiply_by_m_ext(cells);
        ++rd;
    }

    // Second absorb: only 8 elements (24 - 16 = 8), into cells[0..7].
    #pragma unroll
    for (uint j = 0u; j < 8u; ++j) {
        const uint v = src[(16u + j) * count + col];
        cells[j] = bb_add(cells[j], bb_mont_mul(v, r_sq));
    }
    // Cells[8..15] keep their post-perm-1 Mont values (no input added).

    // Perm 2.
    multiply_by_m_ext(cells);
    rd = 0u;
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        add_rc_sbox_layer_mont(cells, round_cs_mont + rd * CELLS);
        multiply_by_m_ext(cells);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_PARTIAL; ++r) {
        cells[0] = bb_add(cells[0], round_cs_mont[rd * CELLS]);
        cells[0] = bb_sbox_mont(cells[0]);
        multiply_by_m_int_mont(cells, m_int_diag_mont);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        add_rc_sbox_layer_mont(cells, round_cs_mont + rd * CELLS);
        multiply_by_m_ext(cells);
        ++rd;
    }

    #pragma unroll
    for (uint c = 0u; c < 8u; ++c) dst[col * 8u + c] = bb_mont_mul(cells[c], 1u);
}

/* hash_fold_vload removed — the patched IGC build is missing the SPIRV OCL
 * vloadn/vstoren builtin tables. The baseline kernel already coalesces
 * scalar loads optimally per IGC ISA dump, so vload16 was redundant anyway. */

/* Scalar-per-cell sbox layer — minimizes register pressure (4 scratch
 * rather than 4*CELLS) at the cost of cell-level ILP. Pair with x2 ILP
 * to recover the lost parallelism via two-state interleaving. */
inline void add_rc_sbox_cell_scalar(uint *cells, __global const uint * restrict rc) {
    #pragma unroll
    for (uint i = 0u; i < CELLS; ++i) {
        uint x  = bb_add(cells[i], rc[i]);
        uint x2 = bb_mont_mul(x, x);
        uint x4 = bb_mont_mul(x2, x2);
        uint x6 = bb_mont_mul(x4, x2);
        cells[i] = bb_mont_mul(x6, x);
    }
}

__kernel
void poseidon2_hash_fold_x2_lp(__global uint * restrict dst,
                               __global const uint * restrict src,
                               __global const uint * restrict round_cs_mont,
                               __global const uint * restrict m_int_diag_mont,
                               const uint num_hashes,
                               const uint r_sq)
{
    const uint gid = get_global_id(0);
    const uint base = gid * 2u;
    if (base + 1u >= num_hashes) {
        if (base >= num_hashes) return;
        uint cells[CELLS];
        #pragma unroll
        for (uint i = 0u; i < 16u; ++i) cells[i] = bb_mont_mul(src[base * 16u + i], r_sq);
        #pragma unroll
        for (uint i = 16u; i < CELLS; ++i) cells[i] = 0u;
        multiply_by_m_ext(cells);
        uint rd = 0u;
        #pragma unroll
        for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
            add_rc_sbox_cell_scalar(cells, round_cs_mont + rd * CELLS);
            multiply_by_m_ext(cells); ++rd;
        }
        #pragma unroll
        for (uint r = 0u; r < ROUNDS_PARTIAL; ++r) {
            cells[0] = bb_add(cells[0], round_cs_mont[rd * CELLS]);
            cells[0] = bb_sbox_mont(cells[0]);
            multiply_by_m_int_mont(cells, m_int_diag_mont); ++rd;
        }
        #pragma unroll
        for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
            add_rc_sbox_cell_scalar(cells, round_cs_mont + rd * CELLS);
            multiply_by_m_ext(cells); ++rd;
        }
        #pragma unroll
        for (uint c = 0u; c < 8u; ++c) dst[base * 8u + c] = bb_mont_mul(cells[c], 1u);
        return;
    }

    uint a[CELLS], b[CELLS];
    #pragma unroll
    for (uint i = 0u; i < 16u; ++i) {
        a[i] = bb_mont_mul(src[(base+0u)*16u + i], r_sq);
        b[i] = bb_mont_mul(src[(base+1u)*16u + i], r_sq);
    }
    #pragma unroll
    for (uint i = 16u; i < CELLS; ++i) { a[i] = 0u; b[i] = 0u; }

    multiply_by_m_ext(a); multiply_by_m_ext(b);
    uint rd = 0u;
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        add_rc_sbox_cell_scalar(a, round_cs_mont + rd * CELLS);
        add_rc_sbox_cell_scalar(b, round_cs_mont + rd * CELLS);
        multiply_by_m_ext(a); multiply_by_m_ext(b); ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_PARTIAL; ++r) {
        a[0] = bb_add(a[0], round_cs_mont[rd * CELLS]);
        b[0] = bb_add(b[0], round_cs_mont[rd * CELLS]);
        a[0] = bb_sbox_mont(a[0]);
        b[0] = bb_sbox_mont(b[0]);
        multiply_by_m_int_mont(a, m_int_diag_mont);
        multiply_by_m_int_mont(b, m_int_diag_mont);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        add_rc_sbox_cell_scalar(a, round_cs_mont + rd * CELLS);
        add_rc_sbox_cell_scalar(b, round_cs_mont + rd * CELLS);
        multiply_by_m_ext(a); multiply_by_m_ext(b); ++rd;
    }

    #pragma unroll
    for (uint c = 0u; c < 8u; ++c) {
        dst[(base+0u)*8u + c] = bb_mont_mul(a[c], 1u);
        dst[(base+1u)*8u + c] = bb_mont_mul(b[c], 1u);
    }
}

/* SIMD32 forced — tests whether IGC's mul.pair fusion (mulpair-fusion patches
 * in /home/user/inteldebug/igc-bug-report/mulpair-fusion/) lifts throughput
 * vs the default SIMD16. */
__kernel
__attribute__((intel_reqd_sub_group_size(32)))
void poseidon2_hash_fold_simd32(__global uint * restrict dst,
                                __global const uint * restrict src,
                                __global const uint * restrict round_cs_mont,
                                __global const uint * restrict m_int_diag_mont,
                                const uint num_hashes,
                                const uint r_sq)
{
    const uint gid = get_global_id(0);
    if (gid >= num_hashes) return;

    uint cells[CELLS];
    #pragma unroll
    for (uint i = 0u; i < 16u; ++i) cells[i] = bb_mont_mul(src[gid * 16u + i], r_sq);
    #pragma unroll
    for (uint i = 16u; i < CELLS; ++i) cells[i] = 0u;

    multiply_by_m_ext(cells);
    uint rd = 0u;
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        add_rc_sbox_layer_mont(cells, round_cs_mont + rd * CELLS);
        multiply_by_m_ext(cells);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_PARTIAL; ++r) {
        cells[0] = bb_add(cells[0], round_cs_mont[rd * CELLS]);
        cells[0] = bb_sbox_mont(cells[0]);
        multiply_by_m_int_mont(cells, m_int_diag_mont);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        add_rc_sbox_layer_mont(cells, round_cs_mont + rd * CELLS);
        multiply_by_m_ext(cells);
        ++rd;
    }

    #pragma unroll
    for (uint c = 0u; c < 8u; ++c) dst[gid * 8u + c] = bb_mont_mul(cells[c], 1u);
}

/* ============================================================================
 * poseidon2_hash_fold_mont_io — same as poseidon2_hash_fold but skips the
 * to_mont(input) and from_mont(output) conversions. Matches the ESIMD
 * convention where input/output buffers are in Mont form throughout the
 * prover pipeline. r_sq is unused (kept in signature for arg-position
 * compat with the host-side set_arg sequence).
 * ============================================================================ */
__kernel
void poseidon2_hash_fold_mont_io(__global uint * restrict dst,
                                 __global const uint * restrict src,
                                 __global const uint * restrict round_cs_mont,
                                 __global const uint * restrict m_int_diag_mont,
                                 const uint num_hashes,
                                 const uint r_sq_unused)
{
    const uint gid = get_global_id(0);
    if (gid >= num_hashes) return;

    uint cells[CELLS];
    #pragma unroll
    for (uint i = 0u; i < 16u; ++i) cells[i] = src[gid * 16u + i];
    #pragma unroll
    for (uint i = 16u; i < CELLS; ++i) cells[i] = 0u;

    multiply_by_m_ext(cells);
    uint rd = 0u;
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        add_rc_sbox_layer_mont(cells, round_cs_mont + rd * CELLS);
        multiply_by_m_ext(cells);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_PARTIAL; ++r) {
        cells[0] = bb_add(cells[0], round_cs_mont[rd * CELLS]);
        cells[0] = bb_sbox_mont(cells[0]);
        multiply_by_m_int_mont(cells, m_int_diag_mont);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        add_rc_sbox_layer_mont(cells, round_cs_mont + rd * CELLS);
        multiply_by_m_ext(cells);
        ++rd;
    }

    #pragma unroll
    for (uint c = 0u; c < 8u; ++c) dst[gid * 8u + c] = cells[c];
}

/* ============================================================================
 * poseidon2_hash_rows_24_mont_io — same as poseidon2_hash_rows_24 but with
 * Mont-form I/O (no to_mont/from_mont). r_sq is unused.
 * ============================================================================ */
__kernel
void poseidon2_hash_rows_24_mont_io(__global uint * restrict dst,
                                    __global const uint * restrict src,
                                    const uint count,
                                    const uint col_size_unused,
                                    const uint r_sq_unused,
                                    __global const uint * restrict round_cs_mont,
                                    __global const uint * restrict m_int_diag_mont)
{
    const uint col = get_global_id(0);
    if (col >= count) return;

    uint cells[CELLS];
    #pragma unroll
    for (uint j = 0u; j < 16u; ++j) cells[j] = src[j * count + col];
    #pragma unroll
    for (uint i = 16u; i < CELLS; ++i) cells[i] = 0u;

    // Perm 1.
    multiply_by_m_ext(cells);
    uint rd = 0u;
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        add_rc_sbox_layer_mont(cells, round_cs_mont + rd * CELLS);
        multiply_by_m_ext(cells);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_PARTIAL; ++r) {
        cells[0] = bb_add(cells[0], round_cs_mont[rd * CELLS]);
        cells[0] = bb_sbox_mont(cells[0]);
        multiply_by_m_int_mont(cells, m_int_diag_mont);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        add_rc_sbox_layer_mont(cells, round_cs_mont + rd * CELLS);
        multiply_by_m_ext(cells);
        ++rd;
    }

    // Second absorb: ESIMD sponge REPLACES cells[0..7] with input and
    // zero-pads cells[8..15]. cells[16..23] (capacity) carries from perm 1.
    #pragma unroll
    for (uint j = 0u; j < 8u; ++j) cells[j] = src[(16u + j) * count + col];
    #pragma unroll
    for (uint j = 8u; j < 16u; ++j) cells[j] = 0u;

    // Perm 2.
    multiply_by_m_ext(cells);
    rd = 0u;
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        add_rc_sbox_layer_mont(cells, round_cs_mont + rd * CELLS);
        multiply_by_m_ext(cells);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_PARTIAL; ++r) {
        cells[0] = bb_add(cells[0], round_cs_mont[rd * CELLS]);
        cells[0] = bb_sbox_mont(cells[0]);
        multiply_by_m_int_mont(cells, m_int_diag_mont);
        ++rd;
    }
    #pragma unroll
    for (uint r = 0u; r < ROUNDS_HALF_FULL; ++r) {
        add_rc_sbox_layer_mont(cells, round_cs_mont + rd * CELLS);
        multiply_by_m_ext(cells);
        ++rd;
    }

    #pragma unroll
    for (uint c = 0u; c < 8u; ++c) dst[col * 8u + c] = cells[c];
}
)OPENCL";
