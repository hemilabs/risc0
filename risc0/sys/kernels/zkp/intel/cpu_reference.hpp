#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

// CPU reference implementation of BabyBear field arithmetic.
// Uses the risc0 fp.h Montgomery convention for simplicity.
// All values stored in Montgomery form: a_mont = a * R mod P where R = 2^32.

namespace bb31_cpu {

static constexpr uint32_t P = 0x78000001; // 15 * 2^27 + 1 = 2013265921
static constexpr uint32_t M = 0x88000001; // Montgomery constant (risc0 convention)
static constexpr uint32_t R2 = 1172168163; // R^2 mod P (for encoding)

// --- Core arithmetic (matching risc0 fp.h exactly) ---

static inline uint32_t add(uint32_t a, uint32_t b) {
    uint32_t r = a + b;
    return (r >= P ? r - P : r);
}

static inline uint32_t sub(uint32_t a, uint32_t b) {
    uint32_t r = a - b;
    return (r > P ? r + P : r);
}

static inline uint32_t mul(uint32_t a, uint32_t b) {
    uint64_t o64 = uint64_t(a) * uint64_t(b);
    uint32_t low = -uint32_t(o64);
    uint32_t red = M * low;
    o64 += uint64_t(red) * uint64_t(P);
    uint32_t ret = uint32_t(o64 >> 32);
    return (ret >= P ? ret - P : ret);
}

static inline uint32_t encode(uint32_t a) { return mul(R2, a); }
static inline uint32_t decode(uint32_t a) { return mul(1, a); }

// --- sppark convention constants (for ESIMD kernel validation) ---
// sppark uses M0 = 0x77ffffff = -P^{-1} mod 2^32 with a different reduction formula.
// Both produce identical results. The ESIMD kernel uses sppark convention; this CPU
// reference uses risc0 convention. Results are bit-identical in Montgomery form.

static constexpr uint32_t SPPARK_M0 = 0x77ffffff;
static constexpr uint32_t SPPARK_RR = 0x45dddde3;
static constexpr uint32_t SPPARK_ONE = 0x0ffffffe;

// Verify conventions agree (call once at startup)
static inline bool verify_conventions() {
    // encode(1) should equal SPPARK_ONE
    uint32_t one_mont = encode(1);
    if (one_mont != SPPARK_ONE) return false;

    // encode(encode(0)) should be 0
    if (encode(0) != 0) return false;

    // mul(encode(a), encode(b)) == encode(a*b mod P) for small values
    for (uint32_t a = 0; a < 100; a++) {
        for (uint32_t b = 0; b < 100; b++) {
            uint32_t expected = encode(uint32_t((uint64_t(a) * b) % P));
            uint32_t got = mul(encode(a), encode(b));
            if (got != expected) return false;
        }
    }

    // (P-1)^2 mod P == 1
    uint32_t pm1 = encode(P - 1);
    uint32_t pm1_sq = mul(pm1, pm1);
    if (pm1_sq != encode(1)) return false;

    return true;
}

// --- Simple radix-2 NTT for validation ---

// Roots of unity from risc0 rou.h (in normal/decoded form).
// These get encoded to Montgomery form when used.
static constexpr uint32_t ROU_FWD[] = {
    1, 2013265920, 284861408, 1801542727, 567209306, 740045640,
    918899846, 1881002012, 1453957774, 65325759, 1538055801, 515192888,
    483885487, 157393079, 1695124103, 2005211659, 1540072241, 88064245,
    1542985445, 1269900459, 1461624142, 825701067, 682402162, 1311873874,
    1164520853, 352275361, 18769, 137
};

static constexpr uint32_t ROU_REV[] = {
    1, 2013265920, 1728404513, 1592366214, 196396260, 1253260071,
    72041623, 1091445674, 145223211, 1446820157, 1030796471, 2010749425,
    1827366325, 1239938613, 246299276, 596347512, 1893145354, 246074437,
    1525739923, 1194341128, 1463599021, 704606912, 95395244, 15672543,
    647517488, 584175179, 137728885, 749463956
};

// In-place bit-reversal permutation
static inline void bit_reverse(uint32_t* data, uint32_t n) {
    uint32_t logn = 0;
    for (uint32_t t = n; t > 1; t >>= 1) logn++;

    for (uint32_t i = 0; i < n; i++) {
        uint32_t rev = 0;
        for (uint32_t j = 0; j < logn; j++) {
            rev |= ((i >> j) & 1) << (logn - 1 - j);
        }
        if (i < rev) {
            uint32_t tmp = data[i];
            data[i] = data[rev];
            data[rev] = tmp;
        }
    }
}

// Forward NTT (radix-2 Cooley-Tukey, decimation-in-time)
// Input/output in Montgomery form. Result is in bit-reversed order (NR ordering).
static inline void forward_ntt(uint32_t* data, uint32_t n) {
    uint32_t logn = 0;
    for (uint32_t t = n; t > 1; t >>= 1) logn++;

    for (uint32_t s = 1; s <= logn; s++) {
        uint32_t m = 1u << s;
        uint32_t half = m >> 1;
        uint32_t w = encode(ROU_FWD[s]); // root of unity for this stage

        for (uint32_t k = 0; k < n; k += m) {
            uint32_t wj = encode(1); // w^0 = 1
            for (uint32_t j = 0; j < half; j++) {
                uint32_t t_val = mul(wj, data[k + j + half]);
                uint32_t u = data[k + j];
                data[k + j] = add(u, t_val);
                data[k + j + half] = sub(u, t_val);
                wj = mul(wj, w);
            }
        }
    }
}

// Inverse NTT (radix-2 Gentleman-Sande, decimation-in-frequency)
// Includes normalization by 1/n.
static inline void inverse_ntt(uint32_t* data, uint32_t n) {
    uint32_t logn = 0;
    for (uint32_t t = n; t > 1; t >>= 1) logn++;

    for (uint32_t s = logn; s >= 1; s--) {
        uint32_t m = 1u << s;
        uint32_t half = m >> 1;
        uint32_t w = encode(ROU_REV[s]);

        for (uint32_t k = 0; k < n; k += m) {
            uint32_t wj = encode(1);
            for (uint32_t j = 0; j < half; j++) {
                uint32_t u = data[k + j];
                uint32_t v = data[k + j + half];
                data[k + j] = add(u, v);
                data[k + j + half] = mul(sub(u, v), wj);
                wj = mul(wj, w);
            }
        }
    }

    // Normalize by 1/n: compute n^{-1} mod P via Fermat's little theorem
    // n^{-1} = n^{P-2} mod P, but for powers of 2, we can just multiply by
    // the precomputed inverse. For simplicity, use repeated halving.
    uint32_t n_inv = encode(1);
    uint32_t half_mod = encode((P + 1) / 2); // (P+1)/2 mod P = inverse of 2
    for (uint32_t i = 0; i < logn; i++) {
        n_inv = mul(n_inv, half_mod);
    }
    for (uint32_t i = 0; i < n; i++) {
        data[i] = mul(data[i], n_inv);
    }
}

// Generate deterministic test data (seeded PRNG, Montgomery-encoded)
static inline void generate_test_data(uint32_t* out, uint32_t count, uint64_t seed) {
    uint64_t state = seed;
    for (uint32_t i = 0; i < count; i++) {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        uint32_t val = uint32_t(state) % P;
        out[i] = encode(val);
    }
}

// ============================================================================
// FpExt (BabyBear^4) CPU Reference
// ============================================================================
// Extension field: Fp[X] / (X^4 - 11)
// NBETA = P - encode(11) in Montgomery form. We compute it at init time.

struct FpExt {
    uint32_t elems[4]; // [c0, c1, c2, c3] in Montgomery form

    FpExt() : elems{0, 0, 0, 0} {}
    FpExt(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3) : elems{c0, c1, c2, c3} {}

    // Construct from scalar (Fp -> FpExt: [val, 0, 0, 0])
    static FpExt from_fp(uint32_t val) { return FpExt(val, 0, 0, 0); }

    bool operator==(const FpExt& o) const {
        return elems[0] == o.elems[0] && elems[1] == o.elems[1] &&
               elems[2] == o.elems[2] && elems[3] == o.elems[3];
    }
    bool operator!=(const FpExt& o) const { return !(*this == o); }
};

// NBETA = Montgomery form of (P - 11) = -11 mod P
// Using RISC Zero convention: NBETA = 0x40000018
static constexpr uint32_t FPEXT_NBETA = 0x40000018;
static constexpr uint32_t FPEXT_BETA  = 0x37ffffe9;

static inline FpExt fpext_add(const FpExt& a, const FpExt& b) {
    return FpExt(add(a.elems[0], b.elems[0]),
                 add(a.elems[1], b.elems[1]),
                 add(a.elems[2], b.elems[2]),
                 add(a.elems[3], b.elems[3]));
}

static inline FpExt fpext_sub(const FpExt& a, const FpExt& b) {
    return FpExt(sub(a.elems[0], b.elems[0]),
                 sub(a.elems[1], b.elems[1]),
                 sub(a.elems[2], b.elems[2]),
                 sub(a.elems[3], b.elems[3]));
}

// FpExt * Fp (scalar multiply)
static inline FpExt fpext_mul_fp(const FpExt& a, uint32_t b) {
    return FpExt(mul(a.elems[0], b), mul(a.elems[1], b),
                 mul(a.elems[2], b), mul(a.elems[3], b));
}

// FpExt * FpExt (full extension field multiply)
// ret[0] = a0*b0 + NBETA*(a1*b3 + a2*b2 + a3*b1)
// ret[1] = a0*b1 + a1*b0 + NBETA*(a2*b3 + a3*b2)
// ret[2] = a0*b2 + a1*b1 + a2*b0 + NBETA*(a3*b3)
// ret[3] = a0*b3 + a1*b2 + a2*b1 + a3*b0
static inline FpExt fpext_mul(const FpExt& a, const FpExt& b) {
    const uint32_t nb = FPEXT_NBETA;
    return FpExt(
        add(mul(a.elems[0], b.elems[0]),
            mul(nb, add(add(mul(a.elems[1], b.elems[3]),
                            mul(a.elems[2], b.elems[2])),
                        mul(a.elems[3], b.elems[1])))),
        add(add(mul(a.elems[0], b.elems[1]),
                mul(a.elems[1], b.elems[0])),
            mul(nb, add(mul(a.elems[2], b.elems[3]),
                        mul(a.elems[3], b.elems[2])))),
        add(add(add(mul(a.elems[0], b.elems[2]),
                    mul(a.elems[1], b.elems[1])),
                mul(a.elems[2], b.elems[0])),
            mul(nb, mul(a.elems[3], b.elems[3]))),
        add(add(add(mul(a.elems[0], b.elems[3]),
                    mul(a.elems[1], b.elems[2])),
                mul(a.elems[2], b.elems[1])),
            mul(a.elems[3], b.elems[0]))
    );
}

// FpExt reciprocal: brute-force via repeated squaring in the extension field.
// Computes val^(P^4 - 2) using the fact that |FpExt*| = P^4 - 1.
// This is slow but guaranteed correct as a reference implementation.
static inline FpExt fpext_reciprocal(const FpExt& val) {
    // a^{-1} = a^{P^4 - 2} in FpExt
    // But P^4 is huge. Instead, use the formula:
    // a^{-1} = a^{(P-1)} * a^{(P-1)*P} * a^{(P-1)*P^2} * a^{(P-1)*P^3} / a^{P^4-1}
    // This is too complex. Simpler: use the Frobenius endomorphism approach.
    //
    // Actually the simplest approach: compute norm to Fp, invert in Fp, scale back.
    // Norm(a) = a * a^P * a^{P^2} * a^{P^3} is in Fp.
    // For X^4 - beta, Frobenius acts as: a^P = [a0, a1*g, a2*g^2, a3*g^3]
    // where g = beta^{(P-1)/4}.
    //
    // Even simpler: just solve a*x = 1 by Gaussian elimination in Fp.
    // FpExt multiplication is a 4x4 system over Fp. Given a = [a0,a1,a2,a3],
    // find x = [x0,x1,x2,x3] such that a*x = [1,0,0,0].
    //
    // This gives a 4x4 linear system:
    //   a0*x0 + nb*(a1*x3 + a2*x2 + a3*x1) = 1
    //   a0*x1 + a1*x0 + nb*(a2*x3 + a3*x2) = 0
    //   a0*x2 + a1*x1 + a2*x0 + nb*a3*x3   = 0
    //   a0*x3 + a1*x2 + a2*x1 + a3*x0       = 0
    //
    // We solve this by building the matrix and using Gaussian elimination.
    uint32_t nb = FPEXT_NBETA;

    // Build the 4x5 augmented matrix (over Fp, in Montgomery form)
    // Row i represents: sum_j M[i][j] * x[j] = rhs[i]
    uint32_t M[4][5]; // 4 rows, 5 columns (4 + augmented)

    // Row 0: a0*x0 + nb*a3*x1 + nb*a2*x2 + nb*a1*x3 = ONE
    M[0][0] = val.elems[0]; M[0][1] = mul(nb, val.elems[3]); M[0][2] = mul(nb, val.elems[2]); M[0][3] = mul(nb, val.elems[1]); M[0][4] = SPPARK_ONE;
    // Row 1: a1*x0 + a0*x1 + nb*a3*x2 + nb*a2*x3 = 0
    M[1][0] = val.elems[1]; M[1][1] = val.elems[0]; M[1][2] = mul(nb, val.elems[3]); M[1][3] = mul(nb, val.elems[2]); M[1][4] = 0;
    // Row 2: a2*x0 + a1*x1 + a0*x2 + nb*a3*x3 = 0
    M[2][0] = val.elems[2]; M[2][1] = val.elems[1]; M[2][2] = val.elems[0]; M[2][3] = mul(nb, val.elems[3]); M[2][4] = 0;
    // Row 3: a3*x0 + a2*x1 + a1*x2 + a0*x3 = 0
    M[3][0] = val.elems[3]; M[3][1] = val.elems[2]; M[3][2] = val.elems[1]; M[3][3] = val.elems[0]; M[3][4] = 0;

    // Gaussian elimination (forward)
    for (int col = 0; col < 4; col++) {
        // Find pivot
        int pivot = -1;
        for (int row = col; row < 4; row++) {
            if (M[row][col] != 0) { pivot = row; break; }
        }
        if (pivot < 0) return FpExt(); // singular

        // Swap rows
        if (pivot != col) {
            for (int j = 0; j < 5; j++) {
                uint32_t tmp = M[col][j]; M[col][j] = M[pivot][j]; M[pivot][j] = tmp;
            }
        }

        // Compute pivot inverse
        uint32_t piv_inv;
        {
            uint32_t base = M[col][col];
            uint32_t result = encode(1);
            uint32_t exp = P - 2;
            while (exp > 0) {
                if (exp & 1) result = mul(result, base);
                base = mul(base, base);
                exp >>= 1;
            }
            piv_inv = result;
        }

        // Scale pivot row
        for (int j = col; j < 5; j++) {
            M[col][j] = mul(M[col][j], piv_inv);
        }

        // Eliminate column in other rows
        for (int row = 0; row < 4; row++) {
            if (row == col) continue;
            uint32_t factor = M[row][col];
            if (factor == 0) continue;
            for (int j = col; j < 5; j++) {
                M[row][j] = sub(M[row][j], mul(factor, M[col][j]));
            }
        }
    }

    // Solution is in the augmented column
    return FpExt(M[0][4], M[1][4], M[2][4], M[3][4]);
}

// Verify FpExt: a * a^{-1} should equal [ONE, 0, 0, 0]
static inline bool verify_fpext() {
    // Test a few random FpExt values
    uint64_t state = 54321;
    for (int t = 0; t < 20; t++) {
        FpExt a;
        for (int i = 0; i < 4; i++) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            a.elems[i] = encode(uint32_t(state) % P);
        }
        // Skip zero
        if (a.elems[0] == 0 && a.elems[1] == 0 && a.elems[2] == 0 && a.elems[3] == 0)
            continue;

        FpExt inv = fpext_reciprocal(a);
        FpExt product = fpext_mul(a, inv);

        // Should be [ONE, 0, 0, 0]
        if (product.elems[0] != SPPARK_ONE || product.elems[1] != 0 ||
            product.elems[2] != 0 || product.elems[3] != 0) {
            fprintf(stderr, "FpExt verify: a*inv(a) != [ONE,0,0,0] at test %d\n", t);
            fprintf(stderr, "  a    = [%08x, %08x, %08x, %08x]\n", a.elems[0], a.elems[1], a.elems[2], a.elems[3]);
            fprintf(stderr, "  inv  = [%08x, %08x, %08x, %08x]\n", inv.elems[0], inv.elems[1], inv.elems[2], inv.elems[3]);
            fprintf(stderr, "  prod = [%08x, %08x, %08x, %08x]\n", product.elems[0], product.elems[1], product.elems[2], product.elems[3]);
            fprintf(stderr, "  ONE  = %08x\n", SPPARK_ONE);
            fflush(stderr);
            return false;
        }
    }

    // Test FpExt multiply commutativity
    state = 99999;
    for (int t = 0; t < 20; t++) {
        FpExt a, b;
        for (int i = 0; i < 4; i++) {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            a.elems[i] = encode(uint32_t(state) % P);
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            b.elems[i] = encode(uint32_t(state) % P);
        }
        FpExt ab = fpext_mul(a, b);
        FpExt ba = fpext_mul(b, a);
        if (ab != ba) return false;
    }

    return true;
}

// ============================================================================
// sppark-style Multi-Pass NTT CPU Reference
// Implements the exact algorithm from sppark's ct_mixed_radix_narrow.cuh
// and gs_mixed_radix_narrow.cuh for validation of the ESIMD port.
// ============================================================================

static constexpr uint32_t MAX_LG_DOMAIN = 27;
static constexpr uint32_t LG_WINDOW_SIZE = 6;  // (27+4)/5
static constexpr uint32_t WINDOW_SIZE = 1u << LG_WINDOW_SIZE; // 64
static constexpr uint32_t WINDOW_NUM = 5; // ceil(27/6)

// Host-side modular exponentiation
static inline uint32_t mont_pow(uint32_t base, uint64_t exp) {
    uint32_t result = encode(1);
    while (exp > 0) {
        if (exp & 1) result = mul(result, base);
        base = mul(base, base);
        exp >>= 1;
    }
    return result;
}

// Bit-reverse val in nbits bits
static inline uint32_t bit_rev_n(uint32_t val, uint32_t nbits) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < nbits; i++) {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    return result;
}

// Windowed partial twiddle table: roots[w][j] = omega^(j << (w * LG_WINDOW_SIZE))
struct SparkTwiddles {
    uint32_t partial[WINDOW_NUM][WINDOW_SIZE]; // windowed omega powers
    uint32_t radix[512]; // radix twiddle table (root10^k for k=0..511)

    void generate(bool forward) {
        // omega = primitive 2^MAX_LG_DOMAIN root of unity
        uint32_t omega = encode(forward ? ROU_FWD[MAX_LG_DOMAIN] : ROU_REV[MAX_LG_DOMAIN]);

        // Partial twiddles: roots[w][j] = omega^(j << (w * LG_WINDOW_SIZE))
        for (uint32_t j = 0; j < WINDOW_SIZE; j++) {
            uint32_t r = mont_pow(omega, j);
            partial[0][j] = r;
            for (uint32_t w = 1; w < WINDOW_NUM; w++) {
                for (uint32_t b = 0; b < LG_WINDOW_SIZE; b++)
                    r = mul(r, r); // square LG_WINDOW_SIZE times = raise to 2^LG_WINDOW_SIZE
                partial[w][j] = r;
            }
        }

        // Radix twiddles: root10^k for k=0..511
        // root10 = primitive 2^10 = 1024th root of unity
        uint32_t root10 = encode(forward ? ROU_FWD[10] : ROU_REV[10]);
        for (uint32_t k = 0; k < 512; k++) {
            radix[k] = mont_pow(root10, k);
        }
    }

    // Reconstruct omega^pow from windowed table
    uint32_t get_root(uint64_t pow) const {
        int win = (WINDOW_NUM - 1) * LG_WINDOW_SIZE;
        int off = WINDOW_NUM - 1;
        uint32_t result = partial[off][(uint32_t)(pow >> win) % WINDOW_SIZE];
        while (off-- > 0) {
            win -= LG_WINDOW_SIZE;
            uint32_t widx = (uint32_t)((pow >> win) % WINDOW_SIZE);
            if (widx != 0)
                result = mul(result, partial[off][widx]);
        }
        return result;
    }

    // Get radix twiddle for butterfly stage s within a pass
    // radix = max(iterations, 6)
    uint32_t get_radix_tw(uint32_t rank, uint32_t s, uint32_t radix_val) const {
        // For stages 1..5: use radix-6 subtable (entries at stride 2^(10-6) = 16)
        // For stages 6+: use radix-N subtable
        // Generic: radix[rank << (10 - (s+1))] where we use the full radix-10 table
        // and adjust the stride for the effective radix
        uint32_t idx = rank << (radix_val - (s + 1));
        // Map from radix_val to radix-10 table by scaling
        idx <<= (10 - radix_val);
        return radix[idx % 512];
    }
};

// One pass of CT DIT NTT (sppark-style)
// Processes `iterations` butterfly stages starting at `stage`.
// data: array of n = 2^lg_domain_size elements in Montgomery form.
// Modifies data in-place with bit-rotated output indices.
static inline void sppark_ct_pass(uint32_t* data, uint32_t lg_domain_size,
                                   uint32_t stage, uint32_t iterations,
                                   const SparkTwiddles& tw) {
    uint32_t n = 1u << lg_domain_size;
    uint32_t num_threads = n / 2;
    uint32_t radix = iterations < 6 ? 6 : iterations;

    uint32_t diff_mask = (1u << (iterations - 1)) - 1;
    uint32_t inp_mask = (1u << stage) - 1;
    uint32_t out_mask = (1u << (stage + iterations - 1)) - 1;

    // Build index tables: for each "thread" tid, compute idx0 and idx1
    std::vector<uint32_t> idx0_tab(num_threads), idx1_tab(num_threads);
    for (uint32_t tid = 0; tid < num_threads; tid++) {
        uint32_t tiz = tid;
        uint32_t thread_ntt_pos = (tiz >> (iterations - 1)) & inp_mask;
        uint32_t idx0 = (tiz & ~out_mask) | ((tiz << stage) & out_mask);
        idx0 = idx0 * 2 + thread_ntt_pos;
        idx0_tab[tid] = idx0;
        idx1_tab[tid] = idx0 + (1u << stage);
    }

    // Working buffer: initially loaded from data at computed indices
    // r0[tid] = data[idx0[tid]], r1[tid] = data[idx1[tid]]
    std::vector<uint32_t> r0(num_threads), r1(num_threads);
    for (uint32_t tid = 0; tid < num_threads; tid++) {
        r0[tid] = data[idx0_tab[tid]];
        r1[tid] = data[idx1_tab[tid]];
    }

    // === Inter-pass twiddle (if stage != 0) ===
    if (stage != 0) {
        for (uint32_t tid = 0; tid < num_threads; tid++) {
            uint32_t tiz = tid;
            uint32_t thread_ntt_pos = (tiz >> (iterations - 1)) & inp_mask;
            uint32_t thread_ntt_idx = (tiz & diff_mask) * 2;
            uint32_t nbits = MAX_LG_DOMAIN - stage;
            uint32_t br_idx = bit_rev_n(thread_ntt_idx, nbits);
            uint64_t ri0 = (uint64_t)br_idx * thread_ntt_pos;
            uint64_t ri1 = ri0 + ((uint64_t)thread_ntt_pos << (nbits - 1));

            r0[tid] = mul(r0[tid], tw.get_root(ri0));
            r1[tid] = mul(r1[tid], tw.get_root(ri1));
        }
    }

    // === Butterfly stages 0..iterations-1 ===
    // Stage 0: no twiddle, no exchange. Simple add/sub per thread.
    for (uint32_t tid = 0; tid < num_threads; tid++) {
        uint32_t t = r1[tid];
        r1[tid] = sub(r0[tid], t);
        r0[tid] = add(r0[tid], t);
    }

    // Stages 1..iterations-1: exchange via XOR on thread index, then butterfly
    for (uint32_t s = 1; s < iterations; s++) {
        uint32_t laneMask = 1u << (s - 1);
        uint32_t thrdMask = (1u << s) - 1;

        // Snapshot current state for exchange reads
        std::vector<uint32_t> r0_snap(r0), r1_snap(r1);

        for (uint32_t tid = 0; tid < num_threads; tid++) {
            // Thread-local position within its block
            // In sppark, threadIdx.x is the local thread within a work-group.
            // The block_size = 2^(radix-1). XOR operates on the local index.
            uint32_t block_size = 1u << (radix - 1);
            uint32_t local_tid = tid % block_size;
            uint32_t block_base = tid - local_tid;

            uint32_t rank = local_tid & thrdMask;
            bool pos = rank < laneMask;

            // Partner is at local_tid ^ laneMask within the same block
            uint32_t partner_local = local_tid ^ laneMask;
            uint32_t partner_tid = block_base + partner_local;

            // sppark csel+exchange:
            // Each thread puts csel(r1, r0, pos) into exchange.
            // "pos" is per-thread: pos=true means this thread is in the "top" half.
            // The partner thread has partner_pos = !pos (they're in opposite halves).
            bool partner_pos = (partner_local & thrdMask) < laneMask;

            // What each thread puts into exchange (their csel result)
            // csel(r1, r0, pos) means: return r1 if pos, else r0
            // uint32_t my_xchg = pos ? r1_snap[tid] : r0_snap[tid];
            uint32_t partner_xchg = partner_pos ? r1_snap[partner_tid] : r0_snap[partner_tid];

            // After exchange: top thread keeps its r0, gets partner's xchg as r1
            // Bottom thread gets partner's xchg as r0, keeps its r1
            uint32_t new_r0 = pos ? r0_snap[tid] : partner_xchg;
            uint32_t new_r1 = pos ? partner_xchg : r1_snap[tid];

            // Twiddle
            uint32_t root = tw.get_radix_tw(rank, s, radix);

            // CT butterfly: t = root * r1; new_r1 = r0 - t; new_r0 = r0 + t
            uint32_t tw_r1 = mul(root, new_r1);
            r1[tid] = sub(new_r0, tw_r1);
            r0[tid] = add(new_r0, tw_r1);
        }
    }

    // === Bit-rotated output ===
    for (uint32_t tid = 0; tid < num_threads; tid++) {
        uint32_t idx0 = idx0_tab[tid];
        uint32_t idx1 = idx1_tab[tid];

        // Right-rotate iterations bits at position stage
        uint32_t mask = ((1u << iterations) - 1) << stage;
        uint32_t rot0 = idx0 & mask;
        rot0 = ((rot0 >> 1) | (rot0 << (iterations - 1))) & mask;
        uint32_t out_idx0 = (idx0 & ~mask) | rot0;

        uint32_t rot1 = idx1 & mask;
        rot1 = ((rot1 >> 1) | (rot1 << (iterations - 1))) & mask;
        uint32_t out_idx1 = (idx1 & ~mask) | rot1;

        data[out_idx0] = r0[tid];
        data[out_idx1] = r1[tid];
    }
}

// Multi-pass CT NTT (sppark decomposition)
static inline void sppark_forward_ntt(uint32_t* data, uint32_t n) {
    uint32_t lg_n = 0;
    for (uint32_t t = n; t > 1; t >>= 1) lg_n++;

    SparkTwiddles tw;
    tw.generate(true);

    int stage = 0;
    if (lg_n <= 10) {
        sppark_ct_pass(data, lg_n, stage, lg_n, tw);
    } else if (lg_n <= 18) {
        int step = lg_n / 2;
        int first = step + lg_n % 2;
        sppark_ct_pass(data, lg_n, stage, first, tw);
        stage += first;
        sppark_ct_pass(data, lg_n, stage, step, tw);
    } else {
        int step = lg_n / 3;
        int rem = lg_n % 3;
        int s0 = step, s1 = step, s2 = step;
        // Match sppark's CT_NTT exactly
        if (lg_n == 29) { s1++; s2++; }
        else { if (rem >= 1) s2++; if (rem >= 2) s1++; }
        // Wait -- sppark CT_NTT for lg_n <= 30:
        // step(step); step(step + (lg==29?1:0)); step(step + (lg==29?1:rem))
        s0 = step;
        s1 = step + (lg_n == 29 ? 1 : 0);
        s2 = step + (lg_n == 29 ? 1 : rem);
        sppark_ct_pass(data, lg_n, stage, s0, tw);
        stage += s0;
        sppark_ct_pass(data, lg_n, stage, s1, tw);
        stage += s1;
        sppark_ct_pass(data, lg_n, stage, s2, tw);
    }
}

// One pass of GS DIF inverse NTT (sppark-style)
static inline void sppark_gs_pass(uint32_t* data, uint32_t lg_domain_size,
                                   uint32_t stage, uint32_t iterations,
                                   const SparkTwiddles& tw, bool is_intt) {
    uint32_t n = 1u << lg_domain_size;
    uint32_t num_threads = n / 2;
    uint32_t radix = iterations < 6 ? 6 : iterations;

    uint32_t diff_mask = (1u << (iterations - 1)) - 1;
    uint32_t inp_mask = ((uint32_t)1 << (stage - 1)) - 1;
    uint32_t out_mask = ((uint32_t)1 << (stage - iterations)) - 1;

    // Build index tables (GS uses different formula from CT)
    std::vector<uint32_t> idx0_tab(num_threads), idx1_tab(num_threads);
    for (uint32_t tid = 0; tid < num_threads; tid++) {
        uint32_t tiz = tid;
        uint32_t idx0 = (tiz & ~inp_mask) * 2;
        idx0 += (tiz << (stage - iterations)) & inp_mask;
        idx0 += (tiz >> (iterations - 1)) & out_mask;
        idx0_tab[tid] = idx0;
        idx1_tab[tid] = idx0 + ((uint32_t)1 << (stage - 1));
    }

    // Load
    std::vector<uint32_t> r0(num_threads), r1(num_threads);
    for (uint32_t tid = 0; tid < num_threads; tid++) {
        r0[tid] = data[idx0_tab[tid]];
        r1[tid] = data[idx1_tab[tid]];
    }

    // GS butterfly stages: process from iterations-1 down to 0
    // (matching sppark's `for (s = iterations; --s >= 1;)` then final add/sub)
    // Stages iterations-1..1: butterfly THEN exchange
    // Stage 0: just add/sub (no twiddle, no exchange)

    for (uint32_t s = iterations - 1; s >= 1; s--) {
        uint32_t laneMask = 1u << (s - 1);
        uint32_t thrdMask = (1u << s) - 1;

        // GS butterfly FIRST: t = root * (r0 - r1); r0 = r0 + r1; r1 = t
        for (uint32_t tid = 0; tid < num_threads; tid++) {
            uint32_t block_size_val = 1u << (radix - 1);
            uint32_t local_tid = tid % block_size_val;
            uint32_t rank = local_tid & thrdMask;

            uint32_t tw_idx = (rank << (10 - (s + 1))) % 512;
            uint32_t root = tw.radix[tw_idx];

            uint32_t diff = sub(r0[tid], r1[tid]);
            r0[tid] = add(r0[tid], r1[tid]);
            r1[tid] = mul(root, diff);
        }

        // THEN exchange
        std::vector<uint32_t> r0s(r0), r1s(r1);
        for (uint32_t tid = 0; tid < num_threads; tid++) {
            uint32_t block_size_val = 1u << (radix - 1);
            uint32_t local_tid = tid % block_size_val;
            uint32_t block_base = tid - local_tid;
            uint32_t rank = local_tid & thrdMask;
            bool pos = rank < laneMask;

            uint32_t partner_local = local_tid ^ laneMask;
            uint32_t partner_tid = block_base + partner_local;
            bool partner_pos = (partner_local & thrdMask) < laneMask;

            uint32_t partner_xchg = partner_pos ? r1s[partner_tid] : r0s[partner_tid];
            r0[tid] = pos ? r0s[tid] : partner_xchg;
            r1[tid] = pos ? partner_xchg : r1s[tid];
        }
    }

    // Stage 0 (innermost): just add/sub, no twiddle, no exchange
    for (uint32_t tid = 0; tid < num_threads; tid++) {
        uint32_t t = sub(r0[tid], r1[tid]);
        r0[tid] = add(r0[tid], r1[tid]);
        r1[tid] = t;
    }

    // Inter-pass twiddle (if stage - iterations != 0, i.e., not the last GS pass)
    if (stage - iterations != 0) {
        for (uint32_t tid = 0; tid < num_threads; tid++) {
            uint32_t tiz = tid;
            uint32_t thread_ntt_pos = (tiz & inp_mask) >> (iterations - 1);
            uint32_t thread_ntt_idx = (tiz & diff_mask) * 2;
            uint32_t nbits = MAX_LG_DOMAIN - (stage - iterations);
            uint32_t br_idx = bit_rev_n(thread_ntt_idx, nbits);
            uint64_t ri0 = (uint64_t)br_idx * thread_ntt_pos;
            uint64_t ri1 = ri0 + ((uint64_t)thread_ntt_pos << (nbits - 1));
            r0[tid] = mul(r0[tid], tw.get_root(ri0));
            r1[tid] = mul(r1[tid], tw.get_root(ri1));
        }
    }

    // 1/N scaling on the last pass (stage == iterations)
    if (is_intt && stage == iterations) {
        uint32_t n_inv = encode(1);
        uint32_t half_mod = encode((P + 1) / 2);
        for (uint32_t i = 0; i < lg_domain_size; i++)
            n_inv = mul(n_inv, half_mod);
        for (uint32_t tid = 0; tid < num_threads; tid++) {
            r0[tid] = mul(r0[tid], n_inv);
            r1[tid] = mul(r1[tid], n_inv);
        }
    }

    // Bit-rotated output: LEFT-rotate for GS
    for (uint32_t tid = 0; tid < num_threads; tid++) {
        uint32_t idx0 = idx0_tab[tid];
        uint32_t idx1 = idx1_tab[tid];
        uint32_t mask = ((1u << iterations) - 1) << (stage - iterations);

        uint32_t rot0 = idx0 & mask;
        rot0 = ((rot0 << 1) | (rot0 >> (iterations - 1))) & mask;
        data[(idx0 & ~mask) | rot0] = r0[tid];

        uint32_t rot1 = idx1 & mask;
        rot1 = ((rot1 << 1) | (rot1 >> (iterations - 1))) & mask;
        data[(idx1 & ~mask) | rot1] = r1[tid];
    }
}

// Multi-pass GS inverse NTT (sppark decomposition)
static inline void sppark_inverse_ntt(uint32_t* data, uint32_t n) {
    uint32_t lg_n = 0;
    for (uint32_t t = n; t > 1; t >>= 1) lg_n++;

    SparkTwiddles tw;
    tw.generate(false); // inverse roots

    int stage = lg_n;
    if (lg_n <= 10) {
        sppark_gs_pass(data, lg_n, stage, lg_n, tw, true);
    } else if (lg_n <= 18) {
        int step = lg_n / 2;
        sppark_gs_pass(data, lg_n, stage, step, tw, true);
        stage -= step;
        sppark_gs_pass(data, lg_n, stage, step + lg_n % 2, tw, true);
    } else {
        int step = lg_n / 3;
        int rem = lg_n % 3;
        int s0 = step + (lg_n == 29 ? 1 : rem);
        int s1 = step + (lg_n == 29 ? 1 : 0);
        int s2 = step;
        sppark_gs_pass(data, lg_n, stage, s0, tw, true);
        stage -= s0;
        sppark_gs_pass(data, lg_n, stage, s1, tw, true);
        stage -= s1;
        sppark_gs_pass(data, lg_n, stage, s2, tw, true);
    }
}

// Validate sppark round-trip: forward + inverse = identity
static inline int32_t validate_sppark_roundtrip(uint32_t lg_n) {
    uint32_t n = 1u << lg_n;
    std::vector<uint32_t> data(n), original(n);

    generate_test_data(data.data(), n, 99999 + lg_n);
    memcpy(original.data(), data.data(), n * sizeof(uint32_t));

    sppark_forward_ntt(data.data(), n);
    sppark_inverse_ntt(data.data(), n);

    int32_t errors = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (data[i] != original[i]) {
            if (errors < 5) {
                fprintf(stderr, "  sppark round-trip mismatch at %u: got=%08x expected=%08x\n",
                        i, data[i], original[i]);
            }
            errors++;
        }
    }
    return errors;
}

// Validate sppark_forward_ntt against existing forward_ntt
static inline int32_t validate_sppark_ntt(uint32_t lg_n) {
    uint32_t n = 1u << lg_n;
    std::vector<uint32_t> data_ref(n), data_sppark(n);

    generate_test_data(data_ref.data(), n, 77777 + lg_n);
    memcpy(data_sppark.data(), data_ref.data(), n * sizeof(uint32_t));

    forward_ntt(data_ref.data(), n);
    sppark_forward_ntt(data_sppark.data(), n);

    int32_t errors = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (data_ref[i] != data_sppark[i]) {
            if (errors < 5) {
                fprintf(stderr, "  sppark NTT mismatch at %u: ref=%08x sppark=%08x (lg_n=%u)\n",
                        i, data_ref[i], data_sppark[i], lg_n);
            }
            errors++;
        }
    }
    return errors;
}

} // namespace bb31_cpu
