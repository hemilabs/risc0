#pragma once
// Optimized Poseidon-254 (BN254 Fr) for Intel ESIMD.
//
// Representation: 8 x simd<u32,16> limbs, lane-per-hash, Montgomery form,
// REDUNDANT range [0, 2r) (r < 2^254 so 4r < 2^256 and Montgomery products of
// operands < 2r land in [0, 2r) without a final subtraction).
//
// Field products use FIPS / product-scanning Montgomery (column-wise, one
// interleaved reduction).  dot3() fuses three products into ONE reduction
// (256 imul instead of 3x128), fma() adds a value into the upper half before
// reduction (s + a*b/R for free), sqr() computes cross products once.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cstdint>

namespace p254o {
namespace esimd = sycl::ext::intel::esimd;
namespace esimd_exp = sycl::ext::intel::experimental::esimd;

using V = esimd::simd<uint32_t, 16>;
using K = esimd::simd<uint32_t, 8>;

struct Fp { V v[8]; };

static constexpr uint32_t RM[8] = {0xf0000001u, 0x43e1f593u, 0x79b97091u, 0x2833e848u,
                                   0x8181585du, 0xb85045b6u, 0xe131a029u, 0x30644e72u};
static constexpr uint32_t R2M[8] = {0xe0000002u, 0x87c3eb27u, 0xf372e122u, 0x5067d090u,
                                    0x0302b0bau, 0x70a08b6du, 0xc2634053u, 0x60c89ce5u};
static constexpr uint32_t RRM[8] = {0xae216da7u, 0x1bb8e645u, 0xe35c59e3u, 0x53fe3ab1u,
                                    0x53bb8085u, 0x8c49833du, 0x7f4e44a5u, 0x0216d0b1u};
static constexpr uint32_t M0 = 0xefffffffu;

// Load one constant Fr (8 dwords, uniform) with a single block load.
ESIMD_INLINE K ldk(const uint32_t* p) {
  return esimd::block_load<uint32_t, 8>(p, esimd::overaligned_tag<16>{});
}

// ---------------------------------------------------------------------------
// Column accumulator: (c0,c1) current column (c1 small), (d0,d1) next column.
// ---------------------------------------------------------------------------
struct Acc { V c0, c1, d0, d1; };

ESIMD_INLINE void acc_zero(Acc& A) { A.c0 = 0; A.c1 = 0; A.d0 = 0; A.d1 = 0; }

ESIMD_INLINE void acc_prod(Acc& A, V a, V b) {
  V lo;
  V hi = esimd_exp::imul(lo, a, b);
  V cy, cy2;
  A.c0 = esimd::addc(cy, A.c0, lo);
  A.c1 += cy;
  A.d0 = esimd::addc(cy2, A.d0, hi);
  A.d1 += cy2;
}
ESIMD_INLINE void acc_add(Acc& A, V x) {
  V cy;
  A.c0 = esimd::addc(cy, A.c0, x);
  A.c1 += cy;
}
ESIMD_INLINE V acc_shift(Acc& A) {
  V out = A.c0;
  V cy;
  A.c0 = esimd::addc(cy, A.c1, A.d0);
  A.c1 = A.d1 + cy;
  A.d0 = 0;
  A.d1 = 0;
  return out;
}

// Generic FIPS Montgomery: `col(A,k)` adds all terms of weight 2^(32k) of the
// 512-bit value T (k = 0..15).  Returns (T + m r) / 2^256.
template <typename F>
ESIMD_INLINE Fp mont_cols(F&& col) {
  Acc A;
  acc_zero(A);
  V m[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    col(A, k);
#pragma unroll
    for (int i = 0; i < k; ++i) acc_prod(A, m[i], V(RM[k - i]));
    m[k] = A.c0 * V(M0);
    acc_prod(A, m[k], V(RM[0]));
    (void)acc_shift(A);  // low word is zero by construction
  }
  Fp out;
#pragma unroll
  for (int k = 8; k < 16; ++k) {
    col(A, k);
#pragma unroll
    for (int i = k - 7; i < 8; ++i) acc_prod(A, m[i], V(RM[k - i]));
    out.v[k - 8] = acc_shift(A);
  }
  return out;
}

// a*b (vector x vector)
ESIMD_INLINE Fp mul(const Fp& a, const Fp& b) {
  return mont_cols([&](Acc& A, int k) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int j = k - i;
      if (j >= 0 && j < 8) acc_prod(A, a.v[i], b.v[j]);
    }
  });
}
// a*c (vector x uniform constant)
ESIMD_INLINE Fp mulk(const Fp& a, const K& c) {
  return mont_cols([&](Acc& A, int k) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int j = k - i;
      if (j >= 0 && j < 8) acc_prod(A, a.v[i], V(c[j]));
    }
  });
}
// s + a*c/R   (s added into the upper half before reduction)
ESIMD_INLINE Fp fmak(const Fp& s, const Fp& a, const K& c) {
  return mont_cols([&](Acc& A, int k) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int j = k - i;
      if (j >= 0 && j < 8) acc_prod(A, a.v[i], V(c[j]));
    }
    if (k >= 8) acc_add(A, s.v[k - 8]);
  });
}
// x0*c0 + x1*c1 + x2*c2 with one reduction
ESIMD_INLINE Fp dot3k(const Fp& x0, const Fp& x1, const Fp& x2, const K& c0,
                      const K& c1, const K& c2) {
  return mont_cols([&](Acc& A, int k) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int j = k - i;
      if (j >= 0 && j < 8) {
        acc_prod(A, x0.v[i], V(c0[j]));
        acc_prod(A, x1.v[i], V(c1[j]));
        acc_prod(A, x2.v[i], V(c2[j]));
      }
    }
  });
}

// Squaring: cross products once, doubled on the fly, plus diagonal.
ESIMD_INLINE Fp sqr(const Fp& a) {
  V t[16];
  {
    Acc A;
    acc_zero(A);
#pragma unroll
    for (int k = 0; k < 16; ++k) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        int j = k - i;
        if (j > i && j < 8) acc_prod(A, a.v[i], a.v[j]);
      }
      t[k] = acc_shift(A);
    }
  }
  V u[16];
  u[0] = t[0] << 1;
#pragma unroll
  for (int k = 1; k < 16; ++k) u[k] = (t[k] << 1) | (t[k - 1] >> 31);
  return mont_cols([&](Acc& A, int k) {
    acc_add(A, u[k]);
    if ((k & 1) == 0) {
      V lo;
      V hi = esimd_exp::imul(lo, a.v[k >> 1], a.v[k >> 1]);
      V cy, cy2;
      A.c0 = esimd::addc(cy, A.c0, lo);
      A.c1 += cy;
      A.d0 = esimd::addc(cy2, A.d0, hi);
      A.d1 += cy2;
    }
  });
}

// x - K if x >= K (K = constant limbs)
ESIMD_INLINE void csub(Fp& x, const uint32_t (&Kc)[8]) {
  V d[8];
  V borrow = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    V b1, b2;
    V tmp = esimd::subb(b1, x.v[j], V(Kc[j]));
    d[j] = esimd::subb(b2, tmp, borrow);
    borrow = b1 + b2;
  }
  esimd::simd_mask<16> ge = borrow == V(0u);
#pragma unroll
  for (int j = 0; j < 8; ++j) x.v[j].merge(d[j], ge);
}
ESIMD_INLINE void csub2r(Fp& x) { csub(x, R2M); }

// x + c (c uniform constant < r), result kept < 2r
ESIMD_INLINE Fp addk(const Fp& x, const K& c) {
  Fp s;
  V carry = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    V c1, c2;
    V t = esimd::addc(c1, x.v[j], V(c[j]));
    s.v[j] = esimd::addc(c2, t, carry);
    carry = c1 + c2;
  }
  csub2r(s);
  return s;
}

ESIMD_INLINE Fp sbox(const Fp& x) {
  return sqr(sqr(sqr(x)));  // x^8 (risc0's Poseidon-254 parameter set)
}

// canonical (< r) -> Montgomery (< 2r)
ESIMD_INLINE Fp to_mont(const Fp& x) {
  K rr;
#pragma unroll
  for (int j = 0; j < 8; ++j) rr[j] = RRM[j];
  return mulk(x, rr);
}
// Montgomery (< 2r) -> canonical (< r)
ESIMD_INLINE Fp from_mont(const Fp& x) {
  Fp r = mont_cols([&](Acc& A, int k) {
    if (k < 8) acc_add(A, x.v[k]);
  });
  csub(r, RM);
  return r;
}

// ---------------------------------------------------------------------------
// Permutations.  Only cells[0] is meaningful on exit (callers never read 1,2:
// rows re-absorb into them, fold/pair read only cell 0).
// ---------------------------------------------------------------------------

// Sparse layout (CUDA poseidon254_constants.cuh order):
//   RC[68] | MDS[9] | PRE_SPARSE[9] | SPARSE[42][4] | M00[1]
static constexpr int SP_RC = 0, SP_MDS = 68, SP_PRE = 77, SP_SPARSE = 86, SP_M00 = 254;

ESIMD_INLINE void full_sp(Fp s[3], const uint32_t* pc, const uint32_t* rc,
                          const uint32_t* mat, bool only0) {
#pragma unroll
  for (int i = 0; i < 3; ++i) s[i] = sbox(addk(s[i], ldk(rc + i * 8)));
  K a0 = ldk(mat + 0 * 8), a1 = ldk(mat + 1 * 8), a2 = ldk(mat + 2 * 8);
  Fp n0 = dot3k(s[0], s[1], s[2], a0, a1, a2);
  csub2r(n0);
  if (!only0) {
    K b0 = ldk(mat + 3 * 8), b1 = ldk(mat + 4 * 8), b2 = ldk(mat + 5 * 8);
    Fp n1 = dot3k(s[0], s[1], s[2], b0, b1, b2);
    csub2r(n1);
    K c0 = ldk(mat + 6 * 8), c1 = ldk(mat + 7 * 8), c2 = ldk(mat + 8 * 8);
    Fp n2 = dot3k(s[0], s[1], s[2], c0, c1, c2);
    csub2r(n2);
    s[1] = n1;
    s[2] = n2;
  }
  s[0] = n0;
}

ESIMD_INLINE void mix_sparse(Fp s[3], const uint32_t* pc) {
  const uint32_t* rc = pc + SP_RC * 8;
  const uint32_t* mds = pc + SP_MDS * 8;
#pragma unroll 1
  for (int r = 0; r < 3; ++r) { full_sp(s, pc, rc, mds, false); rc += 24; }
  full_sp(s, pc, rc, pc + SP_PRE * 8, false);
  rc += 24;
#pragma unroll
  for (int i = 0; i < 3; ++i) s[i] = addk(s[i], ldk(rc + i * 8));
  rc += 24;
  const K m00 = ldk(pc + SP_M00 * 8);
#pragma unroll 1
  for (int pr = 0; pr < 42; ++pr) {
    const uint32_t* sp = pc + (SP_SPARSE + pr * 4) * 8;
    Fp x = sbox(s[0]);
    Fp p0;
    if (pr < 41) {
      p0 = addk(x, ldk(rc));
      rc += 8;
    } else {
      p0 = x;
    }
    K w0 = ldk(sp + 0), w1 = ldk(sp + 8), v0 = ldk(sp + 16), v1 = ldk(sp + 24);
    Fp n0 = dot3k(p0, s[1], s[2], m00, w0, w1);
    csub2r(n0);
    Fp n1 = fmak(s[1], p0, v0);  // < 3.38r
    Fp n2 = fmak(s[2], p0, v1);
    csub2r(n1);
    csub2r(n2);
    s[0] = n0; s[1] = n1; s[2] = n2;
  }
#pragma unroll 1
  for (int r = 0; r < 3; ++r) { full_sp(s, pc, rc, mds, false); rc += 24; }
  full_sp(s, pc, rc, mds, true);
}

}  // namespace p254o
