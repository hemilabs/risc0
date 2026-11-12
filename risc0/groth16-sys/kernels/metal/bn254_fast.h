// Copyright 2026 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Item G9: faster formulations of bn254.h operations, added next to (not replacing) bn254.h's,
// so that bn254.h keeps its semantics for Z10. Every function here returns bit-for-bit what the
// bn254.h function it names returns (same canonical inputs and outputs); the MSM kernels pick
// one or the other at compile time (template parameter), the host picks the pipeline.
#pragma once

#include "bn254.h"

namespace bn254 {

// Montgomery multiplication, gnark's "no-carry" CIOS: valid when the modulus' top limb is below
// 2^31 - 1 (BN254 Fp and Fr: 0x30644e72), so that t never needs the 9th/10th words; the
// product and reduction rows run in one loop. Same result as `a * b` (inputs canonical).
template <class Params> inline Mont<Params> mul_nc(Mont<Params> a, Mont<Params> b) {
  uint32_t t[8];
  for (int i = 0; i < 8; i++)
    t[i] = 0;
  for (int i = 0; i < 8; i++) {
    uint32_t bi = b.l[i];
    ulong A = (ulong)a.l[0] * bi + t[0];
    uint32_t t0 = (uint32_t)A;
    A >>= 32;
    uint32_t m = t0 * Params::m0();
    ulong C = ((ulong)m * Params::p(0) + t0) >> 32;
    for (int j = 1; j < 8; j++) {
      A = (ulong)a.l[j] * bi + t[j] + A;
      uint32_t tj = (uint32_t)A;
      A >>= 32;
      C = (ulong)m * Params::p(j) + tj + C;
      t[j - 1] = (uint32_t)C;
      C >>= 32;
    }
    t[7] = (uint32_t)(C + A);
  }
  return Mont<Params>::reduce_once(t, 0);
}

// `Mont::reduce_once` with the borrow chain as 64-bit subtractions.
template <class Params>
inline Mont<Params> reduce_once_wide(thread const uint32_t* t, uint32_t hi) {
  uint32_t u[8];
  ulong br = 0;
  for (int i = 0; i < 8; i++) {
    ulong d = (ulong)t[i] - Params::p(i) - br;
    u[i] = (uint32_t)d;
    br = (d >> 32) & 1;
  }
  bool keep_t = br && !hi;
  Mont<Params> r;
  for (int i = 0; i < 8; i++)
    r.l[i] = keep_t ? t[i] : u[i];
  return r;
}

// Montgomery multiplication, the CIOS of `Mont::operator*` with the carry chains written as
// 32x32 -> 64-bit multiply-accumulates (`ulong`), which the compiler lowers to mul/mulhi plus a
// 64-bit add, instead of compare-based carries. Same result as `a * b`.
template <class Params> inline Mont<Params> mul_wide(Mont<Params> a, Mont<Params> b) {
  uint32_t t[10];
  for (int i = 0; i < 10; i++)
    t[i] = 0;
  for (int i = 0; i < 8; i++) {
    uint32_t bi = b.l[i];
    ulong c = 0;
    for (int j = 0; j < 8; j++) {
      ulong s = (ulong)a.l[j] * bi + t[j] + c;
      t[j] = (uint32_t)s;
      c = s >> 32;
    }
    ulong s8 = (ulong)t[8] + c;
    t[8] = (uint32_t)s8;
    t[9] = (uint32_t)(s8 >> 32);

    uint32_t m = t[0] * Params::m0();
    ulong r = (ulong)m * Params::p(0) + t[0];
    c = r >> 32;
    for (int j = 1; j < 8; j++) {
      r = (ulong)m * Params::p(j) + t[j] + c;
      t[j - 1] = (uint32_t)r;
      c = r >> 32;
    }
    r = (ulong)t[8] + c;
    t[7] = (uint32_t)r;
    t[8] = t[9] + (uint32_t)(r >> 32);
  }
  return Mont<Params>::reduce_once(t, t[8]);
}

// a + b mod P (`Mont::operator+`), carries as 64-bit adds.
template <class Params> inline Mont<Params> add_wide(Mont<Params> a, Mont<Params> b) {
  uint32_t t[8];
  ulong c = 0;
  for (int i = 0; i < 8; i++) {
    ulong s = (ulong)a.l[i] + b.l[i] + c;
    t[i] = (uint32_t)s;
    c = s >> 32;
  }
  return reduce_once_wide<Params>(t, (uint32_t)c);
}

// a - b mod P (`Mont::operator-`), borrows as 64-bit subtractions.
template <class Params> inline Mont<Params> sub_wide(Mont<Params> a, Mont<Params> b) {
  uint32_t t[8];
  ulong br = 0;
  for (int i = 0; i < 8; i++) {
    ulong d = (ulong)a.l[i] - b.l[i] - br;
    t[i] = (uint32_t)d;
    br = (d >> 32) & 1;
  }
  uint32_t mask = 0u - (uint32_t)br;
  ulong c = 0;
  for (int i = 0; i < 8; i++) {
    ulong s = (ulong)t[i] + (Params::p(i) & mask) + c;
    t[i] = (uint32_t)s;
    c = s >> 32;
  }
  return Mont<Params>::from_limbs(t);
}

// a^2 R^-1 mod P (`a * a`): the 512-bit square with the off-diagonal products computed once
// (36 instead of 64 limb products), then a separated Montgomery reduction. Same result.
template <class Params> inline Mont<Params> sqr_wide(Mont<Params> a) {
  uint32_t t[16];
  for (int i = 0; i < 16; i++)
    t[i] = 0;
  // Off-diagonal products a[i] a[j], i < j.
  for (int i = 0; i < 7; i++) {
    ulong c = 0;
    for (int j = i + 1; j < 8; j++) {
      ulong s = (ulong)a.l[i] * a.l[j] + t[i + j] + c;
      t[i + j] = (uint32_t)s;
      c = s >> 32;
    }
    t[i + 8] = (uint32_t)c;
  }
  // Times two (the top bit of the off-diagonal sum is 0: it is < 2^511).
  for (int i = 15; i > 0; i--)
    t[i] = (t[i] << 1) | (t[i - 1] >> 31);
  t[0] <<= 1;
  // Plus the diagonal a[i]^2 at 2i.
  ulong c = 0;
  for (int i = 0; i < 8; i++) {
    ulong d = (ulong)a.l[i] * a.l[i];
    ulong s = (ulong)t[2 * i] + (uint32_t)d + c;
    t[2 * i] = (uint32_t)s;
    s = (ulong)t[2 * i + 1] + (uint32_t)(d >> 32) + (s >> 32);
    t[2 * i + 1] = (uint32_t)s;
    c = s >> 32;
  }
  // Montgomery reduction, one limb per row; `hi` carries out of t[i + 8] into the next row's
  // t[i + 9] (which that row's inner loop does not touch).
  ulong hi = 0;
  for (int i = 0; i < 8; i++) {
    uint32_t m = t[i] * Params::m0();
    ulong r = (ulong)m * Params::p(0) + t[i];
    r >>= 32;
    for (int j = 1; j < 8; j++) {
      r = (ulong)m * Params::p(j) + t[i + j] + r;
      t[i + j] = (uint32_t)r;
      r >>= 32;
    }
    ulong s = (ulong)t[i + 8] + r + hi;
    t[i + 8] = (uint32_t)s;
    hi = s >> 32;
  }
  return Mont<Params>::reduce_once(t + 8, (uint32_t)hi);
}

// a + b without the reduction, for operands of a multiplication: a, b < P give a + b < 2P <
// 2^255 (see MontF::mul_loose).
template <class Params> inline Mont<Params> add_nored(Mont<Params> a, Mont<Params> b) {
  Mont<Params> r;
  ulong c = 0;
  for (int i = 0; i < 8; i++) {
    ulong s = (ulong)a.l[i] + b.l[i] + c;
    r.l[i] = (uint32_t)s;
    c = s >> 32;
  }
  return r;
}

// ----------------------------------------------------------------------------------------------
// Field types with the fast multiplication: same layout and values as Fp / Fp2 (a `device
// Affine<Fp>` can be read as `Affine<FpF>`), so `Affine`, `Xyzz` instantiated over them compute
// bit-for-bit what they compute over Fp / Fp2. Only `*` (and `sqr`) differ in formulation.
// ----------------------------------------------------------------------------------------------

// `*` is mul_nc. V (benchmark variants, src/metal/tests.rs `g9_fast_field_ops`; rejected in
// the G9 note): bit 0 = `sqr` through sqr_wide, bit 1 = `+`/`-` through add_wide/sub_wide,
// bit 2 = `*` through mul_wide.
template <class Params, int V = 0> struct MontF {
  uint32_t l[8];
  using B = Mont<Params>;

  inline B base() const { return B::from_limbs(l); }
  static inline MontF of(B x) {
    MontF r;
    for (int i = 0; i < 8; i++)
      r.l[i] = x.l[i];
    return r;
  }
  static inline MontF zero() { return of(B::zero()); }
  static inline MontF one() { return of(B::one()); }
  inline bool is_zero() const { return base().is_zero(); }
  inline bool is_canonical() const { return base().is_canonical(); }
  inline bool operator==(MontF o) const { return base() == o.base(); }
  inline bool operator!=(MontF o) const { return !(*this == o); }
  friend inline MontF operator+(MontF a, MontF b) {
    return of((V & 2) ? add_wide(a.base(), b.base()) : a.base() + b.base());
  }
  friend inline MontF operator-(MontF a, MontF b) {
    return of((V & 2) ? sub_wide(a.base(), b.base()) : a.base() - b.base());
  }
  friend inline MontF operator*(MontF a, MontF b) {
    return of((V & 4) ? mul_wide(a.base(), b.base()) : mul_nc(a.base(), b.base()));
  }
  inline MontF neg() const { return of(base().neg()); }
  inline MontF dbl() const { return of(base().dbl()); }
  inline MontF sqr() const { return (V & 1) ? of(sqr_wide(base())) : *this * *this; }
  // a * b for a, b < 2P (unreduced sums from add_nored): canonical a b R^-1.
  // a * b for a, b < 2P: mul_nc's rows stay below 3P < 2^256 and the result below 2P (4P < R),
  // so reduce_once makes it canonical; the same holds for mul_wide.
  static inline MontF mul_loose(MontF a, MontF b) { return a * b; }
  static inline MontF add_loose(MontF a, MontF b) { return of(add_nored(a.base(), b.base())); }
  inline MontF inv() const { return of(base().inv()); }
};

using FpF = MontF<FpParams>;
using FrF = MontF<FrParams>;
using FpS = MontF<FpParams, 1>;
using FpA = MontF<FpParams, 2>;
using FpW = MontF<FpParams, 4>;

// Fp2 over FpF: the formulas of bn254.h's Fp2 (Karatsuba mul, 2-mul sqr).
// LAZY: Karatsuba's (a0 + a1)(b0 + b1) and sqr's (c0 + c1) without the reduction of the sums.
template <class FpF, bool LAZY = false> struct Fp2T {
  using Fp2F = Fp2T;
  FpF c0, c1;

  static inline Fp2F zero() { return {FpF::zero(), FpF::zero()}; }
  static inline Fp2F one() { return {FpF::one(), FpF::zero()}; }
  inline bool is_zero() const { return c0.is_zero() && c1.is_zero(); }
  inline bool is_canonical() const { return c0.is_canonical() && c1.is_canonical(); }
  inline bool operator==(Fp2F o) const { return c0 == o.c0 && c1 == o.c1; }
  inline bool operator!=(Fp2F o) const { return !(*this == o); }
  friend inline Fp2F operator+(Fp2F a, Fp2F b) { return {a.c0 + b.c0, a.c1 + b.c1}; }
  friend inline Fp2F operator-(Fp2F a, Fp2F b) { return {a.c0 - b.c0, a.c1 - b.c1}; }
  friend inline Fp2F operator*(Fp2F a, Fp2F b) {
    FpF t0 = a.c0 * b.c0;
    FpF t1 = a.c1 * b.c1;
    FpF t2 = LAZY ? FpF::mul_loose(FpF::add_loose(a.c0, a.c1), FpF::add_loose(b.c0, b.c1))
                  : (a.c0 + a.c1) * (b.c0 + b.c1);
    return {t0 - t1, t2 - t0 - t1};
  }
  inline Fp2F sqr() const {
    FpF t = c0 * c1;
    FpF s = LAZY ? FpF::mul_loose(FpF::add_loose(c0, c1), c0 - c1) : (c0 + c1) * (c0 - c1);
    return {s, t + t};
  }
  inline Fp2F neg() const { return {c0.neg(), c1.neg()}; }
  inline Fp2F dbl() const { return *this + *this; }
  inline Fp2F inv() const {
    FpF n = (c0.sqr() + c1.sqr()).inv();
    return {c0 * n, (c1 * n).neg()};
  }
};

// Fp2L: the MSM's G2 field (lazy Karatsuba sums). Fp2F, Fp2W: benchmark variants.
using Fp2L = Fp2T<FpF, true>;
using Fp2F = Fp2T<FpF>;
using Fp2W = Fp2T<FpW, true>;


// ----------------------------------------------------------------------------------------------
// G2 on lane pairs: a thread pair (lanes 2k, 2k + 1 of a SIMD group) holds one G2 value, the
// even lane its c0 components, the odd lane its c1 components (half the registers of a thread
// holding the whole Fp2 element). Fp2 products exchange the partner's operands with
// simd_shuffle_xor(., 1): schoolbook, 2 Fp products per lane (lane 0 a0 b0 - a1 b1, lane 1
// a1 b0 + a0 b1); squares 1 per lane. Branch conditions are combined over the pair, so both
// lanes always take the same path. Values are canonical, as bn254.h's Fp2.
// ----------------------------------------------------------------------------------------------

namespace pair {

inline FpF other(FpF a) {
  return FpF{{metal::simd_shuffle_xor(a.l[0], 1u), metal::simd_shuffle_xor(a.l[1], 1u),
              metal::simd_shuffle_xor(a.l[2], 1u), metal::simd_shuffle_xor(a.l[3], 1u),
              metal::simd_shuffle_xor(a.l[4], 1u), metal::simd_shuffle_xor(a.l[5], 1u),
              metal::simd_shuffle_xor(a.l[6], 1u), metal::simd_shuffle_xor(a.l[7], 1u)}};
}

// Both lanes' components zero (the Fp2 element is zero).
inline bool is_zero(FpF a) {
  uint32_t z = a.is_zero() ? 1u : 0u;
  return (z & metal::simd_shuffle_xor(z, 1u)) != 0;
}

inline FpF mul(FpF a, FpF b, bool odd) {
  FpF ao = other(a), bo = other(b);
  FpF x = a * (odd ? bo : b);
  FpF y = ao * (odd ? b : bo);
  return odd ? x + y : x - y;
}

inline FpF sqr(FpF a, bool odd) {
  FpF ao = other(a);
  FpF x = odd ? ao.dbl() : a + ao;
  FpF y = odd ? a : a - ao;
  return x * y;
}

// Fp2 one on this lane: (1, 0).
inline FpF one(bool odd) { return odd ? FpF::zero() : FpF::one(); }

struct Aff {
  FpF x, y;
};

struct Xyzz2 {
  FpF x, y, zz, zzz;
};

inline Xyzz2 inf() { return {FpF::zero(), FpF::zero(), FpF::zero(), FpF::zero()}; }

// Xyzz<Fp2>::dbl_affine.
inline Xyzz2 dbl_affine(Aff q, bool odd) {
  if (is_zero(q.y))
    return inf();
  FpF u = q.y.dbl();
  FpF v = sqr(u, odd);
  FpF w = mul(u, v, odd);
  FpF s = mul(q.x, v, odd);
  FpF x2 = sqr(q.x, odd);
  FpF m = x2.dbl() + x2;
  FpF x3 = sqr(m, odd) - s.dbl();
  FpF y3 = mul(m, s - x3, odd) - mul(w, q.y, odd);
  return {x3, y3, v, w};
}

// Xyzz<Fp2>::add_affine (madd-2008-s), same branches.
inline Xyzz2 add_affine(Xyzz2 a, Aff q, bool odd) {
  if (is_zero(q.x) && is_zero(q.y))
    return a;
  if (is_zero(a.zz))
    return {q.x, q.y, one(odd), one(odd)};
  FpF u2 = mul(q.x, a.zz, odd);
  FpF s2 = mul(q.y, a.zzz, odd);
  FpF p = u2 - a.x;
  FpF r = s2 - a.y;
  if (is_zero(p)) {
    if (is_zero(r))
      return dbl_affine(q, odd);
    return inf();
  }
  FpF pp = sqr(p, odd);
  FpF ppp = mul(p, pp, odd);
  FpF qq = mul(a.x, pp, odd);
  FpF x3 = sqr(r, odd) - ppp - qq.dbl();
  FpF y3 = mul(r, qq - x3, odd) - mul(a.y, ppp, odd);
  return {x3, y3, mul(a.zz, pp, odd), mul(a.zzz, ppp, odd)};
}

} // namespace pair

} // namespace bn254
