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

// Item G12: Fp2 multiplication with lazy Montgomery reduction for the G2 MSM, next to (not
// replacing) bn254.h / bn254_fast.h. Every function returns bit for bit what the bn254.h / Fp2L
// operation it stands for returns (canonical inputs, canonical outputs).
//
// R17 (docs/metal-port/notes/R17.md): every function here is always_inline, the loops have
// constant bounds and no calls, and the 512-bit values are returned by value from inlined
// functions only, so no private aggregate is written element-wise around a call.
#pragma once

#include "bn254_fast.h"

#define G12_INLINE inline __attribute__((always_inline))

namespace bn254 {

// A 512-bit unsigned integer, little-endian words.
struct W512 {
  uint32_t l[16];
};

// P^2 for BN254 Fp (the offset that keeps T0 - T1 non-negative below).
constant uint32_t FP_P2[16] = {0x275d69b1u, 0x3b5458a2u, 0x09eac101u, 0xa602072du,
                               0x6d96cadcu, 0x4a50189cu, 0x7a1242c8u, 0x04689e95u,
                               0x34c6b38du, 0x26edfa5cu, 0x16375606u, 0xb00b8551u,
                               0x0348d21cu, 0x599a6f7cu, 0x763cbf9cu, 0x0925c4b8u};

// The full 512-bit product a * b of any two 256-bit integers (row by row, 64-bit accumulate).
template <class Params> G12_INLINE W512 mul_512(Mont<Params> a, Mont<Params> b) {
  W512 r;
  _Pragma("unroll") for (int i = 0; i < 16; i++)
    r.l[i] = 0;
  _Pragma("unroll") for (int i = 0; i < 8; i++) {
    uint32_t bi = b.l[i];
    uint32_t c = 0;
    _Pragma("unroll") for (int j = 0; j < 8; j++) {
      ulong s = (ulong)a.l[j] * bi + r.l[i + j] + c;
      r.l[i + j] = (uint32_t)s;
      c = (uint32_t)(s >> 32);
    }
    r.l[i + 8] = c;
  }
  return r;
}

// a + b mod 2^512.
G12_INLINE W512 add_512(W512 a, W512 b) {
  W512 r;
  ulong c = 0;
  _Pragma("unroll") for (int i = 0; i < 16; i++) {
    ulong s = (ulong)a.l[i] + b.l[i] + c;
    r.l[i] = (uint32_t)s;
    c = s >> 32;
  }
  return r;
}

// a - b mod 2^512.
G12_INLINE W512 sub_512(W512 a, W512 b) {
  W512 r;
  ulong br = 0;
  _Pragma("unroll") for (int i = 0; i < 16; i++) {
    ulong d = (ulong)a.l[i] - b.l[i] - br;
    r.l[i] = (uint32_t)d;
    br = (d >> 32) & 1;
  }
  return r;
}

// a - b + P^2 mod 2^512 (exact when b < P^2 + a).
G12_INLINE W512 sub_512_p2(W512 a, W512 b) {
  W512 r;
  ulong br = 0;
  ulong c = 0;
  _Pragma("unroll") for (int i = 0; i < 16; i++) {
    ulong d = (ulong)a.l[i] - b.l[i] - br;
    br = (d >> 32) & 1;
    ulong s = (ulong)(uint32_t)d + FP_P2[i] + c;
    r.l[i] = (uint32_t)s;
    c = s >> 32;
  }
  return r;
}

// Montgomery reduction of t < P R: t R^-1 mod P, canonical. The first word of each row uses
// lo(m p0) = -t_i mod 2^32, so (t_i + m p0) >> 32 = mulhi(m, p0) + (t_i != 0) (Z9's identity).
// The rows leave t R^-1 + (something) < t / R + P < 2P, so one conditional subtraction.
template <class Params> G12_INLINE Mont<Params> redc_512(W512 t) {
  uint32_t hi = 0;
  _Pragma("unroll") for (int i = 0; i < 8; i++) {
    uint32_t m = t.l[i] * Params::m0();
    uint32_t c = metal::mulhi(m, Params::p(0)) + (t.l[i] != 0 ? 1u : 0u);
    _Pragma("unroll") for (int j = 1; j < 8; j++) {
      ulong s = (ulong)m * Params::p(j) + t.l[i + j] + c;
      t.l[i + j] = (uint32_t)s;
      c = (uint32_t)(s >> 32);
    }
    ulong s = (ulong)t.l[i + 8] + c + hi;
    t.l[i + 8] = (uint32_t)s;
    hi = (uint32_t)(s >> 32);
  }
  return Mont<Params>::reduce_once(t.l + 8, hi);
}

// mul_nc (bn254_fast.h) with the first word of each reduction row as mulhi(m, p0) + (t0 != 0)
// (see redc_512). Same result.
template <class Params> G12_INLINE Mont<Params> mul_nch(Mont<Params> a, Mont<Params> b) {
  uint32_t t[8];
  _Pragma("unroll") for (int i = 0; i < 8; i++)
    t[i] = 0;
  _Pragma("unroll") for (int i = 0; i < 8; i++) {
    uint32_t bi = b.l[i];
    ulong A = (ulong)a.l[0] * bi + t[0];
    uint32_t t0 = (uint32_t)A;
    A >>= 32;
    uint32_t m = t0 * Params::m0();
    ulong C = (ulong)(metal::mulhi(m, Params::p(0)) + (t0 != 0 ? 1u : 0u));
    _Pragma("unroll") for (int j = 1; j < 8; j++) {
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

// Fp field type with mul_nch (layout and values of Fp / FpF).
template <class Params> struct MontH {
  uint32_t l[8];
  using B = Mont<Params>;

  G12_INLINE B base() const { return B::from_limbs(l); }
  static G12_INLINE MontH of(B x) {
    MontH r;
    _Pragma("unroll") for (int i = 0; i < 8; i++)
      r.l[i] = x.l[i];
    return r;
  }
  static G12_INLINE MontH zero() { return of(B::zero()); }
  static G12_INLINE MontH one() { return of(B::one()); }
  G12_INLINE bool is_zero() const { return base().is_zero(); }
  G12_INLINE bool is_canonical() const { return base().is_canonical(); }
  G12_INLINE bool operator==(MontH o) const { return base() == o.base(); }
  G12_INLINE bool operator!=(MontH o) const { return !(*this == o); }
  friend G12_INLINE MontH operator+(MontH a, MontH b) { return of(a.base() + b.base()); }
  friend G12_INLINE MontH operator-(MontH a, MontH b) { return of(a.base() - b.base()); }
  friend G12_INLINE MontH operator*(MontH a, MontH b) { return of(mul_nch(a.base(), b.base())); }
  G12_INLINE MontH neg() const { return of(base().neg()); }
  G12_INLINE MontH dbl() const { return of(base().dbl()); }
  G12_INLINE MontH sqr() const { return *this * *this; }
  static G12_INLINE MontH mul_loose(MontH a, MontH b) { return a * b; }
  static G12_INLINE MontH add_loose(MontH a, MontH b) { return of(add_nored(a.base(), b.base())); }
  G12_INLINE MontH inv() const { return of(base().inv()); }
};

using FpH = MontH<FpParams>;

// Fp2 = Fp[u]/(u^2 + 1) over F (FpF or FpH: same values) with the product's Karatsuba terms
// kept as 512-bit integers and reduced twice instead of three times:
//   T0 = a0 b0, T1 = a1 b1, T2 = (a0 + a1)(b0 + b1) (sums unreduced, < 2P)
//   c0 = REDC(T0 - T1 + P^2)   (T1 < P^2, so non-negative; < 2 P^2 < P R)
//   c1 = REDC(T2 - T0 - T1)    (= a0 b1 + a1 b0 >= 0; < 2 P^2)
// Each REDC result is canonical and equals (bn254.h) Fp2's c0, c1. Squaring is Fp2L's.
template <class F> struct Fp2R {
  F c0, c1;
  using M = Mont<FpParams>;

  static G12_INLINE Fp2R zero() { return {F::zero(), F::zero()}; }
  static G12_INLINE Fp2R one() { return {F::one(), F::zero()}; }
  G12_INLINE bool is_zero() const { return c0.is_zero() && c1.is_zero(); }
  G12_INLINE bool is_canonical() const { return c0.is_canonical() && c1.is_canonical(); }
  G12_INLINE bool operator==(Fp2R o) const { return c0 == o.c0 && c1 == o.c1; }
  G12_INLINE bool operator!=(Fp2R o) const { return !(*this == o); }
  friend G12_INLINE Fp2R operator+(Fp2R a, Fp2R b) { return {a.c0 + b.c0, a.c1 + b.c1}; }
  friend G12_INLINE Fp2R operator-(Fp2R a, Fp2R b) { return {a.c0 - b.c0, a.c1 - b.c1}; }
  friend G12_INLINE Fp2R operator*(Fp2R a, Fp2R b) {
    M a0 = a.c0.base(), a1 = a.c1.base(), b0 = b.c0.base(), b1 = b.c1.base();
    W512 t0 = mul_512(a0, b0);
    W512 t1 = mul_512(a1, b1);
    W512 s = add_512(t0, t1);
    F r0 = F::of(redc_512<FpParams>(sub_512_p2(t0, t1)));
    W512 t2 = mul_512(add_nored(a0, a1), add_nored(b0, b1));
    F r1 = F::of(redc_512<FpParams>(sub_512(t2, s)));
    return {r0, r1};
  }
  G12_INLINE Fp2R sqr() const {
    F t = c0 * c1;
    F s = F::mul_loose(F::add_loose(c0, c1), c0 - c1);
    return {s, t + t};
  }
  G12_INLINE Fp2R neg() const { return {c0.neg(), c1.neg()}; }
  G12_INLINE Fp2R dbl() const { return *this + *this; }
  G12_INLINE Fp2R inv() const {
    F n = (c0.sqr() + c1.sqr()).inv();
    return {c0 * n, (c1 * n).neg()};
  }
};

// Fp2R over FpF (mul_nc) and over FpH (mul_nch); Fp2LH: Fp2L's formulas over FpH.
using Fp2RF = Fp2R<FpF>;
using Fp2RH = Fp2R<FpH>;
using Fp2LH = Fp2T<FpH, true>;

} // namespace bn254
