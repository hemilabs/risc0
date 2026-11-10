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

// BN254 (alt_bn128) field and curve arithmetic for the Metal Groth16 prover (and, later, the
// Poseidon254 hash, PLAN item Z10).
//
// Port of the Intel SYCL prover's kernels/intel/bn254.hpp (same algorithms, same layout); see
// docs/metal-port/notes/G2.md.
//
// Representation
// - An element is 8 x 32-bit limbs, least significant first, in Montgomery form with R = 2^256
//   (bit-for-bit sppark's fp_t / fr_t and ark-ff's in-memory BigInt), kept fully reduced.
// - Arithmetic (+, -, *, sqr, neg, dbl, inv, pow) REQUIRES canonical inputs (value < P); results
//   are canonical. Non-canonical inputs are not detected by the arithmetic; callers that load
//   untrusted data check `is_canonical()` (and reject), or reduce raw integers with `to_mont()`.
// - `to_mont()` accepts ANY 256-bit integer v (including v >= P) and returns (v mod P) * R mod P.
//   `from_mont()` accepts any 256-bit value too and returns v * R^-1 mod P (canonical).
// - Only 32-bit multiplies (`a * b`, `mulhi(a, b)`) are used: on Apple GPUs 64-bit multiplies are
//   emulated (D0 measured 64-bit mulhi ~4.7x slower than 32-bit mulhi).
//
// Curves (a = 0): G1 over Fp (y^2 = x^3 + 3), G2 over Fp2 = Fp[u]/(u^2 + 1).
// - Affine (x, y); (0, 0) is the point at infinity (the proving key's convention).
// - Jacobian (X, Y, Z): x = X/Z^2, y = Y/Z^3; Z = 0 is infinity (sppark point_t layout).
// - Xyzz (X, Y, ZZ, ZZZ): x = X/ZZ, y = Y/ZZZ, ZZ^3 = ZZZ^2; ZZ = 0 is infinity (X, Y ignored).
//
// Everything is `thread`-address-space value code: kernels copy `device` data into locals, operate,
// and store. No 64-bit integers, no recursion.
#pragma once

#include <metal_stdlib>

#ifndef BN254_OUTLINE
// Point operations off the MSM hot path. Empty by default (everything inlines); define it to
// `__attribute__((noinline))` to trade speed for front-end time in kernels that use many of them.
#define BN254_OUTLINE
#endif

namespace bn254 {

// ----------------------------------------------------------------------------------------------
// 32-bit carry helpers
// ----------------------------------------------------------------------------------------------

// a + b + carry_in; carry_out set to the carry (0 or 1). carry_in must be 0 or 1.
inline uint32_t addc(uint32_t a, uint32_t b, uint32_t carry_in, thread uint32_t& carry_out) {
  uint32_t s = a + b;
  uint32_t c = s < a;
  uint32_t s2 = s + carry_in;
  carry_out = c + (s2 < s);
  return s2;
}

// a - b - borrow_in; borrow_out set to the borrow (0 or 1). borrow_in must be 0 or 1.
inline uint32_t subb(uint32_t a, uint32_t b, uint32_t borrow_in, thread uint32_t& borrow_out) {
  uint32_t d = a - b;
  uint32_t br = a < b;
  uint32_t d2 = d - borrow_in;
  borrow_out = br + (d < borrow_in);
  return d2;
}

// ----------------------------------------------------------------------------------------------
// Field parameters (identical to kernels/intel/bn254.hpp, i.e. sppark's ALT_BN128 constants)
// ----------------------------------------------------------------------------------------------

constant uint32_t FP_P[8] = {0xd87cfd47u, 0x3c208c16u, 0x6871ca8du, 0x97816a91u,
                             0x8181585du, 0xb85045b6u, 0xe131a029u, 0x30644e72u};
constant uint32_t FP_RR[8] = {0x538afa89u, 0xf32cfc5bu, 0xd44501fbu, 0xb5e71911u,
                              0x0a417ff6u, 0x47ab1effu, 0xcab8351fu, 0x06d89f71u}; // R^2 mod P
constant uint32_t FP_ONE[8] = {0xc58f0d9du, 0xd35d438du, 0xf5c70b3du, 0x0a78eb28u,
                               0x7879462cu, 0x666ea36fu, 0x9a07df2fu, 0x0e0a77c1u}; // R mod P

constant uint32_t FR_P[8] = {0xf0000001u, 0x43e1f593u, 0x79b97091u, 0x2833e848u,
                             0x8181585du, 0xb85045b6u, 0xe131a029u, 0x30644e72u};
constant uint32_t FR_RR[8] = {0xae216da7u, 0x1bb8e645u, 0xe35c59e3u, 0x53fe3ab1u,
                              0x53bb8085u, 0x8c49833du, 0x7f4e44a5u, 0x0216d0b1u};
constant uint32_t FR_ONE[8] = {0x4ffffffbu, 0xac96341cu, 0x9f60cd29u, 0x36fc7695u,
                               0x7879462eu, 0x666ea36fu, 0x9a07df2fu, 0x0e0a77c1u};

struct FpParams {
  static inline uint32_t p(int i) { return FP_P[i]; }
  static inline uint32_t rr(int i) { return FP_RR[i]; }
  static inline uint32_t one(int i) { return FP_ONE[i]; }
  static inline uint32_t m0() { return 0xe4866389u; } // -P^-1 mod 2^32
};

struct FrParams {
  static inline uint32_t p(int i) { return FR_P[i]; }
  static inline uint32_t rr(int i) { return FR_RR[i]; }
  static inline uint32_t one(int i) { return FR_ONE[i]; }
  static inline uint32_t m0() { return 0xefffffffu; } // -P^-1 mod 2^32
};

// ----------------------------------------------------------------------------------------------
// Montgomery field element
// ----------------------------------------------------------------------------------------------

template <class Params> struct Mont {
  uint32_t l[8];

  static inline Mont zero() {
    Mont r;
    for (int i = 0; i < 8; i++)
      r.l[i] = 0;
    return r;
  }
  static inline Mont one() {
    Mont r;
    for (int i = 0; i < 8; i++)
      r.l[i] = Params::one(i);
    return r;
  }
  // R^2 mod P as an element (multiplying by it converts into Montgomery form).
  static inline Mont rr() {
    Mont r;
    for (int i = 0; i < 8; i++)
      r.l[i] = Params::rr(i);
    return r;
  }
  // The raw limbs as an element (no conversion).
  static inline Mont from_limbs(thread const uint32_t* v) {
    Mont r;
    for (int i = 0; i < 8; i++)
      r.l[i] = v[i];
    return r;
  }
  static inline Mont from_limbs(const device uint32_t* v) {
    Mont r;
    for (int i = 0; i < 8; i++)
      r.l[i] = v[i];
    return r;
  }
  static inline Mont from_limbs(constant uint32_t* v) {
    Mont r;
    for (int i = 0; i < 8; i++)
      r.l[i] = v[i];
    return r;
  }

  inline bool is_zero() const {
    uint32_t acc = 0;
    for (int i = 0; i < 8; i++)
      acc |= l[i];
    return acc == 0;
  }
  inline bool operator==(Mont o) const {
    uint32_t acc = 0;
    for (int i = 0; i < 8; i++)
      acc |= l[i] ^ o.l[i];
    return acc == 0;
  }
  inline bool operator!=(Mont o) const { return !(*this == o); }

  // value < P, i.e. a valid (fully reduced) element.
  inline bool is_canonical() const {
    uint32_t borrow = 0;
    for (int i = 0; i < 8; i++)
      (void)subb(l[i], Params::p(i), borrow, borrow);
    return borrow != 0;
  }

  // t (8 limbs, plus a carry-out bit `hi`) reduced once by P. Precondition t + hi*2^256 < 2P.
  static inline Mont reduce_once(thread const uint32_t* t, uint32_t hi) {
    uint32_t u[8];
    uint32_t borrow = 0;
    for (int i = 0; i < 8; i++)
      u[i] = subb(t[i], Params::p(i), borrow, borrow);
    // Keep t - P unless it went negative (and t had no carry-out).
    bool keep_t = borrow && !hi;
    Mont r;
    for (int i = 0; i < 8; i++)
      r.l[i] = keep_t ? t[i] : u[i];
    return r;
  }

  friend inline Mont operator+(Mont a, Mont b) {
    uint32_t t[8];
    uint32_t c = 0;
    for (int i = 0; i < 8; i++)
      t[i] = addc(a.l[i], b.l[i], c, c);
    return reduce_once(t, c);
  }

  friend inline Mont operator-(Mont a, Mont b) {
    uint32_t t[8];
    uint32_t borrow = 0;
    for (int i = 0; i < 8; i++)
      t[i] = subb(a.l[i], b.l[i], borrow, borrow);
    // Add P back if it went negative (masked, no branch).
    uint32_t mask = 0u - borrow;
    uint32_t c = 0;
    for (int i = 0; i < 8; i++)
      t[i] = addc(t[i], Params::p(i) & mask, c, c);
    return from_limbs(t);
  }

  inline Mont neg() const { return is_zero() ? *this : zero() - *this; }
  inline Mont dbl() const { return *this + *this; }

  // Montgomery multiplication (CIOS, 32-bit operations only): a * b * R^-1 mod P.
  // Correct (result canonical) whenever a * b < P * R, in particular for a < 2^256 and b < P.
  friend inline Mont operator*(Mont a, Mont b) {
    uint32_t t[10];
    for (int i = 0; i < 10; i++)
      t[i] = 0;
    for (int i = 0; i < 8; i++) {
      uint32_t c = 0;
      uint32_t bi = b.l[i];
      for (int j = 0; j < 8; j++) {
        // t[j] + a[j] * bi + c < 2^64: (hi, lo).
        uint32_t lo = a.l[j] * bi, hi = metal::mulhi(a.l[j], bi);
        uint32_t k;
        lo = addc(lo, t[j], 0, k);
        hi += k;
        lo = addc(lo, c, 0, k);
        hi += k;
        t[j] = lo;
        c = hi;
      }
      uint32_t k;
      t[8] = addc(t[8], c, 0, k);
      t[9] = k;

      uint32_t m = t[0] * Params::m0();
      // t[0] + m * P[0] is 0 mod 2^32; only its carry matters.
      uint32_t lo = m * Params::p(0), hi = metal::mulhi(m, Params::p(0));
      (void)addc(lo, t[0], 0, k);
      c = hi + k;
      for (int j = 1; j < 8; j++) {
        lo = m * Params::p(j);
        hi = metal::mulhi(m, Params::p(j));
        lo = addc(lo, t[j], 0, k);
        hi += k;
        lo = addc(lo, c, 0, k);
        hi += k;
        t[j - 1] = lo;
        c = hi;
      }
      t[7] = addc(t[8], c, 0, k);
      t[8] = t[9] + k;
    }
    return reduce_once(t, t[8]);
  }

  inline Mont sqr() const { return *this * *this; }

  thread Mont& operator+=(Mont o) { return *this = *this + o; }
  thread Mont& operator-=(Mont o) { return *this = *this - o; }
  thread Mont& operator*=(Mont o) { return *this = *this * o; }

  // Into Montgomery form: any 256-bit integer v -> (v mod P) * R mod P.
  inline Mont to_mont() const { return *this * rr(); }
  // Out of Montgomery form: v * R^-1 mod P, canonical for any 256-bit v.
  inline Mont from_mont() const {
    Mont one_raw = zero();
    one_raw.l[0] = 1;
    return *this * one_raw;
  }

  // self^e for a little-endian 256-bit exponent e (plain integer, not Montgomery).
  inline Mont pow(thread const uint32_t* e) const {
    Mont r = one();
    for (int i = 255; i >= 0; i--) {
      r = r.sqr();
      if ((e[i / 32] >> (i % 32)) & 1)
        r *= *this;
    }
    return r;
  }
  // Inverse by Fermat (self^(P-2)); zero maps to zero. ~256 S + ~128 M: not for hot loops.
  inline Mont inv() const {
    uint32_t e[8];
    uint32_t borrow = 0;
    for (int i = 0; i < 8; i++)
      e[i] = subb(Params::p(i), i == 0 ? 2u : 0u, borrow, borrow);
    return pow(e);
  }
};

using Fp = Mont<FpParams>;
using Fr = Mont<FrParams>;

// ----------------------------------------------------------------------------------------------
// Fp2 = Fp[u] / (u^2 + 1), layout (c0, c1) = c0 + c1 u (sppark fp2_t / ark Fq2 order)
// ----------------------------------------------------------------------------------------------

struct Fp2 {
  Fp c0, c1;

  static inline Fp2 zero() { return {Fp::zero(), Fp::zero()}; }
  static inline Fp2 one() { return {Fp::one(), Fp::zero()}; }
  inline bool is_zero() const { return c0.is_zero() && c1.is_zero(); }
  inline bool is_canonical() const { return c0.is_canonical() && c1.is_canonical(); }
  inline bool operator==(Fp2 o) const { return c0 == o.c0 && c1 == o.c1; }
  inline bool operator!=(Fp2 o) const { return !(*this == o); }

  friend inline Fp2 operator+(Fp2 a, Fp2 b) { return {a.c0 + b.c0, a.c1 + b.c1}; }
  friend inline Fp2 operator-(Fp2 a, Fp2 b) { return {a.c0 - b.c0, a.c1 - b.c1}; }
  // Karatsuba: 3 M.
  friend inline Fp2 operator*(Fp2 a, Fp2 b) {
    Fp t0 = a.c0 * b.c0;
    Fp t1 = a.c1 * b.c1;
    Fp t2 = (a.c0 + a.c1) * (b.c0 + b.c1);
    return {t0 - t1, t2 - t0 - t1};
  }
  // (c0 + c1)(c0 - c1), 2 c0 c1: 2 M.
  inline Fp2 sqr() const {
    Fp t = c0 * c1;
    return {(c0 + c1) * (c0 - c1), t + t};
  }
  inline Fp2 neg() const { return {c0.neg(), c1.neg()}; }
  inline Fp2 dbl() const { return *this + *this; }
  // 1 / (c0 + c1 u) = (c0 - c1 u) / (c0^2 + c1^2); zero maps to zero.
  inline Fp2 inv() const {
    Fp n = (c0.sqr() + c1.sqr()).inv();
    return {c0 * n, (c1 * n).neg()};
  }
  thread Fp2& operator+=(Fp2 o) { return *this = *this + o; }
  thread Fp2& operator-=(Fp2 o) { return *this = *this - o; }
  thread Fp2& operator*=(Fp2 o) { return *this = *this * o; }
};

// ----------------------------------------------------------------------------------------------
// Points on y^2 = x^3 + b (a = 0) over F (Fp for G1, Fp2 for G2). b is not needed by the formulas.
// ----------------------------------------------------------------------------------------------

template <class F> struct Affine {
  F x, y;
  inline bool is_inf() const { return x.is_zero() && y.is_zero(); }
  static inline Affine inf() { return {F::zero(), F::zero()}; }
  inline Affine neg() const { return is_inf() ? *this : Affine{x, y.neg()}; }
};

template <class F> struct Jacobian {
  F x, y, z;
  inline bool is_inf() const { return z.is_zero(); }
  static inline Jacobian inf() { return {F::zero(), F::zero(), F::zero()}; }
};

template <class F> struct Xyzz {
  F x, y, zz, zzz;

  static inline Xyzz inf() { return {F::zero(), F::zero(), F::zero(), F::zero()}; }
  inline bool is_inf() const { return zz.is_zero(); }

  static inline Xyzz from_affine(Affine<F> p) {
    if (p.is_inf())
      return inf();
    return {p.x, p.y, F::one(), F::one()};
  }
  // 1 S + 1 M.
  static inline Xyzz from_jacobian(Jacobian<F> p) {
    if (p.z.is_zero())
      return inf();
    F z2 = p.z.sqr();
    return {p.x, p.y, z2, z2 * p.z};
  }
  // (X ZZ^2, Y ZZZ^2, ZZZ): Z = ZZZ / ZZ would need an inversion, but with ZZ^3 = ZZZ^2,
  // Z' = ZZZ gives Z'^2 = ZZ^3 and Z'^3 = ZZZ^3, so x = X ZZ^2 / ZZ^3 and y = Y ZZZ^2 / ZZZ^3.
  // 2 S + 2 M. Infinity maps to (0, 0, 0).
  inline Jacobian<F> to_jacobian() const {
    if (is_inf())
      return Jacobian<F>::inf();
    return {x * zz.sqr(), y * zzz.sqr(), zzz};
  }
  // One inversion: t = 1/(ZZ ZZZ), 1/ZZ = t ZZZ, 1/ZZZ = t ZZ. Infinity maps to (0, 0).
  inline Affine<F> to_affine() const {
    if (is_inf())
      return Affine<F>::inf();
    F t = (zz * zzz).inv();
    return {x * (t * zzz), y * (t * zz)};
  }

  // 2 * self (dbl-2008-s-1, a = 0): 6 M + 3 S.
  BN254_OUTLINE Xyzz dbl() const {
    if (is_inf() || y.is_zero())
      return inf();
    F u = y.dbl();
    F v = u.sqr();
    F w = u * v;
    F s = x * v;
    F x2 = x.sqr();
    F m = x2.dbl() + x2;
    F x3 = m.sqr() - s.dbl();
    F y3 = m * (s - x3) - w * y;
    return {x3, y3, v * zz, w * zzz};
  }

  // 2 * q for affine q (mdbl-2008-s-1), inline: it is a branch of add_affine.
  static inline Xyzz dbl_affine(Affine<F> q) {
    if (q.y.is_zero()) // includes infinity (0, 0)
      return inf();
    F u = q.y.dbl();
    F v = u.sqr();
    F w = u * v;
    F s = q.x * v;
    F x2 = q.x.sqr();
    F m = x2.dbl() + x2;
    F x3 = m.sqr() - s.dbl();
    F y3 = m * (s - x3) - w * q.y;
    return {x3, y3, v, w};
  }

  // self + q for affine q (madd-2008-s): 8 M + 2 S on the generic path.
  inline Xyzz add_affine(Affine<F> q) const {
    if (q.is_inf())
      return *this;
    if (is_inf())
      return from_affine(q);
    F u2 = q.x * zz;
    F s2 = q.y * zzz;
    F p = u2 - x;
    F r = s2 - y;
    if (p.is_zero()) {
      if (r.is_zero())
        return dbl_affine(q);
      return inf();
    }
    F pp = p.sqr();
    F ppp = p * pp;
    F qq = x * pp;
    F x3 = r.sqr() - ppp - qq.dbl();
    F y3 = r * (qq - x3) - y * ppp;
    return {x3, y3, zz * pp, zzz * ppp};
  }

  // self + q (add-2008-s): 12 M + 2 S on the generic path.
  BN254_OUTLINE Xyzz add(Xyzz q) const {
    if (q.is_inf())
      return *this;
    if (is_inf())
      return q;
    F u1 = x * q.zz;
    F u2 = q.x * zz;
    F s1 = y * q.zzz;
    F s2 = q.y * zzz;
    F p = u2 - u1;
    F r = s2 - s1;
    if (p.is_zero()) {
      if (r.is_zero())
        return dbl();
      return inf();
    }
    F pp = p.sqr();
    F ppp = p * pp;
    F qq = u1 * pp;
    F x3 = r.sqr() - ppp - qq.dbl();
    F y3 = r * (qq - x3) - s1 * ppp;
    return {x3, y3, zz * q.zz * pp, zzz * q.zzz * ppp};
  }

  inline Xyzz neg() const { return {x, y.neg(), zz, zzz}; }

  // k * self for a 32-bit multiplier (MSM bucket/window reduction).
  BN254_OUTLINE Xyzz mul_small(uint32_t k) const {
    Xyzz acc = inf();
    for (int i = 31; i >= 0; i--) {
      acc = acc.dbl();
      if ((k >> i) & 1)
        acc = acc.add(*this);
    }
    return acc;
  }
};

using G1Affine = Affine<Fp>;
using G2Affine = Affine<Fp2>;
using G1Jacobian = Jacobian<Fp>;
using G2Jacobian = Jacobian<Fp2>;
using G1Xyzz = Xyzz<Fp>;
using G2Xyzz = Xyzz<Fp2>;

} // namespace bn254
