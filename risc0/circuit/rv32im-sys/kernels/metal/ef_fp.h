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

#pragma once

/// \file
/// Fast Fp forms for the evalfast eval_check and the witgen inverse (port of the device paths of
/// kernels/cuda/ef_fp.h and the `inv_chain` of kernels/cuda/witgen.h / intel/witgen.h).
///
/// CUDA's ef_fp.h is a second copy of the Fp class whose operators switch to faster forms under
/// EF_FP_FAST. On Metal only build_kernel's fp.h defines Fp (plan 2.2), so this header adds the
/// same forms as free functions over fp.h's `Fp` / raw Montgomery words instead. Every function is
/// bit-identical to fp.h's operator for canonical inputs (< P), which is all fp.h produces.
/// Translation rules for generated code (R12's evalfast `to_metal.py`):
///   `Fp::mulM(x)`                       -> `ef_mulM(x)`
///   `a + b`, `a - b`, `a * b` under EF_FP_FAST -> `ef_add(a, b)`, `ef_sub(a, b)`, `ef_mul(a, b)`
///                                          (or keep fp.h's operators: same bits)
///   `mulhi`/`rk_mulhi` lambdas          -> `ef_mulhi(x, y)` (32-bit `metal::mulhi`)

#include "rv32im_fp.h"

#include <metal_stdlib>

/// High 32 bits of a 32x32-bit product (native 32-bit `mulhi`, ~4.7x cheaper than a 64-bit one).
inline uint32_t ef_mulhi(uint32_t x, uint32_t y) {
  return metal::mulhi(x, y);
}

/// x * M mod 2^32: the Montgomery factor of the lazy reductions (CUDA `Fp::mulM`).
inline uint32_t ef_mulM(uint32_t x) {
  return x * Fp::M;
}

/// Raw a + b (mod P) for canonical a, b: the EF_FP_FAST min() form.
inline uint32_t ef_add_raw(uint32_t a, uint32_t b) {
  uint32_t r = a + b;
  uint32_t s = r - Fp::P;
  return r < s ? r : s;
}

/// Raw a - b (mod P) for canonical a, b: the EF_FP_FAST min() form.
inline uint32_t ef_sub_raw(uint32_t a, uint32_t b) {
  uint32_t r = a - b;
  uint32_t s = r + Fp::P;
  return r < s ? r : s;
}

/// Montgomery a * b for canonical a, b, subtraction-form REDC (EF_FP_FAST): with
/// m = lo * P^-1 mod 2^32 the low words of a*b and m*P cancel, so (a*b - m*P) / 2^32 =
/// hi - mulhi(m, P), which lies in (-P, P); min() maps it into [0, P). 32-bit operations only.
inline uint32_t ef_mul_raw(uint32_t a, uint32_t b) {
  uint32_t lo = a * b;
  uint32_t hi = ef_mulhi(a, b);
  uint32_t m = lo * Fp::M;
  uint32_t r = hi - ef_mulhi(m, Fp::P);
  uint32_t s = r + Fp::P;
  return r < s ? r : s;
}

inline Fp ef_add(Fp a, Fp b) {
  return Fp::fromRaw(ef_add_raw(a.asRaw(), b.asRaw()));
}

inline Fp ef_sub(Fp a, Fp b) {
  return Fp::fromRaw(ef_sub_raw(a.asRaw(), b.asRaw()));
}

inline Fp ef_neg(Fp a) {
  return Fp::fromRaw(ef_sub_raw(0, a.asRaw()));
}

inline Fp ef_mul(Fp a, Fp b) {
  return Fp::fromRaw(ef_mul_raw(a.asRaw(), b.asRaw()));
}

/// Montgomery reduction of an exact 64-bit sum hi:lo < 4P^2 (four products of canonical values):
/// one conditional fold makes hi < P (hi < 4P^2/2^32 < 1.875P), then the subtraction-form REDC.
inline Fp ef_redc_lohi(uint32_t lo, uint32_t hi) {
  uint32_t hf = hi - Fp::P;
  hi = hi < hf ? hi : hf;
  uint32_t r = hi - ef_mulhi(ef_mulM(lo), Fp::P);
  uint32_t t = r + Fp::P;
  return Fp::fromRaw(r < t ? r : t);
}

/// Addition-chain inverse x^(P-2), P-2 = 0x77FFFFFF: 37 mul/sqr instead of fp.h's
/// square-and-multiply `inv` (same value, including inv(0) = 0). From kernels/cuda/witgen.h.
inline Fp inv_chain(Fp x) {
  Fp x2 = x * x;
  Fp x3 = x2 * x;
  Fp x6 = x3 * x3;
  Fp y3 = x6 * x; // x^(2^3-1)
  Fp t = y3;
#pragma unroll
  for (int i = 0; i < 3; i++)
    t = t * t;
  Fp y6 = t * y3; // x^(2^6-1)
  Fp z = y3;
#pragma unroll
  for (int i = 0; i < 4; i++)
    z = z * z;
  z = z * y3; // x^0x77
#pragma unroll
  for (int k = 0; k < 4; k++) {
#pragma unroll
    for (int i = 0; i < 6; i++)
      z = z * z;
    z = z * y6;
  }
  return z;
}
