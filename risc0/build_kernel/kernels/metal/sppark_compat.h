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
/// sppark's baby-bear names mapped onto fp.h/fpext.h, for ports of the supra/sppark kernels
/// (Metal port plan section 2.2). sppark's member functions become free functions, since a type
/// alias can't add members:
///   `x.zero()`              -> `zero(x)`
///   `T::csel(a, b, sel_a)`  -> `csel(a, b, sel_a)`
///   `x.reciprocal()`        -> `reciprocal(x)`
///   `x ^ p`                 -> unchanged (free `operator^`)

#include <metal_stdlib>

#include "fp.h"
#include "fpext.h"

typedef Fp bb31_t;
typedef Fp fr_t;
typedef FpExt bb31_4_t;
typedef FpExt fr4_t;
typedef FpExt md4_t;
typedef uint index_t;

/// Set `x` to zero (sppark `x.zero()`).
inline void zero(thread Fp& x) {
  x = Fp(0);
}
inline void zero(threadgroup Fp& x) {
  x = Fp(0);
}
inline void zero(device Fp& x) {
  x = Fp(0);
}
inline void zero(thread FpExt& x) {
  x = FpExt();
}
inline void zero(threadgroup FpExt& x) {
  x = FpExt();
}
inline void zero(device FpExt& x) {
  x = FpExt();
}

/// `sel_a ? a : b` (sppark `T::csel(a, b, sel_a)`).
inline Fp csel(Fp a, Fp b, int sel_a) {
  return sel_a ? a : b;
}
inline FpExt csel(FpExt a, FpExt b, int sel_a) {
  return sel_a ? a : b;
}

/// `x` raised to the power `p` (sppark `x ^ p`).
inline Fp operator^(Fp x, unsigned p) {
  return pow(x, size_t(p));
}
inline FpExt operator^(FpExt x, unsigned p) {
  return pow(x, size_t(p));
}

/// Multiplicative inverse, 0 for 0 (sppark `x.reciprocal()`).
inline Fp reciprocal(Fp x) {
  return inv(x);
}
inline FpExt reciprocal(FpExt x) {
  return inv(x);
}
