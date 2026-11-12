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
/// Lazy-reduction FpExt multiply and the addition-chain FpExt inverse (port of the
/// EF_EXT_LAZY device path of kernels/cuda/ef_fpext.h and `inv_ext_chain` of
/// kernels/cuda/witgen.h). As with ef_fp.h, fp.h/fpext.h stay the only field types (plan 2.2):
/// these are free functions over `FpExt`, bit-identical to fpext.h's operator* / inv for
/// canonical inputs. Generated evalfast code calls `ef_ext_mul(a, b)` where CUDA relied on
/// EF_EXT_LAZY changing `operator*`.
///
/// `ef_ext_mul`: fold NBETA into b (3 Montgomery muls), then each output coefficient is a 4-term
/// dot product of raw Montgomery words summed exactly (64 bits as lo/hi 32-bit words with
/// explicit carries, CUDA's RISC0_INTEL_EVAL_LOHI form, which needs no 64-bit integer ops) and
/// reduced once by `ef_redc_lohi`. Bounds: each product < P^2, a sum < 4P^2 < 2^64.
/// Defining EF_WIDE accumulates in `ulong` instead (CUDA's IMAD.WIDE form; emulated on Apple
/// GPUs, kept selectable for the R15 A/B per plan 2.4).

#include "ef_fp.h"

// R17b: always_inline in witgen_tu.metal (witgen.h, RV32IM_WITGEN_EXT_INLINE).
#ifndef RV32IM_WITGEN_EXT_INLINE
#define RV32IM_WITGEN_EXT_INLINE
#endif
// R17: a TU may substitute its own copy of fpext.h (witgen_tu.metal: witgen_fpext.h, the shared
// header with its 4-element loops unrolled; kernels/cuda/witgen_tu_gen.py gen_fpext).
#ifdef RV32IM_FPEXT_HEADER
#include RV32IM_FPEXT_HEADER
#else
#include "fpext.h"
#endif

/// Exact sum x0*y0 + x1*y1 + x2*y2 + x3*y3 of canonical raw words, reduced to an Fp.
inline Fp ef_redc4(uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1,
                   uint32_t x2, uint32_t y2, uint32_t x3, uint32_t y3) {
#ifdef EF_WIDE
  ulong t = ulong(x0) * y0 + ulong(x1) * y1 + ulong(x2) * y2 + ulong(x3) * y3;
  return ef_redc_lohi(uint32_t(t), uint32_t(t >> 32));
#else
  uint32_t lo = x0 * y0, hi = ef_mulhi(x0, y0);
  uint32_t p = x1 * y1, n = lo + p;
  hi += ef_mulhi(x1, y1) + (n < p ? 1u : 0u);
  lo = n;
  p = x2 * y2;
  n = lo + p;
  hi += ef_mulhi(x2, y2) + (n < p ? 1u : 0u);
  lo = n;
  p = x3 * y3;
  n = lo + p;
  hi += ef_mulhi(x3, y3) + (n < p ? 1u : 0u);
  lo = n;
  return ef_redc_lohi(lo, hi);
#endif
}

/// a * b in F_p[X]/(X^4 - 11), lazy reduction (CUDA ef_fpext.h EF_EXT_LAZY).
RV32IM_WITGEN_EXT_INLINE inline FpExt ef_ext_mul(FpExt lhs, FpExt rhs) {
  const Fp nbeta = Fp(Fp::P - 11);
  const uint32_t a0 = lhs.elems[0].asRaw(), a1 = lhs.elems[1].asRaw();
  const uint32_t a2 = lhs.elems[2].asRaw(), a3 = lhs.elems[3].asRaw();
  const uint32_t b0 = rhs.elems[0].asRaw(), b1 = rhs.elems[1].asRaw();
  const uint32_t b2 = rhs.elems[2].asRaw(), b3 = rhs.elems[3].asRaw();
  const uint32_t b1n = (rhs.elems[1] * nbeta).asRaw();
  const uint32_t b2n = (rhs.elems[2] * nbeta).asRaw();
  const uint32_t b3n = (rhs.elems[3] * nbeta).asRaw();
  return FpExt(ef_redc4(a0, b0, a1, b3n, a2, b2n, a3, b1n),
               ef_redc4(a0, b1, a1, b0, a2, b3n, a3, b2n),
               ef_redc4(a0, b2, a1, b1, a2, b0, a3, b3n),
               ef_redc4(a0, b3, a1, b2, a2, b1, a3, b0));
}

/// FpExt inverse through `inv_chain` (same value as fpext.h's `inv`, including inv(0) = 0).
RV32IM_WITGEN_EXT_INLINE inline FpExt inv_ext_chain(FpExt in) {
  thread const Fp* a = in.elems;
  const Fp beta(11), nbeta(Fp::P - 11);
  Fp b0 = a[0] * a[0] + beta * (a[1] * (a[3] + a[3]) - a[2] * a[2]);
  Fp b2 = a[0] * (a[2] + a[2]) - a[1] * a[1] + beta * (a[3] * a[3]);
  Fp c = b0 * b0 + beta * b2 * b2;
  Fp ic = inv_chain(c);
  b0 *= ic;
  b2 *= ic;
  return FpExt(a[0] * b0 + beta * a[2] * b2,
               -a[1] * b0 + nbeta * a[3] * b2,
               -a[0] * b2 + a[2] * b0,
               a[1] * b2 - a[3] * b0);
}
