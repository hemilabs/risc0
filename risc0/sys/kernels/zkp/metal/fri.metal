// Copyright 2022 Risc0, Inc.
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

// R17b (docs/metal-port/notes/R17.md, "R17b hardening"): FpExt * FpExt is always_inline in this
// TU. Declared before fpext.h so that the friend definition there inherits the attribute. At -O2
// the compiler otherwise calls it out of line inside this file's FpExt loops (ir_scan.py tier A:
// a default-constructed private FpExt / FpExt array written after the call and read after the
// loop, the Apple GPU lost-store shape). Same arithmetic, bit-exact.
#include "fp.h"
struct FpExt;
__attribute__((always_inline)) constexpr FpExt operator*(FpExt lhs, FpExt rhs);
#include "fp.h"
#include "fpext.h"

using namespace metal;

constant size_t kFriFold = 16;

/// Compute `ceil(log_2(in))`, i.e. find the smallest value `out` such that `2^out >= in`.
inline constexpr size_t log2Ceil(size_t in) {
  size_t r = 0;
  while ((1 << r) < in) {
    r++;
  }
  return r;
}

// Bindings: 0 out (Fp[4 * count]), 1 in (Fp[4 * count * kFriFold]), 2 mix (FpExt as four raw
// Montgomery words, set_bytes), 3 count (set_bytes). Scalars are `constant` (plan 2.1: set_bytes
// data is read-only); the FpExt is rebuilt in a thread local before any arithmetic, since
// fpext.h has no `constant` operators.
kernel void fri_fold(device Fp* out [[buffer(0)]],
                     const device Fp* in [[buffer(1)]],
                     constant uint32_t* mixRaw [[buffer(2)]],
                     constant uint32_t& countRef [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
  const uint32_t count = countRef;
  if (gid >= count) {
    return;
  }
  const FpExt mix(Fp::fromRaw(mixRaw[0]),
                  Fp::fromRaw(mixRaw[1]),
                  Fp::fromRaw(mixRaw[2]),
                  Fp::fromRaw(mixRaw[3]));
  FpExt tot;
  FpExt curMix(1);
  for (uint32_t i = 0; i < kFriFold; i++) {
    size_t rev_i = reverse_bits(i) >> (32 - log2Ceil(kFriFold));
    size_t rev_idx = rev_i * count + gid;
    FpExt factor(in[0 * count * kFriFold + rev_idx],
                 in[1 * count * kFriFold + rev_idx],
                 in[2 * count * kFriFold + rev_idx],
                 in[3 * count * kFriFold + rev_idx]);
    tot += curMix * factor;
    curMix = curMix * mix;
  }
  for (size_t i = 0; i < 4; i++) {
    out[count * i + gid] = tot.elems[i];
  }
}
