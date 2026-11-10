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

// Twiddle / coset table generators of the Metal Groth16 prover (item G4). Port of the Intel
// prover's `powers` and the coset `*= n^-1` pass (kernels/intel/groth16_intel.hpp, NttPlan), which
// replace the CUDA prover's generate_partial_twiddles. Values are bn254::Fr (Montgomery form).
//
// Buffer indices (mirrored by src/metal/small.rs):
//
// fr_powers   0 out Fr[n]   1 constant FrPowersParams
//   out[j] = scale * base^j for j < n, from the host-built table sq[k] = base^(2^k), k < 32:
//   r = scale; for each set bit k of j: r *= sq[k]   (at most 32 multiplications, j < 2^32).
// fr_scale    0 out Fr[n]   1 in Fr[n]   2 constant FrScaleParams
//   out[i] = in[i] * scale                     (out may alias in)
//
// Field elements inside the constant params are raw uint32 words (8 per element, Montgomery
// form), loaded with Fr::from_limbs: a `constant` Fr cannot be used directly (plan 2.1).

#include "bn254.h"

using namespace bn254;

struct FrPowersParams {
  uint32_t sq[32][8]; // base^(2^k), k = 0..31
  uint32_t scale[8];
  uint32_t count;
  uint32_t _pad[3];
};

struct FrScaleParams {
  uint32_t scale[8];
  uint32_t count;
  uint32_t _pad[3];
};

[[max_total_threads_per_threadgroup(256)]]
kernel void fr_powers(device Fr* out [[buffer(0)]],
                      constant FrPowersParams& p [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count)
    return;
  Fr r = Fr::from_limbs(p.scale);
  uint32_t j = gid;
  for (int k = 0; j != 0; k++, j >>= 1) {
    if (j & 1u)
      r *= Fr::from_limbs(p.sq[k]);
  }
  out[gid] = r;
}

[[max_total_threads_per_threadgroup(256)]]
kernel void fr_scale(device Fr* out [[buffer(0)]],
                     const device Fr* in [[buffer(1)]],
                     constant FrScaleParams& p [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count)
    return;
  Fr x = in[gid];
  out[gid] = x * Fr::from_limbs(p.scale);
}
