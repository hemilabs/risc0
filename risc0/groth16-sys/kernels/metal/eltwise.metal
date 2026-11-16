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

// Elementwise Fr kernels of the Metal Groth16 prover (item G4), ports of the CUDA prover's
// coeff_wise_mul / coeff_wise_mul_and_sub / coeff_wise_add (kernels/cuda/util.cuh) and of the
// Intel prover's h = (a * b - c).from_mont() step (kernels/intel/groth16_intel.hpp, prove()).
// All values are bn254::Fr (8 x 32-bit limbs, Montgomery form, canonical). One thread per element
// (the CUDA grid-stride loop becomes a 1-D dispatch_threads grid, bounds-checked).
//
// Buffer indices (mirrored by src/metal/small.rs):
//
// fr_mul                 0 out Fr[n]    1 a Fr[n]   2 b Fr[n]   3 constant FrEltParams
//   out[i] = a[i] * b[i]                         (out may alias a or b)
// fr_mul_sub_from_mont   0 inout Fr[n]  1 b Fr[n]   2 c Fr[n]   3 constant FrEltParams
//   inout[i] = (inout[i] * b[i] - c[i]).from_mont()
// fr_add                 0 inout Fr[n]  1 b Fr[n]               2 constant FrEltParams
//   inout[i] = inout[i] + b[i]

#include "bn254.h"

using namespace bn254;

struct FrEltParams {
  uint32_t count;
  uint32_t _pad[3];
};

[[max_total_threads_per_threadgroup(256)]]
kernel void fr_mul(device Fr* out [[buffer(0)]],
                   const device Fr* a [[buffer(1)]],
                   const device Fr* b [[buffer(2)]],
                   constant FrEltParams& p [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count)
    return;
  Fr x = a[gid];
  Fr y = b[gid];
  out[gid] = x * y;
}

[[max_total_threads_per_threadgroup(256)]]
kernel void fr_mul_sub_from_mont(device Fr* inout [[buffer(0)]],
                                 const device Fr* b [[buffer(1)]],
                                 const device Fr* c [[buffer(2)]],
                                 constant FrEltParams& p [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count)
    return;
  Fr x = inout[gid];
  Fr y = b[gid];
  Fr z = c[gid];
  inout[gid] = (x * y - z).from_mont();
}

[[max_total_threads_per_threadgroup(256)]]
kernel void fr_add(device Fr* inout [[buffer(0)]],
                   const device Fr* b [[buffer(1)]],
                   constant FrEltParams& p [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count)
    return;
  Fr x = inout[gid];
  Fr y = b[gid];
  inout[gid] = x + y;
}
