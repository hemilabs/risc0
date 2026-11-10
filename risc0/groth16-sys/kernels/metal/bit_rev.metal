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

// In-place bit-reversal permutation of BN254 Fr vectors (item G5, driven by src/metal/ntt.rs).
// Port of the Intel prover's NttPlan::bit_reverse (kernels/intel/groth16_intel.hpp), which
// replaces sppark's bit_rev permutation for the radix-2 NTT.
//
// Buffer indices (mirrored by `buf::BIT_REV_*` in src/metal/ntt.rs):
//
// bit_reverse   0 x Fr[2^lg] (in place)   1 constant NttBitRevParams
//   thread i < 2^lg swaps x[i] and x[rev_lg(i)] when i < rev_lg(i).

#include "bn254.h"

using namespace metal;
using namespace bn254;

struct NttBitRevParams {
  uint32_t lg;
  uint32_t _pad[3];
};

[[max_total_threads_per_threadgroup(256)]]
kernel void bit_reverse(device Fr* x [[buffer(0)]],
                        constant NttBitRevParams& p [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
  if (p.lg == 0 || p.lg > 31 || gid >= (1u << p.lg))
    return;
  uint32_t j = reverse_bits(gid) >> (32 - p.lg);
  if (gid < j) {
    Fr t = x[gid];
    x[gid] = x[j];
    x[j] = t;
  }
}
