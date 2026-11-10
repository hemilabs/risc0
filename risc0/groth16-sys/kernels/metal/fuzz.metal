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

// chacha_fuzz of the Metal Groth16 prover (item G4): the CUDA prover's
// chacha_generate_random_scalars<8> (kernels/cuda/util.cuh, chacha.cuh) bit for bit, as the Intel
// prover's generate_fuzz (kernels/intel/groth16_intel.hpp).
//
// Block i (0 <= i < blocks) is one ChaCha8 block of the initial state (constants, key, word 12 =
// counter, 13..15 = nonce) with the 64-bit counter (word 12, carry into 13) advanced by i. Its 16
// output words are two Fr values, v0 = words 0..7 and v1 = words 8..15, each with
// l[7] &= 0x1FFFFFFF (raw limbs, < 2^253, so canonical). They land at the CUDA kernel's
// warp-interleaved positions (WARP_SZ = 32, independent of the Metal SIMD width):
//
//   lane = i % 32, wid = i / 32, idx = lane + wid * 64
//   out[idx] = v0;  out[idx + 32] = v1 if idx + 32 < 2 * blocks.
//
// Slots of [0, 2 * blocks) that no block writes (the second half of a partial last warp) are zero
// (CUDA bzeroes the buffer first): the dispatch covers blocks rounded up to a multiple of 32, and
// threads i >= blocks write zeros at their two positions, so no prior fill is needed.
//
// Buffer indices (mirrored by src/metal/small.rs):
//   0 out Fr[2 * blocks]   1 constant ChachaFuzzParams   (grid: round_up(blocks, 32) threads)

#include "bn254.h"

using namespace bn254;

struct ChachaFuzzParams {
  uint32_t state[16];
  uint32_t blocks;
  uint32_t threads; // blocks rounded up to a multiple of 32
  uint32_t _pad[2];
};

static inline uint32_t rotl(uint32_t v, uint32_t s) { return (v << s) | (v >> (32 - s)); }

#define QR(a, b, c, d)                                                                             \
  a += b;                                                                                          \
  d = rotl(d ^ a, 16);                                                                             \
  c += d;                                                                                          \
  b = rotl(b ^ c, 12);                                                                             \
  a += b;                                                                                          \
  d = rotl(d ^ a, 8);                                                                              \
  c += d;                                                                                          \
  b = rotl(b ^ c, 7);

[[max_total_threads_per_threadgroup(256)]]
kernel void chacha_fuzz(device Fr* out [[buffer(0)]],
                        constant ChachaFuzzParams& p [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid >= p.threads)
    return;
  uint32_t lane = gid % 32u;
  uint32_t wid = gid / 32u;
  uint32_t idx = lane + wid * 64u;
  uint32_t total = p.blocks * 2u;
  if (gid >= p.blocks) {
    if (idx < total)
      out[idx] = Fr::zero();
    if (idx + 32u < total)
      out[idx + 32u] = Fr::zero();
    return;
  }

  uint32_t x[16];
  for (int k = 0; k < 16; k++)
    x[k] = p.state[k];
  // chacha_mix: counter += i (64-bit, words 12/13); i < 2^32 here.
  x[12] += gid;
  if (x[12] < gid)
    x[13]++;
  uint32_t t[16];
  for (int k = 0; k < 16; k++)
    t[k] = x[k];
  for (int r = 0; r < 4; r++) { // ChaCha8: 4 double rounds
    QR(x[0], x[4], x[8], x[12]);
    QR(x[1], x[5], x[9], x[13]);
    QR(x[2], x[6], x[10], x[14]);
    QR(x[3], x[7], x[11], x[15]);
    QR(x[0], x[5], x[10], x[15]);
    QR(x[1], x[6], x[11], x[12]);
    QR(x[2], x[7], x[8], x[13]);
    QR(x[3], x[4], x[9], x[14]);
  }
  for (int k = 0; k < 16; k++)
    x[k] += t[k];
  x[7] &= 0x1FFFFFFFu;
  x[15] &= 0x1FFFFFFFu;

  Fr v0 = Fr::from_limbs(&x[0]);
  out[idx] = v0;
  if (idx + 32u < total) {
    Fr v1 = Fr::from_limbs(&x[8]);
    out[idx + 32u] = v1;
  }
}
