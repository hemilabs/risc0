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

// witness_into_poly of the Metal Groth16 prover (item G4). Port of the CUDA prover's
// witness_into_poly (kernels/cuda/groth16_prover.cuh) in the Intel prover's form
// (kernels/intel/groth16_intel.hpp, witness_into_poly): one thread per row i of the preprocessed
// coefficients, summing sequentially (same order as CUDA, so bit-identical)
//
//   out[c] = sum_{j in [indices[i], indices[i+1])} witness[coeffs[j].s] * coeffs[j].value
//
// where c = the `c` field of the row's (last) coefficient; every coefficient of a row has the same
// `c`. Rows with no coefficients write nothing (CUDA would write to an uninitialised index; the
// preprocessed files never contain one). Rows that are not written keep their previous value: the
// host zero-fills `out` first.
//
// Untrusted input (the coefficient file) is checked, never trusted for addressing:
//   fault |= 1  indices not monotone or past the coefficient array (row skipped)
//   fault |= 2  a coefficient's s >= witness_len or c >= out_len (row skipped)
//   fault |= 4  a non-canonical coefficient value or witness element (row computed anyway)
//
// Buffer indices (mirrored by src/metal/small.rs):
//   0 out Fr[out_len]   1 coeffs Coeff[coeff_len]   2 indices uint[rows + 1]
//   3 witness Fr[witness_len]   4 fault atomic_uint   5 constant WitnessPolyParams

#include "bn254.h"

using namespace metal;
using namespace bn254;

// sppark's coeff_t / the Intel prover's Coeff: 48 bytes (fr_t is 8-byte aligned in sppark).
struct Coeff {
  uint32_t m, c, s, pad;
  Fr value;
};
static_assert(sizeof(Coeff) == 48, "Coeff must match sppark's coeff_t");

struct WitnessPolyParams {
  uint32_t rows; // indices has rows + 1 entries
  uint32_t coeff_len;
  uint32_t witness_len;
  uint32_t out_len;
};

[[max_total_threads_per_threadgroup(256)]]
kernel void witness_into_poly(device Fr* out [[buffer(0)]],
                              const device Coeff* coeffs [[buffer(1)]],
                              const device uint32_t* indices [[buffer(2)]],
                              const device Fr* witness [[buffer(3)]],
                              device atomic_uint* fault [[buffer(4)]],
                              constant WitnessPolyParams& p [[buffer(5)]],
                              uint gid [[thread_position_in_grid]]) {
  if (gid >= p.rows)
    return;
  uint32_t start = indices[gid];
  uint32_t end = indices[gid + 1];
  if (end < start || end > p.coeff_len) {
    atomic_fetch_or_explicit(fault, 1u, memory_order_relaxed);
    return;
  }
  if (end == start)
    return;
  uint32_t row = coeffs[start].c;
  if (row >= p.out_len) {
    atomic_fetch_or_explicit(fault, 2u, memory_order_relaxed);
    return;
  }
  uint32_t bad = 0;
  Fr sum = Fr::zero();
  for (uint32_t j = start; j < end; j++) {
    uint32_t s = coeffs[j].s;
    row = coeffs[j].c;
    if (s >= p.witness_len || row >= p.out_len) {
      atomic_fetch_or_explicit(fault, 2u, memory_order_relaxed);
      return;
    }
    Fr v = coeffs[j].value;
    Fr w = witness[s];
    if (!v.is_canonical() || !w.is_canonical())
      bad = 4u;
    sum += w * v;
  }
  if (bad)
    atomic_fetch_or_explicit(fault, bad, memory_order_relaxed);
  out[row] = sum;
}
