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

// Poseidon2 (BabyBear, width 24) row hashing and folding with lazy Montgomery reductions:
// fewer instructions than poseidon2.cuh for the same results (same I/O: Montgomery-form u32).
// CUDA uses PTX carry chains; ROCm the same arithmetic in 64-bit C++, which wraps identically.
//
// Value bounds, checked exhaustively over all canonical inputs (p/2^32 = 0.46875):
// - mred(t) = t * 2^-32 mod p, in [0, t/2^32 + p); exact (congruent) whenever the high word
//   of t is at most 0x87fffffe (~1.133p), e.g. for t < 2^32 * p.
// - sbox: x^2 < 1.469p, x^3 < 1.689p before its reduction, x^4 < 2.011p, x^7 < 1.915p before
//   its reduction: every product stays in mred's domain.
// - Partial rounds keep the cells below 1.87p (at most 1.8693p over all DH_i, x in [0, 2p)):
//   cells < 2p hold for any number of rounds, by induction.

#pragma once

#include <cstdint>

#include "pv_consts.h"

namespace pv {

constexpr uint32_t P = 0x78000001u, M0 = 0x77ffffffu; // M0 = -p^-1 mod 2^32

__device__ __forceinline__ uint32_t red1(uint32_t r) { // [0, 2p) -> [0, p)
  return r >= P ? r - P : r;
}

__device__ __forceinline__ uint32_t addm(uint32_t a, uint32_t b) {
  return red1(a + b);
}

// Montgomery reduction (see the bounds above).
__device__ __forceinline__ uint32_t mred(uint64_t t) {
#ifdef __HIPCC__
  // The high word of t + m*p, modulo 2^64 like the PTX carry chain below.
  uint32_t m = uint32_t(t) * M0;
  return uint32_t((t + uint64_t(m) * P) >> 32);
#else
  uint32_t r;
  asm("{\n\t.reg .u32 m, lo, hi, d;\n\tmov.b64 {lo,hi}, %1;\n\tmul.lo.u32 m, lo, %2;\n\t"
      "mad.lo.cc.u32 d, m, %3, lo;\n\tmadc.hi.u32 %0, m, %3, hi;\n\t}"
      : "=r"(r)
      : "l"(t), "n"(M0), "n"(P));
  return r;
#endif
}

__device__ __forceinline__ uint32_t mmul_lazy(uint32_t a, uint32_t b) {
  return mred((uint64_t)a * b);
}

__device__ __forceinline__ uint32_t mmul(uint32_t a, uint32_t b) {
  return red1(mmul_lazy(a, b));
}

// x canonical -> x^7 canonical.
__device__ __forceinline__ uint32_t sbox(uint32_t x) {
  uint32_t x2 = mmul_lazy(x, x);
  uint32_t x3 = mmul(x2, x);
  uint32_t x4 = mmul_lazy(x2, x2);
  return mmul(x3, x4);
}

__device__ __forceinline__ void m4(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
  uint32_t t0 = addm(a, b), t1 = addm(c, d);
  uint32_t t2 = addm(addm(b, b), t1), t3 = addm(addm(d, d), t0);
  uint32_t t1_2 = addm(t1, t1), t0_2 = addm(t0, t0);
  uint32_t t4 = addm(addm(t1_2, t1_2), t3), t5 = addm(addm(t0_2, t0_2), t2);
  a = addm(t3, t5);
  b = t5;
  c = addm(t2, t4);
  d = t4;
}

// External (full-round) linear layer.
__device__ __forceinline__ void m_ext(uint32_t* x) {
  uint32_t s[4] = {0, 0, 0, 0};
#pragma unroll
  for (int g = 0; g < 6; g++) {
    m4(x[4 * g], x[4 * g + 1], x[4 * g + 2], x[4 * g + 3]);
#pragma unroll
    for (int j = 0; j < 4; j++)
      s[j] = g ? addm(s[j], x[4 * g + j]) : x[4 * g + j];
  }
#pragma unroll
  for (int i = 0; i < 24; i++)
    x[i] = addm(x[i], s[i & 3]);
}

__device__ __forceinline__ void full_round(uint32_t* x, int r) {
#pragma unroll
  for (int i = 0; i < 24; i++)
    x[i] = sbox(addm(x[i], PV_RCF[r * 24 + i]));
  m_ext(x);
}

// The 21 partial rounds, with the cells kept in [0, 1.87p) between rounds (no per-cell
// reduction): e'_i = sum_j e_j + DH_i * e_i * 2^-32, computed as mred(DH_i * e_i + w) with
// w = mred(mred(S) * R^3) == S * 2^32 mod p, S = sum_j e_j.
__device__ __forceinline__ void partial_rounds(uint32_t* x) {
#pragma unroll 1
  for (int r = 0; r < 21; r++) {
    x[0] = sbox(addm(red1(x[0]), PV_RCP[r]));
#ifdef __HIPCC__
    uint64_t sum = x[0];
#pragma unroll
    for (int i = 1; i < 24; i++)
      sum += x[i];
    uint32_t st = mred(sum); // < p + 22
#else
    uint32_t lo, hi;
    asm("add.cc.u32 %0, %2, %3; addc.u32 %1, 0, 0;" : "=r"(lo), "=r"(hi) : "r"(x[0]), "r"(x[1]));
#pragma unroll
    for (int i = 2; i < 24; i++)
      asm("add.cc.u32 %0, %0, %2; addc.u32 %1, %1, 0;" : "+r"(lo), "+r"(hi) : "r"(x[i]));
    uint32_t st = mred(((uint64_t)hi << 32) | lo); // < p + 22
#endif
    uint32_t w = mred((uint64_t)st * 0x12f37bfbu); // == S * 2^32 mod p, < 1.074p
#pragma unroll
    for (int i = 0; i < 24; i++)
      x[i] = mred((uint64_t)PV_DH[i] * x[i] + w);
  }
#pragma unroll
  for (int i = 0; i < 24; i++)
    x[i] = red1(x[i]);
}

__device__ __forceinline__ void permute(uint32_t* x) {
  m_ext(x);
#pragma unroll 1
  for (int r = 0; r < 4; r++)
    full_round(x, r);
  partial_rounds(x);
#pragma unroll 1
  for (int r = 4; r < 8; r++)
    full_round(x, r);
}

// out[i] = hash of the two digests in[2i], in[2i + 1] (n outputs).
__global__ __launch_bounds__(128, 6) void fold_k(uint4* out, const uint4* in, uint32_t n) {
  uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
  if (gid >= n)
    return;
  uint32_t x[24];
#pragma unroll
  for (int q = 0; q < 4; q++) {
    uint4 v = in[(size_t)gid * 4 + q];
    x[4 * q] = v.x;
    x[4 * q + 1] = v.y;
    x[4 * q + 2] = v.z;
    x[4 * q + 3] = v.w;
  }
#pragma unroll
  for (int i = 16; i < 24; i++)
    x[i] = 0;
  permute(x);
  out[(size_t)gid * 2] = make_uint4(x[0], x[1], x[2], x[3]);
  out[(size_t)gid * 2 + 1] = make_uint4(x[4], x[5], x[6], x[7]);
}

// out[r] = hash of row r of the column-major dim_x x dim_y matrix m (16 elements per absorb).
__global__ __launch_bounds__(128, 6) void rows_k(uint4* out,
                                                 const uint32_t* __restrict__ m,
                                                 uint32_t dim_x,
                                                 uint32_t dim_y) {
  uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
  if (gid >= dim_x)
    return;
  uint32_t x[24];
#pragma unroll
  for (int i = 0; i < 24; i++)
    x[i] = 0;
  m += gid;
  uint32_t i = 0;
  do {
#pragma unroll
    for (int j = 0; j < 16; j++, i++)
      x[j] = i < dim_y ? __ldg(m + (size_t)i * dim_x) : 0u;
    permute(x);
  } while (i < dim_y);
  out[(size_t)gid * 2] = make_uint4(x[0], x[1], x[2], x[3]);
  out[(size_t)gid * 2 + 1] = make_uint4(x[4], x[5], x[6], x[7]);
}

} // namespace pv
