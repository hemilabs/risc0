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

// NTT v1 over BN254 Fr for the Metal Groth16 prover (item G5, driven by src/metal/ntt.rs).
//
// Ports the Intel prover's radix-2 NttPlan (kernels/intel/groth16_intel.hpp "NTT over Fr") and
// the CUDA prover's LDE sequence (kernels/cuda/groth16_prover.cuh NTT_sequence: sppark inverse
// NTT natural -> bit-reversed, LDE_distribute_powers by shift^k in bit-reversed position,
// forward NTT bit-reversed -> natural) with one twiddle table for both directions.
//
// Notation: n = 2^lg, w = the primitive n-th root (forward), tw[j] = w^j for j < n/2.
// A butterfly "level" q (0 <= q < lg) pairs i0 and i1 = i0 + 2^q with i0 bit q clear; its
// twiddle is w^jj, jj = (i0 mod 2^q) << (lg - 1 - q).
//   DIT (Cooley-Tukey; bit-reversed in, natural out; levels 0 .. lg-1):
//     (u, v) -> (u + v t, u - v t)
//   DIF (Gentleman-Sande; natural in, bit-reversed out; levels lg-1 .. 0):
//     (u, v) -> (u + v, (u - v) t)
// with t = w^jj forward and t = w^-jj inverse. The inverse twiddle comes from the same table:
// w^-jj = w^(n - jj) = -w^(n/2 - jj) = -tw[n/2 - jj] for jj > 0 (and 1 for jj = 0), the sign
// being folded into the butterfly. Inverse transforms do not scale by 1/n here (the host adds
// ntt_scale or folds 1/n into the coset table, like sppark's Type::standard inverse).
//
// Buffer indices (mirrored by `buf::*` in src/metal/ntt.rs):
//
// ntt_stage         0 x Fr[n] (in place)  1 tw Fr[n/2]  2 constant NttStageParams
//   one radix-2 level q, thread t < n/2 does butterfly t.
// ntt_fused_stages  0 x Fr[n] (in place)  1 tw Fr[n/2]  2 constant NttFusedParams
//                   threadgroup(0) 2^(ns+cols) Fr (<= 512 Fr = 16 KB; host asserts <= 32 KB)
//   levels q0 .. q0+ns-1 (ascending DIT, descending DIF) of 2^cols adjacent columns: threadgroup g
//   owns the elements (h << (q0+ns)) | (r << q0) | (b << cols) | c for r < 2^ns, c < 2^cols,
//   with b = g mod 2^(q0-cols), h = g >> (q0-cols); 2^(ns+cols-1) threads, one butterfly each per
//   level. Columns keep the loads of later passes (q0 > 0) contiguous over 2^cols Fr.
// ntt_powers        0 out Fr[count]  1 constant NttPowersParams
//   out[j] = c * base^j, from base^(2^k) for k < 32 (Intel prover's powers()).
// ntt_mul_powers    0 x Fr[n] (in place)  1 lo Fr[2^lg_lo]  2 hi Fr[2^(lg-lg_lo)]
//                   3 constant NttMulPowersParams
//   x[i] *= lo[k mod 2^lg_lo] * hi[k >> lg_lo], k = i, or rev_lg(i) when bitrev != 0
//   (CUDA's LDE_distribute_powers with two windows; lo/hi hold s^k and c * s^(k 2^lg_lo)).
// ntt_scale         0 x Fr[count] (in place)  1 constant NttScaleParams   x[i] *= c

#include "bn254.h"
#include "bn254_fast.h"

using namespace metal;
using namespace bn254;

#define NTT_INVERSE 1u
#define NTT_DIF 2u

struct NttStageParams {
  uint32_t lg;
  uint32_t q;
  uint32_t flags; // NTT_INVERSE | NTT_DIF
  uint32_t _pad;
};

struct NttFusedParams {
  uint32_t lg;
  uint32_t q0;
  uint32_t ns;
  uint32_t cols; // log2 of the number of adjacent columns
  uint32_t flags;
  uint32_t _pad[3];
};

struct NttPowersParams {
  uint32_t count;
  uint32_t _pad[7];
  uint32_t c[8];
  uint32_t sq[32][8];
};

struct NttMulPowersParams {
  uint32_t lg;
  uint32_t lg_lo;
  uint32_t bitrev;
  uint32_t _pad;
};

struct NttScaleParams {
  uint32_t count;
  uint32_t _pad[7];
  uint32_t c[8];
};

// F: Fr (bn254.h) or FrF (bn254_fast.h, item G9: the same values with the 64-bit-accumulate
// multiplication); the `_w` kernels are the FrF instantiations.
template <class F>
inline void ntt_butterfly(thread F& a, thread F& b, const device F* tw, uint32_t jj,
                          uint32_t half_n, uint32_t flags) {
  F u = a;
  F v = b;
  if (jj == 0) {
    a = u + v;
    b = u - v;
    return;
  }
  bool inv = (flags & NTT_INVERSE) != 0;
  F t = tw[inv ? half_n - jj : jj];
  if ((flags & NTT_DIF) == 0) {
    F vt = v * t;
    a = inv ? u - vt : u + vt;
    b = inv ? u + vt : u - vt;
  } else {
    a = u + v;
    b = inv ? (v - u) * t : (u - v) * t;
  }
}

template <class F>
[[max_total_threads_per_threadgroup(256)]]
kernel void ntt_stage(device F* x [[buffer(0)]],
                      const device F* tw [[buffer(1)]],
                      constant NttStageParams& p [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
  uint32_t half_n = (1u << p.lg) >> 1;
  if (gid >= half_n)
    return;
  uint32_t q = p.q;
  uint32_t j = gid & ((1u << q) - 1);
  uint32_t i0 = ((gid >> q) << (q + 1)) | j;
  uint32_t i1 = i0 + (1u << q);
  uint32_t jj = j << (p.lg - 1 - q);
  F a = x[i0];
  F b = x[i1];
  ntt_butterfly<F>(a, b, tw, jj, half_n, p.flags);
  x[i0] = a;
  x[i1] = b;
}

template <class F>
[[max_total_threads_per_threadgroup(256)]]
kernel void ntt_fused_stages(device F* x [[buffer(0)]],
                             const device F* tw [[buffer(1)]],
                             constant NttFusedParams& p [[buffer(2)]],
                             threadgroup F* sh [[threadgroup(0)]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint g [[threadgroup_position_in_grid]]) {
  const uint32_t cols = p.cols;
  const uint32_t ns = p.ns;
  const uint32_t q0 = p.q0;
  const uint32_t m = 1u << (ns + cols);
  const uint32_t nthreads = m >> 1;
  const uint32_t half_n = (1u << p.lg) >> 1;
  const uint32_t col_mask = (1u << cols) - 1;
  const uint32_t lowbits = q0 - cols;
  const uint32_t blk = g & ((1u << lowbits) - 1);
  const uint32_t base = ((g >> lowbits) << (q0 + ns)) | (blk << cols);

  for (uint32_t e = tid; e < m; e += nthreads)
    sh[e] = x[base | ((e >> cols) << q0) | (e & col_mask)];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint32_t c = tid & col_mask;
  const uint32_t bf = tid >> cols;
  const uint32_t low = (blk << cols) | c; // i0 mod 2^q0
  const bool dif = (p.flags & NTT_DIF) != 0;
  for (uint32_t s = 0; s < ns; s++) {
    uint32_t lq = dif ? ns - 1 - s : s;
    uint32_t jr = bf & ((1u << lq) - 1);
    uint32_t r0 = ((bf >> lq) << (lq + 1)) | jr;
    uint32_t r1 = r0 + (1u << lq);
    uint32_t q = q0 + lq;
    uint32_t jj = ((jr << q0) | low) << (p.lg - 1 - q);
    uint32_t e0 = (r0 << cols) | c;
    uint32_t e1 = (r1 << cols) | c;
    F a = sh[e0];
    F b = sh[e1];
    ntt_butterfly<F>(a, b, tw, jj, half_n, p.flags);
    sh[e0] = a;
    sh[e1] = b;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint32_t e = tid; e < m; e += nthreads)
    x[base | ((e >> cols) << q0) | (e & col_mask)] = sh[e];
}

[[max_total_threads_per_threadgroup(256)]]
kernel void ntt_powers(device Fr* out [[buffer(0)]],
                       constant NttPowersParams& p [[buffer(1)]],
                       uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count)
    return;
  Fr r = Fr::from_limbs(p.c);
  uint32_t j = gid;
  for (uint32_t k = 0; j != 0; k++, j >>= 1)
    if (j & 1)
      r = r * Fr::from_limbs(p.sq[k]);
  out[gid] = r;
}

template <class F>
[[max_total_threads_per_threadgroup(256)]]
kernel void ntt_mul_powers(device F* x [[buffer(0)]],
                           const device F* lo [[buffer(1)]],
                           const device F* hi [[buffer(2)]],
                           constant NttMulPowersParams& p [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
  if (gid >= (1u << p.lg))
    return;
  uint32_t k = (p.bitrev != 0 && p.lg != 0) ? reverse_bits(gid) >> (32 - p.lg) : gid;
  F f = lo[k & ((1u << p.lg_lo) - 1)] * hi[k >> p.lg_lo];
  x[gid] = x[gid] * f;
}

[[max_total_threads_per_threadgroup(256)]]
kernel void ntt_scale(device Fr* x [[buffer(0)]],
                      constant NttScaleParams& p [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count)
    return;
  x[gid] = x[gid] * Fr::from_limbs(p.c);
}

// Explicit instantiations (build.rs reads the host_name strings literally).
template [[host_name("ntt_stage")]] kernel void
ntt_stage<Fr>(device Fr*, const device Fr*, constant NttStageParams&, uint);
template [[host_name("ntt_stage_w")]] kernel void
ntt_stage<FrF>(device FrF*, const device FrF*, constant NttStageParams&, uint);
template [[host_name("ntt_fused_stages")]] kernel void
ntt_fused_stages<Fr>(device Fr*, const device Fr*, constant NttFusedParams&, threadgroup Fr*,
                     uint, uint);
template [[host_name("ntt_fused_stages_w")]] kernel void
ntt_fused_stages<FrF>(device FrF*, const device FrF*, constant NttFusedParams&,
                      threadgroup FrF*, uint, uint);
template [[host_name("ntt_mul_powers")]] kernel void
ntt_mul_powers<Fr>(device Fr*, const device Fr*, const device Fr*, constant NttMulPowersParams&,
                   uint);
template [[host_name("ntt_mul_powers_w")]] kernel void
ntt_mul_powers<FrF>(device FrF*, const device FrF*, const device FrF*,
                    constant NttMulPowersParams&, uint);
