// Copyright 2025 RISC Zero, Inc.
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

// eval_u (batch_evaluate_any) for Metal: a transliteration of the CUDA (non-HIP) kernels in
// sys/kernels/zkp/cuda/kernels.cu (batch_evaluate_any: lines 46-78; eu2_*: lines 187-387).
// Host dispatcher: zkp/src/hal/metal/eval.rs (ports ffi.cu risc0_zkp_cuda_batch_evaluate_any).
//
// All arithmetic is exact in canonical Montgomery form, so every kernel here is bit-identical to
// the CPU HAL for any input.
//
// Bindings (mirrored by the Rust side in eval.rs):
//   batch_evaluate_any: 0 out (FpExt[nEvals]), 1 coeffs (Fp), 2 which (u32[nEvals]),
//                       3 xs (FpExt[nEvals]), 4 EvalAnyParams; threadgroup(0) = 16 B * threads.
//                       grid = nEvals groups x 256 threads.
//   eu2_pows:   0 pw (FpExt[nEvals*256]), 1 ps (FpExt[nEvals*S]), 2 xs, 3 Eu2Params.
//               grid = nEvals groups x 256 threads.
//   eu2_main:   0 partial (FpExt[nEvals*S]), 1 coeffs (16-B aligned), 2 which, 3 xs, 4 pw, 5 ps,
//               6 Eu2Params. grid = (S, nEvals) groups x 256 threads.
//   eu2_reduce: 0 out, 1 partial, 2 Eu2Params. grid = nEvals groups x 32 threads.

#include "fp.h"
#include "fpext.h"

using namespace metal;

#define EU_J 16
#define EU_THREADS 256

struct EvalAnyParams {
  uint32_t deg;
  uint32_t _pad[3];
};

struct Eu2Params {
  uint32_t deg;
  uint32_t chunk;
  uint32_t S;
  uint32_t _pad;
};

// ---- Horner fallback (CUDA batch_evaluate_any) ----

kernel void batch_evaluate_any(device FpExt* out [[buffer(0)]],
                               const device Fp* coeffs [[buffer(1)]],
                               const device uint32_t* which [[buffer(2)]],
                               const device FpExt* xs [[buffer(3)]],
                               constant EvalAnyParams& p [[buffer(4)]],
                               threadgroup uint32_t* totsBuf [[threadgroup(0)]],
                               uint3 blockIdx [[threadgroup_position_in_grid]],
                               uint3 threadIdx [[thread_position_in_threadgroup]],
                               uint3 blockDim [[threads_per_threadgroup]]) {
  const uint32_t deg = p.deg;
  const device Fp* cur_poly = coeffs + (size_t)which[blockIdx.x] * deg;
  FpExt x = xs[blockIdx.x];
  FpExt stepx = pow(x, (size_t)blockDim.x);
  // Horner's method: evaluate each thread's strided subsequence from high to low degree.
  uint32_t K = (deg + blockDim.x - 1) / blockDim.x;
  FpExt tot;
  for (int k = K - 1; k >= 0; k--) {
    tot *= stepx;
    size_t i = threadIdx.x + (size_t)k * blockDim.x;
    if (i < deg) {
      tot += FpExt(Fp(cur_poly[i]));
    }
  }
  tot *= pow(x, (size_t)threadIdx.x);
  threadgroup FpExt* tots = reinterpret_cast<threadgroup FpExt*>(totsBuf);
  tots[threadIdx.x] = tot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  unsigned cur = blockDim.x;
  while (cur) {
    cur /= 2;
    if (threadIdx.x < cur) {
      tots[threadIdx.x] = FpExt(tots[threadIdx.x]) + FpExt(tots[threadIdx.x + cur]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (threadIdx.x == 0) {
    out[blockIdx.x] = FpExt(tots[0]);
  }
}

// ---- eval_u v2 ----
// Split-K over (chunk, eval point), per-point power tables, J coefficients per thread per step,
// one REDC per pair of products. The CUDA (non-HIP) branch; SIMD width 32 = CUDA warp size, so
// the 32-lane reductions map 1:1 (simd_shuffle_down returns the lane's own value when the source
// lane is out of range, as __shfl_down_sync does; only lane 0's result is used).

inline uint32_t eu_addmod(uint32_t a, uint32_t b) {
  uint32_t r = a + b;
  return r >= Fp::P ? r - Fp::P : r;
}

// (a0*b0 + a1*b1) * 2^-32 mod p, inputs < p. Same REDC formulation as Fp::mul.
inline uint32_t eu_redc_pair(uint32_t a0, uint32_t b0, uint32_t a1, uint32_t b1) {
  uint64_t o64 = (uint64_t)a0 * b0 + (uint64_t)a1 * b1;  // < 2p^2 < p*2^32
  uint32_t low = -uint32_t(o64);
  uint32_t red = Fp::M * low;
  o64 += uint64_t(red) * uint64_t(Fp::P);  // < 2^64
  uint32_t ret = o64 >> 32;                 // < 2p
  return ret >= Fp::P ? ret - Fp::P : ret;
}

inline FpExt fpext_shfl_down(FpExt v, ushort d) {
  FpExt r;
  for (int i = 0; i < 4; i++) {
    r.elems[i] = Fp::fromRaw(simd_shuffle_down(v.elems[i].asRaw(), d));
  }
  return r;
}

// pw[e*EU_THREADS + t] = x_e^(J*t);  ps[e*S + s] = x_e^(s*chunk)
[[max_total_threads_per_threadgroup(EU_THREADS)]]
kernel void eu2_pows(device FpExt* pw [[buffer(0)]],
                     device FpExt* ps [[buffer(1)]],
                     const device FpExt* xs [[buffer(2)]],
                     constant Eu2Params& p [[buffer(3)]],
                     uint3 blockIdx [[threadgroup_position_in_grid]],
                     uint3 threadIdx [[thread_position_in_threadgroup]],
                     uint3 blockDim [[threads_per_threadgroup]]) {
  const uint32_t e = blockIdx.x;
  const uint32_t t = threadIdx.x;
  const uint32_t S = p.S;
  const uint32_t chunk = p.chunk;
  FpExt x = xs[e];
  pw[(size_t)e * EU_THREADS + t] = pow(x, (size_t)EU_J * t);
  for (uint32_t s = t; s < S; s += blockDim.x) {
    ps[(size_t)e * S + s] = pow(x, (size_t)s * chunk);
  }
}

// grid = (S, nEvals), block = EU_THREADS
[[max_total_threads_per_threadgroup(EU_THREADS)]]
kernel void eu2_main(device FpExt* partial [[buffer(0)]],
                     const device Fp* coeffs [[buffer(1)]],
                     const device uint32_t* which [[buffer(2)]],
                     const device FpExt* xs [[buffer(3)]],
                     const device FpExt* pw [[buffer(4)]],
                     const device FpExt* ps [[buffer(5)]],
                     constant Eu2Params& p [[buffer(6)]],
                     uint3 blockIdx [[threadgroup_position_in_grid]],
                     uint3 threadIdx [[thread_position_in_threadgroup]],
                     uint3 gridDim [[threadgroups_per_grid]]) {
  const uint32_t s = blockIdx.x;
  const uint32_t e = blockIdx.y;
  const uint32_t S = gridDim.x;
  const uint32_t t = threadIdx.x;
  const uint32_t deg = p.deg;
  const uint32_t chunk = p.chunk;
  const device Fp* poly = coeffs + (size_t)which[e] * deg + (size_t)s * chunk;
  FpExt x = xs[e];
  FpExt xp[EU_J];
  xp[0] = FpExt(1);
  for (int j = 1; j < EU_J; j++) {
    xp[j] = xp[j - 1] * x;
  }
  // step = x^(J*THREADS)
  FpExt step = FpExt(pw[(size_t)e * EU_THREADS + (EU_THREADS - 1)]) * (xp[EU_J - 1] * x);
  const uint32_t iters = chunk / (EU_J * EU_THREADS);
  const device uint4* p4 = reinterpret_cast<const device uint4*>(poly);
  FpExt acc;
  for (int k = (int)iters - 1; k >= 0; k--) {
    size_t base = ((size_t)k * EU_THREADS + t) * (EU_J / 4);
    uint32_t c[EU_J];
    for (int q = 0; q < EU_J / 4; q++) {
      uint4 v = p4[base + q];
      c[4 * q + 0] = v.x;
      c[4 * q + 1] = v.y;
      c[4 * q + 2] = v.z;
      c[4 * q + 3] = v.w;
    }
    // Pair-wise lazy reduction: one REDC per two products.
    FpExt inner;
    for (int i = 0; i < 4; i++) {
      uint32_t sacc = 0;
      for (int j = 0; j < EU_J; j += 2) {
        sacc = eu_addmod(sacc,
                         eu_redc_pair(c[j],
                                      xp[j].elems[i].asRaw(),
                                      c[j + 1],
                                      xp[j + 1].elems[i].asRaw()));
      }
      inner.elems[i] = Fp::fromRaw(sacc);
    }
    acc = acc * step + inner;
  }
  acc = acc * FpExt(pw[(size_t)e * EU_THREADS + t]);
  // block reduce: 32-lane groups, then across the 8 SIMD groups.
  for (ushort d = 16; d > 0; d >>= 1) {
    acc += fpext_shfl_down(acc, d);
  }
  threadgroup uint32_t red_raw[EU_THREADS / 32 * 4];
  if ((t & 31) == 0) {
    for (int i = 0; i < 4; i++) {
      red_raw[(t >> 5) * 4 + i] = acc.elems[i].asRaw();
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 32) {
    FpExt v;
    if (t < EU_THREADS / 32) {
      v = FpExt(Fp::fromRaw(red_raw[t * 4 + 0]),
                Fp::fromRaw(red_raw[t * 4 + 1]),
                Fp::fromRaw(red_raw[t * 4 + 2]),
                Fp::fromRaw(red_raw[t * 4 + 3]));
    }
    for (ushort d = 16; d > 0; d >>= 1) {
      v += fpext_shfl_down(v, d);
    }
    if (t == 0) {
      partial[(size_t)e * S + s] = v * FpExt(ps[(size_t)e * S + s]);
    }
  }
}

// grid = nEvals, block = 32
[[max_total_threads_per_threadgroup(32)]]
kernel void eu2_reduce(device FpExt* out [[buffer(0)]],
                       const device FpExt* partial [[buffer(1)]],
                       constant Eu2Params& p [[buffer(2)]],
                       uint3 blockIdx [[threadgroup_position_in_grid]],
                       uint3 threadIdx [[thread_position_in_threadgroup]]) {
  const uint32_t e = blockIdx.x;
  const uint32_t S = p.S;
  FpExt v;
  for (uint32_t s = threadIdx.x; s < S; s += 32) {
    v += FpExt(partial[(size_t)e * S + s]);
  }
  for (ushort d = 16; d > 0; d >>= 1) {
    v += fpext_shfl_down(v, d);
  }
  if (threadIdx.x == 0) {
    out[e] = v;
  }
}
