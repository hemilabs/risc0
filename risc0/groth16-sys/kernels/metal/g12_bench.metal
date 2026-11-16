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

// Item G12 micro-benchmarks of the bn254_g12.h formulations (src/metal/tests.rs `g12_*`).
// Buffers: 0 out, 1 in, 2 constant G12BenchParams (same layout as G9BenchParams).

#include "bn254_g12.h"

using namespace metal;
using namespace bn254;

struct G12BenchParams {
  uint32_t count;
  uint32_t op;
  uint32_t iters;
  uint32_t k;
};

// x = x * y, `iters` times, for Fp2 pairs (in[gid], in[gid + 1 mod count]).
template <class F2>
kernel void g12_bench_fp2_mul(device F2* out [[buffer(0)]],
                              const device F2* in [[buffer(1)]],
                              constant G12BenchParams& p [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count)
    return;
  F2 x = in[gid];
  F2 y = in[(gid + 1) % p.count];
  for (uint32_t i = 0; i < p.iters; i++)
    x = x * y;
  out[gid] = x;
}

template [[host_name("g12_bench_fp2_mul_l")]] kernel void
g12_bench_fp2_mul<Fp2L>(device Fp2L*, const device Fp2L*, constant G12BenchParams&, uint);
template [[host_name("g12_bench_fp2_mul_r")]] kernel void
g12_bench_fp2_mul<Fp2RF>(device Fp2RF*, const device Fp2RF*, constant G12BenchParams&, uint);
template [[host_name("g12_bench_fp2_mul_rh")]] kernel void
g12_bench_fp2_mul<Fp2RH>(device Fp2RH*, const device Fp2RH*, constant G12BenchParams&, uint);
template [[host_name("g12_bench_fp2_mul_lh")]] kernel void
g12_bench_fp2_mul<Fp2LH>(device Fp2LH*, const device Fp2LH*, constant G12BenchParams&, uint);

// Mixed additions, as g9_bench_madd.
template <class F>
kernel void g12_bench_madd(device Xyzz<F>* out [[buffer(0)]],
                           const device Affine<F>* points [[buffer(1)]],
                           constant G12BenchParams& p [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count)
    return;
  Xyzz<F> acc = Xyzz<F>::from_affine(points[gid % p.k]);
  for (uint32_t j = 1; j < p.iters; j++)
    acc = acc.add_affine(points[(gid + j) % p.k]);
  out[gid] = acc;
}

template [[host_name("g12_bench_madd_g1h")]] kernel void
g12_bench_madd<FpH>(device Xyzz<FpH>*, const device Affine<FpH>*, constant G12BenchParams&, uint);
template [[host_name("g12_bench_madd_g2r")]] kernel void
g12_bench_madd<Fp2RF>(device Xyzz<Fp2RF>*, const device Affine<Fp2RF>*, constant G12BenchParams&,
                      uint);
template [[host_name("g12_bench_madd_g2rh")]] kernel void
g12_bench_madd<Fp2RH>(device Xyzz<Fp2RH>*, const device Affine<Fp2RH>*, constant G12BenchParams&,
                      uint);
template [[host_name("g12_bench_madd_g2lh")]] kernel void
g12_bench_madd<Fp2LH>(device Xyzz<Fp2LH>*, const device Affine<Fp2LH>*, constant G12BenchParams&,
                      uint);

// Feasibility probe of batch-affine accumulation (G12 track 2): each thread owns K = p.op slots
// (affine accumulators in `acc`, device memory), and every step adds one point to each slot with
// affine formulas, the K denominators inverted together (Montgomery's trick, one Fp2 inversion
// by Fermat). No special cases (probe on random field data; d = 0 does not occur).
// Slot-major layout (slot k of thread t at k * threads + t).
// Buffers: 0 acc Affine<F2>[threads * K] (in/out), 1 points Affine<F2>[k],
// 2 pre F2[threads * K] (scratch), 3 params.
template <class F2>
kernel void g12_bench_batch_affine(device Affine<F2>* acc [[buffer(0)]],
                                   const device Affine<F2>* points [[buffer(1)]],
                                   device F2* pre [[buffer(2)]],
                                   constant G12BenchParams& p [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]]) {
  const uint32_t K = p.op;
  const uint32_t T = p.count / K; // threads; slot k of thread t at k * T + t (coalesced)
  if (gid >= T)
    return;
  device Affine<F2>* a = acc + gid;
  device F2* pr = pre + gid;
  for (uint32_t j = 1; j < p.iters; j++) {
    F2 prod = F2::one();
    for (uint32_t k = 0; k < K; k++) {
      F2 d = points[(gid * K + k + j) % p.k].x - a[k * T].x;
      pr[k * T] = prod;
      prod = prod * d;
    }
    F2 inv = prod.inv();
    for (uint32_t k = K; k-- > 0;) {
      Affine<F2> q = points[(gid * K + k + j) % p.k];
      Affine<F2> s = a[k * T];
      F2 d = q.x - s.x;
      F2 id = inv * pr[k * T];
      inv = inv * d;
      F2 l = (q.y - s.y) * id;
      F2 x3 = l.sqr() - s.x - q.x;
      F2 y3 = l * (s.x - x3) - s.y;
      a[k * T] = Affine<F2>{x3, y3};
    }
  }
}

template [[host_name("g12_bench_batch_affine_g2")]] kernel void
g12_bench_batch_affine<Fp2L>(device Affine<Fp2L>*, const device Affine<Fp2L>*, device Fp2L*,
                             constant G12BenchParams&, uint);
template [[host_name("g12_bench_batch_affine_g1")]] kernel void
g12_bench_batch_affine<FpF>(device Affine<FpF>*, const device Affine<FpF>*, device FpF*,
                            constant G12BenchParams&, uint);

// Fp2 inversions: x = (x + y).inv(), `iters` times.
kernel void g12_bench_fp2_inv(device Fp2L* out [[buffer(0)]],
                              const device Fp2L* in [[buffer(1)]],
                              constant G12BenchParams& p [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count)
    return;
  Fp2L x = in[gid];
  Fp2L y = in[(gid + 1) % p.count];
  for (uint32_t i = 0; i < p.iters; i++)
    x = (x + y).inv();
  out[gid] = x;
}
