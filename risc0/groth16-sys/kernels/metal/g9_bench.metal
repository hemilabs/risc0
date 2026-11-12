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

// Item G9 micro-benchmarks of the bn254_fast.h formulations (src/metal/tests.rs `g9_*`).
// Buffers: 0 out Fp[count], 1 in Fp[count], 2 constant G9BenchParams.

#include "bn254_fast.h"

using namespace metal;
using namespace bn254;

// Same layout as Bn254TestParams (bn254_test.metal).
struct G9BenchParams {
  uint32_t count;
  uint32_t op;
  uint32_t iters;
  uint32_t k;
};

kernel void g9_bench_fp_mul_wide(device Fp* out [[buffer(0)]],
                                 const device Fp* in [[buffer(1)]],
                                 constant G9BenchParams& p [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count)
    return;
  Fp x = in[gid];
  Fp y = in[(gid + 1) % p.count];
  for (uint32_t i = 0; i < p.iters; i++)
    x = mul_wide(x, y);
  out[gid] = x;
}

kernel void g9_bench_fp_add(device Fp* out [[buffer(0)]],
                            const device Fp* in [[buffer(1)]],
                            constant G9BenchParams& p [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count)
    return;
  Fp x = in[gid];
  Fp y = in[(gid + 1) % p.count];
  for (uint32_t i = 0; i < p.iters; i++) {
    x = x + y;
    y = x - y;
  }
  out[gid] = x;
}

kernel void g9_bench_fp_add_wide(device Fp* out [[buffer(0)]],
                                 const device Fp* in [[buffer(1)]],
                                 constant G9BenchParams& p [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count)
    return;
  Fp x = in[gid];
  Fp y = in[(gid + 1) % p.count];
  for (uint32_t i = 0; i < p.iters; i++) {
    x = add_wide(x, y);
    y = sub_wide(x, y);
  }
  out[gid] = x;
}

// Mixed additions acc += points[(gid + j) % k], j = 1..iters-1 (any field data: both
// formulations compute the same values whether or not the points are on the curve).
template <class F>
kernel void g9_bench_madd(device Xyzz<F>* out [[buffer(0)]],
                          const device Affine<F>* points [[buffer(1)]],
                          constant G9BenchParams& p [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count)
    return;
  Xyzz<F> acc = Xyzz<F>::from_affine(points[gid % p.k]);
  for (uint32_t j = 1; j < p.iters; j++)
    acc = acc.add_affine(points[(gid + j) % p.k]);
  out[gid] = acc;
}

template [[host_name("g9_bench_madd_g1")]] kernel void
g9_bench_madd<Fp>(device Xyzz<Fp>*, const device Affine<Fp>*, constant G9BenchParams&, uint);
template [[host_name("g9_bench_madd_g1f")]] kernel void
g9_bench_madd<FpF>(device Xyzz<FpF>*, const device Affine<FpF>*, constant G9BenchParams&, uint);
template [[host_name("g9_bench_madd_g1s")]] kernel void
g9_bench_madd<FpS>(device Xyzz<FpS>*, const device Affine<FpS>*, constant G9BenchParams&, uint);
template [[host_name("g9_bench_madd_g1a")]] kernel void
g9_bench_madd<FpA>(device Xyzz<FpA>*, const device Affine<FpA>*, constant G9BenchParams&, uint);
template [[host_name("g9_bench_madd_g1w")]] kernel void
g9_bench_madd<FpW>(device Xyzz<FpW>*, const device Affine<FpW>*, constant G9BenchParams&, uint);
template [[host_name("g9_bench_madd_g2")]] kernel void
g9_bench_madd<Fp2>(device Xyzz<Fp2>*, const device Affine<Fp2>*, constant G9BenchParams&, uint);
template [[host_name("g9_bench_madd_g2l")]] kernel void
g9_bench_madd<Fp2L>(device Xyzz<Fp2L>*, const device Affine<Fp2L>*, constant G9BenchParams&, uint);
template [[host_name("g9_bench_madd_g2f")]] kernel void
g9_bench_madd<Fp2F>(device Xyzz<Fp2F>*, const device Affine<Fp2F>*, constant G9BenchParams&, uint);
template [[host_name("g9_bench_madd_g2w")]] kernel void
g9_bench_madd<Fp2W>(device Xyzz<Fp2W>*, const device Affine<Fp2W>*, constant G9BenchParams&, uint);

// g9_bench_madd for G2 on lane pairs (bn254_fast.h `pair`): threads 2 i and 2 i + 1 compute
// out[i]; grid 2 count.
kernel void g9_bench_madd_g2p(device Xyzz<Fp2>* out [[buffer(0)]],
                              const device Affine<Fp2>* points [[buffer(1)]],
                              constant G9BenchParams& p [[buffer(2)]],
                              uint gid2 [[thread_position_in_grid]]) {
  const uint gid = gid2 >> 1;
  const bool odd = (gid2 & 1) != 0;
  const uint o = odd ? 1 : 0;
  if (gid >= p.count)
    return;
  const device FpF* q0 = (const device FpF*)&points[gid % p.k];
  pair::Xyzz2 acc = {q0[o], q0[2 + o], pair::one(odd), pair::one(odd)};
  for (uint32_t j = 1; j < p.iters; j++) {
    const device FpF* q = (const device FpF*)&points[(gid + j) % p.k];
    acc = pair::add_affine(acc, pair::Aff{q[o], q[2 + o]}, odd);
  }
  device FpF* d = (device FpF*)&out[gid];
  d[o] = acc.x;
  d[2 + o] = acc.y;
  d[4 + o] = acc.zz;
  d[6 + o] = acc.zzz;
}

kernel void g9_bench_fp_mul_nc(device Fp* out [[buffer(0)]],
                               const device Fp* in [[buffer(1)]],
                               constant G9BenchParams& p [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count)
    return;
  Fp x = in[gid];
  Fp y = in[(gid + 1) % p.count];
  for (uint32_t i = 0; i < p.iters; i++)
    x = mul_nc(x, y);
  out[gid] = x;
}
