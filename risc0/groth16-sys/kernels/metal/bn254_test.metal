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

// Test and benchmark kernels for bn254.h, driven by src/metal/tests.rs (item G2). Not used by
// the prover. Every kernel is elementwise: thread i reads element i of each input array.
//
// Buffer indices (mirrored by the `buf` constants in src/metal/tests.rs):
//
// bn254_test_{fp,fr,fp2}_op   0 out F[count]   1 a F[count]   2 b F[count]
//                             3 constant Bn254TestParams
//   op: 0 a+b  1 a-b  2 a*b  3 a.sqr()  4 a.neg()  5 a.dbl()  6 a.inv()
//       Fp/Fr only: 7 a.to_mont()  8 a.from_mont()  9 a.is_canonical() (out = {flag, 0, ...})
//
// bn254_test_{g1,g2}_op       0 out Xyzz[count]  1 out Affine[count]  2 out Jacobian[count]
//                             3 a Xyzz[count]    4 b Affine[count]    5 c Xyzz[count]
//                             6 d Jacobian[count] 7 k uint[count]     8 constant Bn254TestParams
//   op: 0 a.add_affine(b)  1 a.add(c)  2 a.dbl()  3 Xyzz::dbl_affine(b)  4 a.neg()
//       5 Xyzz::from_affine(b)  6 Xyzz::from_jacobian(d)  7 a.to_affine() -> out Affine
//       8 a.to_jacobian() -> out Jacobian  9 a.mul_small(k[i])
//
// bn254_bench_fp_mul          0 out Fp[count]  1 in Fp[count]  2 constant Bn254TestParams
//   x = in[i], y = in[(i + 1) % count]; iters times x = x * y; out[i] = x.
//
// bn254_bench_g1_add_affine   0 out G1Xyzz[count]  1 points G1Affine[k]  2 constant Bn254TestParams
//   acc = from_affine(points[i % k]); for j in 1..iters: acc = acc.add_affine(points[(i + j) % k]).

#include "bn254.h"

using namespace bn254;

struct Bn254TestParams {
  uint32_t count;
  uint32_t op;
  uint32_t iters;
  uint32_t k;
};

template <class F> inline F field_common_op(uint32_t op, F a, F b) {
  switch (op) {
  case 0:
    return a + b;
  case 1:
    return a - b;
  case 2:
    return a * b;
  case 3:
    return a.sqr();
  case 4:
    return a.neg();
  case 5:
    return a.dbl();
  case 6:
    return a.inv();
  default:
    return F::zero();
  }
}

template <class F>
kernel void bn254_test_mont_op(device F* out [[buffer(0)]],
                               const device F* a [[buffer(1)]],
                               const device F* b [[buffer(2)]],
                               constant Bn254TestParams& p [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count)
    return;
  F x = a[gid];
  F y = b[gid];
  F r;
  switch (p.op) {
  case 7:
    r = x.to_mont();
    break;
  case 8:
    r = x.from_mont();
    break;
  case 9:
    r = F::zero();
    r.l[0] = x.is_canonical() ? 1u : 0u;
    break;
  default:
    r = field_common_op<F>(p.op, x, y);
    break;
  }
  out[gid] = r;
}

template [[host_name("bn254_test_fp_op")]] kernel void
bn254_test_mont_op<Fp>(device Fp*, const device Fp*, const device Fp*,
                       constant Bn254TestParams&, uint);
template [[host_name("bn254_test_fr_op")]] kernel void
bn254_test_mont_op<Fr>(device Fr*, const device Fr*, const device Fr*,
                       constant Bn254TestParams&, uint);

kernel void bn254_test_fp2_op(device Fp2* out [[buffer(0)]],
                              const device Fp2* a [[buffer(1)]],
                              const device Fp2* b [[buffer(2)]],
                              constant Bn254TestParams& p [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count)
    return;
  out[gid] = field_common_op<Fp2>(p.op, a[gid], b[gid]);
}

template <class F>
kernel void bn254_test_curve_op(device Xyzz<F>* out [[buffer(0)]],
                                device Affine<F>* out_aff [[buffer(1)]],
                                device Jacobian<F>* out_jac [[buffer(2)]],
                                const device Xyzz<F>* a [[buffer(3)]],
                                const device Affine<F>* b [[buffer(4)]],
                                const device Xyzz<F>* c [[buffer(5)]],
                                const device Jacobian<F>* d [[buffer(6)]],
                                const device uint32_t* k [[buffer(7)]],
                                constant Bn254TestParams& p [[buffer(8)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count)
    return;
  using P = Xyzz<F>;
  switch (p.op) {
  case 0:
    out[gid] = P(a[gid]).add_affine(b[gid]);
    break;
  case 1:
    out[gid] = P(a[gid]).add(c[gid]);
    break;
  case 2:
    out[gid] = P(a[gid]).dbl();
    break;
  case 3:
    out[gid] = P::dbl_affine(b[gid]);
    break;
  case 4:
    out[gid] = P(a[gid]).neg();
    break;
  case 5:
    out[gid] = P::from_affine(b[gid]);
    break;
  case 6:
    out[gid] = P::from_jacobian(d[gid]);
    break;
  case 7:
    out_aff[gid] = P(a[gid]).to_affine();
    break;
  case 8:
    out_jac[gid] = P(a[gid]).to_jacobian();
    break;
  case 9:
    out[gid] = P(a[gid]).mul_small(k[gid]);
    break;
  default:
    break;
  }
}

template [[host_name("bn254_test_g1_op")]] kernel void
bn254_test_curve_op<Fp>(device G1Xyzz*, device G1Affine*, device G1Jacobian*,
                        const device G1Xyzz*, const device G1Affine*, const device G1Xyzz*,
                        const device G1Jacobian*, const device uint32_t*,
                        constant Bn254TestParams&, uint);
template [[host_name("bn254_test_g2_op")]] kernel void
bn254_test_curve_op<Fp2>(device G2Xyzz*, device G2Affine*, device G2Jacobian*,
                         const device G2Xyzz*, const device G2Affine*, const device G2Xyzz*,
                         const device G2Jacobian*, const device uint32_t*,
                         constant Bn254TestParams&, uint);

kernel void bn254_bench_fp_mul(device Fp* out [[buffer(0)]],
                               const device Fp* in [[buffer(1)]],
                               constant Bn254TestParams& p [[buffer(2)]],
                               uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count)
    return;
  Fp x = in[gid];
  Fp y = in[(gid + 1) % p.count];
  for (uint32_t i = 0; i < p.iters; i++)
    x = x * y;
  out[gid] = x;
}

kernel void bn254_bench_g1_add_affine(device G1Xyzz* out [[buffer(0)]],
                                      const device G1Affine* points [[buffer(1)]],
                                      constant Bn254TestParams& p [[buffer(2)]],
                                      uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count)
    return;
  G1Xyzz acc = G1Xyzz::from_affine(points[gid % p.k]);
  for (uint32_t j = 1; j < p.iters; j++)
    acc = acc.add_affine(points[(gid + j) % p.k]);
  out[gid] = acc;
}
