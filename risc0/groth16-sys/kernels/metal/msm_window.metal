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

// msm_window_reduce (item G6; moved out of msm_reduce.metal by item G12): the window sums of the
// MSM, one threadgroup per window, a tree over the window's segment partials.
//
// msm_window_reduce_{g1,g2,g1w,g2w}  0 partial Xyzz[nwins * max_segs]  1 totals uint[nwins * 2]
//                                    2 wsum Xyzz[nwins] (out)  3 constant MsmParams
//                                    threadgroup(0) Xyzz[threads]
//   Grid: nwins threadgroups of a power-of-two width; wsum[w] = sum of window w's partials.
//
// R17 (docs/metal-port/notes/R17.md, G12.md): every function of this TU, the bn254 headers
// included, is always_inline. With the point additions out of line (the shared msm_reduce.metal
// build), the G2 kernel returned wrong window sums for some windows under concurrent GPU load
// (item G12: 3-5 of 2000 re-runs of the same inputs, n 1000, wbits 1), the R17 shape of a private
// aggregate written in a loop around an out-of-line call (ir_scan.py tier B). Inlined, the TU has
// no loop with a call.

#include "msm_common.h"

#pragma clang attribute push(__attribute__((always_inline)), apply_to = function)
#include "bn254.h"
#include "bn254_fast.h"

using namespace metal;
using namespace bn254;

template <class F>
kernel void msm_window_reduce(const device Xyzz<F>* partial [[buffer(0)]],
                              const device uint32_t* totals [[buffer(1)]],
                              device Xyzz<F>* wsum [[buffer(2)]],
                              constant MsmParams& p [[buffer(3)]],
                              threadgroup Xyzz<F>* tg [[threadgroup(0)]],
                              uint w [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint nthr [[threads_per_threadgroup]]) {
  if (w >= p.nwins)
    return;
  const uint32_t np = min(totals[2 * w + 1], p.max_pieces);
  const uint32_t nsegs = (np + p.seg - 1) / p.seg;
  Xyzz<F> acc = Xyzz<F>::inf();
  for (uint32_t j = tid; j < nsegs; j += nthr)
    acc = acc.add(partial[(size_t)w * p.max_segs + j]);
  tg[tid] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint32_t stride = nthr >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      Xyzz<F> a = tg[tid];
      Xyzz<F> b = tg[tid + stride];
      tg[tid] = a.add(b);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0)
    wsum[w] = tg[0];
}

// Explicit instantiations, written out (build.rs reads the host_name strings literally).
template [[host_name("msm_window_reduce_g1")]] kernel void
msm_window_reduce<Fp>(const device Xyzz<Fp>*, const device uint32_t*, device Xyzz<Fp>*,
                       constant MsmParams&, threadgroup Xyzz<Fp>*, uint, uint, uint);
template [[host_name("msm_window_reduce_g2")]] kernel void
msm_window_reduce<Fp2>(const device Xyzz<Fp2>*, const device uint32_t*, device Xyzz<Fp2>*,
                       constant MsmParams&, threadgroup Xyzz<Fp2>*, uint, uint, uint);
template [[host_name("msm_window_reduce_g1w")]] kernel void
msm_window_reduce<FpF>(const device Xyzz<FpF>*, const device uint32_t*, device Xyzz<FpF>*,
                       constant MsmParams&, threadgroup Xyzz<FpF>*, uint, uint, uint);
template [[host_name("msm_window_reduce_g2w")]] kernel void
msm_window_reduce<Fp2L>(const device Xyzz<Fp2L>*, const device uint32_t*, device Xyzz<Fp2L>*,
                       constant MsmParams&, threadgroup Xyzz<Fp2L>*, uint, uint, uint);

#pragma clang attribute pop
