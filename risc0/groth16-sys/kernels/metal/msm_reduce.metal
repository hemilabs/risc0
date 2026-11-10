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

// MSM point arithmetic (item G6), after msm_sort.metal sorted the digits. Port of the Intel
// prover's msm_sorted (kernels/intel/groth16_intel.hpp:646-706), with these changes:
// - Bucket sums are load-balanced: a thread sums one "piece" (at most `piece` consecutive entries
//   of one bucket) instead of a whole bucket.
// - The segment reduction runs over pieces instead of buckets, weighting each piece by its bucket
//   index (running sums; a gap of g empty buckets costs one g-multiple). Every segment does the
//   same amount of work whatever the bucket occupancy.
// - The window sums are one kernel, msm_window_reduce (one threadgroup per window, a tree over the
//   window's segment partials; what the CUDA prover's integrate_split/integrate_combine pair does).
//   sum_w 2^(c w) S_w stays on the host (Intel step 6): 256 serial doublings on one GPU thread
//   cost ~15 ms for G2, far more than in ark on the CPU.
//
// Points are affine, Montgomery form, (0, 0) = infinity (bn254.h); results are Xyzz.
// Every kernel exists for G1 (Fp) and G2 (Fp2): `<name>_g1`, `<name>_g2`, explicit instantiations
// at the end of the file.
//
// Buffer indices (mirrored in src/metal/msm.rs, `mod buf`):
//
// msm_buckets_{g1,g2}         0 points Affine[npoints]   1 sorted uint[nwins * n]
//                             2 offset uint[nwins * B] (bucket starts)
//                             3 hist uint[nwins * B] (bucket ends, after msm_scatter)
//                             4 piece_off uint[nwins * B]  5 totals uint[nwins * 2]
//                             6 pieces Xyzz[wcount * max_pieces] (out)
//                             7 piece_bucket uint[wcount * max_pieces] (out)
//                             8 constant MsmParams
//   Grid: (max_pieces, wcount) threads; thread (p, wl) handles piece p of window w0 + wl.
// msm_segment_reduce_{g1,g2}  0 pieces  1 piece_bucket  2 totals
//                             3 partial Xyzz[nwins * max_segs] (out)  4 constant MsmParams
//   Grid: (max_segs, wcount) threads; partial[w][s] = sum over the segment's pieces of
//   bucket(piece) * piece.
// msm_window_reduce_{g1,g2}   0 partial  1 totals  2 wsum Xyzz[nwins] (out)
//                             3 constant MsmParams  threadgroup(0) Xyzz[threads]
//   Grid: nwins threadgroups of a power-of-two width; wsum[w] = sum of window w's partials.

#include "bn254.h"
#include "msm_common.h"

using namespace metal;
using namespace bn254;

// k * p for any 32-bit k, double-and-add from the top set bit (cheap for small k, unlike
// Xyzz::mul_small, which always runs 32 rounds).
template <class F> inline Xyzz<F> msm_mul_var(Xyzz<F> p, uint32_t k) {
  if (k == 0)
    return Xyzz<F>::inf();
  int top = 31 - (int)clz(k);
  Xyzz<F> acc = p;
  for (int i = top - 1; i >= 0; i--) {
    acc = acc.dbl();
    if ((k >> i) & 1)
      acc = acc.add(p);
  }
  return acc;
}

template <class F>
kernel void msm_buckets(const device Affine<F>* points [[buffer(0)]],
                        const device uint32_t* sorted [[buffer(1)]],
                        const device uint32_t* offset [[buffer(2)]],
                        const device uint32_t* hist [[buffer(3)]],
                        const device uint32_t* piece_off [[buffer(4)]],
                        const device uint32_t* totals [[buffer(5)]],
                        device Xyzz<F>* pieces [[buffer(6)]],
                        device uint32_t* piece_bucket [[buffer(7)]],
                        constant MsmParams& p [[buffer(8)]],
                        uint2 gid [[thread_position_in_grid]]) {
  const uint32_t pc = gid.x;
  const uint32_t wl = gid.y;
  if (wl >= p.wcount)
    return;
  const uint32_t w = p.w0 + wl;
  if (pc >= totals[2 * w + 1] || pc >= p.max_pieces)
    return;
  const uint32_t nb = p.nbuckets;
  const device uint32_t* poff = piece_off + (size_t)w * nb;

  // The bucket holding piece pc: the last b with poff[b] <= pc (poff[0] = 0, non-decreasing).
  uint32_t lo = 0, hi = nb;
  while (hi - lo > 1) {
    uint32_t mid = (lo + hi) >> 1;
    if (poff[mid] <= pc)
      lo = mid;
    else
      hi = mid;
  }
  const uint32_t b = lo;
  const uint32_t start = offset[(size_t)w * nb + b] + (pc - poff[b]) * p.piece;
  const uint32_t end = min(start + p.piece, hist[(size_t)w * nb + b]);

  const device uint32_t* idx = sorted + (size_t)w * p.n;
  Xyzz<F> acc = Xyzz<F>::inf();
  for (uint32_t k = start; k < end; k++) {
    uint32_t i = idx[k];
    if (i >= p.first) {
      uint32_t j = i - p.first;
      if (j < p.npoints) // guaranteed by the host; keeps a bad call in bounds
        acc = acc.add_affine(points[j]);
    }
  }
  const size_t out = (size_t)wl * p.max_pieces + pc;
  pieces[out] = acc;
  piece_bucket[out] = b;
}

template <class F>
kernel void msm_segment_reduce(const device Xyzz<F>* pieces [[buffer(0)]],
                               const device uint32_t* piece_bucket [[buffer(1)]],
                               const device uint32_t* totals [[buffer(2)]],
                               device Xyzz<F>* partial [[buffer(3)]],
                               constant MsmParams& p [[buffer(4)]],
                               uint2 gid [[thread_position_in_grid]]) {
  const uint32_t s = gid.x;
  const uint32_t wl = gid.y;
  if (wl >= p.wcount || s >= p.max_segs)
    return;
  const uint32_t w = p.w0 + wl;
  const uint32_t np = min(totals[2 * w + 1], p.max_pieces);
  const uint32_t lo = s * p.seg;
  if (lo >= np)
    return;
  const uint32_t hi = min(lo + p.seg, np);
  const device Xyzz<F>* pc = pieces + (size_t)wl * p.max_pieces;
  const device uint32_t* pb = piece_bucket + (size_t)wl * p.max_pieces;

  // Descending over the pieces: running = sum of the pieces seen, total = sum (b_k - b) P_k
  // for the current bucket b. Moving down g buckets adds g * running.
  uint32_t bprev = pb[hi - 1];
  Xyzz<F> running = pc[hi - 1];
  Xyzz<F> total = Xyzz<F>::inf();
  for (uint32_t k = hi - 1; k-- > lo;) {
    uint32_t b = pb[k];
    if (b != bprev) {
      uint32_t g = bprev - b;
      total = total.add(g == 1 ? running : msm_mul_var(running, g));
      bprev = b;
    }
    running = running.add(pc[k]);
  }
  total = total.add(msm_mul_var(running, bprev));
  partial[(size_t)w * p.max_segs + s] = total;
}

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
template [[host_name("msm_buckets_g1")]] kernel void
msm_buckets<Fp>(const device Affine<Fp>*, const device uint32_t*, const device uint32_t*,
                 const device uint32_t*, const device uint32_t*, const device uint32_t*,
                 device Xyzz<Fp>*, device uint32_t*, constant MsmParams&, uint2);
template [[host_name("msm_segment_reduce_g1")]] kernel void
msm_segment_reduce<Fp>(const device Xyzz<Fp>*, const device uint32_t*, const device uint32_t*,
                        device Xyzz<Fp>*, constant MsmParams&, uint2);
template [[host_name("msm_window_reduce_g1")]] kernel void
msm_window_reduce<Fp>(const device Xyzz<Fp>*, const device uint32_t*, device Xyzz<Fp>*,
                       constant MsmParams&, threadgroup Xyzz<Fp>*, uint, uint, uint);
template [[host_name("msm_buckets_g2")]] kernel void
msm_buckets<Fp2>(const device Affine<Fp2>*, const device uint32_t*, const device uint32_t*,
                 const device uint32_t*, const device uint32_t*, const device uint32_t*,
                 device Xyzz<Fp2>*, device uint32_t*, constant MsmParams&, uint2);
template [[host_name("msm_segment_reduce_g2")]] kernel void
msm_segment_reduce<Fp2>(const device Xyzz<Fp2>*, const device uint32_t*, const device uint32_t*,
                        device Xyzz<Fp2>*, constant MsmParams&, uint2);
template [[host_name("msm_window_reduce_g2")]] kernel void
msm_window_reduce<Fp2>(const device Xyzz<Fp2>*, const device uint32_t*, device Xyzz<Fp2>*,
                       constant MsmParams&, threadgroup Xyzz<Fp2>*, uint, uint, uint);
