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

// MSM digit sort (item G6): the scalars' c-bit digits, per window, counting-sorted by bucket.
// Port of the Intel prover's msm_sort (kernels/intel/groth16_intel.hpp:583-644) with the CUDA
// prover's grid-synchronised sort replaced by separate dispatches (plan 2.5): count, scan,
// scatter. The scan additionally cuts every bucket into "pieces" of at most `piece` entries, so
// that the bucket kernel (msm_reduce.metal) is load-balanced even when one bucket holds a large
// share of the scalars (witness bits: digit 1 of window 0).
//
// Scalars are canonical little-endian 256-bit integers (8 x u32, NOT Montgomery). Any 256-bit
// value is accepted (the MSM then computes s * P for the integer s).
//
// Buffer indices (mirrored in src/metal/msm.rs, `mod buf`):
//
// msm_clear        0 hist uint[nwins * nbuckets]               1 constant MsmParams
//   hist = 0. Grid: nwins * nbuckets threads.
// msm_hist_count   0 scalars uint[n * 8]   1 hist atomic_uint[nwins * nbuckets]
//                  2 constant MsmParams   3 tr uint[8 * n] (out; written only with MsmParams::tr)
//   hist[w][d] += 1 for every scalar and window with digit d != 0. Grid: n threads.
//   Item G15: with MsmParams::tr, also tr[k * n + i] = word k of scalar i (the transposed copy
//   msm_scatter then reads: per window only the one or two words that hold its digit).
// msm_window_scan  0 hist uint[nwins * nbuckets] (counts in; bucket starts out)
//                  1 offset uint[nwins * nbuckets] (bucket starts)
//                  2 piece_off uint[nwins * nbuckets] (first piece of each bucket)
//                  3 totals uint[nwins * 2] ({entries, pieces} per window)
//                  4 constant MsmParams     threadgroup(0) uint[2 * threads]
//   Exclusive prefix sums of each window's counts and piece counts ceil(count / piece).
//   Grid: nwins threadgroups (one per window) of any width <= 1024.
// msm_scatter      0 scalars uint[n * 8]   1 hist atomic_uint (cursors; bucket ends afterwards)
//                  2 sorted uint[nwins * n] 3 constant MsmParams
//   sorted[w * n + pos] = i for each (i, w) with digit d != 0, pos taken from the cursor of
//   bucket d. Grid: n threads. The order inside a bucket is not deterministic (the sum is).
//   Item G15: windows [w0, w0 + wcount) only when wcount != 0 (the host then dispatches one
//   window group after another, so that the scattered stores of a dispatch stay in one window's
//   n entries); with MsmParams::tr, buffer 0 is msm_hist_count's transposed copy.
//
// Length order (item G12; MsmParams::lenperm), after msm_scatter, over the bucket-cut pieces:
// msm_len_clear    0 lhist uint[nwins * MSM_LEN_BINS]  1 constant MsmParams
//   lhist = 0. Grid: nwins * MSM_LEN_BINS threads.
// msm_len_count    0 offset  1 hist (bucket ends)  2 lhist atomic_uint  3 constant MsmParams
//   lhist[w][min(len, bins - 1)] += 1 for every piece of window w. Grid: (nbuckets, nwins).
// msm_len_scan     0 lhist (counts in; exclusive starts out, longest bin first)
//                  1 constant MsmParams. Grid: nwins threads.
// msm_len_scatter  0 offset  1 hist  2 piece_off  3 lhist atomic_uint (cursors)
//                  4 perm uint[nwins * max_pieces] (out)  5 constant MsmParams
//   perm[w][pos] = piece index, pos from the cursor of the piece's length bin: the pieces of a
//   window listed longest first, so that the 32 lanes of a SIMD group of msm_buckets_perm sum
//   pieces of about the same length (msm_buckets: consecutive pieces, i.e. buckets of any size,
//   and a SIMD group runs as long as its largest). The order inside a bin is not deterministic;
//   every piece is still summed by one thread in entry order, so the outputs are.
//   Grid: (nbuckets, nwins).
//
// Booth digits (MsmParams::booth, item G9): bucket |d| of a signed digit (msm_digit_booth),
// nbuckets = 2^(c-1) + 1, and sorted entries i | MSM_NEG for d < 0 (n < 2^31, host-checked).

#include "msm_common.h"

using namespace metal;

[[max_total_threads_per_threadgroup(256)]]
kernel void msm_clear(device uint32_t* hist [[buffer(0)]],
                      constant MsmParams& p [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
  if (gid >= p.nwins * p.nbuckets)
    return;
  hist[gid] = 0;
}

[[max_total_threads_per_threadgroup(256)]]
kernel void msm_hist_count(const device uint32_t* scalars [[buffer(0)]],
                           device atomic_uint* hist [[buffer(1)]],
                           constant MsmParams& p [[buffer(2)]],
                           device uint32_t* tr [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n)
    return;
  const device uint32_t* s = scalars + (size_t)gid * 8;
  if (p.tr) {
    // Item G15: the transposed copy for msm_scatter (word k of every scalar contiguous).
    for (uint32_t k = 0; k < 8; k++)
      tr[(size_t)k * p.n + gid] = s[k];
  }
  for (uint32_t w = 0; w < p.nwins; w++) {
    bool neg = false;
    uint32_t d = p.booth ? msm_digit_booth(s, w, p.wbits, neg) : msm_digit(s, w, p.wbits);
    if (d)
      atomic_fetch_add_explicit(&hist[w * p.nbuckets + d], 1u, memory_order_relaxed);
  }
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void msm_window_scan(device uint32_t* hist [[buffer(0)]],
                            device uint32_t* offset [[buffer(1)]],
                            device uint32_t* piece_off [[buffer(2)]],
                            device uint32_t* totals [[buffer(3)]],
                            constant MsmParams& p [[buffer(4)]],
                            threadgroup uint32_t* tg [[threadgroup(0)]],
                            uint w [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint nthr [[threads_per_threadgroup]]) {
  const uint32_t nb = p.nbuckets;
  const uint32_t T = p.piece;
  const uint32_t chunk = (nb + nthr - 1) / nthr;
  const uint32_t lo = min(tid * chunk, nb);
  const uint32_t hi = min(lo + chunk, nb);
  device uint32_t* h = hist + (size_t)w * nb;
  device uint32_t* off = offset + (size_t)w * nb;
  device uint32_t* poff = piece_off + (size_t)w * nb;
  threadgroup uint32_t* tg_cnt = tg;
  threadgroup uint32_t* tg_pcs = tg + nthr;

  // 1. This thread's chunk totals.
  uint32_t cnt = 0, pcs = 0;
  for (uint32_t b = lo; b < hi; b++) {
    uint32_t c = h[b];
    cnt += c;
    pcs += (c + T - 1) / T;
  }
  tg_cnt[tid] = cnt;
  tg_pcs[tid] = pcs;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 2. Exclusive scan over the threads (one thread; at most 1024 entries).
  if (tid == 0) {
    uint32_t a = 0, q = 0;
    for (uint32_t t = 0; t < nthr; t++) {
      uint32_t c = tg_cnt[t], k = tg_pcs[t];
      tg_cnt[t] = a;
      tg_pcs[t] = q;
      a += c;
      q += k;
    }
    totals[2 * w] = a;
    totals[2 * w + 1] = q;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 3. Rewrite the chunk: bucket starts (also the scatter cursors) and first pieces.
  uint32_t a = tg_cnt[tid], q = tg_pcs[tid];
  for (uint32_t b = lo; b < hi; b++) {
    uint32_t c = h[b];
    off[b] = a;
    h[b] = a;
    poff[b] = q;
    a += c;
    q += (c + T - 1) / T;
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void msm_scatter(const device uint32_t* scalars [[buffer(0)]],
                        device atomic_uint* hist [[buffer(1)]],
                        device uint32_t* sorted [[buffer(2)]],
                        constant MsmParams& p [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n)
    return;
  // Item G15: `scalars` transposed (p.tr; see msm_hist_count), and windows [w0, w0 + wcount)
  // only (wcount = 0: all), one dispatch per window group.
  const uint32_t stride = p.tr ? p.n : 1u;
  const device uint32_t* s = p.tr ? scalars + gid : scalars + (size_t)gid * 8;
  const uint32_t wend = p.wcount ? min(p.w0 + p.wcount, p.nwins) : p.nwins;
  for (uint32_t w = p.w0; w < wend; w++) {
    bool neg = false;
    uint32_t d = p.booth ? msm_digit_booth(s, w, p.wbits, neg, stride)
                         : msm_digit(s, w, p.wbits, stride);
    if (d) {
      uint32_t pos =
          atomic_fetch_add_explicit(&hist[w * p.nbuckets + d], 1u, memory_order_relaxed);
      sorted[(size_t)w * p.n + pos] = gid | (neg ? MSM_NEG : 0u);
    }
  }
}

// Item G12 length order (see the header).
[[max_total_threads_per_threadgroup(256)]]
kernel void msm_len_clear(device uint32_t* lhist [[buffer(0)]],
                          constant MsmParams& p [[buffer(1)]],
                          uint gid [[thread_position_in_grid]]) {
  if (gid >= p.nwins * MSM_LEN_BINS)
    return;
  lhist[gid] = 0;
}

[[max_total_threads_per_threadgroup(256)]]
kernel void msm_len_count(const device uint32_t* offset [[buffer(0)]],
                          const device uint32_t* hist [[buffer(1)]],
                          device atomic_uint* lhist [[buffer(2)]],
                          constant MsmParams& p [[buffer(3)]],
                          uint2 gid [[thread_position_in_grid]]) {
  const uint32_t b = gid.x;
  const uint32_t w = gid.y;
  if (b >= p.nbuckets || w >= p.nwins)
    return;
  const size_t at = (size_t)w * p.nbuckets + b;
  const uint32_t c = hist[at] - offset[at];
  if (c == 0)
    return;
  const uint32_t T = p.piece;
  device atomic_uint* lh = lhist + (size_t)w * MSM_LEN_BINS;
  if (c / T)
    atomic_fetch_add_explicit(&lh[min(T, MSM_LEN_BINS - 1)], c / T, memory_order_relaxed);
  if (c % T)
    atomic_fetch_add_explicit(&lh[min(c % T, MSM_LEN_BINS - 1)], 1u, memory_order_relaxed);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void msm_len_scan(device uint32_t* lhist [[buffer(0)]],
                         constant MsmParams& p [[buffer(1)]],
                         uint w [[thread_position_in_grid]]) {
  if (w >= p.nwins)
    return;
  device uint32_t* lh = lhist + (size_t)w * MSM_LEN_BINS;
  uint32_t a = 0;
  for (uint32_t k = MSM_LEN_BINS; k-- > 0;) {
    uint32_t c = lh[k];
    lh[k] = a;
    a += c;
  }
}

[[max_total_threads_per_threadgroup(256)]]
kernel void msm_len_scatter(const device uint32_t* offset [[buffer(0)]],
                            const device uint32_t* hist [[buffer(1)]],
                            const device uint32_t* piece_off [[buffer(2)]],
                            device atomic_uint* lhist [[buffer(3)]],
                            device uint32_t* perm [[buffer(4)]],
                            constant MsmParams& p [[buffer(5)]],
                            uint2 gid [[thread_position_in_grid]]) {
  const uint32_t b = gid.x;
  const uint32_t w = gid.y;
  if (b >= p.nbuckets || w >= p.nwins)
    return;
  const size_t at = (size_t)w * p.nbuckets + b;
  const uint32_t c = hist[at] - offset[at];
  const uint32_t T = p.piece;
  device atomic_uint* lh = lhist + (size_t)w * MSM_LEN_BINS;
  device uint32_t* pw = perm + (size_t)w * p.max_pieces;
  uint32_t pc = piece_off[at];
  for (uint32_t done = 0; done < c; done += T, pc++) {
    uint32_t len = min(T, c - done);
    uint32_t pos = atomic_fetch_add_explicit(&lh[min(len, MSM_LEN_BINS - 1)], 1u,
                                             memory_order_relaxed);
    if (pos < p.max_pieces) // the pieces of a window fit (host bound); keeps a bad call in bounds
      pw[pos] = pc;
  }
}
