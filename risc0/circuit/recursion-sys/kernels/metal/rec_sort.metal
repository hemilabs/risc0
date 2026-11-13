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

// C14 (docs/metal-port/notes/C14.md): the recursion witgen's WOM sort on the GPU, replacing the
// host counting sort (`sort_wom_rows`, a port of cxx/ffi.cpp `sortWomFast`) between parStepExec
// and injectWomBacks.
//
// A counting sort on the canonical address. Rows with equal canonical address land in one bucket
// in an arbitrary (atomic) order; recSortCheck flags any bucket whose rows are not bit-identical,
// and any address outside [0, numAddrs) or count > 9. Without a flag the result is exactly the
// unique `std::sort` order the host sort gives (rows with equal keys are bit-identical); with one
// the host falls back to `sort_wom_rows` on the untouched exec output.
//
// | kernel         | 0         | 1        | 2         | 3        | 4       | 5    | 6     | grid                          |
// |----------------|-----------|----------|-----------|----------|---------|------|-------|-------------------------------|
// | recSortHist    | rows      | counts   | hist      | flags    | P       |      |       | threads numSlots              |
// | recScanBlocks  | in        | out      | sums      | P (n)    |         |      |       | groups ceil(n/1024) x 256     |
// | recScanSums    | sums      | P (n)    |           |          |         |      |       | groups 1 x 256                |
// | recScanAdd     | data      | sums     | P (n)     |          |         |      |       | threads n                     |
// | recSortScatter | rows      | counts   | cursor    | out      | flags   | nv   | P     | threads numSlots              |
// | recSortCheck   | out       | flags    | nv        | P        |         |      |       | threads numSlots              |
// | recCopyWords   | src       | dst      | n (uint)  |          |         |      |       | threads n                     |
//
// recCopyWords copies the staged preflight cycles into the witgen's scratch (C14 staging,
// recursion/src/prove/hal/metal/staging.rs).
//
// All plain uint32 code: no private aggregates, no function calls in loops (R17).

#include <metal_stdlib>

#include "fp.h"

using namespace metal;

/// RecSortParams (Rust `RecSortParams` in recursion/src/prove/hal/metal/witgen.rs).
struct RecSortParams {
  uint32_t numSlots; // numCycles * 9
  uint32_t numAddrs; // histogram size
  uint32_t n;        // scan length (scan kernels)
  uint32_t nvIndex;  // index of nv (the counts' total) in the nv buffer
  uint32_t _pad[4];
};

constant constexpr uint32_t kRowWords = 5;
constant constexpr uint32_t kRowsPerCycle = 9;
constant constexpr uint32_t kScanTg = 256;
constant constexpr uint32_t kScanBlock = 1024; // 4 elements per thread

constant constexpr uint32_t kFlagAddr = 1;   // canonical address >= numAddrs
constant constexpr uint32_t kFlagCount = 2;  // a cycle wrote more than 9 rows
constant constexpr uint32_t kFlagBucket = 4; // rows of one address are not identical

static inline uint32_t canonAddr(const device uint32_t* rows, uint32_t slot) {
  return Fp::fromRaw(rows[slot * kRowWords]).asUInt32();
}

kernel void recSortHist(const device uint32_t* rows [[buffer(0)]],
                        const device uint32_t* counts [[buffer(1)]],
                        device atomic_uint* hist [[buffer(2)]],
                        device atomic_uint* flags [[buffer(3)]],
                        constant RecSortParams& p [[buffer(4)]],
                        uint slot [[thread_position_in_grid]]) {
  if (slot >= p.numSlots) {
    return;
  }
  uint32_t cycle = slot / kRowsPerCycle;
  uint32_t k = slot - cycle * kRowsPerCycle;
  uint32_t n = counts[cycle];
  if (k == 0 && n > kRowsPerCycle) {
    atomic_fetch_or_explicit(flags, kFlagCount, memory_order_relaxed);
  }
  if (k >= n) {
    return;
  }
  uint32_t a = canonAddr(rows, slot);
  if (a >= p.numAddrs) {
    atomic_fetch_or_explicit(flags, kFlagAddr, memory_order_relaxed);
    return;
  }
  atomic_fetch_add_explicit(&hist[a], 1u, memory_order_relaxed);
}

/// Exclusive scan of 1024-element blocks of `in` into `out` (may alias), block totals to `sums`.
[[max_total_threads_per_threadgroup(256)]]
kernel void recScanBlocks(const device uint32_t* in [[buffer(0)]],
                          device uint32_t* out [[buffer(1)]],
                          device uint32_t* sums [[buffer(2)]],
                          constant RecSortParams& p [[buffer(3)]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint group [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint numSg [[simdgroups_per_threadgroup]],
                          uint simdW [[threads_per_simdgroup]]) {
  threadgroup uint32_t sgTotals[32];
  uint32_t base = group * kScanBlock + tid * 4;
  uint32_t v0 = base + 0 < p.n ? in[base + 0] : 0;
  uint32_t v1 = base + 1 < p.n ? in[base + 1] : 0;
  uint32_t v2 = base + 2 < p.n ? in[base + 2] : 0;
  uint32_t v3 = base + 3 < p.n ? in[base + 3] : 0;
  uint32_t t = v0 + v1 + v2 + v3;
  uint32_t ex = simd_prefix_exclusive_sum(t);
  uint32_t simdTotal = simd_broadcast(ex + t, simdW - 1);
  if (lane == 0) {
    sgTotals[sg] = simdTotal;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint32_t off = 0;
  uint32_t all = 0;
  for (uint32_t i = 0; i < numSg; i++) {
    uint32_t s = sgTotals[i];
    off += i < sg ? s : 0;
    all += s;
  }
  ex += off;
  if (base + 0 < p.n) out[base + 0] = ex;
  if (base + 1 < p.n) out[base + 1] = ex + v0;
  if (base + 2 < p.n) out[base + 2] = ex + v0 + v1;
  if (base + 3 < p.n) out[base + 3] = ex + v0 + v1 + v2;
  if (tid == 0) {
    sums[group] = all;
  }
}

/// Exclusive scan of the `p.n` block totals in place (one threadgroup, chunks of 1024 with a
/// carry); `sums[p.n]` gets the grand total.
[[max_total_threads_per_threadgroup(256)]]
kernel void recScanSums(device uint32_t* sums [[buffer(0)]],
                        constant RecSortParams& p [[buffer(1)]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint numSg [[simdgroups_per_threadgroup]],
                        uint simdW [[threads_per_simdgroup]]) {
  threadgroup uint32_t sgTotals[32];
  uint32_t carry = 0;
  for (uint32_t chunk = 0; chunk < p.n; chunk += kScanBlock) {
    uint32_t base = chunk + tid * 4;
    uint32_t v0 = base + 0 < p.n ? sums[base + 0] : 0;
    uint32_t v1 = base + 1 < p.n ? sums[base + 1] : 0;
    uint32_t v2 = base + 2 < p.n ? sums[base + 2] : 0;
    uint32_t v3 = base + 3 < p.n ? sums[base + 3] : 0;
    uint32_t t = v0 + v1 + v2 + v3;
    uint32_t ex = simd_prefix_exclusive_sum(t);
    uint32_t simdTotal = simd_broadcast(ex + t, simdW - 1);
    if (lane == 0) {
      sgTotals[sg] = simdTotal;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint32_t off = 0;
    uint32_t all = 0;
    for (uint32_t i = 0; i < numSg; i++) {
      uint32_t s = sgTotals[i];
      off += i < sg ? s : 0;
      all += s;
    }
    ex += off + carry;
    if (base + 0 < p.n) sums[base + 0] = ex;
    if (base + 1 < p.n) sums[base + 1] = ex + v0;
    if (base + 2 < p.n) sums[base + 2] = ex + v0 + v1;
    if (base + 3 < p.n) sums[base + 3] = ex + v0 + v1 + v2;
    carry += all;
    // sgTotals is rewritten by the next chunk.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    sums[p.n] = carry;
  }
}

/// Add each 1024-element block's offset (`recScanSums` output) to its elements.
kernel void recScanAdd(device uint32_t* data [[buffer(0)]],
                       const device uint32_t* sums [[buffer(1)]],
                       constant RecSortParams& p [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
  if (i >= p.n || i < kScanBlock) {
    return;
  }
  data[i] += sums[i / kScanBlock];
}

/// Scatter every written row to its bucket (`cursor` = the exclusive scan of the histogram,
/// advanced atomically) and fill the slots [nv, numSlots) of `out` with the invalid pattern.
kernel void recSortScatter(const device uint32_t* rows [[buffer(0)]],
                           const device uint32_t* counts [[buffer(1)]],
                           device atomic_uint* cursor [[buffer(2)]],
                           device uint32_t* out [[buffer(3)]],
                           device atomic_uint* flags [[buffer(4)]],
                           const device uint32_t* nvBuf [[buffer(5)]],
                           constant RecSortParams& p [[buffer(6)]],
                           uint slot [[thread_position_in_grid]]) {
  if (slot >= p.numSlots) {
    return;
  }
  uint32_t nv = nvBuf[p.nvIndex];
  if (slot >= nv) {
    uint32_t o = slot * kRowWords;
    out[o + 0] = 0xffffffffu;
    out[o + 1] = 0xffffffffu;
    out[o + 2] = 0xffffffffu;
    out[o + 3] = 0xffffffffu;
    out[o + 4] = 0xffffffffu;
  }
  uint32_t cycle = slot / kRowsPerCycle;
  uint32_t k = slot - cycle * kRowsPerCycle;
  if (k >= counts[cycle]) {
    return;
  }
  uint32_t a = canonAddr(rows, slot);
  if (a >= p.numAddrs) {
    return; // flagged by recSortHist
  }
  uint32_t pos = atomic_fetch_add_explicit(&cursor[a], 1u, memory_order_relaxed);
  if (pos >= nv) {
    atomic_fetch_or_explicit(flags, kFlagAddr, memory_order_relaxed);
    return;
  }
  uint32_t s = slot * kRowWords;
  uint32_t o = pos * kRowWords;
  out[o + 0] = rows[s + 0];
  out[o + 1] = rows[s + 1];
  out[o + 2] = rows[s + 2];
  out[o + 3] = rows[s + 3];
  out[o + 4] = rows[s + 4];
}

/// Flag any sorted row [1, nv) whose canonical address equals its predecessor's while the rows
/// differ: then the bucket's order is not unique and the host sort must decide it.
kernel void recSortCheck(const device uint32_t* out [[buffer(0)]],
                         device atomic_uint* flags [[buffer(1)]],
                         const device uint32_t* nvBuf [[buffer(2)]],
                         constant RecSortParams& p [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
  uint32_t nv = nvBuf[p.nvIndex];
  if (i == 0 || i >= nv || i >= p.numSlots) {
    return;
  }
  uint32_t a = i * kRowWords;
  uint32_t b = a - kRowWords;
  if (Fp::fromRaw(out[a]).asUInt32() != Fp::fromRaw(out[b]).asUInt32()) {
    return;
  }
  bool same = out[a] == out[b] && out[a + 1] == out[b + 1] && out[a + 2] == out[b + 2] &&
              out[a + 3] == out[b + 3] && out[a + 4] == out[b + 4];
  if (!same) {
    atomic_fetch_or_explicit(flags, kFlagBucket, memory_order_relaxed);
  }
}

kernel void recCopyWords(const device uint32_t* src [[buffer(0)]],
                         device uint32_t* dst [[buffer(1)]],
                         constant uint32_t& n [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
  if (i < n) {
    dst[i] = src[i];
  }
}
