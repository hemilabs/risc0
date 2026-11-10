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

// Hand-written recursion kernels (plan 4.3, item C1), 1:1 ports of kernels/cuda/ffi.cu.
// Entry names keep the CUDA `__global__` names. Every kernel is listed in
// recursion-sys/src/lib.rs METAL_KERNEL_NAMES.
//
// Bindings (`[[buffer(n)]]`; P = params struct bound with set_bytes, F = `device atomic_uint*`
// fault word ORed once per thread on bad input, plan 2.9):
//
// | kernel               | 0        | 1          | 2        | 3      | 4      | 5      | 6       | 7 | 8 | dispatch                                  |
// |----------------------|----------|------------|----------|--------|--------|--------|---------|---|---|-------------------------------------------|
// | injectWomBacks       | data     | womIndex   | womRows  | F      | P Exec |        |         |   |   | threads numCycles                         |
// | fillOneExt           | p FpExt  | n (uint)   |          |        |        |        |         |   |   | threads n                                 |
// | prefixProductsChunks | io FpExt | n (uint)   | totals   |        |        |        |         |   |   | groups ceil(n/2048) x 256 threads (exact) |
// | prefixProductsTotals | totals   | nblocks    |          |        |        |        |         |   |   | threads 1                                 |
// | prefixProductsFixup  | io FpExt | n (uint)   | totals   |        |        |        |         |   |   | threads n - 2048 (only if nblocks > 1)    |
// | recInitRows          | buf u32  | P Row      |          |        |        |        |         |   |   | threads (cols << logRows) / 4             |
// | recFillTail          | buf u32  | P Row      |          |        |        |        |         |   |   | threads zk * cols                         |
// | recZeroizeRows       | buf u32  | P Row      |          |        |        |        |         |   |   | threads 2D (ceil(live/4), cols)           |
// | markHostChains       | cycles   | heads      | F        | n      | numCyc |        |         |   |   | threads n                                 |
// | scatterHostChains    | data u32 | womIndex   | womRows  | cycles | rows   | womIdx | srcWom  | F | P Scatter | threads numRows * dataCols        |
//
// P Exec = RecExecParams, P Row = RecRowParams, P Scatter = RecScatterParams (context.h).

#include <metal_stdlib>

#include "context.h"
#include "cuda2metal.h"

using namespace metal;

// ---------------------------------------------------------------------------------------------
// Witgen: injectWomBacks (ffi.cu `injectWomBacks`, cxx/ffi.cpp MachineContext::injectWomBacks).
// After the WOM rows are sorted and womIndex is exclusive-scanned, row `cycle - 1` of data
// columns 0..4 gets the sorted row just before cycle's first row (or zeros if there is none).

kernel void injectWomBacks(device Fp* data [[buffer(0)]],
                           const device uint32_t* womIndex [[buffer(1)]],
                           const device WomArgumentRow* womRows [[buffer(2)]],
                           device atomic_uint* fault [[buffer(3)]],
                           constant RecExecParams& p [[buffer(4)]],
                           uint cycle [[thread_position_in_grid]]) {
  if (cycle < 1 || cycle >= p.numCycles || cycle > p.totalCycles) {
    return;
  }
  uint32_t totalCycles = p.totalCycles;
  uint32_t idx = womIndex[cycle];
  uint32_t row = cycle - 1;
  if (idx != 0 && idx <= p.numWomRows) {
    WomArgumentRow prev = womRows[idx - 1];
    data[0 * totalCycles + row] = prev.addr;
    data[1 * totalCycles + row] = prev.value.elems[0];
    data[2 * totalCycles + row] = prev.value.elems[1];
    data[3 * totalCycles + row] = prev.value.elems[2];
    data[4 * totalCycles + row] = prev.value.elems[3];
  } else {
    // idx > numWomRows is a corrupt index (plan 2.9): fault, and write the zeros of idx == 0.
    r0_fault_flush(fault, idx != 0 ? 1u : 0u);
    data[0 * totalCycles + row] = Fp(0);
    data[1 * totalCycles + row] = Fp(0);
    data[2 * totalCycles + row] = Fp(0);
    data[3 * totalCycles + row] = Fp(0);
    data[4 * totalCycles + row] = Fp(0);
  }
}

// ---------------------------------------------------------------------------------------------
// Accum: fillOneExt + the SPPARK_NO_COOPERATIVE_LAUNCH prefix-product trio (ffi.cu
// `fillOneExt`, `prefixProductsChunks/Totals/Fixup`). In-place inclusive prefix products of an
// FpExt array without a grid-wide sync: each threadgroup scans a 2048-element chunk and writes
// its total, one thread scans the totals, then every chunk past the first is multiplied by the
// product of the chunks before it. FpExt multiplication is associative and exact, so the result
// is bit-identical to the sequential scan.

kernel void fillOneExt(device FpExt* p [[buffer(0)]],
                       constant uint32_t& n [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
  if (i < n) {
    p[i] = FpExt(1u);
  }
}

constant constexpr uint32_t kScanThreads = 256;
constant constexpr uint32_t kScanPerThread = 8;
constant constexpr uint32_t kScanChunk = kScanThreads * kScanPerThread;

// Must be dispatched with exactly kScanThreads threads per threadgroup and ceil(n / kScanChunk)
// threadgroups. Indices are 32-bit: n <= 2^32 - kScanChunk (the host checks).
[[max_total_threads_per_threadgroup(256)]]
kernel void prefixProductsChunks(device FpExt* io [[buffer(0)]],
                                 constant uint32_t& n [[buffer(1)]],
                                 device FpExt* totals [[buffer(2)]],
                                 uint t [[thread_position_in_threadgroup]],
                                 uint block [[threadgroup_position_in_grid]]) {
  threadgroup FpExt part[kScanThreads]; // 4 KB static
  uint32_t base = block * kScanChunk + t * kScanPerThread;
  FpExt run(1u);
  for (uint32_t k = 0; k < kScanPerThread; k++) {
    uint32_t i = base + k;
    if (i < n) {
      run = run * io[i];
      io[i] = run;
    }
  }
  part[t] = run;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Inclusive scan of the threads' products (Hillis-Steele), as in CUDA.
  for (uint32_t d = 1; d < kScanThreads; d <<= 1) {
    FpExt v = part[t];
    if (t >= d) {
      v = part[t - d] * v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    part[t] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (t > 0) {
    FpExt before = part[t - 1];
    for (uint32_t k = 0; k < kScanPerThread; k++) {
      uint32_t i = base + k;
      if (i < n) {
        io[i] = before * io[i];
      }
    }
  }
  if (t == kScanThreads - 1) {
    totals[block] = part[t];
  }
}

// One thread: totals[b] becomes the product of chunks 0..b (inclusive).
kernel void prefixProductsTotals(device FpExt* totals [[buffer(0)]],
                                 constant uint32_t& nblocks [[buffer(1)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid != 0) {
    return;
  }
  FpExt acc = totals[0];
  for (uint32_t b = 1; b < nblocks; b++) {
    acc = acc * totals[b];
    totals[b] = acc;
  }
}

// Thread gid handles element gid + kScanChunk (chunk 0 is already final).
kernel void prefixProductsFixup(device FpExt* io [[buffer(0)]],
                                constant uint32_t& n [[buffer(1)]],
                                const device FpExt* totals [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
  uint32_t i = gid + kScanChunk;
  if (gid < n && i < n) {
    io[i] = totals[i / kScanChunk - 1] * io[i];
  }
}

// ---------------------------------------------------------------------------------------------
// Witness-buffer row passes (ffi.cu `recInitRows/recFillTail/recZeroizeRows`, the HIP
// has_row_passes path). Column-major buffer, 2^logRows rows per column; the witgen writes only
// rows below `live`. Equivalent to the generic INVALID fill + ZK tail copy + zeroize.

// INVALID in rows [0, live) and zero in rows [live, 2^logRows), four words per thread.
// Requires logRows >= 2 (the host checks), so a quad never crosses a column.
kernel void recInitRows(device uint32_t* buf [[buffer(0)]],
                        constant RecRowParams& p [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
  ulong i = ulong(gid) * 4;
  ulong total = ulong(p.cols) << p.logRows;
  if (i >= total) {
    return;
  }
  uint32_t row = uint32_t(i & ((ulong(1) << p.logRows) - 1));
  uint32_t live = p.live;
  device uint4* q = reinterpret_cast<device uint4*>(buf + i);
  if (row + 4 <= live) {
    *q = uint4(kInvalidPattern);
  } else if (row >= live) {
    *q = uint4(0u);
  } else {
    for (uint32_t k = 0; k < 4; k++) {
      buf[i + k] = row + k < live ? kInvalidPattern : 0u;
    }
  }
}

// `value` (a raw Fp word) in the last `zk` rows of every column.
kernel void recFillTail(device uint32_t* buf [[buffer(0)]],
                        constant RecRowParams& p [[buffer(1)]],
                        uint t [[thread_position_in_grid]]) {
  if (t >= p.zk * p.cols) {
    return;
  }
  uint32_t col = t / p.zk, r = t % p.zk;
  buf[(ulong(col) << p.logRows) + (ulong(1) << p.logRows) - p.zk + r] = p.value;
}

// INVALID -> 0 in rows [0, live) of every column; gid.x = row quad, gid.y = column.
kernel void recZeroizeRows(device uint32_t* buf [[buffer(0)]],
                           constant RecRowParams& p [[buffer(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
  uint32_t row = gid.x * 4;
  uint32_t live = p.live;
  if (row >= live || gid.y >= p.cols) {
    return;
  }
  device uint32_t* col = buf + (ulong(gid.y) << p.logRows) + row;
  if (row + 4 <= live) {
    device uint4* q = reinterpret_cast<device uint4*>(col);
    uint4 v = *q;
    bool4 inv = v == uint4(kInvalidPattern);
    if (any(inv)) {
      *q = select(v, uint4(0u), inv);
    }
  } else {
    for (uint32_t k = 0; row + k < live; k++) {
      if (col[k] == kInvalidPattern) {
        col[k] = 0;
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Optional host-chain kernels (ffi.cu `markHostChains/scatterHostChains`). Round 1 does this on
// the CPU over the shared buffers (plan 4.3); these exist for C11 to measure against.

kernel void markHostChains(device PreflightCycle* cycles [[buffer(0)]],
                           const device uint32_t* heads [[buffer(1)]],
                           device atomic_uint* fault [[buffer(2)]],
                           constant uint32_t& n [[buffer(3)]],
                           constant uint32_t& numCycles [[buffer(4)]],
                           uint i [[thread_position_in_grid]]) {
  if (i >= n) {
    return;
  }
  uint32_t c = heads[i];
  if (c < numCycles) {
    cycles[c].isParSafe = 2;
  } else {
    r0_fault_flush(fault, 1u);
  }
}

kernel void scatterHostChains(device uint32_t* data [[buffer(0)]],
                              device uint32_t* womIndex [[buffer(1)]],
                              device WomArgumentRow* womRows [[buffer(2)]],
                              const device uint32_t* cycles [[buffer(3)]],
                              const device uint32_t* rows [[buffer(4)]],
                              const device uint32_t* womIdx [[buffer(5)]],
                              const device WomArgumentRow* srcWomRows [[buffer(6)]],
                              device atomic_uint* fault [[buffer(7)]],
                              constant RecScatterParams& p [[buffer(8)]],
                              uint t [[thread_position_in_grid]]) {
  if (p.dataCols == 0 || t >= p.numRows * p.dataCols) {
    return;
  }
  uint32_t i = t / p.dataCols, k = t % p.dataCols, c = cycles[i];
  if (c >= p.numCycles || c >= p.totalCycles) {
    r0_fault_flush(fault, 1u);
    return;
  }
  data[ulong(k) * p.totalCycles + c] = rows[t];
  if (k == 0) {
    womIndex[c] = womIdx[i];
  }
  if (k < kMaxWomRowsPerCycle) {
    womRows[ulong(c) * kMaxWomRowsPerCycle + k] =
        srcWomRows[ulong(i) * kMaxWomRowsPerCycle + k];
  }
}
