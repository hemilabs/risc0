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

/// \file
/// Small hand-ported rv32im kernels (plan 4.2, item R1).
///
/// | kernel               | CUDA source            | grid                         |
/// |----------------------|------------------------|------------------------------|
/// | injectCycleCols      | kernels/cuda/ffi.cu:674 | 1-D, n threads (one per row) |
/// | finalizeZeroizeAccum | kernels/cuda/ffi.cu:1190 | 1-D, rows threads           |
/// | fpScanReduce         | cub::DeviceScan::InclusiveScan (ffi.cu accum), R13 | groups (blocks, 4) |
/// | fpScanBlockSums      | as above                                           | groups (1, 4)      |
/// | fpScanApply          | as above                                           | groups (blocks, 4) |
/// | seq_fwd_probe        | seq_exec.h drivers with a probe step (tests only) | 1x1 per chunk |
/// | seq_rev_probe        | as above, reverse order                           | 1x1 per chunk |
///
/// Buffer bindings (explicit, plan 2.4); every Params struct is POD, passed with set_bytes and
/// mirrored #[repr(C)] on the Rust side with the sizes asserted below:
///
///   injectCycleCols:      0 data (device Fp*, rows x cols column-major)
///                         1 cycles (const device PreflightCycle*, >= n entries)
///                         2 InjectParams
///   finalizeZeroizeAccum: 0 accum (device Fp*, rows x cols column-major)
///                         1 FinalizeAccumParams
///   fpScanReduce:         0 buf (const device uint*), 1 sums, 2 fault (atomic_uint), 3 FpScanParams
///   fpScanBlockSums:      0 sums, 1 FpScanParams
///   fpScanApply:          0 buf, 1 sums (const), 2 FpScanParams
///   seq_fwd_probe/rev:    0 data (device Fp*, rows x 2), 1 counter (device uint*, 1 entry),
///                         2 SeqParams, 3 SeqProbeParams
///
/// The host asserts rows * cols < 2^32 (32-bit indexing, cuda_compat.h).

#include "cuda_compat.h"
#include "fp.h"
#include "preflight.h"
#include "seq_exec.h"

using namespace rv32im_metal;

// ---------------------------------------------------------------------------------------------
// injectCycleCols
// ---------------------------------------------------------------------------------------------

/// Data columns written by injectCycleCols (the host's `cycle_cols`) and the extent.
struct InjectParams {
  uint32_t n;       // rows to write: lastCycle
  uint32_t rows;    // rows of `data`
  uint32_t cCycle;  // column: cycle number
  uint32_t cPcLow;  // column: next pc & 0xffff
  uint32_t cPcHigh; // column: next pc >> 16
  uint32_t cState;  // column: next state
  uint32_t cMode;   // column: next machine mode
  uint32_t _pad;
};
RV32IM_ASSERT_SIZE(InjectParams, 32);

/// The five per-cycle columns the host injector used to scatter for every row (cycle, next pc
/// low/high, next state, next machine mode), written from the uploaded preflight cycles. Runs
/// after the back scatter and before par_stepExec, so the same values win as before.
kernel void injectCycleCols(device Fp* data [[buffer(0)]],
                            const device PreflightCycle* cycles [[buffer(1)]],
                            constant InjectParams& p [[buffer(2)]],
                            uint row [[thread_position_in_grid]]) {
  if (row >= p.n || row >= p.rows) {
    return;
  }
  const device PreflightCycle& c = cycles[row];
  uint32_t pc = c.pc;
  data[rv32im_cell(row, p.cCycle, p.rows)] = Fp(row);
  data[rv32im_cell(row, p.cPcLow, p.rows)] = Fp(pc & 0xffff);
  data[rv32im_cell(row, p.cPcHigh, p.rows)] = Fp(pc >> 16);
  data[rv32im_cell(row, p.cState, p.rows)] = Fp(c.state);
  data[rv32im_cell(row, p.cMode, p.rows)] = Fp(uint32_t(c.machineMode));
}

// ---------------------------------------------------------------------------------------------
// finalizeZeroizeAccum
// ---------------------------------------------------------------------------------------------

struct FinalizeAccumParams {
  uint32_t rows;           // rows of `accum` (threads)
  uint32_t cols;           // columns of `accum`
  uint32_t lastCycle;      // cycles in the segment (rows below it are finalized)
  uint32_t userAccumSplit; // kLayout_TopAccum.columns[0].col (CUDA kUserAccumSplit, 23 today)
};
RV32IM_ASSERT_SIZE(FinalizeAccumParams, 16);

/// INVALID -> 0 for one cell (CUDA eltwise_zeroize_fp).
inline void zeroizeCell(device Fp* buf, rv32im_index_t i) {
  Fp cur = buf[i];
  if (cur.asRaw() == Fp::INVALID) {
    buf[i] = Fp(0);
  }
}

/// One finalized machine cell: cur + prev, an unset (INVALID) cell counting as 0: INVALID +
/// prev is unreduced when prev is 0, and the NTT mishandles it (see phase 3 of
/// risc0_circuit_rv32im_cpu_accum).
static inline __attribute__((always_inline)) void finalizeCell(device Fp* buf, rv32im_index_t i,
                                                               Fp prev) {
  Fp cur = buf[i];
  if (cur.asRaw() == Fp::INVALID) {
    cur = Fp(0);
  }
  buf[i] = cur + prev;
}

/// finalizeAccum fused with eltwise_zeroize_fp(accum) (INVALID -> 0), one pass over all rows.
/// Machine cells (columns [split, cols - 4)) of rows < lastCycle get the previous row's running
/// sums (the last 4 columns, already inclusive-scanned by fpScanApply or the host) added, an
/// unset (INVALID) cell counting as 0; every other cell is only zeroized. prev0..prev3 (last 4
/// columns of row back1) are read raw while row back1's thread may zeroize them concurrently;
/// those cells are scan outputs and never INVALID, so the zeroize cannot change them (as on CUDA).
/// The CUDA kernel takes kUserAccumSplit from the generated layout; here it is a param (the host
/// passes the layout's value), so this kernel does not depend on the generated .inc files.
kernel void finalizeZeroizeAccum(device Fp* buf [[buffer(0)]],
                                 constant FinalizeAccumParams& p [[buffer(1)]],
                                 uint cycle [[thread_position_in_grid]]) {
  const uint32_t rows = p.rows, cols = p.cols, split = p.userAccumSplit;
  if (cycle >= rows) {
    return;
  }
  // Finalized columns: [split, finEnd), finEnd = split + (machineColumns - 1) * 4. Malformed
  // params (no complete machine column group, lastCycle past rows) only zeroize, where CUDA's
  // size_t arithmetic would wrap.
  const uint32_t machineColumns = (cols >= split + 4) ? (cols - split) / 4 : 0;
  const uint32_t finEnd = split + (machineColumns ? machineColumns - 1 : 0) * 4;
  if (cycle < p.lastCycle && machineColumns && p.lastCycle <= rows) {
    const uint32_t back1 = (cycle + p.lastCycle - 1) % p.lastCycle;
    // Four scalars, not a default-constructed `Fp prev[4]` filled in a loop (R13: that array was
    // ir_scan.py tier A at -O0, the R17 defect shape; the -O2 build had no finding).
    const Fp prev0 = buf[rv32im_cell(back1, cols - 4, rows)];
    const Fp prev1 = buf[rv32im_cell(back1, cols - 3, rows)];
    const Fp prev2 = buf[rv32im_cell(back1, cols - 2, rows)];
    const Fp prev3 = buf[rv32im_cell(back1, cols - 1, rows)];
    for (uint32_t j = split; j < finEnd; j += 4) {
      finalizeCell(buf, rv32im_cell(cycle, j + 0, rows), prev0);
      finalizeCell(buf, rv32im_cell(cycle, j + 1, rows), prev1);
      finalizeCell(buf, rv32im_cell(cycle, j + 2, rows), prev2);
      finalizeCell(buf, rv32im_cell(cycle, j + 3, rows), prev3);
    }
    for (uint32_t c = 0; c < split; c++) {
      zeroizeCell(buf, rv32im_cell(cycle, c, rows));
    }
    for (uint32_t c = finEnd; c < cols; c++) {
      zeroizeCell(buf, rv32im_cell(cycle, c, rows));
    }
  } else {
    for (uint32_t c = 0; c < cols; c++) {
      zeroizeCell(buf, rv32im_cell(cycle, c, rows));
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Fp inclusive prefix scans of the last 4 accum columns (R13; CUDA: 4 cub::DeviceScan::
// InclusiveScan in risc0_circuit_rv32im_cuda_accum, CPU: std::inclusive_scan)
// ---------------------------------------------------------------------------------------------
//
// Reduce-then-scan, all 4 columns per dispatch (threadgroup grid (blocks, columns)):
//   1. fpScanReduce:    the Fp sum of each block of `perThread * tg` rows -> sums[col][block];
//                       also the canonical-input check (fault bit RV32IM_FAULT_ACCUM_SCAN).
//   2. fpScanBlockSums: one threadgroup per column turns sums[col][*] into the exclusive
//                       prefix (each block's carry-in), in chunks with a running carry.
//   3. fpScanApply:     each block scans its rows (threadgroup scan) from its carry-in and
//                       writes them back in place.
// Fp addition of canonical words (< P) is the addition of the group Z/P, associative and
// commutative, and Fp::add of canonical words is canonical; so any grouping gives the same words
// as the host's sequential scan, bit for bit. A non-canonical input (INVALID: a row stepAccumA
// did not write) has no such guarantee (the CPU scan then depends on the order, as cub's does on
// CUDA), so pass 1 reports it through the fault word instead of producing a silently different
// result.
//
// Every value is a scalar uint32_t and every helper is always_inline (R17: no private aggregate
// written element-wise around calls). Threadgroup memory (host-sized, Rust accum.rs):
//   index 0: perThread * tg words (fpScanApply's block staging; unused by the others)
//   index 1: one word per simdgroup (<= 32: a threadgroup has at most 1024 threads and a
//            simdgroup at least 32 on Apple GPUs; checked on the host from the PSO)
// The simdgroup width comes from [[threads_per_simdgroup]], the threadgroup size from the
// dispatch, so nothing assumes a core count or a width.

/// Not in cuda_compat.h's list (owned by R1/R17): kept next to its only kernel, distinct from
/// every bit there.
#define RV32IM_FAULT_ACCUM_SCAN 0x100u // fpScanReduce: a scanned accum cell is not < P
static_assert((RV32IM_FAULT_ACCUM_SCAN &
               (RV32IM_FAULT_LOOKUP_TABLE | RV32IM_FAULT_LOOKUP_INDEX | RV32IM_FAULT_TXN_INDEX |
                RV32IM_FAULT_BIGINT_INDEX | RV32IM_FAULT_ASSERT | RV32IM_FAULT_CYCLE_INDEX |
                RV32IM_FAULT_ACCUM_PERM | RV32IM_FAULT_DIFF_COUNT_INDEX)) == 0,
              "RV32IM_FAULT_ACCUM_SCAN collides with a cuda_compat.h fault bit");

/// Rows per thread of one block (block = FP_SCAN_PER_THREAD * threads per threadgroup); the host
/// mirrors it (accum.rs FP_SCAN_PER_THREAD).
#define FP_SCAN_PER_THREAD 4u

struct FpScanParams {
  uint32_t n;      // rows to scan: [0, n) of each column
  uint32_t rows;   // rows of the buffer (column stride)
  uint32_t col0;   // first scanned column; threadgroup row y scans column col0 + y
  uint32_t blocks; // ceil(n / (FP_SCAN_PER_THREAD * threads per threadgroup))
};
RV32IM_ASSERT_SIZE(FpScanParams, 16);

/// Fp::add on raw words (fp.h), the host's fp_add.
static inline __attribute__((always_inline)) uint32_t fpScanAdd(uint32_t a, uint32_t b) {
  uint32_t r = a + b;
  return r >= Fp::P ? r - Fp::P : r;
}

/// a - b in Fp for canonical words (the group's subtraction; canonical result).
static inline __attribute__((always_inline)) uint32_t fpScanSub(uint32_t a, uint32_t b) {
  return fpScanAdd(a, b == 0 ? 0 : Fp::P - b);
}

/// Inclusive Fp scan of `x` over the lanes of the simdgroup (Hillis-Steele, log2(width) steps).
static inline __attribute__((always_inline)) uint32_t fpScanSimdInclusive(uint32_t x, uint lane,
                                                                          uint width) {
  for (uint d = 1; d < width; d <<= 1) {
    uint32_t y = metal::simd_shuffle_up(x, ushort(d));
    if (lane >= d) {
      x = fpScanAdd(x, y);
    }
  }
  return x;
}

/// Inclusive Fp scan of one word per thread over the threadgroup; `total` gets the sum of all.
/// Every thread of the threadgroup must call it (barriers). `wsum`: one word per simdgroup.
static inline __attribute__((always_inline)) uint32_t
fpScanGroupInclusive(uint32_t x, threadgroup uint32_t* wsum, uint lane, uint width, uint sid,
                     uint nsimd, thread uint32_t& total) {
  x = fpScanSimdInclusive(x, lane, width);
  // wsum may still be read by a previous call's threads.
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (lane == width - 1) {
    wsum[sid] = x;
  }
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (sid == 0) {
    uint32_t w = lane < nsimd ? wsum[lane] : 0;
    w = fpScanSimdInclusive(w, lane, width);
    if (lane < nsimd) {
      wsum[lane] = w;
    }
  }
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (sid > 0) {
    x = fpScanAdd(wsum[sid - 1], x);
  }
  total = wsum[nsimd - 1];
  return x;
}

/// Pass 1: sums[col * blocks + block] = Fp sum of the block's rows (< n); fault on x >= P.
kernel void fpScanReduce(const device uint32_t* buf [[buffer(0)]],
                         device uint32_t* sums [[buffer(1)]],
                         device metal::atomic_uint* fault [[buffer(2)]],
                         constant FpScanParams& p [[buffer(3)]],
                         threadgroup uint32_t* wsum [[threadgroup(1)]],
                         uint2 tgid [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint2 tgs [[threads_per_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint width [[threads_per_simdgroup]],
                         uint sid [[simdgroup_index_in_threadgroup]],
                         uint nsimd [[simdgroups_per_threadgroup]]) {
  const uint tg = tgs.x;
  const uint32_t block = tgid.x, col = p.col0 + tgid.y;
  const uint32_t base = block * (FP_SCAN_PER_THREAD * tg);
  const rv32im_index_t colBase = rv32im_cell(0, col, p.rows);
  uint32_t acc = 0;
  bool bad = false;
#pragma unroll
  for (uint32_t k = 0; k < FP_SCAN_PER_THREAD; k++) {
    // Coalesced: consecutive threads read consecutive rows. The order of the sum is irrelevant
    // for canonical words (see above).
    uint32_t i = base + k * tg + tid;
    if (i < p.n) {
      uint32_t v = buf[colBase + i];
      bad = bad || v >= Fp::P;
      acc = fpScanAdd(acc, v);
    }
  }
  if (bad) {
    metal::atomic_fetch_or_explicit(fault, RV32IM_FAULT_ACCUM_SCAN, metal::memory_order_relaxed);
  }
  uint32_t total;
  fpScanGroupInclusive(acc, wsum, lane, width, sid, nsimd, total);
  if (tid == 0) {
    sums[tgid.y * p.blocks + block] = total;
  }
}

/// Pass 2: sums[col][*] -> exclusive prefix sums (block b's carry-in), one threadgroup per
/// column (threadgroup grid (1, columns)), FP_SCAN_PER_THREAD * tg sums per round.
kernel void fpScanBlockSums(device uint32_t* sums [[buffer(0)]],
                            constant FpScanParams& p [[buffer(1)]],
                            threadgroup uint32_t* wsum [[threadgroup(1)]],
                            uint2 tgid [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint2 tgs [[threads_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint width [[threads_per_simdgroup]],
                            uint sid [[simdgroup_index_in_threadgroup]],
                            uint nsimd [[simdgroups_per_threadgroup]]) {
  const uint tg = tgs.x;
  device uint32_t* s = sums + tgid.y * p.blocks;
  uint32_t carry = 0;
  for (uint32_t base = 0; base < p.blocks; base += FP_SCAN_PER_THREAD * tg) {
    // Each thread owns FP_SCAN_PER_THREAD consecutive sums (scalars, no private array).
    const uint32_t i0 = base + tid * FP_SCAN_PER_THREAD;
    uint32_t v0 = i0 + 0 < p.blocks ? s[i0 + 0] : 0;
    uint32_t v1 = i0 + 1 < p.blocks ? s[i0 + 1] : 0;
    uint32_t v2 = i0 + 2 < p.blocks ? s[i0 + 2] : 0;
    uint32_t v3 = i0 + 3 < p.blocks ? s[i0 + 3] : 0;
    uint32_t t = fpScanAdd(fpScanAdd(v0, v1), fpScanAdd(v2, v3));
    uint32_t total;
    uint32_t incl = fpScanGroupInclusive(t, wsum, lane, width, sid, nsimd, total);
    // Exclusive prefix of this thread's first sum.
    uint32_t e0 = fpScanAdd(carry, fpScanSub(incl, t));
    uint32_t e1 = fpScanAdd(e0, v0);
    uint32_t e2 = fpScanAdd(e1, v1);
    uint32_t e3 = fpScanAdd(e2, v2);
    if (i0 + 0 < p.blocks) {
      s[i0 + 0] = e0;
    }
    if (i0 + 1 < p.blocks) {
      s[i0 + 1] = e1;
    }
    if (i0 + 2 < p.blocks) {
      s[i0 + 2] = e2;
    }
    if (i0 + 3 < p.blocks) {
      s[i0 + 3] = e3;
    }
    carry = fpScanAdd(carry, total);
  }
}

/// Pass 3: in-place inclusive scan of each block's rows (< n) from its carry-in sums[col][block].
/// Rows are staged through threadgroup memory so the global loads and stores stay coalesced
/// while each thread scans FP_SCAN_PER_THREAD consecutive rows.
kernel void fpScanApply(device uint32_t* buf [[buffer(0)]],
                        const device uint32_t* sums [[buffer(1)]],
                        constant FpScanParams& p [[buffer(2)]],
                        threadgroup uint32_t* stage [[threadgroup(0)]],
                        threadgroup uint32_t* wsum [[threadgroup(1)]],
                        uint2 tgid [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint2 tgs [[threads_per_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint width [[threads_per_simdgroup]],
                        uint sid [[simdgroup_index_in_threadgroup]],
                        uint nsimd [[simdgroups_per_threadgroup]]) {
  const uint tg = tgs.x;
  const uint32_t block = tgid.x, col = p.col0 + tgid.y;
  const uint32_t base = block * (FP_SCAN_PER_THREAD * tg);
  device uint32_t* c = buf + rv32im_cell(0, col, p.rows);
#pragma unroll
  for (uint32_t k = 0; k < FP_SCAN_PER_THREAD; k++) {
    uint32_t j = k * tg + tid;
    stage[j] = base + j < p.n ? c[base + j] : 0;
  }
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  const uint32_t j0 = tid * FP_SCAN_PER_THREAD;
  uint32_t v0 = stage[j0 + 0];
  uint32_t v1 = fpScanAdd(v0, stage[j0 + 1]);
  uint32_t v2 = fpScanAdd(v1, stage[j0 + 2]);
  uint32_t v3 = fpScanAdd(v2, stage[j0 + 3]);
  uint32_t total;
  uint32_t incl = fpScanGroupInclusive(v3, wsum, lane, width, sid, nsimd, total);
  // Carry-in of this thread's rows: the block's carry plus the sums of the earlier threads.
  uint32_t carry = fpScanAdd(sums[tgid.y * p.blocks + block], fpScanSub(incl, v3));
  // stage[] is only rewritten by its owner thread after every thread has read its rows (the
  // barriers inside fpScanGroupInclusive).
  stage[j0 + 0] = fpScanAdd(carry, v0);
  stage[j0 + 1] = fpScanAdd(carry, v1);
  stage[j0 + 2] = fpScanAdd(carry, v2);
  stage[j0 + 3] = fpScanAdd(carry, v3);
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
#pragma unroll
  for (uint32_t k = 0; k < FP_SCAN_PER_THREAD; k++) {
    uint32_t j = k * tg + tid;
    if (base + j < p.n) {
      c[base + j] = stage[j];
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Chunked sequential drivers, probe instantiation (seq_exec.h)
// ---------------------------------------------------------------------------------------------

struct SeqProbeParams {
  uint32_t rows; // rows of `data` (2 columns)
  uint32_t _pad[3];
};
RV32IM_ASSERT_SIZE(SeqProbeParams, 16);

/// A step whose output depends on the visiting order: column 0 of row `cycle` becomes
/// 3 * (column 0 of row cycle - 1, mod rows) + cycle + 1, and column 1 records the visit number.
struct SeqProbeStep {
  device Fp* data;
  device uint32_t* counter;
  uint32_t rows;

  void operator()(uint32_t cycle) thread {
    uint32_t seq = counter[0];
    counter[0] = seq + 1;
    data[rv32im_cell(cycle, 1, rows)] = Fp(seq);
    Fp prev = data[rv32im_cell((cycle + rows - 1) % rows, 0, rows)];
    data[rv32im_cell(cycle, 0, rows)] = prev * Fp(3) + Fp(cycle + 1);
  }
};

kernel void seq_fwd_probe(device Fp* data [[buffer(0)]],
                          device uint32_t* counter [[buffer(1)]],
                          constant SeqParams& p [[buffer(2)]],
                          constant SeqProbeParams& q [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
  if (gid != 0) {
    return;
  }
  SeqProbeStep step{data, counter, q.rows};
  rv32im_seq_fwd(step, p);
}

kernel void seq_rev_probe(device Fp* data [[buffer(0)]],
                          device uint32_t* counter [[buffer(1)]],
                          constant SeqParams& p [[buffer(2)]],
                          constant SeqProbeParams& q [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
  if (gid != 0) {
    return;
  }
  SeqProbeStep step{data, counter, q.rows};
  rv32im_seq_rev(step, p);
}
