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

/// finalizeAccum fused with eltwise_zeroize_fp(accum) (INVALID -> 0), one pass over all rows.
/// Machine cells (columns [split, cols - 4)) of rows < lastCycle get the previous row's running
/// sums (the last 4 columns, already inclusive-scanned on the host) added, an unset (INVALID)
/// cell counting as 0; every other cell is only zeroized. prev[] (last 4 columns of row back1) is
/// read raw while row back1's thread may zeroize it concurrently; those cells are scan outputs and
/// never INVALID, so the zeroize cannot change them (as on CUDA).
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
    Fp prev[4];
    for (uint32_t k = 0; k < 4; k++) {
      prev[k] = buf[rv32im_cell(back1, cols - 4 + k, rows)];
    }
    for (uint32_t j = split; j < finEnd; j += 4) {
#pragma unroll
      for (uint32_t k = 0; k < 4; k++) {
        rv32im_index_t i = rv32im_cell(cycle, j + k, rows);
        Fp cur = buf[i];
        // An unset (INVALID) cell counts as 0: INVALID + prev[k] is unreduced when prev[k] is
        // 0, and the NTT mishandles it (see phase 3 of risc0_circuit_rv32im_cpu_accum).
        if (cur.asRaw() == Fp::INVALID) {
          cur = Fp(0);
        }
        buf[i] = cur + prev[k];
      }
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
