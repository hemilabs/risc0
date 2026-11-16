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

#pragma once

/// \file
/// Metal port of kernels/cuda/context.h (plan 4.3, item C1).
///
/// CUDA builds one `ExecContext`/`AccumContext` in managed memory and passes a pointer to it to
/// every kernel. Metal kernels take explicit `[[buffer(n)]]` bindings plus a POD params struct
/// bound with `set_bytes` (plan 2.0 "Kernel context passing"), and build a **thread-local**
/// context of `device` pointers from them (`ExecCtx`, `AccumCtx` below). The generated step
/// functions (C2) take `thread ExecCtx* ctx` / `thread AccumCtx* ctx`; the extern handlers in
/// extern.h read the context and record bad inputs in `ctx->bad` (plan 2.9), which the kernel
/// wrapper ORs into the `fault` buffer once at the end (`r0_fault_flush`).
///
/// Every params struct here is mirrored by a `#[repr(C)]` Rust struct; sizes are static_asserted
/// on both sides.

#include "fpext.h"

constant constexpr uint32_t kBabyBearExtSize = 4;
constant constexpr uint32_t kMaxWomRowsPerCycle = 9;
constant constexpr uint32_t kInvalidPattern = 0xffffffff;

/// One write-once-memory argument row (CUDA context.h `WomArgumentRow`): 20 bytes, no padding,
/// so the host can sort it as `[u32; 5]` (CPU `sortWomFast` port, C6) and kernels index it
/// directly. `addr` is an Fp in Montgomery form exactly as the CUDA struct stores it.
struct WomArgumentRow {
  Fp addr;
  FpExt value;
};
static_assert(sizeof(WomArgumentRow) == 20, "WomArgumentRow must be 20 bytes (CUDA layout)");
static_assert(sizeof(FpExt) == 16, "FpExt must be 16 bytes");

/// CUDA `PreflightCycle` (8 bytes). `iopIdx` is advanced by extern_readIOPBody; `isParSafe` is 0
/// (sequential chain row), 1 (parallel-safe head) or 2 (head of a chain the host computed).
struct PreflightCycle {
  uint32_t iopIdx;
  uint32_t isParSafe;
};
static_assert(sizeof(PreflightCycle) == 8, "PreflightCycle must be 8 bytes");

/// Scalars of the witgen kernels (CUDA `ExecContext::totalCycles` + `PreflightTrace` sizes).
/// `numWomRows` is the element count of the `womRows` buffer (numCycles * kMaxWomRowsPerCycle
/// when the host allocates it as CUDA does); it bounds every womRows index.
struct RecExecParams {
  uint32_t totalCycles; // rows per column of ctrl/data (2^po2)
  uint32_t numCycles;   // trace cycles (<= totalCycles)
  uint32_t numWoms;     // elements of `wom`
  uint32_t numIops;     // elements of `iops`
  uint32_t numWomRows;  // elements of `womRows`
  uint32_t _pad0;
  uint32_t _pad1;
  uint32_t _pad2;
};
static_assert(sizeof(RecExecParams) == 32, "RecExecParams must be 32 bytes");

/// Scalars of the accum kernels (CUDA `AccumContext::{totalCycles, workCycles}`).
struct RecAccumParams {
  uint32_t totalCycles;
  uint32_t workCycles; // elements of the accum scratch
  uint32_t _pad0;
  uint32_t _pad1;
};
static_assert(sizeof(RecAccumParams) == 16, "RecAccumParams must be 16 bytes");

/// Thread-local witgen context (CUDA `ExecContext` + `PreflightTrace`, flattened). Build it with
/// `make_exec_ctx` in a kernel wrapper; pass `thread ExecCtx*` to the step functions.
///
/// Canonical binding order for kernels that need the whole context (C2 wrappers):
///   0 ctrl, 1 data, 2 global, 3 wom, 4 cycles, 5 iops, 6 womRows, 7 womIndex, 8 fault,
///   9 params (RecExecParams, set_bytes).
struct ExecCtx {
  device Fp* ctrl;
  device Fp* data;
  device Fp* global;
  const device FpExt* wom;
  device PreflightCycle* cycles;
  const device FpExt* iops;
  device WomArgumentRow* womRows;
  device uint32_t* womIndex;
  uint32_t totalCycles;
  uint32_t numCycles;
  uint32_t numWoms;
  uint32_t numIops;
  uint32_t numWomRows;
  /// Set to 1 by any extern or translated assert that sees a bad input (plan 2.9).
  uint32_t bad;

  bool isParSafeExec(uint32_t cycle) const { return cycles[cycle].isParSafe != 0; }
};

inline ExecCtx make_exec_ctx(device Fp* ctrl,
                             device Fp* data,
                             device Fp* global,
                             const device FpExt* wom,
                             device PreflightCycle* cycles,
                             const device FpExt* iops,
                             device WomArgumentRow* womRows,
                             device uint32_t* womIndex,
                             constant RecExecParams& p) {
  ExecCtx ctx;
  ctx.ctrl = ctrl;
  ctx.data = data;
  ctx.global = global;
  ctx.wom = wom;
  ctx.cycles = cycles;
  ctx.iops = iops;
  ctx.womRows = womRows;
  ctx.womIndex = womIndex;
  ctx.totalCycles = p.totalCycles;
  ctx.numCycles = p.numCycles;
  ctx.numWoms = p.numWoms;
  ctx.numIops = p.numIops;
  ctx.numWomRows = p.numWomRows;
  ctx.bad = 0;
  return ctx;
}

/// Thread-local accum context (CUDA `AccumContext`).
///
/// Canonical binding order for the accum wrappers (C2):
///   0 ctrl, 1 global, 2 data, 3 mix, 4 accum (the circuit's accum group), 5 accumTmp (the
///   per-cycle FpExt scratch, `AccumContext::accum`), 6 fault, 7 params (RecAccumParams).
struct AccumCtx {
  device Fp* ctrl;
  device Fp* global;
  device Fp* data;
  device Fp* mix;
  device Fp* accumGroup;
  device FpExt* accum;
  uint32_t totalCycles;
  uint32_t workCycles;
  uint32_t bad;
};

inline AccumCtx make_accum_ctx(device Fp* ctrl,
                               device Fp* global,
                               device Fp* data,
                               device Fp* mix,
                               device Fp* accumGroup,
                               device FpExt* accum,
                               constant RecAccumParams& p) {
  AccumCtx ctx;
  ctx.ctrl = ctrl;
  ctx.global = global;
  ctx.data = data;
  ctx.mix = mix;
  ctx.accumGroup = accumGroup;
  ctx.accum = accum;
  ctx.totalCycles = p.totalCycles;
  ctx.workCycles = p.workCycles;
  ctx.bad = 0;
  return ctx;
}

/// Params of the witness-buffer row passes (CUDA recInitRows/recFillTail/recZeroizeRows
/// arguments): a column-major buffer of `cols` columns with 2^logRows rows each.
struct RecRowParams {
  uint32_t logRows;
  uint32_t live;  // rows the witgen writes (its cycle count)
  uint32_t cols;
  uint32_t zk;    // rows at the end of every column that get `value` (recFillTail)
  uint32_t value; // raw Fp word (recFillTail)
  uint32_t _pad0;
  uint32_t _pad1;
  uint32_t _pad2;
};
static_assert(sizeof(RecRowParams) == 32, "RecRowParams must be 32 bytes");

/// Params of scatterHostChains (CUDA `HostChains` scalars).
struct RecScatterParams {
  uint32_t numRows;     // host chain rows
  uint32_t dataCols;    // data columns per row
  uint32_t totalCycles; // rows per data column
  uint32_t numCycles;   // trace cycles: bounds every target cycle
};
static_assert(sizeof(RecScatterParams) == 16, "RecScatterParams must be 16 bytes");
