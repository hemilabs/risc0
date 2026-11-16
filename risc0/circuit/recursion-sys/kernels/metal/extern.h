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
/// Metal port of kernels/cuda/extern.cuh (+ the declarations of extern.h): the extern handlers
/// the generated step functions call (plan 4.3, item C1).
///
/// Signature the C2 translator targets (CUDA `(void* ctx, size_t cycle, const char* extra,
/// Fp* args, Fp* outs)`):
///   exec externs:  `(thread ExecCtx* ctx, uint32_t cycle, constant char* extra,
///                    thread Fp* args, thread Fp* outs)`
///   accum externs: same with `thread AccumCtx* ctx`.
/// `extra` is the string literal of the call site (MSL places literals in `constant`).
///
/// Differences from CUDA, all per plan 2.9 (bad input never reads or writes out of bounds):
/// - `assert(idx < kMaxWomRowsPerCycle)` in plonkWrite_wom becomes `ctx->bad` + skipped write.
/// - every index derived from trace data (`addr` in womRead, `iopIdx` in readIOPBody, the global
///   `idx` in plonkRead_wom, `cycle` in the accum externs) is range-checked; out of range sets
///   `ctx->bad = 1` and reads as zero instead of reading out of bounds.
/// - the CUDA no-op externs (readIOPHeader, womWrite, readCoefficients, log) stay no-ops.

#include "context.h"
#include "cuda2metal.h"

inline void extern_readIOPHeader(
    thread ExecCtx* ctx, uint32_t cycle, constant char* extra, thread Fp* args, thread Fp* outs) {
  // no-op (CUDA extern.cuh)
}

inline void extern_readIOPBody(
    thread ExecCtx* ctx, uint32_t cycle, constant char* extra, thread Fp* args, thread Fp* outs) {
  // CUDA: iopIdx = trace->cycles[cycle].iopIdx++; body = trace->iops[iopIdx].
  // `cycle` comes from the kernel's own bounded index, so only iopIdx (trace data) is clamped.
  uint32_t iopIdx = ctx->cycles[cycle].iopIdx++;
  FpExt body;
  if (iopIdx < ctx->numIops) {
    body = ctx->iops[iopIdx];
  } else {
    ctx->bad = 1;
  }
  outs[0] = body.elems[0];
  outs[1] = body.elems[1];
  outs[2] = body.elems[2];
  outs[3] = body.elems[3];
}

inline void extern_womRead(
    thread ExecCtx* ctx, uint32_t cycle, constant char* extra, thread Fp* args, thread Fp* outs) {
  uint32_t addr = args[0].asUInt32();
  FpExt val;
  if (addr < ctx->numWoms) {
    val = ctx->wom[addr];
  } else {
    ctx->bad = 1;
  }
  outs[0] = val.elems[0];
  outs[1] = val.elems[1];
  outs[2] = val.elems[2];
  outs[3] = val.elems[3];
}

inline void extern_womWrite(
    thread ExecCtx* ctx, uint32_t cycle, constant char* extra, thread Fp* args, thread Fp* outs) {
  // no-op (CUDA extern.cuh)
}

inline void extern_plonkWrite_wom(
    thread ExecCtx* ctx, uint32_t cycle, constant char* extra, thread Fp* args, thread Fp* outs) {
  // Rows of cycle c go to slots [c * 9, c * 9 + 9) in write order; womIndex[c] counts them.
  uint32_t idx = ctx->womIndex[cycle]++;
  uint32_t slot = cycle * kMaxWomRowsPerCycle + idx;
  if (idx >= kMaxWomRowsPerCycle || slot >= ctx->numWomRows) {
    ctx->bad = 1;
    return;
  }
  device WomArgumentRow& row = ctx->womRows[slot];
  row.addr = args[0];
  row.value.elems[0] = args[1];
  row.value.elems[1] = args[2];
  row.value.elems[2] = args[3];
  row.value.elems[3] = args[4];
}

inline void extern_plonkRead_wom(
    thread ExecCtx* ctx, uint32_t cycle, constant char* extra, thread Fp* args, thread Fp* outs) {
  // After the sort and the exclusive scan, womIndex[c] is a global index into the sorted rows.
  uint32_t idx = ctx->womIndex[cycle]++;
  WomArgumentRow row;
  if (idx < ctx->numWomRows) {
    row = ctx->womRows[idx];
  } else {
    ctx->bad = 1;
  }
  outs[0] = row.addr;
  outs[1] = row.value.elems[0];
  outs[2] = row.value.elems[1];
  outs[3] = row.value.elems[2];
  outs[4] = row.value.elems[3];
}

inline void extern_readCoefficients(
    thread ExecCtx* ctx, uint32_t cycle, constant char* extra, thread Fp* args, thread Fp* outs) {
  // no-op (CUDA extern.cuh)
}

inline void extern_log(
    thread ExecCtx* ctx, uint32_t cycle, constant char* extra, thread Fp* args, thread Fp* outs) {
  // no-op (CUDA extern.cuh)
}

inline void extern_plonkWriteAccum_wom(
    thread AccumCtx* ctx, uint32_t cycle, constant char* extra, thread Fp* args, thread Fp* outs) {
  if (cycle >= ctx->workCycles) {
    ctx->bad = 1;
    return;
  }
  device FpExt& dst = ctx->accum[cycle];
  dst.elems[0] = args[0];
  dst.elems[1] = args[1];
  dst.elems[2] = args[2];
  dst.elems[3] = args[3];
}

inline void extern_plonkReadAccum_wom(
    thread AccumCtx* ctx, uint32_t cycle, constant char* extra, thread Fp* args, thread Fp* outs) {
  FpExt value;
  if (cycle < ctx->workCycles) {
    value = ctx->accum[cycle];
  } else {
    ctx->bad = 1;
  }
  outs[0] = value.elems[0];
  outs[1] = value.elems[1];
  outs[2] = value.elems[2];
  outs[3] = value.elems[3];
}
