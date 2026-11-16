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

// Included first by every generated keccak witgen file ($OUT_DIR/metal/witgen_arm_<K>.metal,
// written by keccak-sys/build_metal_witgen.rs from the CUDA step code kernels/cuda/steps_*.cu).
// The Metal counterpart of kernels/cuda/witgen.h + buffers.h + the `nextStep`/`par_stepExec` part
// of kernels/cuda/ffi.cu (item K6, docs/metal-port/PLAN.md 4.4):
// - `BufferObj`'s virtual load/store (MutableBufObj / GlobalBufObj) become two small by-value
//   structs, `MutableBuf` and `GlobalBuf` (no virtual calls in MSL); the generated code only ever
//   calls `get`/`set` on a MutableBuf and `setGlobal` on a GlobalBuf (the translator checks it);
// - `Buffer::set`'s "Inconsistent set" and `Buffer::get`'s "Read of unset value" (an `assert` on
//   CUDA, an exception on the CPU) set a bit of the thread's `ctx.fault`, which the kernel ORs into
//   the `fault` buffer once at the end; the host turns a non-zero flag into `Err`. A rejected `set`
//   writes nothing, as the CPU path (which throws before writing);
// - the externs read the preimages as 32-bit words (a KeccakState is 25 u64, i.e. 50 words), so
//   no 64-bit integer math is needed;
// - indices are 32-bit: the host asserts rows * cols < 2^32 and the translator that every column
//   literal is below the buffer's column count.
//
// Bindings of every step_arm_<K> kernel (mirrored by WITGEN_BUF_* in $OUT_DIR/keccak_metal_witgen.rs):
//   buffer(0) data          device Fp*            witness, column-major: data[col * rows + row]
//   buffer(1) global        device Fp*            globals (one row)
//   buffer(2) run_order     const device uint*    cycle of each run slot (all buckets, flattened)
//   buffer(3) preimage_idx  const device uint*    preimage index of each run slot
//   buffer(4) preimages     const device uint*    preimages_count * 50 words
//   buffer(5) fault         device atomic_uint*   ORed with the WITGEN_FAULT_* bits
//   buffer(6) p             constant WitgenParams& set_bytes
// Thread t of a dispatch runs the slot `p.base + t` (t < p.count): one dispatch per
// (cycle_type, sub_type) bucket, with the arm kernel of that cycle type.

#pragma once

#include <metal_stdlib>

#include "fp.h"

using namespace metal;

// Threadgroup size limit of the arm kernels (the HAL may dispatch smaller groups).
#ifndef R0_KECCAK_WITGEN_TG
#define R0_KECCAK_WITGEN_TG 256
#endif

// How the generated step functions are emitted. By default every step stays a call
// (`noinline`), as CUDA's `__device__` functions in separate translation units are: the GPU
// back end then compiles each function once instead of one inlined body per arm, which cuts cold
// PSO creation of the 12 arms from ~207 s to ~30 s (19 s concurrently) with no measurable loss
// (docs/metal-port/notes/K6.md). -DR0_KECCAK_WITGEN_INLINE leaves inlining to the compiler.
#ifdef R0_KECCAK_WITGEN_INLINE
#define R0_KECCAK_WITGEN_FN static inline
#else
#define R0_KECCAK_WITGEN_FN static __attribute__((noinline))
#endif

// Fault bits (WITGEN_FAULT_* on the Rust side).
#define WITGEN_FAULT_INCONSISTENT_SET 1u // set() of a cell holding another valid value
#define WITGEN_FAULT_UNSET_READ 2u       // get() of an INVALID cell with checked reads
#define WITGEN_FAULT_WRONG_ARM 4u        // the cycle's one-hot selects an arm this kernel lacks
#define WITGEN_FAULT_BAD_SLOT 8u         // run_order / preimage_idx out of range
#define WITGEN_FAULT_BAD_PREIMAGE 16u    // getPreimage with no current preimage

/// Per-dispatch parameters (set_bytes; 32 bytes, WITGEN_PARAMS_SIZE on the Rust side).
struct WitgenParams {
  uint32_t rows;            // rows of `data` (the cycle count)
  uint32_t data_cols;       // columns of `data`
  uint32_t global_cols;     // columns of `global`
  uint32_t count;           // threads (run slots) in this dispatch
  uint32_t base;            // first run slot of this dispatch
  uint32_t preimages_count; // number of preimages
  uint32_t checked_reads;   // bit 0: data, bit 1: global
  uint32_t _pad;
};
static_assert(sizeof(WitgenParams) == 32, "WitgenParams must be 32 bytes");

namespace keccak_witgen {

using Val = Fp;

/// kernels/cuda/witgen.h `ExecContext`, plus the preflight it points to and the fault bits.
struct ExecContext {
  uint32_t cycle;
  uint32_t curPreimage;
  uint32_t preimagesSize;
  const device uint32_t* preimages;
  uint32_t fault;
};

/// kernels/cuda/witgen.h `MutableBufObj` over kernels/cuda/buffers.h `Buffer`.
struct MutableBuf {
  device Fp* buf;
  uint32_t rows;
  bool checkedReads;
};

/// kernels/cuda/witgen.h `GlobalBufObj` (one row).
struct GlobalBuf {
  device Fp* buf;
  bool checkedReads;
};

// Buffer::set: a cell may be written if INVALID or already equal to the value.
inline void bufSet(thread ExecContext& ctx, device Fp* buf, uint32_t idx, Val val) {
  Fp cur = buf[idx];
  if (cur != Fp::invalid() && cur != val) {
    ctx.fault |= WITGEN_FAULT_INCONSISTENT_SET;
    return;
  }
  buf[idx] = val;
}

// Buffer::get
inline Val bufGet(thread ExecContext& ctx, const device Fp* buf, uint32_t idx, bool checked) {
  Fp ret = buf[idx];
  if (checked && ret == Fp::invalid()) {
    ctx.fault |= WITGEN_FAULT_UNSET_READ;
  }
  return ret;
}

// MutableBufObj::load: rows before the first read as zero.
inline Val get(thread ExecContext& ctx, MutableBuf buf, uint32_t col, uint32_t back) {
  if (back > ctx.cycle) {
    return Val(0);
  }
  return bufGet(ctx, buf.buf, col * buf.rows + (ctx.cycle - back), buf.checkedReads);
}

// MutableBufObj::store
inline void set(thread ExecContext& ctx, MutableBuf buf, uint32_t col, Val val) {
  bufSet(ctx, buf.buf, col * buf.rows + ctx.cycle, val);
}

// GlobalBufObj::store
inline void setGlobal(thread ExecContext& ctx, GlobalBuf buf, uint32_t col, Val val) {
  bufSet(ctx, buf.buf, col, val);
}

inline uint32_t to_size_t(Val v) {
  return v.asUInt32();
}

inline Val isz(Val x) {
  return Val(x == Val(0) ? 1u : 0u);
}

inline Val inv_0(Val x) {
  return inv(x);
}

inline Val bitAnd(Val a, Val b) {
  return Val(a.asUInt32() & b.asUInt32());
}

inline Val extern_getCycle(thread ExecContext& ctx) {
  return Val(ctx.cycle);
}

// (preimages[cur][idx / 4] >> (16 * (idx % 4))) & 0xffff, on the little-endian 32-bit halves.
inline Val extern_getPreimage(thread ExecContext& ctx, Val idx) {
  uint32_t i = idx.asUInt32();
  uint32_t idxLow = i % 4;
  uint32_t idxHigh = i / 4;
  if (ctx.curPreimage >= ctx.preimagesSize || idxHigh >= 25) {
    // CUDA would read past the preimages; the generated code never asks then.
    ctx.fault |= WITGEN_FAULT_BAD_PREIMAGE;
    return Val(0);
  }
  uint32_t word = ctx.preimages[ctx.curPreimage * 50 + idxHigh * 2 + idxLow / 2];
  return Val((word >> (16 * (idxLow % 2))) & 0xffff);
}

inline Val extern_nextPreimage(thread ExecContext& ctx) {
  return Val(ctx.curPreimage != ctx.preimagesSize ? 1u : 0u);
}

} // namespace keccak_witgen

// The kernel of one arm: `nextStep` of kernels/cuda/ffi.cu for the run slot `p.base + gid`, with
// `TOP` (the generated step_Top variant of this arm) in place of step_Top.
#define R0_KECCAK_WITGEN_KERNEL(NAME, TOP)                                                         \
  [[max_total_threads_per_threadgroup(R0_KECCAK_WITGEN_TG)]]                                       \
  kernel void NAME(device Fp* data [[buffer(0)]],                                                  \
                   device Fp* global [[buffer(1)]],                                                \
                   const device uint32_t* run_order [[buffer(2)]],                                 \
                   const device uint32_t* preimage_idx [[buffer(3)]],                              \
                   const device uint32_t* preimages [[buffer(4)]],                                 \
                   device atomic_uint* fault [[buffer(5)]],                                        \
                   constant WitgenParams& p [[buffer(6)]],                                         \
                   uint gid [[thread_position_in_grid]]) {                                         \
    if (gid >= p.count) {                                                                          \
      return;                                                                                      \
    }                                                                                              \
    uint32_t slot = p.base + gid;                                                                  \
    keccak_witgen::ExecContext ctx;                                                                \
    ctx.cycle = run_order[slot];                                                                   \
    ctx.curPreimage = preimage_idx[slot];                                                          \
    ctx.preimagesSize = p.preimages_count;                                                         \
    ctx.preimages = preimages;                                                                     \
    ctx.fault = 0;                                                                                 \
    if (ctx.cycle >= p.rows || ctx.curPreimage > p.preimages_count) {                              \
      atomic_fetch_or_explicit(fault, WITGEN_FAULT_BAD_SLOT, memory_order_relaxed);                \
      return;                                                                                      \
    }                                                                                              \
    keccak_witgen::MutableBuf dataBuf = {data, p.rows, (p.checked_reads & 1u) != 0};               \
    keccak_witgen::GlobalBuf globalBuf = {global, (p.checked_reads & 2u) != 0};                    \
    TOP(ctx, dataBuf, globalBuf);                                                                  \
    if (ctx.fault != 0) {                                                                          \
      atomic_fetch_or_explicit(fault, ctx.fault, memory_order_relaxed);                            \
    }                                                                                              \
  }
