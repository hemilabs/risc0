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
/// Runtime support for the generated rv32im step functions on Metal (port of
/// kernels/cuda/witgen.h, its non-virtual CUDA branch, with the Intel SYCL port intel/witgen.h as
/// the precedent for removing virtual dispatch and spelling every address space).
///
/// What the translated code (R2's cuda2metal.py, R3's .metal.inc files) can rely on:
/// - `Val`/`ExtVal` = fp.h's `Fp`/fpext.h's `FpExt` (global types, no namespace).
/// - `ExecContext` (thread): `preflight`, `tables` (thread references) and `cycle`. Generated
///   signatures take `thread ExecContext& ctx`.
/// - `MutableBuf`/`GlobalBuf` = `thread BufferObj*`. One concrete, kind-tagged `BufferObj`
///   replaces CUDA's base class + `MutableBufObj`/`GlobalBufObj` (MSL has no virtual functions,
///   and a single struct needs no static_cast); kernels build them with `makeMutableBuf(...)` and
///   `makeGlobalBuf(...)` where CUDA wrote `MutableBufObj data(...)` / `GlobalBufObj g(...)`.
/// - `BoundLayout<T>` holds a `const constant T&`: every layout is a program-scope
///   `constant constexpr` table (R3 emits kLayout_* that way). MSL 3.2 is C++14, so there is no
///   class template argument deduction: BIND_LAYOUT / LAYOUT_LOOKUP / LAYOUT_SUBSCRIPT expand to
///   `bindLayout(...)`, and generated code that names `BoundLayout(x, y)` directly must call
///   `bindLayout(x, y)` instead.
/// - map/reduce over `::cuda::std::array` (r0::array) take functors; MSL 3.2 accepts lambdas
///   (verified with metal 32023.922), so the generated lambdas need no expansion for these.
/// - String literals are `constant char*` (eqz, extern_log, extern_assert).
/// - The extern_* functions are declared here and defined by the TU that owns them (R5/R8), as
///   in CUDA (ffi.cu / witgen_tu.cu).
/// - The generated .inc files (defs/types/layout.metal.inc in this directory: cuda2metal.py
///   `incs` translations of the zirgen .cu.inc files, re-checked by build.rs on every build)
///   are not included by default: a TU defines RV32IM_METAL_INCLUDE_GENERATED to include them at
///   the end of the namespace, as CUDA's witgen.h includes the .cu.inc files. Layouts are
///   `constant constexpr` tables; r0::array's `operator[]` is not constexpr, so a constant
///   expression over a layout array spells `.elems[i]` (CUDA's
///   `kLayout_TopAccum.columns[0].col` -> `kLayout_TopAccum.columns.elems[0].col`).
///
/// Defines a TU may set: RISC0_ROWS_POW2 (rows are a power of two: mask instead of modulo, as in
/// CUDA), RV32IM_ORDERED_ROWS (buffers.h), RV32IM_WIDE_INDEX (cuda_compat.h).

#include "buffers.h"
#include "cuda_compat.h"
#include "ef_fp.h"
#include "ef_fpext.h"
#include "fp.h"
#include "fpext.h"
#include "preflight.h"
#include "tables.h"

#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-variable"

namespace rv32im_metal {

using Val = Fp;
using ExtVal = FpExt;

struct ExecContext {
  ExecContext(thread PreflightTrace& preflight, thread LookupTables& tables, uint32_t cycle)
      : preflight(preflight), tables(tables), cycle(cycle) {}
  thread PreflightTrace& preflight;
  thread LookupTables& tables;
  uint32_t cycle;
};

// ============================================================================================
// Buffer objects: one kind-tagged struct, no virtual dispatch.
// ============================================================================================

struct BufferObj {
  enum Kind : uint32_t { MUTABLE = 0, GLOBAL = 1 };

  uint32_t kind;
  Buffer buf;
  Buffer preDataBuf; // MUTABLE only: back > 0 loads come from here when its buf is non-null
  uint32_t zeroBack; // MUTABLE only: back > 0 loads of columns > zeroBack read 0 (accum)

  /// CUDA MutableBufObj::load.
  Val loadMutable(thread const ExecContext& ctx, uint32_t col, uint32_t back) const {
    if (zeroBack && col > zeroBack && back > 0) {
      return Val(0);
    }
    uint32_t rows = buf.rows;
#ifdef RISC0_ROWS_POW2
    uint32_t backRow = (ctx.cycle - back) & (rows - 1);
#else
    uint32_t backRow = ((rows & (rows - 1)) == 0) ? ((ctx.cycle - back) & (rows - 1))
                                                  : ((rows + ctx.cycle - back) % rows);
#endif
    if (back > 0 && preDataBuf.buf) {
      return preDataBuf.get(backRow, col);
    }
    return buf.get(backRow, col);
  }

  /// CUDA MutableBufObj::store.
  void storeMutable(thread const ExecContext& ctx, uint32_t col, Val val) const {
    buf.set(ctx.cycle, col, val);
  }

  /// CUDA GlobalBufObj::load / store: row 0.
  Val loadGlobal(uint32_t col) const { return buf.get(0, col); }
  void storeGlobal(uint32_t col, Val val) const { buf.set(0, col, val); }

  /// CUDA BufferObj::load / store: dispatch on the kind.
  Val load(thread const ExecContext& ctx, uint32_t col, uint32_t back) const {
    if (kind == GLOBAL) {
      return loadGlobal(col);
    }
    return loadMutable(ctx, col, back);
  }

  void store(thread const ExecContext& ctx, uint32_t col, Val val) const {
    if (kind == GLOBAL) {
      storeGlobal(col, val);
    } else {
      storeMutable(ctx, col, val);
    }
  }
};

/// CUDA `MutableBufObj(buf[, preDataBuf], zeroBack)`.
inline BufferObj makeMutableBuf(Buffer buf, Buffer preDataBuf, uint32_t zeroBack = 0) {
  BufferObj o;
  o.kind = BufferObj::MUTABLE;
  o.buf = buf;
  o.preDataBuf = preDataBuf;
  o.zeroBack = zeroBack;
  return o;
}

inline BufferObj makeMutableBuf(Buffer buf) {
  return makeMutableBuf(buf, nullBuffer(), 0);
}

/// CUDA `GlobalBufObj(buf)`.
inline BufferObj makeGlobalBuf(Buffer buf) {
  BufferObj o;
  o.kind = BufferObj::GLOBAL;
  o.buf = buf;
  o.preDataBuf = nullBuffer();
  o.zeroBack = 0;
  return o;
}

using MutableBuf = thread BufferObj*;
using GlobalBuf = thread BufferObj*;

template <typename T> struct BoundLayout {
  BoundLayout(const constant T& layout, thread BufferObj* buf) : layout(layout), buf(buf) {}

  const constant T& layout;
  thread BufferObj* buf;
};

/// CUDA `BoundLayout(layout, buf)` (class template argument deduction is C++17).
template <typename T>
inline BoundLayout<T> bindLayout(const constant T& layout, thread BufferObj* buf) {
  return BoundLayout<T>(layout, buf);
}

inline size_t to_size_t(Val v) {
  return v.asUInt32();
}

inline Val mod(Val a, Val b) {
  return Val(a.asUInt32() % b.asUInt32());
}

constant constexpr size_t EXT_SIZE = 4;

// Built in field operations
inline Val isz(Val x) {
  return Val(x == Val(0));
}

inline Val neg_0(Val x) {
  return -x;
}

inline Val inv_0(Val x) {
  return inv_chain(x);
}

inline ExtVal inv_0(ExtVal x) {
  return inv_ext_chain(x);
}

inline Val bitAnd(Val a, Val b) {
  return Val(a.asUInt32() & b.asUInt32());
}

inline Val inRange(Val low, Val mid, Val high) {
  return Val(low <= mid && mid < high);
}

// Non-fatal, as on CUDA and Intel: parallel witgen can see transient violations (union layout
// overwrites); eval_check verifies the constraints independently.
inline void eqz(thread ExecContext& ctx, Val a, constant char* loc) {}

inline void eqz(thread ExecContext& ctx, ExtVal a, constant char* loc) {}

// Define index type (used in back)
using Index = size_t;

struct Reg {
  constexpr Reg(size_t col) : col(uint32_t(col)) {}
  constexpr Reg(size_t col) constant : col(uint32_t(col)) {}
  uint32_t col; // CUDA: size_t (32 bits suffice: < cols)
};

#define BIND_LAYOUT(orig, buf) bindLayout(orig, buf)
#define LAYOUT_LOOKUP(orig, elem) bindLayout(orig.layout.elem, orig.buf)
#define LAYOUT_SUBSCRIPT(orig, index) bindLayout(orig.layout[index], orig.buf)
#define EQZ(val, loc) eqz(ctx, val, loc)

inline void store(thread ExecContext& ctx, BoundLayout<Reg> reg, Val val) {
  reg.buf->store(ctx, reg.layout.col, val);
}

/// CUDA casts to MutableBufObj: a mutable store at the current cycle whatever the kind.
inline void set(thread ExecContext& ctx, thread BufferObj* buf, size_t offset, Val val) {
  buf->storeMutable(ctx, offset, val);
}

inline void setGlobal(thread ExecContext& ctx, thread BufferObj* buf, size_t offset, Val val) {
  buf->storeGlobal(offset, val);
}

inline void storeExt(thread ExecContext& ctx, BoundLayout<Reg> reg, ExtVal val) {
  for (uint32_t i = 0; i < EXT_SIZE; i++) {
    reg.buf->store(ctx, reg.layout.col + i, val.elems[i]);
  }
}

inline Val load(thread ExecContext& ctx, BoundLayout<Reg> reg, size_t back) {
  return reg.buf->load(ctx, reg.layout.col, back);
}

inline ExtVal loadExt(thread ExecContext& ctx, BoundLayout<Reg> reg, size_t back) {
  Fp elems[EXT_SIZE];
  for (uint32_t i = 0; i < EXT_SIZE; i++) {
    elems[i] = reg.buf->load(ctx, reg.layout.col + i, back);
  }
  return FpExt(elems[0], elems[1], elems[2], elems[3]);
}

/// CUDA casts to MutableBufObj: a mutable load whatever the kind.
inline Val get(thread ExecContext& ctx, thread BufferObj* buf, size_t offset, size_t back) {
  return buf->loadMutable(ctx, offset, back);
}

inline Val getGlobal(thread ExecContext& ctx, thread BufferObj* buf, size_t offset) {
  return buf->loadGlobal(offset);
}

#define LOAD(reg, back) load(ctx, reg, back)
#define LOAD_EXT(reg, back) loadExt(ctx, reg, back)
#define STORE(reg, val) store(ctx, reg, val)
#define STORE_EXT(reg, val) storeExt(ctx, reg, val)

// Map + reduce support
template <typename T1, typename F, size_t N> inline auto map(::cuda::std::array<T1, N> a, F f) {
  ::cuda::std::array<decltype(f(a[0])), N> out;
  for (size_t i = 0; i < N; i++) {
    out[i] = f(a[i]);
  }
  return out;
}

template <typename T1, typename T2, typename F, size_t N>
inline auto map(::cuda::std::array<T1, N> a, ::cuda::std::array<T2, N> b, F f) {
  ::cuda::std::array<decltype(f(a[0], b[0])), N> out;
  for (size_t i = 0; i < N; i++) {
    out[i] = f(a[i], b[i]);
  }
  return out;
}

template <typename T1, typename T2, typename F, size_t N>
inline auto map(::cuda::std::array<T1, N> a, thread const BoundLayout<T2>& b, F f) {
  ::cuda::std::array<decltype(f(a[0], bindLayout(b.layout[0], b.buf))), N> out;
  for (size_t i = 0; i < N; i++) {
    out[i] = f(a[i], bindLayout(b.layout[i], b.buf));
  }
  return out;
}

template <typename T1, typename T2, typename F, size_t N>
inline auto reduce(::cuda::std::array<T1, N> elems, T2 start, F f) {
  T2 cur = start;
  for (size_t i = 0; i < N; i++) {
    cur = f(cur, elems[i]);
  }
  return cur;
}

template <typename T1, typename T2, typename T3, typename F, size_t N>
inline auto
reduce(::cuda::std::array<T1, N> elems, T2 start, thread const BoundLayout<T3>& b, F f) {
  T2 cur = start;
  for (size_t i = 0; i < N; i++) {
    cur = f(cur, elems[i], bindLayout(b.layout[i], b.buf));
  }
  return cur;
}

// All the extern handling (definitions live in the TU that runs the step functions).
#define INVOKE_EXTERN(ctx, name, ...) extern_##name(ctx, ##__VA_ARGS__)

::cuda::std::array<Val, 5> extern_getMemoryTxn(thread ExecContext& ctx, Val addrElem);
void extern_lookupDelta(thread ExecContext& ctx, Val table, Val index, Val count);
Val extern_lookupCurrent(thread ExecContext& ctx, Val table, Val index);
void extern_memoryDelta(
    thread ExecContext& ctx, Val addr, Val cycle, Val dataLow, Val dataHigh, Val count);
uint32_t extern_getDiffCount(thread ExecContext& ctx, Val cycle);
Val extern_isFirstCycle_0(thread ExecContext& ctx);
::cuda::std::array<Val, 4> extern_divide(thread ExecContext& ctx,
                                         Val numerLow,
                                         Val numerHigh,
                                         Val denomLow,
                                         Val denomHigh,
                                         Val signType);
void extern_print(thread ExecContext& ctx, Val v);
::cuda::std::array<Val, 2> extern_getMajorMinor(thread ExecContext& ctx);
Val extern_hostReadPrepare(thread ExecContext& ctx, Val fp, Val len);
Val extern_hostWrite(thread ExecContext& ctx, Val fdVal, Val addrLow, Val addrHigh, Val lenVal);
::cuda::std::array<Val, 2> extern_nextPagingIdx(thread ExecContext& ctx);
::cuda::std::array<Val, 16> extern_bigIntExtern(thread ExecContext& ctx);

template <typename T>
inline void extern_log(thread ExecContext& ctx, constant char* message, T vals) {}

inline void extern_assert(thread ExecContext& ctx, Val cond, constant char* message) {}

// Setup the basic field stuff
#define SET_FIELD(x) /**/

#ifdef RV32IM_METAL_INCLUDE_GENERATED
#include "defs.metal.inc"

#include "types.metal.inc"

#include "layout.metal.inc"
#endif

} // namespace rv32im_metal
