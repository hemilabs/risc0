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
/// cuda2metal.py --selftest stub: a minimal stand-in for the rv32im Metal runtime header
/// (`rv32im-sys/kernels/metal/witgen.h`, item R1), used only when that header is not on the
/// branch. It declares the API the translated step functions use with R1's spelling
/// (namespace `rv32im_metal`, `thread ExecContext&`, kind-tagged `BufferObj`, `bindLayout`,
/// `constant char*` strings, `ExecContext::tables.bad` fault bits) so the translator output can
/// be compile-checked. It is a compile fixture: nothing here is meant to run.

#include "cuda2metal.h"
#include "fp.h"
#include "fpext.h"

#include <metal_stdlib>

#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-variable"

namespace std {
template <typename T> struct initializer_list {
  constexpr initializer_list() {}
  template <typename... A> constexpr initializer_list(A...) {}
};
} // namespace std

// R1's ef_fp.h / ef_fpext.h define the addition-chain inverses at global scope.
inline Fp inv_chain(Fp x) {
  return inv(x);
}
inline FpExt inv_ext_chain(FpExt x) {
  return inv(x);
}

namespace rv32im_metal {

using Val = Fp;
using ExtVal = FpExt;

struct Buffer {
  device Fp* buf;
  uint32_t rows;
  uint32_t cols;
  Fp get(uint32_t row, uint32_t col) const { return buf[col * rows + row]; }
  void set(uint32_t row, uint32_t col, Fp v) const { buf[col * rows + row] = v; }
};

struct LookupTables {
  device metal::atomic_uint* tableU8;
  device metal::atomic_uint* tableU16;
  uint32_t bad;
};

struct PreflightCycle {
  uint32_t state;
  uint32_t pc;
  uint8_t major;
  uint8_t minor;
  uint8_t machineMode;
  uint8_t padding;
  uint32_t userCycle;
  uint32_t txnIdx;
  uint32_t pagingIdx;
  uint32_t bigintIdx;
  uint32_t diffCount[2];
};

struct PreflightTrace {
  const device uint32_t* cycles;
  uint32_t cyclesLen;
};

struct ExecContext {
  thread PreflightTrace& preflight;
  thread LookupTables& tables;
  uint32_t cycle;
};

struct BufferObj {
  uint32_t kind;
  Buffer buf;
  Buffer preDataBuf;
  uint32_t zeroBack;
  Val load(thread const ExecContext& ctx, uint32_t col, uint32_t back) const {
    return buf.get(kind ? 0 : (ctx.cycle - back) & (buf.rows - 1), col);
  }
  void store(thread const ExecContext& ctx, uint32_t col, Val val) const {
    buf.set(kind ? 0 : ctx.cycle, col, val);
  }
};

using MutableBuf = thread BufferObj*;
using GlobalBuf = thread BufferObj*;

template <typename T> struct BoundLayout {
  BoundLayout(const constant T& layout, thread BufferObj* buf) : layout(layout), buf(buf) {}
  const constant T& layout;
  thread BufferObj* buf;
};

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
inline Val isz(Val x) {
  return Val(x == Val(0));
}
inline Val neg_0(Val x) {
  return -x;
}
inline Val inv_0(Val x) {
  return inv(x);
}
inline ExtVal inv_0(ExtVal x) {
  return inv(x);
}
inline Val bitAnd(Val a, Val b) {
  return Val(a.asUInt32() & b.asUInt32());
}
inline Val inRange(Val low, Val mid, Val high) {
  return Val(low <= mid && mid < high);
}
inline void eqz(thread ExecContext& ctx, Val a, constant char* loc) {}
inline void eqz(thread ExecContext& ctx, ExtVal a, constant char* loc) {}

using Index = size_t;

struct Reg {
  constexpr Reg(size_t col) : col(uint32_t(col)) {}
  constexpr Reg(size_t col) constant : col(uint32_t(col)) {}
  uint32_t col;
};

#define BIND_LAYOUT(orig, buf) bindLayout(orig, buf)
#define LAYOUT_LOOKUP(orig, elem) bindLayout(orig.layout.elem, orig.buf)
#define LAYOUT_SUBSCRIPT(orig, index) bindLayout(orig.layout[index], orig.buf)
#define EQZ(val, loc) eqz(ctx, val, loc)

inline void store(thread ExecContext& ctx, BoundLayout<Reg> reg, Val val) {
  reg.buf->store(ctx, reg.layout.col, val);
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
  Fp e[EXT_SIZE];
  for (uint32_t i = 0; i < EXT_SIZE; i++) {
    e[i] = reg.buf->load(ctx, reg.layout.col + i, back);
  }
  return FpExt(e[0], e[1], e[2], e[3]);
}
inline void set(thread ExecContext& ctx, thread BufferObj* buf, size_t offset, Val val) {
  buf->store(ctx, offset, val);
}
inline void setGlobal(thread ExecContext& ctx, thread BufferObj* buf, size_t offset, Val val) {
  buf->store(ctx, offset, val);
}
inline Val get(thread ExecContext& ctx, thread BufferObj* buf, size_t offset, size_t back) {
  return buf->load(ctx, offset, back);
}
inline Val getGlobal(thread ExecContext& ctx, thread BufferObj* buf, size_t offset) {
  return buf->load(ctx, offset, 0);
}

#define LOAD(reg, back) load(ctx, reg, back)
#define LOAD_EXT(reg, back) loadExt(ctx, reg, back)
#define STORE(reg, val) store(ctx, reg, val)
#define STORE_EXT(reg, val) storeExt(ctx, reg, val)

// No map/reduce templates on purpose: cuda2metal.py expands every map/reduce site into a loop,
// so a site it missed fails to compile against this stub.

#define INVOKE_EXTERN(ctx, name, ...) extern_##name(ctx, ##__VA_ARGS__)

r0::array<Val, 5> extern_getMemoryTxn(thread ExecContext& ctx, Val addrElem);
void extern_lookupDelta(thread ExecContext& ctx, Val table, Val index, Val count);
Val extern_lookupCurrent(thread ExecContext& ctx, Val table, Val index);
void extern_memoryDelta(
    thread ExecContext& ctx, Val addr, Val cycle, Val dataLow, Val dataHigh, Val count);
uint32_t extern_getDiffCount(thread ExecContext& ctx, Val cycle);
Val extern_isFirstCycle_0(thread ExecContext& ctx);
r0::array<Val, 4> extern_divide(thread ExecContext& ctx,
                                Val numerLow,
                                Val numerHigh,
                                Val denomLow,
                                Val denomHigh,
                                Val signType);
void extern_print(thread ExecContext& ctx, Val v);
r0::array<Val, 2> extern_getMajorMinor(thread ExecContext& ctx);
Val extern_hostReadPrepare(thread ExecContext& ctx, Val fp, Val len);
Val extern_hostWrite(thread ExecContext& ctx, Val fdVal, Val addrLow, Val addrHigh, Val lenVal);
r0::array<Val, 2> extern_nextPagingIdx(thread ExecContext& ctx);
r0::array<Val, 16> extern_bigIntExtern(thread ExecContext& ctx);

template <typename T>
inline void extern_log(thread ExecContext& ctx, constant char* message, T vals) {}
inline void extern_assert(thread ExecContext& ctx, Val cond, constant char* message) {}

#define SET_FIELD(x) /**/

#ifdef RV32IM_METAL_INCLUDE_GENERATED
#include "defs.metal.inc"

#include "types.metal.inc"

#include "layout.metal.inc"
#endif

} // namespace rv32im_metal
