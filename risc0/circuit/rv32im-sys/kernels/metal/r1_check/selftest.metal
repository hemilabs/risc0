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
/// Self-test kernels for the R1 headers (not part of the crate's metallib): compiled by the
/// r1_check runner with `xcrun metal -std=metal3.2 -O2` (once plain, once with -DEF_WIDE) and
/// checked against a CPU model in src/main.rs. They use the headers the way the translated
/// rv32im code will: BoundLayout over `constant` layouts, the LOAD/STORE macros, map/reduce with
/// lambdas, extern_log with an initializer list, LookupTables and the fault flag.

#include "witgen.h"

using namespace metal;
using namespace rv32im_metal;

// ---------------------------------------------------------------------------------------------
// ef_fp.h / ef_fpext.h vs fp.h / fpext.h
// ---------------------------------------------------------------------------------------------

/// out[i * 16 + ...]: raw words of ef_add, ef_sub, ef_mul, ef_neg, inv_chain (5), ef_ext_mul (4),
/// inv_ext_chain (4), then 3 unused. bad[0] counts mismatches against fp.h/fpext.h's operators.
kernel void ef_selftest(const device uint* a [[buffer(0)]],
                        const device uint* b [[buffer(1)]],
                        device uint* out [[buffer(2)]],
                        device atomic_uint* bad [[buffer(3)]],
                        constant uint& n [[buffer(4)]],
                        uint i [[thread_position_in_grid]]) {
  if (i >= n) {
    return;
  }
  Fp x = Fp::fromRaw(a[i]);
  Fp y = Fp::fromRaw(b[i]);
  uint mism = 0;
  Fp r0 = ef_add(x, y), r1 = ef_sub(x, y), r2 = ef_mul(x, y), r3 = ef_neg(x), r4 = inv_chain(x);
  mism += r0 != x + y;
  mism += r1 != x - y;
  mism += r2 != x * y;
  mism += r3 != -x;
  mism += r4 != inv(x);
  uint j = (i + 1) % n, k = (i + 2) % n, l = (i + 3) % n;
  FpExt ex(x, Fp::fromRaw(a[j]), Fp::fromRaw(a[k]), Fp::fromRaw(a[l]));
  FpExt ey(y, Fp::fromRaw(b[j]), Fp::fromRaw(b[k]), Fp::fromRaw(b[l]));
  FpExt em = ef_ext_mul(ex, ey);
  FpExt ei = inv_ext_chain(ex);
  mism += em != ex * ey;
  mism += ei != inv(ex);
  device uint* o = out + i * 16;
  o[0] = r0.asRaw();
  o[1] = r1.asRaw();
  o[2] = r2.asRaw();
  o[3] = r3.asRaw();
  o[4] = r4.asRaw();
  for (uint e = 0; e < 4; e++) {
    o[5 + e] = em.elems[e].asRaw();
    o[9 + e] = ei.elems[e].asRaw();
  }
  if (mism) {
    atomic_fetch_add_explicit(bad, mism, memory_order_relaxed);
  }
}

// ---------------------------------------------------------------------------------------------
// witgen.h / buffers.h / tables.h / preflight.h
// ---------------------------------------------------------------------------------------------

struct PairLayout {
  Reg lo;
  Reg hi;
};

constant constexpr ::cuda::std::array<Reg, 4> kRegs = {{Reg(0), Reg(1), Reg(2), Reg(3)}};
constant constexpr PairLayout kPair = {Reg(1), Reg(3)};
constant constexpr Reg kOut0 = Reg(0);
constant constexpr Reg kOut4 = Reg(4);

struct WitParams {
  uint32_t rows;     // rows of data, pre, out
  uint32_t usePre;   // back loads from pre
  uint32_t zeroBack; // MutableBufObj zeroBack
  uint32_t _pad;
};

/// One thread per cycle. data/pre: rows x 4, global: 1 x 4, out: rows x 8 (written only through
/// BufferObj stores). Thread 0 also stores global col 3 = 77 and reads preflight fields.
kernel void witgen_selftest(device Fp* data [[buffer(0)]],
                            device Fp* pre [[buffer(1)]],
                            device Fp* global [[buffer(2)]],
                            device Fp* outp [[buffer(3)]],
                            device atomic_uint* t8 [[buffer(4)]],
                            device atomic_uint* t16 [[buffer(5)]],
                            device PreflightCycle* cycles [[buffer(6)]],
                            const device MemoryTransaction* txns [[buffer(7)]],
                            device atomic_uint* fault [[buffer(8)]],
                            constant WitParams& p [[buffer(9)]],
                            uint gid [[thread_position_in_grid]]) {
  if (gid >= p.rows) {
    return;
  }
  PreflightTrace pf;
  pf.cycles = cycles;
  pf.txns = txns;
  pf.bigintBytes = nullptr;
  pf.txnsLen = 0;
  pf.bigintBytesLen = 0;
  pf.tableSplitCycle = 0;
  LookupTables tables = makeLookupTables(t8, t16);
  ExecContext ctx(pf, tables, gid);

  BufferObj d = makeMutableBuf(makeBuffer(data, p.rows, 4),
                               p.usePre ? makeBuffer(pre, p.rows, 4) : nullBuffer(),
                               p.zeroBack);
  BufferObj g = makeGlobalBuf(makeBuffer(global, 1, 4));
  BufferObj o = makeMutableBuf(makeBuffer(outp, p.rows, 8));
  MutableBuf dp = &d;
  GlobalBuf gp = &g;

  auto regs = BIND_LAYOUT(kRegs, dp);
  auto pair = BIND_LAYOUT(kPair, dp);
  Val v0 = LOAD(LAYOUT_SUBSCRIPT(regs, 0), 0);          // data[0][c]
  Val v1 = LOAD(LAYOUT_LOOKUP(pair, lo), 1);            // col 1, back 1
  Val v2 = LOAD(LAYOUT_LOOKUP(pair, hi), 5);            // col 3, back 5 (zeroBack)
  Val v3 = getGlobal(ctx, gp, 2);                       // global col 2
  ::cuda::std::array<Val, 3> three = {{v0, v1, Val(gid)}};
  auto doubled = map(three, [&](Val x) { return x + x; });
  Val v4 = reduce(doubled, Val(0), [&](Val acc, Val x) { return acc * Val(3) + x; });
  // map/reduce over a bound layout: sum of back-2 loads of cols 0..3.
  ::cuda::std::array<Val, 4> ones = {{Val(1), Val(1), Val(1), Val(1)}};
  Val v5 = reduce(ones, Val(0), regs, [&](Val acc, Val w, BoundLayout<Reg> r) {
    return acc + w * LOAD(r, 2);
  });
  ExtVal e = LOAD_EXT(LAYOUT_SUBSCRIPT(regs, 0), 3);    // cols 0..3, back 3
  Val v6 = get(ctx, dp, 2, 0);                          // mutable get

  INVOKE_EXTERN(ctx, log, "selftest", std::initializer_list<Val>{v0, v1});
  INVOKE_EXTERN(ctx, log, "selftest", std::initializer_list<Val>{});
  EQZ(v0 - v0, "loc");
  EQZ(e - e, "loc");

  // Stores: out cols 0..3 via STORE/set, cols 4..7 via STORE_EXT.
  STORE(bindLayout(kOut0, &o), v0 + v1);
  set(ctx, &o, 1, v2);
  set(ctx, &o, 2, v3 + v4);
  set(ctx, &o, 3, v5 + v6 * isz(Val(0)) + inv_0(Val(gid + 1)) * bitAnd(Val(gid), Val(5)));
  STORE_EXT(bindLayout(kOut4, &o), e); // cols 4..7
  if (gid == 0) {
    setGlobal(ctx, gp, 3, Val(77));
  }

  // Lookups: one valid u8 and u16 lookup per cycle, and two bad ones.
  tables.lookupDelta(Val(8), Val(gid & 255), Val(1));
  tables.lookupDelta(Val(16), Val((gid * 7) & 0xffff), Val(1));
  tables.lookupDelta(Val(0), Val(123456), Val(1)); // cycle table: no-op
  if (gid == 5) {
    tables.lookupDelta(Val(8), Val(300), Val(1)); // index out of range
  }
  if (gid == 6) {
    (void)tables.lookupCurrent(Val(9), Val(0)); // bad table
  }
  r0_fault_flush(fault, tables.bad);
}
