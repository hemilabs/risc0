// Intel SYCL witgen kernel wrapper + extern implementations.
// Ported from kernels/cuda/ffi.cu, adapted for SYCL device code.

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>

// sycl/sycl.hpp pulls in <cassert> which re-defines assert.
// We need assert to be a no-op for the step functions (which hit
// "unreachable mux arm" on padding cycles — benign, eval_check validates).
#ifdef assert
#undef assert
#endif
#define assert(x) ((void)0)

// Include Intel SYCL witgen headers (which include steps.h → all types)
#include "fp.h"
#include "fpext.h"

// The steps.cpp body is included via the amalgamation created by build.rs.
// This file provides the extern implementations and kernel wrappers.

namespace risc0::circuit::rv32im_v2::intel {

using Val = Fp;
using ExtVal = FpExt;

constexpr size_t kUserAccumSplit = 23;

// ============================================================================
// RISC-V integer division (matching CUDA's divide_rv32im)
// ============================================================================

inline std::pair<uint32_t, uint32_t> divide_rv32im(uint32_t numer, uint32_t denom, uint32_t signType) {
  uint32_t onesComp = (denom == 0) ? 0 : 1;
  uint32_t negNumer = 0, negDenom = 0;
  if (signType == 2) {
    // Signed
    negNumer = numer >> 31;
    negDenom = denom >> 31;
    if (negNumer) numer = -numer;
    if (negDenom) denom = -denom;
  }
  uint32_t quot, rem;
  if (denom == 0) {
    quot = 0xFFFFFFFF;
    rem = numer;
  } else {
    quot = numer / denom;
    rem = numer % denom;
  }
  uint32_t quotNegOut = negNumer ^ negDenom;
  uint32_t remNegOut = negNumer;
  if (quotNegOut) quot = -quot - onesComp;
  if (remNegOut) rem = -rem - onesComp;
  return {quot, rem};
}

// ============================================================================
// Extern function implementations (device-callable)
// ============================================================================

inline std::array<Val, 5> extern_getMemoryTxn(ExecContext& ctx, Val addrElem) {
  size_t txnIdx = ctx.preflight.cycles[ctx.cycle].txnIdx++;
  const MemoryTransaction& txn = ctx.preflight.txns[txnIdx];
  return {
      Val(txn.prevCycle),
      Val(txn.prevWord & 0xffff),
      Val(txn.prevWord >> 16),
      Val(txn.word & 0xffff),
      Val(txn.word >> 16),
  };
}

inline void extern_lookupDelta(ExecContext& ctx, Val table, Val index, Val count) {
  ctx.tables.lookupDelta(table, index, count);
}

inline Val extern_lookupCurrent(ExecContext& ctx, Val table, Val index) {
  return ctx.tables.lookupCurrent(table, index);
}

inline void extern_memoryDelta(ExecContext& ctx, Val addr, Val cycle, Val dataLow, Val dataHigh, Val count) {
  // no-op (same as CUDA)
}

inline uint32_t extern_getDiffCount(ExecContext& ctx, Val cycle) {
  uint32_t cycleU32 = cycle.asUInt32();
  return ctx.preflight.cycles[cycleU32 / 2].diffCount[cycleU32 % 2];
}

inline Val extern_isFirstCycle_0(ExecContext& ctx) {
  return Val(ctx.cycle == 0);
}

inline std::array<Val, 4> extern_divide(
    ExecContext& ctx, Val numerLow, Val numerHigh, Val denomLow, Val denomHigh, Val signType) {
  uint32_t numer = numerLow.asUInt32() | (numerHigh.asUInt32() << 16);
  uint32_t denom = denomLow.asUInt32() | (denomHigh.asUInt32() << 16);
  auto [quot, rem] = divide_rv32im(numer, denom, signType.asUInt32());
  return {Val(quot & 0xffff), Val(quot >> 16), Val(rem & 0xffff), Val(rem >> 16)};
}

inline void extern_print(ExecContext& ctx, Val v) {
  // no-op
}

inline std::array<Val, 2> extern_getMajorMinor(ExecContext& ctx) {
  return {Val(ctx.preflight.cycles[ctx.cycle].major),
          Val(ctx.preflight.cycles[ctx.cycle].minor)};
}

inline Val extern_hostReadPrepare(ExecContext& ctx, Val fp, Val len) {
  size_t txnIdx = ctx.preflight.cycles[ctx.cycle].txnIdx;
  return Val(ctx.preflight.txns[txnIdx].word);
}

inline Val extern_hostWrite(ExecContext& ctx, Val fdVal, Val addrLow, Val addrHigh, Val lenVal) {
  size_t txnIdx = ctx.preflight.cycles[ctx.cycle].txnIdx;
  return Val(ctx.preflight.txns[txnIdx].word);
}

inline std::array<Val, 2> extern_nextPagingIdx(ExecContext& ctx) {
  return {Val(ctx.preflight.cycles[ctx.cycle].pagingIdx),
          Val(ctx.preflight.cycles[ctx.cycle].machineMode)};
}

inline std::array<Val, 16> extern_bigIntExtern(ExecContext& ctx) {
  std::array<Val, 16> ret;
  size_t bigintIdx = ctx.preflight.cycles[ctx.cycle].bigintIdx;
  for (size_t i = 0; i < 16; i++) {
    ret[i] = Val(ctx.preflight.bigintBytes[bigintIdx + i]);
  }
  return ret;
}

// ============================================================================
// Device-side step dispatch (matches CUDA's nextStep)
// ============================================================================

inline void nextStep(Buffer& dataBuf, Buffer& preDataBuf, Buffer& globalBuf,
                     PreflightTrace& preflight, LookupTables& tables, uint32_t cycle) {
  ExecContext execCtx(preflight, tables, cycle);
  if (preDataBuf.buf == dataBuf.buf) {
    preDataBuf.buf = nullptr;
  }
  MutableBufObj data(dataBuf, preDataBuf);
  GlobalBufObj global(globalBuf);
  BufferObj dataBo(&data);
  BufferObj globalBo(&global);
  step_Top(execCtx, &dataBo, &globalBo);
}

} // namespace risc0::circuit::rv32im_v2::intel

// ============================================================================
// SYCL kernel launcher + FFI entry points
// ============================================================================

constexpr size_t kStepModeParallel = 0;
constexpr size_t kStepModeSeqForward = 1;
constexpr size_t kStepModeSeqReverse = 2;

// Device cache for persistent preflight allocations (grow-only)
struct IntelWitgenCache {
  // Device preflight data
  risc0::circuit::rv32im_v2::intel::PreflightCycle* d_cycles = nullptr;
  risc0::circuit::rv32im_v2::intel::MemoryTransaction* d_txns = nullptr;
  uint8_t* d_bigintBytes = nullptr;
  size_t cycles_cap = 0, txns_cap = 0, bigint_cap = 0;

  // Lookup tables
  uint32_t* d_tableU8 = nullptr;
  uint32_t* d_tableU16 = nullptr;
  bool tables_allocated = false;

  // Track whether preflight was uploaded by witgen (for accum reuse)
  bool preflight_on_device = false;

  void ensure(sycl::queue& q, size_t nCycles, size_t nTxns, size_t nBigint) {
    if (nCycles > cycles_cap) {
      if (d_cycles) sycl::free(d_cycles, q);
      d_cycles = sycl::malloc_device<risc0::circuit::rv32im_v2::intel::PreflightCycle>(nCycles, q);
      cycles_cap = nCycles;
    }
    if (nTxns > txns_cap) {
      if (d_txns) sycl::free(d_txns, q);
      d_txns = sycl::malloc_device<risc0::circuit::rv32im_v2::intel::MemoryTransaction>(nTxns, q);
      txns_cap = nTxns;
    }
    if (nBigint > bigint_cap) {
      if (d_bigintBytes) sycl::free(d_bigintBytes, q);
      d_bigintBytes = sycl::malloc_device<uint8_t>(nBigint > 0 ? nBigint : 16, q);
      bigint_cap = nBigint > 0 ? nBigint : 16;
    }
    if (!tables_allocated) {
      d_tableU8 = sycl::malloc_device<uint32_t>(256, q);
      d_tableU16 = sycl::malloc_device<uint32_t>(65536, q);
      tables_allocated = true;
    }
  }
};

static IntelWitgenCache g_cache;

extern "C" {

using namespace risc0::circuit::rv32im_v2::intel;

const char* risc0_circuit_rv32im_intel_witgen(
    void* queue_ptr,
    uint32_t mode,
    void* d_data_ptr, uint32_t data_rows, uint32_t data_cols,
    void* d_pre_data_ptr,
    void* d_global_ptr, uint32_t global_cols,
    const PreflightCycle* h_cycles, uint32_t cycles_len,
    const MemoryTransaction* h_txns, uint32_t txns_len,
    const uint8_t* h_bigint, uint32_t bigint_len,
    uint32_t table_split_cycle,
    uint32_t last_cycle)
{
  try {
    auto& q = *static_cast<sycl::queue*>(queue_ptr);
    auto t0 = std::chrono::steady_clock::now();

    // Ensure device allocations
    g_cache.ensure(q, cycles_len, txns_len, bigint_len);

    // Upload preflight data to device
    q.memcpy(g_cache.d_cycles, h_cycles, cycles_len * sizeof(PreflightCycle));
    q.memcpy(g_cache.d_txns, h_txns, txns_len * sizeof(MemoryTransaction));
    if (bigint_len > 0) {
      q.memcpy(g_cache.d_bigintBytes, h_bigint, bigint_len);
    }

    // Zero lookup tables
    q.memset(g_cache.d_tableU8, 0, 256 * sizeof(uint32_t));
    q.memset(g_cache.d_tableU16, 0, 65536 * sizeof(uint32_t));
    q.wait();
    g_cache.preflight_on_device = true;

    auto t1 = std::chrono::steady_clock::now();

    // Build device-side structs (captured by value in kernel lambda)
    Buffer dataBuf{static_cast<risc0::Fp*>(d_data_ptr), data_rows, data_cols, false};
    Buffer preDataBuf{static_cast<risc0::Fp*>(d_pre_data_ptr), data_rows, data_cols, false};
    Buffer globalBuf{static_cast<risc0::Fp*>(d_global_ptr), 1, global_cols, false};

    PreflightTrace pf;
    pf.cycles = g_cache.d_cycles;
    pf.txns = g_cache.d_txns;
    pf.bigintBytes = g_cache.d_bigintBytes;
    pf.txnsLen = txns_len;
    pf.bigintBytesLen = bigint_len;
    pf.tableSplitCycle = table_split_cycle;

    LookupTables tables;
    tables.tableU8 = g_cache.d_tableU8;
    tables.tableU16 = g_cache.d_tableU16;

    constexpr uint32_t WG_SIZE = 256;

    if (mode == kStepModeParallel) {
      uint32_t split = table_split_cycle;
      uint32_t phase2Count = last_cycle - split;

      // Phase 1: body cycles [0, split)
      if (split > 0) {
        uint32_t global1 = ((split + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
        q.parallel_for(sycl::nd_range<1>(global1, WG_SIZE),
            [=](sycl::nd_item<1> item) {
              uint32_t cycle = item.get_global_id(0);
              if (cycle >= split) return;
              Buffer db = dataBuf, pdb = preDataBuf, gb = globalBuf;
              PreflightTrace pft = pf;
              LookupTables tbl = tables;
              ExecContext execCtx(pft, tbl, cycle);
              if (pdb.buf == db.buf) pdb.buf = nullptr;
              MutableBufObj data(db, pdb);
              GlobalBufObj global(gb);
              BufferObj dataBo(&data);
              BufferObj globalBo(&global);
              step_Top(execCtx, &dataBo, &globalBo);
            });
      }

      // Phase 2: table-fill cycles [split, last_cycle)
      if (phase2Count > 0) {
        uint32_t global2 = ((phase2Count + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
        q.parallel_for(sycl::nd_range<1>(global2, WG_SIZE),
            [=](sycl::nd_item<1> item) {
              uint32_t cycle = item.get_global_id(0);
              if (cycle >= phase2Count) return;
              uint32_t actualCycle = split + cycle;
              Buffer db = dataBuf, pdb = preDataBuf, gb = globalBuf;
              PreflightTrace pft = pf;
              LookupTables tbl = tables;
              ExecContext execCtx(pft, tbl, actualCycle);
              if (pdb.buf == db.buf) pdb.buf = nullptr;
              MutableBufObj data(db, pdb);
              GlobalBufObj global(gb);
              BufferObj dataBo(&data);
              BufferObj globalBo(&global);
              step_Top(execCtx, &dataBo, &globalBo);
            });
      }

      q.wait();
    } else {
      // Sequential modes (for debugging)
      uint32_t global_size = WG_SIZE;
      q.parallel_for(sycl::nd_range<1>(global_size, WG_SIZE),
          [=](sycl::nd_item<1> item) {
            if (item.get_global_id(0) != 0) return;
            Buffer db = dataBuf, pdb = preDataBuf, gb = globalBuf;
            PreflightTrace pft = pf;
            LookupTables tbl = tables;
            // Null out pre_data for sequential modes
            Buffer nullBuf{nullptr, 0, 0, false};
            if (mode == kStepModeSeqForward) {
              for (uint32_t cycle = 0; cycle < last_cycle; cycle++) {
                ExecContext execCtx(pft, tbl, cycle);
                MutableBufObj data(db, nullBuf);
                GlobalBufObj global(gb);
                BufferObj dataBo(&data);
                BufferObj globalBo(&global);
                step_Top(execCtx, &dataBo, &globalBo);
              }
            } else {
              uint32_t split = pft.tableSplitCycle;
              for (uint32_t cycle = split; cycle-- > 0;) {
                ExecContext execCtx(pft, tbl, cycle);
                MutableBufObj data(db, nullBuf);
                GlobalBufObj global(gb);
                BufferObj dataBo(&data);
                BufferObj globalBo(&global);
                step_Top(execCtx, &dataBo, &globalBo);
              }
              for (uint32_t cycle = last_cycle; cycle-- > split;) {
                ExecContext execCtx(pft, tbl, cycle);
                MutableBufObj data(db, nullBuf);
                GlobalBufObj global(gb);
                BufferObj dataBo(&data);
                BufferObj globalBo(&global);
                step_Top(execCtx, &dataBo, &globalBo);
              }
            }
          });
      q.wait();
    }

    auto t2 = std::chrono::steady_clock::now();
    bool verbose = (getenv("RISC0_VERBOSE") != nullptr);
    if (verbose) {
      fprintf(stderr, "      [ffi_witgen_intel] upload: %.1fms, kernel: %.1fms\n",
              std::chrono::duration<double, std::milli>(t1 - t0).count(),
              std::chrono::duration<double, std::milli>(t2 - t1).count());
    }
    return nullptr;
  } catch (const sycl::exception& e) {
    static thread_local std::string err;
    err = std::string("SYCL witgen error: ") + e.what();
    return err.c_str();
  } catch (const std::exception& e) {
    static thread_local std::string err;
    err = std::string("witgen error: ") + e.what();
    return err.c_str();
  }
}

const char* risc0_circuit_rv32im_intel_accum(
    void* queue_ptr,
    void* d_data_ptr, uint32_t data_rows, uint32_t data_cols,
    void* d_accum_ptr, uint32_t accum_rows, uint32_t accum_cols,
    void* d_global_ptr, uint32_t global_cols,
    void* d_mix_ptr, uint32_t mix_cols,
    const PreflightCycle* h_cycles, uint32_t cycles_len,
    const MemoryTransaction* h_txns, uint32_t txns_len,
    const uint8_t* h_bigint, uint32_t bigint_len,
    uint32_t table_split_cycle,
    uint32_t last_cycle)
{
  try {
    auto& q = *static_cast<sycl::queue*>(queue_ptr);
    auto t0 = std::chrono::steady_clock::now();

    // Reuse cached preflight from witgen phase (already on device).
    // DO NOT re-upload — witgen modified txnIdx in-place on device.
    g_cache.ensure(q, cycles_len, txns_len, bigint_len);
    if (!g_cache.preflight_on_device) {
      // Standalone accum (no prior witgen) — must upload
      q.memcpy(g_cache.d_cycles, h_cycles, cycles_len * sizeof(PreflightCycle));
      q.memcpy(g_cache.d_txns, h_txns, txns_len * sizeof(MemoryTransaction));
      if (bigint_len > 0) {
        q.memcpy(g_cache.d_bigintBytes, h_bigint, bigint_len);
      }
    }
    // Zero lookup tables for accum phase
    q.memset(g_cache.d_tableU8, 0, 256 * sizeof(uint32_t));
    q.memset(g_cache.d_tableU16, 0, 65536 * sizeof(uint32_t));
    q.wait();

    auto t1 = std::chrono::steady_clock::now();

    Buffer dataBuf{static_cast<risc0::Fp*>(d_data_ptr), data_rows, data_cols, false};
    Buffer accumBuf{static_cast<risc0::Fp*>(d_accum_ptr), accum_rows, accum_cols, false};
    Buffer globalBuf{static_cast<risc0::Fp*>(d_global_ptr), 1, global_cols, false};
    Buffer mixBuf{static_cast<risc0::Fp*>(d_mix_ptr), 1, mix_cols, false};

    PreflightTrace pf;
    pf.cycles = g_cache.d_cycles;
    pf.txns = g_cache.d_txns;
    pf.bigintBytes = g_cache.d_bigintBytes;
    pf.txnsLen = txns_len;
    pf.bigintBytesLen = bigint_len;
    pf.tableSplitCycle = table_split_cycle;

    LookupTables tables;
    tables.tableU8 = g_cache.d_tableU8;
    tables.tableU16 = g_cache.d_tableU16;

    constexpr uint32_t WG_SIZE = 256;

    // Phase 1: step_TopAccum — parallel, one work-item per cycle
    {
      uint32_t global_size = ((last_cycle + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
      q.parallel_for(sycl::nd_range<1>(global_size, WG_SIZE),
          [=](sycl::nd_item<1> item) {
            uint32_t cycle = item.get_global_id(0);
            if (cycle >= last_cycle) return;
            Buffer db = dataBuf, ab = accumBuf, gb = globalBuf, mb = mixBuf;
            PreflightTrace pft = pf;
            LookupTables tbl = tables;
            ExecContext execCtx(pft, tbl, cycle);
            MutableBufObj data(db);
            MutableBufObj accum(ab, kUserAccumSplit);
            GlobalBufObj global(gb);
            GlobalBufObj mix(mb);
            BufferObj accumBo(&accum);
            BufferObj dataBo(&data);
            BufferObj globalBo(&global);
            BufferObj mixBo(&mix);
            step_TopAccum(execCtx, &accumBo, &dataBo, &globalBo, &mixBo);
          });
      q.wait();
    }

    auto t2 = std::chrono::steady_clock::now();

    // Phase 2: inclusive scan on last 4 columns (CPU — download, scan, upload)
    // The scan operator is Fp::operator+ (ADDITIVE, not multiplicative).
    {
      risc0::Fp* d_accum = static_cast<risc0::Fp*>(d_accum_ptr);
      size_t rows = accum_rows;
      size_t cols = accum_cols;

      for (size_t j = 0; j < 4; j++) {
        size_t col = cols - 4 + j;
        size_t offset = col * rows;
        // Download this column
        std::vector<risc0::Fp> col_data(last_cycle);
        q.memcpy(col_data.data(), d_accum + offset, last_cycle * sizeof(risc0::Fp));
        q.wait();
        // Sequential inclusive scan (Fp addition)
        for (size_t i = 1; i < last_cycle; i++) {
          col_data[i] = col_data[i] + col_data[i - 1];
        }
        // Upload back
        q.memcpy(d_accum + offset, col_data.data(), last_cycle * sizeof(risc0::Fp));
        q.wait();
      }
    }

    auto t3 = std::chrono::steady_clock::now();

    // Phase 3: finalizeAccum — parallel, add prefix sums to machine columns
    {
      uint32_t global_size = ((last_cycle + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;
      q.parallel_for(sycl::nd_range<1>(global_size, WG_SIZE),
          [=](sycl::nd_item<1> item) {
            uint32_t cycle = item.get_global_id(0);
            if (cycle >= last_cycle) return;
            Buffer ab = accumBuf;
            size_t machineColumns = (ab.cols - kUserAccumSplit) / 4;
            size_t back1 = (cycle + last_cycle - 1) % last_cycle;
            risc0::Fp prev[4];
            for (size_t k = 0; k < 4; k++) {
              prev[k] = ab.get(back1, ab.cols - 4 + k);
            }
            for (size_t j = 0; j < machineColumns - 1; j++) {
              for (size_t k = 0; k < 4; k++) {
                size_t col = kUserAccumSplit + j * 4 + k;
                ab.set(cycle, col, ab.get(cycle, col) + prev[k]);
              }
            }
          });
      q.wait();
    }

    auto t4 = std::chrono::steady_clock::now();

    bool verbose = (getenv("RISC0_VERBOSE") != nullptr);
    if (verbose) {
      fprintf(stderr, "      [ffi_accum_intel] phase1: %.1fms, scan: %.1fms, phase3: %.1fms\n",
              std::chrono::duration<double, std::milli>(t2 - t1).count(),
              std::chrono::duration<double, std::milli>(t3 - t2).count(),
              std::chrono::duration<double, std::milli>(t4 - t3).count());
    }
    return nullptr;
  } catch (const sycl::exception& e) {
    static thread_local std::string err;
    err = std::string("SYCL accum error: ") + e.what();
    return err.c_str();
  } catch (const std::exception& e) {
    static thread_local std::string err;
    err = std::string("accum error: ") + e.what();
    return err.c_str();
  }
}

} // extern "C"
