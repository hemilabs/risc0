// Copyright 2024 RISC Zero, Inc.
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

#include "cuda.h"
#include "steps.cuh"
#include "witgen.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-braces"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-braces"
#endif

#include "vendor/nvtx3/nvtx3.hpp"

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <cstdint>
#include <cstdio>
#include <cuda/std/array>
#include <string.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace risc0::circuit::rv32im_v2::cuda {

constexpr size_t kUserAccumSplit = kLayout_TopAccum.columns[0].col;

struct ExecBuffers {
  Buffer global;
  Buffer data;
};

struct DeviceExecContext {
  Buffer* data;
  Buffer* global;
  PreflightTrace* preflight;
  LookupTables* tables;
};

struct AccumBuffers {
  Buffer data;
  Buffer accum;
  Buffer global;
  Buffer mix;
};

struct DeviceAccumContext {
  Buffer* data;
  Buffer* accum;
  Buffer* global;
  Buffer* mix;
  PreflightTrace* preflight;
  LookupTables* tables;
};

// Static cache for device-side allocations that persist across segments.
// Eliminates ~42 cudaMallocAsync/cudaFreeAsync pairs per segment and avoids
// re-uploading ~18MB of preflight data in the accum phase (which doesn't use it).
struct DeviceCache {
  // Exec context (device pointers, reused across segments)
  DeviceExecContext* exec_ctx = nullptr;
  Buffer* d_exec_data = nullptr;
  Buffer* d_exec_global = nullptr;
  PreflightTrace* d_exec_preflight = nullptr;
  LookupTables* d_exec_tables = nullptr;

  // Accum context (device pointers, reused across segments)
  DeviceAccumContext* accum_ctx = nullptr;
  Buffer* d_accum_data = nullptr;
  Buffer* d_accum_accum = nullptr;
  Buffer* d_accum_global = nullptr;
  Buffer* d_accum_mix = nullptr;
  PreflightTrace* d_accum_preflight = nullptr;
  LookupTables* d_accum_tables = nullptr;

  // Shared lookup table memory (reused, zeroed between phases)
  ::cuda::atomic<uint32_t>* d_tableU8 = nullptr;
  ::cuda::atomic<uint32_t>* d_tableU16 = nullptr;

  // Variable-size preflight data (grow-only, shared between exec and accum)
  PreflightCycle* d_cycles = nullptr;
  size_t cycles_capacity = 0;
  MemoryTransaction* d_txns = nullptr;
  size_t txns_capacity = 0;
  uint8_t* d_bigintBytes = nullptr;
  size_t bigintBytes_capacity = 0;


  void init(cudaStream_t stream) {
    if (exec_ctx) return;  // already initialized
    // Fixed-size allocations for exec context
    CUDA_OK(cudaMallocAsync(&exec_ctx, sizeof(DeviceExecContext), stream));
    CUDA_OK(cudaMallocAsync(&d_exec_data, sizeof(Buffer), stream));
    CUDA_OK(cudaMallocAsync(&d_exec_global, sizeof(Buffer), stream));
    CUDA_OK(cudaMallocAsync(&d_exec_preflight, sizeof(PreflightTrace), stream));
    CUDA_OK(cudaMallocAsync(&d_exec_tables, sizeof(LookupTables), stream));
    // Fixed-size allocations for accum context
    CUDA_OK(cudaMallocAsync(&accum_ctx, sizeof(DeviceAccumContext), stream));
    CUDA_OK(cudaMallocAsync(&d_accum_data, sizeof(Buffer), stream));
    CUDA_OK(cudaMallocAsync(&d_accum_accum, sizeof(Buffer), stream));
    CUDA_OK(cudaMallocAsync(&d_accum_global, sizeof(Buffer), stream));
    CUDA_OK(cudaMallocAsync(&d_accum_mix, sizeof(Buffer), stream));
    CUDA_OK(cudaMallocAsync(&d_accum_preflight, sizeof(PreflightTrace), stream));
    CUDA_OK(cudaMallocAsync(&d_accum_tables, sizeof(LookupTables), stream));
    // Shared lookup tables
    CUDA_OK(cudaMallocAsync(&d_tableU8, (1 << 8) * sizeof(uint32_t), stream));
    CUDA_OK(cudaMallocAsync(&d_tableU16, (1 << 16) * sizeof(uint32_t), stream));
  }

  // Grow-only reallocation for variable-size preflight buffers
  void ensure_preflight(cudaStream_t stream, size_t nCycles, size_t txnsLen, size_t bigintBytesLen) {
    if (nCycles > cycles_capacity) {
      if (d_cycles) CUDA_OK(cudaFreeAsync(d_cycles, stream));
      CUDA_OK(cudaMallocAsync(&d_cycles, nCycles * sizeof(PreflightCycle), stream));
      cycles_capacity = nCycles;
    }
    if (txnsLen > txns_capacity) {
      if (d_txns) CUDA_OK(cudaFreeAsync(d_txns, stream));
      CUDA_OK(cudaMallocAsync(&d_txns, txnsLen * sizeof(MemoryTransaction), stream));
      txns_capacity = txnsLen;
    }
    if (bigintBytesLen > bigintBytes_capacity) {
      if (d_bigintBytes) CUDA_OK(cudaFreeAsync(d_bigintBytes, stream));
      CUDA_OK(cudaMallocAsync(&d_bigintBytes, bigintBytesLen, stream));
      bigintBytes_capacity = bigintBytesLen;
    }
  }

  // Setup exec context: upload buffers + preflight data + zero tables
  DeviceExecContext* setup_exec(ExecBuffers* buffers, PreflightTrace* preflight,
                                size_t lastCycle, cudaStream_t stream) {
    init(stream);
    ensure_preflight(stream, lastCycle, preflight->txnsLen, preflight->bigintBytesLen);

    // Upload buffer descriptors
    CUDA_OK(cudaMemcpyAsync(d_exec_data, &buffers->data, sizeof(Buffer), cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(d_exec_global, &buffers->global, sizeof(Buffer), cudaMemcpyHostToDevice, stream));

    // Upload preflight data (async H2D, CUDA driver stages pageable memory).
    CUDA_OK(cudaMemcpyAsync(d_cycles, preflight->cycles,
                            lastCycle * sizeof(PreflightCycle), cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(d_txns, preflight->txns,
                            preflight->txnsLen * sizeof(MemoryTransaction), cudaMemcpyHostToDevice, stream));
    if (preflight->bigintBytesLen > 0) {
      CUDA_OK(cudaMemcpyAsync(d_bigintBytes, preflight->bigintBytes,
                              preflight->bigintBytesLen, cudaMemcpyHostToDevice, stream));
    }

    // Build and upload PreflightTrace descriptor
    PreflightTrace h_pf;
    h_pf.cycles = d_cycles;
    h_pf.txns = d_txns;
    h_pf.bigintBytes = d_bigintBytes;
    h_pf.txnsLen = preflight->txnsLen;
    h_pf.bigintBytesLen = preflight->bigintBytesLen;
    h_pf.tableSplitCycle = preflight->tableSplitCycle;
    CUDA_OK(cudaMemcpyAsync(d_exec_preflight, &h_pf, sizeof(PreflightTrace), cudaMemcpyHostToDevice, stream));

    // Zero and upload lookup tables
    CUDA_OK(cudaMemsetAsync(d_tableU8, 0, (1 << 8) * sizeof(uint32_t), stream));
    CUDA_OK(cudaMemsetAsync(d_tableU16, 0, (1 << 16) * sizeof(uint32_t), stream));
    LookupTables h_tables;
    h_tables.tableU8 = d_tableU8;
    h_tables.tableU16 = d_tableU16;
    CUDA_OK(cudaMemcpyAsync(d_exec_tables, &h_tables, sizeof(LookupTables), cudaMemcpyHostToDevice, stream));

    // Build and upload DeviceExecContext
    DeviceExecContext h_ctx;
    h_ctx.data = d_exec_data;
    h_ctx.global = d_exec_global;
    h_ctx.preflight = d_exec_preflight;
    h_ctx.tables = d_exec_tables;
    CUDA_OK(cudaMemcpyAsync(exec_ctx, &h_ctx, sizeof(DeviceExecContext), cudaMemcpyHostToDevice, stream));

    return exec_ctx;
  }

  // Setup accum context: upload buffers only, reuse preflight from exec phase.
  // The accum circuit (step_TopAccum) only uses lookupDelta/lookupCurrent - it never
  // accesses preflight cycles/txns/bigintBytes, so we skip the ~18MB re-upload.
  DeviceAccumContext* setup_accum(AccumBuffers* buffers, PreflightTrace* preflight,
                                  size_t lastCycle, cudaStream_t stream) {
    // Upload buffer descriptors
    CUDA_OK(cudaMemcpyAsync(d_accum_data, &buffers->data, sizeof(Buffer), cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(d_accum_accum, &buffers->accum, sizeof(Buffer), cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(d_accum_global, &buffers->global, sizeof(Buffer), cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(d_accum_mix, &buffers->mix, sizeof(Buffer), cudaMemcpyHostToDevice, stream));

    // Reuse preflight descriptor from exec phase (already on device)
    // Just update tableSplitCycle in case it changed (it shouldn't, but be safe)
    PreflightTrace h_pf;
    h_pf.cycles = d_cycles;
    h_pf.txns = d_txns;
    h_pf.bigintBytes = d_bigintBytes;
    h_pf.txnsLen = preflight->txnsLen;
    h_pf.bigintBytesLen = preflight->bigintBytesLen;
    h_pf.tableSplitCycle = preflight->tableSplitCycle;
    CUDA_OK(cudaMemcpyAsync(d_accum_preflight, &h_pf, sizeof(PreflightTrace), cudaMemcpyHostToDevice, stream));

    // Zero and upload lookup tables (must be fresh for accum phase)
    CUDA_OK(cudaMemsetAsync(d_tableU8, 0, (1 << 8) * sizeof(uint32_t), stream));
    CUDA_OK(cudaMemsetAsync(d_tableU16, 0, (1 << 16) * sizeof(uint32_t), stream));
    LookupTables h_tables;
    h_tables.tableU8 = d_tableU8;
    h_tables.tableU16 = d_tableU16;
    CUDA_OK(cudaMemcpyAsync(d_accum_tables, &h_tables, sizeof(LookupTables), cudaMemcpyHostToDevice, stream));

    // Build and upload DeviceAccumContext
    DeviceAccumContext h_ctx;
    h_ctx.data = d_accum_data;
    h_ctx.accum = d_accum_accum;
    h_ctx.global = d_accum_global;
    h_ctx.mix = d_accum_mix;
    h_ctx.preflight = d_accum_preflight;
    h_ctx.tables = d_accum_tables;
    CUDA_OK(cudaMemcpyAsync(accum_ctx, &h_ctx, sizeof(DeviceAccumContext), cudaMemcpyHostToDevice, stream));

    return accum_ctx;
  }
};

// Thread-local cache: each proving thread gets its own device-side buffers,
// enabling concurrent segment proving without data races.
static thread_local DeviceCache g_cache;

__device__ ::cuda::std::array<uint32_t, 2>
divide_rv32im(uint32_t numer, uint32_t denom, uint32_t signType) {
  uint32_t onesComp = (signType == 2);
  bool negNumer = signType && int32_t(numer) < 0;
  bool negDenom = signType == 1 && int32_t(denom) < 0;
  if (negNumer) {
    numer = -numer - onesComp;
  }
  if (negDenom) {
    denom = -denom - onesComp;
  }
  uint32_t quot;
  uint32_t rem;
  if (denom == 0) {
    quot = 0xffffffff;
    rem = numer;
  } else {
    quot = numer / denom;
    rem = numer % denom;
  }
  uint32_t quotNegOut = (negNumer ^ negDenom) - ((denom == 0) * negNumer);
  uint32_t remNegOut = negNumer;
  if (quotNegOut) {
    quot = -quot - onesComp;
  }
  if (remNegOut) {
    rem = -rem - onesComp;
  }
  return {quot, rem};
}

__device__ ::cuda::std::array<Val, 5> extern_getMemoryTxn(ExecContext& ctx, Val addrElem) {
  uint32_t addr = addrElem.asUInt32();
  size_t txnIdx = ctx.preflight.cycles[ctx.cycle].txnIdx++;
  const MemoryTransaction& txn = ctx.preflight.txns[txnIdx];
  // printf("getMemoryTxn(%lu, 0x%08x): txn(%u, 0x%08x, 0x%08x)\n",
  //        ctx.cycle,
  //        addr,
  //        txn.cycle,
  //        txn.addr,
  //        txn.word);

  if (txn.cycle / 2 != ctx.cycle) {
    printf("txn.cycle: %u, ctx.cycle: %zu\n", txn.cycle, ctx.cycle);
    assert(false && "txn cycle mismatch");
  }

  if (txn.addr != addr) {
    printf("txn.addr: 0x%08x, addr: 0x%08x\n", txn.addr, addr);
    assert(false && "memory peek not in preflight");
  }
  return {
      txn.prevCycle,
      txn.prevWord & 0xffff,
      txn.prevWord >> 16,
      txn.word & 0xffff,
      txn.word >> 16,
  };
}

__device__ void extern_lookupDelta(ExecContext& ctx, Val table, Val index, Val count) {
  // printf("lookupDelta(table: %u, index: %u, count: %u, P: %u)\n",
  //        table.asUInt32(),
  //        index.asUInt32(),
  //        count.asUInt32(),
  //        Fp::P);
  ctx.tables.lookupDelta(table, index, count);
}

__device__ Val extern_lookupCurrent(ExecContext& ctx, Val table, Val index) {
  Val ret = ctx.tables.lookupCurrent(table, index);
  // printf("lookupCurrent(table: %u, index: %u): %u\n",
  //        table.asUInt32(),
  //        index.asUInt32(),
  //        ret.asUInt32());
  return ret;
}

__device__ void
extern_memoryDelta(ExecContext& ctx, Val addr, Val cycle, Val dataLow, Val dataHigh, Val count) {
  // printf("memoryDelta\n");
  // ctx.tables.memoryDelta(
  //     addr.asUInt32(), cycle.asUInt32(), dataLow.asUInt32() | (dataHigh.asUInt32() << 16),
  //     count);
}

__device__ uint32_t extern_getDiffCount(ExecContext& ctx, Val cycle) {
  // printf("getDiffCount\n");
  uint32_t cycleU32 = cycle.asUInt32();
  return ctx.preflight.cycles[cycleU32 / 2].diffCount[cycleU32 % 2];
}

__device__ Val extern_isFirstCycle_0(ExecContext& ctx) {
  // printf("isFirstCycle\n");
  return ctx.cycle == 0;
}

__device__ ::cuda::std::array<Val, 4> extern_divide(
    ExecContext& ctx, Val numerLow, Val numerHigh, Val denomLow, Val denomHigh, Val signType) {
  // printf("divide\n");
  uint32_t numer = numerLow.asUInt32() | (numerHigh.asUInt32() << 16);
  uint32_t denom = denomLow.asUInt32() | (denomHigh.asUInt32() << 16);
  auto [quot, rem] = divide_rv32im(numer, denom, signType.asUInt32());
  ::cuda::std::array<Val, 4> ret;
  ret[0] = quot & 0xffff;
  ret[1] = quot >> 16;
  ret[2] = rem & 0xffff;
  ret[3] = rem >> 16;
  return ret;
}

__device__ void extern_print(ExecContext& ctx, Val v) {
  // printf("LOG: %u\n", v.asUInt32());
}

__device__ ::cuda::std::array<Val, 2> extern_getMajorMinor(ExecContext& ctx) {
  uint8_t major = ctx.preflight.cycles[ctx.cycle].major;
  uint8_t minor = ctx.preflight.cycles[ctx.cycle].minor;
  // printf("getMajorMinor: %u, %u\n", major, minor);
  return {major, minor};
}

__device__ Val extern_hostReadPrepare(ExecContext& ctx, Val fp, Val len) {
  size_t txnIdx = ctx.preflight.cycles[ctx.cycle].txnIdx;
  uint32_t word = ctx.preflight.txns[txnIdx].word;
  // printf("[%lu]: hostReadPrepare(txnIdx: %zu, word: 0x%08x)\n", ctx.cycle, txnIdx, word);
  return word;
}

__device__ Val
extern_hostWrite(ExecContext& ctx, Val fdVal, Val addrLow, Val addrHigh, Val lenVal) {
  // printf("hostWrite\n");
  size_t txnIdx = ctx.preflight.cycles[ctx.cycle].txnIdx;
  return ctx.preflight.txns[txnIdx].word;
}

__device__ ::cuda::std::array<Val, 2> extern_nextPagingIdx(ExecContext& ctx) {
  uint32_t pagingIdx = ctx.preflight.cycles[ctx.cycle].pagingIdx;
  uint32_t machineMode = ctx.preflight.cycles[ctx.cycle].machineMode;
  // printf("nextPagingIdx: (0x%05x, %u)\n", pagingIdx, machineMode);
  return {pagingIdx, machineMode};
}

__device__ ::cuda::std::array<Val, 16> extern_bigIntExtern(ExecContext& ctx) {
  ::cuda::std::array<Val, 16> ret;
  size_t bigintIdx = ctx.preflight.cycles[ctx.cycle].bigintIdx;
  for (size_t i = 0; i < 16; i++) {
    ret[i] = ctx.preflight.bigintBytes[bigintIdx + i];
  }
  return ret;
}

__device__ void nextStep(DeviceExecContext* ctx, uint32_t cycle) {
  // printf("nextStep: %u\n", cycle);
  ExecContext execCtx(*ctx->preflight, *ctx->tables, cycle);
  MutableBufObj data(*ctx->data);
  GlobalBufObj global(*ctx->global);
  step_Top(execCtx, &data, &global);
}

__global__ void par_stepExec(DeviceExecContext* ctx, uint32_t start, uint32_t count) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle >= count) {
    return;
  }
  nextStep(ctx, start + cycle);
}

__global__ void rev_stepExec(DeviceExecContext* ctx, uint32_t split, uint32_t lastCycle) {
  for (uint32_t cycle = split; cycle-- > 0;) {
    nextStep(ctx, cycle);
  }
  for (uint32_t cycle = lastCycle; cycle-- > split;) {
    nextStep(ctx, cycle);
  }
}

__global__ void fwd_stepExec(DeviceExecContext* ctx, uint32_t count) {
  for (uint32_t cycle = 0; cycle < count; cycle++) {
    nextStep(ctx, cycle);
  }
}

__global__ void stepAccum(DeviceAccumContext* ctx, uint32_t count) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle >= count) {
    return;
  }

  ExecContext execCtx(*ctx->preflight, *ctx->tables, cycle);
  MutableBufObj data(*ctx->data);
  MutableBufObj accum(*ctx->accum, /*zeroBack=*/kUserAccumSplit);
  GlobalBufObj mix(*ctx->mix);
  GlobalBufObj global(*ctx->global);
  step_TopAccum(execCtx, &accum, &data, &global, &mix);
}

__global__ void finalizeAccum(DeviceAccumContext* ctx, uint32_t lastCycle) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle >= lastCycle) {
    return;
  }

  Buffer& accum = *ctx->accum;

  size_t machineColumns = (accum.cols - kUserAccumSplit) / 4;
  size_t back1 = (cycle + lastCycle - 1) % lastCycle;
  Fp prev[4];
  for (size_t k = 0; k < 4; k++) {
    prev[k] = accum.get(back1, accum.cols - 4 + k);
  }
  for (size_t j = 0; j < machineColumns - 1; j++) {
    for (size_t k = 0; k < 4; k++) {
      size_t col = kUserAccumSplit + j * 4 + k;
      accum.set(cycle, col, accum.get(cycle, col) + prev[k]);
    }
  }
}

} // namespace risc0::circuit::rv32im_v2::cuda

constexpr size_t kStepModeParallel = 0;
constexpr size_t kStepModeSeqForward = 1;
constexpr size_t kStepModeSeqReverse = 2;

extern "C" {

using namespace risc0::circuit::rv32im_v2::cuda;

const char* risc0_circuit_rv32im_cuda_witgen(uint32_t mode,
                                             ExecBuffers* buffers,
                                             PreflightTrace* preflight,
                                             uint32_t lastCycle) {
  try {
    cudaStream_t stream = getPersistentStream();
    DeviceExecContext* d_ctx = g_cache.setup_exec(buffers, preflight, lastCycle, stream);
    size_t split = preflight->tableSplitCycle;

    switch (mode) {
    case kStepModeParallel: {
      auto cfg1 = getSimpleConfig(split);
      size_t phase2Count = lastCycle - split;
      auto cfg2 = getSimpleConfig(phase2Count);
      {
        nvtx3::scoped_range range("par_stepExec");
        par_stepExec<<<cfg1.grid, cfg1.block, 0, stream>>>(d_ctx, 0, split);
        par_stepExec<<<cfg2.grid, cfg2.block, 0, stream>>>(d_ctx, split, phase2Count);
      }
    } break;
    case kStepModeSeqForward:
      fwd_stepExec<<<1, 1, 0, stream>>>(d_ctx, lastCycle);
      CUDA_OK(cudaStreamSynchronize(stream));
      break;
    case kStepModeSeqReverse:
      rev_stepExec<<<1, 1, 0, stream>>>(d_ctx, split, lastCycle);
      CUDA_OK(cudaStreamSynchronize(stream));
      break;
    }
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

const char* risc0_circuit_rv32im_cuda_accum(AccumBuffers* buffers,
                                            PreflightTrace* preflight,
                                            uint32_t lastCycle) {
  try {
    cudaStream_t stream = getPersistentStream();
    DeviceAccumContext* d_ctx = g_cache.setup_accum(buffers, preflight, lastCycle, stream);
    auto cfg = getSimpleConfig(lastCycle);

    {
      nvtx3::scoped_range range("stepAccum");
      stepAccum<<<cfg.grid, cfg.block, 0, stream>>>(d_ctx, lastCycle);

      // par_nosync (CUDA 12.2+) uses cudaMallocAsync for temp storage instead
      // of cudaMalloc, avoiding implicit device-wide synchronization per scan call.
      auto policy = thrust::cuda::par_nosync.on(stream);
      size_t rows = buffers->accum.rows;
      for (size_t j = 0; j < 4; j++) {
        size_t col = buffers->accum.cols - 4 + j;
        Fp* itBegin = buffers->accum.buf + col * rows;
        Fp* itEnd = buffers->accum.buf + col * rows + lastCycle;
        thrust::inclusive_scan(policy, itBegin, itEnd, itBegin);
      }

      finalizeAccum<<<cfg.grid, cfg.block, 0, stream>>>(d_ctx, lastCycle);
    }

  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

const char* risc0_circuit_rv32im_cuda_warmup() {
  try {
    cudaStream_t stream = getPersistentStream();
    // Launch major kernels with count=0 to trigger full binary loading.
    // With count=0: "if (cycle >= count) return;" exits immediately, no memory access.
    // This avoids ~5-10ms of first-launch stalls during the actual proof.
    par_stepExec<<<1, 1, 0, stream>>>(nullptr, 0, 0);
    stepAccum<<<1, 1, 0, stream>>>(nullptr, 0);
    finalizeAccum<<<1, 1, 0, stream>>>(nullptr, 0);
    CUDA_OK(cudaStreamSynchronize(stream));
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

} // extern "C"
