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

#include <chrono>
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

struct HostExecContext {
  DeviceExecContext* ctx;
  DeviceExecContext h_ctx;  // host-side copy for filling in pointers
  PreflightTrace d_preflight;
  LookupTables d_tables;
  void* alloc_ptrs[10];
  size_t alloc_count = 0;

  HostExecContext(ExecBuffers* buffers, PreflightTrace* preflight, size_t cycles) {
    cudaStream_t stream = getPersistentStream();

    CUDA_OK(cudaMallocAsync(&ctx, sizeof(DeviceExecContext), stream));
    alloc_ptrs[alloc_count++] = ctx;

    CUDA_OK(cudaMallocAsync(&h_ctx.data, sizeof(Buffer), stream));
    CUDA_OK(cudaMemcpyAsync(h_ctx.data, &buffers->data, sizeof(Buffer), cudaMemcpyHostToDevice, stream));
    alloc_ptrs[alloc_count++] = h_ctx.data;

    CUDA_OK(cudaMallocAsync(&h_ctx.global, sizeof(Buffer), stream));
    CUDA_OK(cudaMemcpyAsync(h_ctx.global, &buffers->global, sizeof(Buffer), cudaMemcpyHostToDevice, stream));
    alloc_ptrs[alloc_count++] = h_ctx.global;

    CUDA_OK(cudaMallocAsync(&d_preflight.cycles, cycles * sizeof(PreflightCycle), stream));
    CUDA_OK(cudaMemcpyAsync(d_preflight.cycles,
                       preflight->cycles,
                       cycles * sizeof(PreflightCycle),
                       cudaMemcpyHostToDevice, stream));
    alloc_ptrs[alloc_count++] = d_preflight.cycles;

    CUDA_OK(cudaMallocAsync(&d_preflight.txns, preflight->txnsLen * sizeof(MemoryTransaction), stream));
    CUDA_OK(cudaMemcpyAsync(d_preflight.txns,
                       preflight->txns,
                       preflight->txnsLen * sizeof(MemoryTransaction),
                       cudaMemcpyHostToDevice, stream));
    alloc_ptrs[alloc_count++] = d_preflight.txns;

    CUDA_OK(cudaMallocAsync(&d_preflight.bigintBytes, preflight->bigintBytesLen, stream));
    CUDA_OK(cudaMemcpyAsync(d_preflight.bigintBytes,
                       preflight->bigintBytes,
                       preflight->bigintBytesLen,
                       cudaMemcpyHostToDevice, stream));
    alloc_ptrs[alloc_count++] = d_preflight.bigintBytes;

    d_preflight.txnsLen = preflight->txnsLen;
    d_preflight.bigintBytesLen = preflight->bigintBytesLen;
    d_preflight.tableSplitCycle = preflight->tableSplitCycle;

    CUDA_OK(cudaMallocAsync(&h_ctx.preflight, sizeof(PreflightTrace), stream));
    CUDA_OK(
        cudaMemcpyAsync(h_ctx.preflight, &d_preflight, sizeof(PreflightTrace), cudaMemcpyHostToDevice, stream));
    alloc_ptrs[alloc_count++] = h_ctx.preflight;

    CUDA_OK(cudaMallocAsync(&d_tables.tableU8, (1 << 8) * sizeof(uint32_t), stream));
    CUDA_OK(cudaMemsetAsync(d_tables.tableU8, 0, (1 << 8) * sizeof(uint32_t), stream));
    alloc_ptrs[alloc_count++] = d_tables.tableU8;

    CUDA_OK(cudaMallocAsync(&d_tables.tableU16, (1 << 16) * sizeof(uint32_t), stream));
    CUDA_OK(cudaMemsetAsync(d_tables.tableU16, 0, (1 << 16) * sizeof(uint32_t), stream));
    alloc_ptrs[alloc_count++] = d_tables.tableU16;

    CUDA_OK(cudaMallocAsync(&h_ctx.tables, sizeof(LookupTables), stream));
    CUDA_OK(cudaMemcpyAsync(h_ctx.tables, &d_tables, sizeof(LookupTables), cudaMemcpyHostToDevice, stream));
    alloc_ptrs[alloc_count++] = h_ctx.tables;

    CUDA_OK(cudaMemcpyAsync(ctx, &h_ctx, sizeof(DeviceExecContext), cudaMemcpyHostToDevice, stream));
  }

  DeviceExecContext* get_ctx() const { return ctx; }

  ~HostExecContext() {
    cudaStream_t stream = getPersistentStream();
    for (size_t i = 0; i < alloc_count; i++) {
      cudaFreeAsync(alloc_ptrs[i], stream);
    }
  }
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

struct HostAccumContext {
  DeviceAccumContext* ctx;
  DeviceAccumContext h_ctx;  // host-side copy for filling in pointers
  PreflightTrace d_preflight;
  LookupTables d_tables;
  void* alloc_ptrs[11];
  size_t alloc_count = 0;

  HostAccumContext(AccumBuffers* buffers, PreflightTrace* preflight, size_t cycles) {
    cudaStream_t stream = getPersistentStream();

    CUDA_OK(cudaMallocAsync(&ctx, sizeof(DeviceAccumContext), stream));
    alloc_ptrs[alloc_count++] = ctx;

    CUDA_OK(cudaMallocAsync(&h_ctx.data, sizeof(Buffer), stream));
    CUDA_OK(cudaMemcpyAsync(h_ctx.data, &buffers->data, sizeof(Buffer), cudaMemcpyHostToDevice, stream));
    alloc_ptrs[alloc_count++] = h_ctx.data;

    CUDA_OK(cudaMallocAsync(&h_ctx.accum, sizeof(Buffer), stream));
    CUDA_OK(cudaMemcpyAsync(h_ctx.accum, &buffers->accum, sizeof(Buffer), cudaMemcpyHostToDevice, stream));
    alloc_ptrs[alloc_count++] = h_ctx.accum;

    CUDA_OK(cudaMallocAsync(&h_ctx.global, sizeof(Buffer), stream));
    CUDA_OK(cudaMemcpyAsync(h_ctx.global, &buffers->global, sizeof(Buffer), cudaMemcpyHostToDevice, stream));
    alloc_ptrs[alloc_count++] = h_ctx.global;

    CUDA_OK(cudaMallocAsync(&h_ctx.mix, sizeof(Buffer), stream));
    CUDA_OK(cudaMemcpyAsync(h_ctx.mix, &buffers->mix, sizeof(Buffer), cudaMemcpyHostToDevice, stream));
    alloc_ptrs[alloc_count++] = h_ctx.mix;

    CUDA_OK(cudaMallocAsync(&d_preflight.cycles, cycles * sizeof(PreflightCycle), stream));
    CUDA_OK(cudaMemcpyAsync(d_preflight.cycles,
                       preflight->cycles,
                       cycles * sizeof(PreflightCycle),
                       cudaMemcpyHostToDevice, stream));
    alloc_ptrs[alloc_count++] = d_preflight.cycles;

    CUDA_OK(cudaMallocAsync(&d_preflight.txns, preflight->txnsLen * sizeof(MemoryTransaction), stream));
    CUDA_OK(cudaMemcpyAsync(d_preflight.txns,
                       preflight->txns,
                       preflight->txnsLen * sizeof(MemoryTransaction),
                       cudaMemcpyHostToDevice, stream));
    alloc_ptrs[alloc_count++] = d_preflight.txns;

    d_preflight.txnsLen = preflight->txnsLen;
    d_preflight.tableSplitCycle = preflight->tableSplitCycle;

    CUDA_OK(cudaMallocAsync(&h_ctx.preflight, sizeof(PreflightTrace), stream));
    CUDA_OK(
        cudaMemcpyAsync(h_ctx.preflight, &d_preflight, sizeof(PreflightTrace), cudaMemcpyHostToDevice, stream));
    alloc_ptrs[alloc_count++] = h_ctx.preflight;

    CUDA_OK(cudaMallocAsync(&d_tables.tableU8, (1 << 8) * sizeof(uint32_t), stream));
    CUDA_OK(cudaMemsetAsync(d_tables.tableU8, 0, (1 << 8) * sizeof(uint32_t), stream));
    alloc_ptrs[alloc_count++] = d_tables.tableU8;

    CUDA_OK(cudaMallocAsync(&d_tables.tableU16, (1 << 16) * sizeof(uint32_t), stream));
    CUDA_OK(cudaMemsetAsync(d_tables.tableU16, 0, (1 << 16) * sizeof(uint32_t), stream));
    alloc_ptrs[alloc_count++] = d_tables.tableU16;

    CUDA_OK(cudaMallocAsync(&h_ctx.tables, sizeof(LookupTables), stream));
    CUDA_OK(cudaMemcpyAsync(h_ctx.tables, &d_tables, sizeof(LookupTables), cudaMemcpyHostToDevice, stream));
    alloc_ptrs[alloc_count++] = h_ctx.tables;

    CUDA_OK(cudaMemcpyAsync(ctx, &h_ctx, sizeof(DeviceAccumContext), cudaMemcpyHostToDevice, stream));
  }

  DeviceAccumContext* get_ctx() const { return ctx; }

  ~HostAccumContext() {
    cudaStream_t stream = getPersistentStream();
    for (size_t i = 0; i < alloc_count; i++) {
      cudaFreeAsync(alloc_ptrs[i], stream);
    }
  }
};

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
    auto t0 = std::chrono::steady_clock::now();
    // No explicit event sync needed: the persistent stream is a blocking stream
    // (created with cudaStreamCreate, no NonBlocking flag), so it automatically
    // synchronizes with default stream operations (buffer allocs, memsets).
    HostExecContext ctx(buffers, preflight, lastCycle);
    auto t1 = std::chrono::steady_clock::now();
    size_t split = preflight->tableSplitCycle;

    switch (mode) {
    case kStepModeParallel: {
      auto cfg1 = getSimpleConfig(split);
      size_t phase2Count = lastCycle - split;
      auto cfg2 = getSimpleConfig(phase2Count);
      // Launch both phases back-to-back on the same stream (GPU serializes automatically).
      // Single sync at the end catches errors from both.
      {
        nvtx3::scoped_range range("par_stepExec");
        par_stepExec<<<cfg1.grid, cfg1.block, 0, stream>>>(ctx.get_ctx(), 0, split);
        par_stepExec<<<cfg2.grid, cfg2.block, 0, stream>>>(ctx.get_ctx(), split, phase2Count);
        CUDA_OK(cudaStreamSynchronize(stream));
      }
    } break;
    case kStepModeSeqForward:
      fwd_stepExec<<<1, 1, 0, stream>>>(ctx.get_ctx(), lastCycle);
      CUDA_OK(cudaStreamSynchronize(stream));
      break;
    case kStepModeSeqReverse:
      rev_stepExec<<<1, 1, 0, stream>>>(ctx.get_ctx(), split, lastCycle);
      CUDA_OK(cudaStreamSynchronize(stream));
      break;
    }
    auto t2 = std::chrono::steady_clock::now();
    fprintf(stderr, "      [ffi_witgen] ctx_setup: %.1fms, kernels+sync: %.1fms\n",
            std::chrono::duration<double, std::milli>(t1 - t0).count(),
            std::chrono::duration<double, std::milli>(t2 - t1).count());
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

    auto t0 = std::chrono::steady_clock::now();
    // No explicit event sync needed: the persistent stream is a blocking stream
    // (created with cudaStreamCreate, no NonBlocking flag), so it automatically
    // synchronizes with default stream operations (scatter H2D, mix upload).
    HostAccumContext ctx(buffers, preflight, lastCycle);
    auto t1 = std::chrono::steady_clock::now();
    auto cfg = getSimpleConfig(lastCycle);

    {
      nvtx3::scoped_range range("stepAccum");
      stepAccum<<<cfg.grid, cfg.block, 0, stream>>>(ctx.get_ctx(), lastCycle);

      // Run thrust scans on the same persistent stream as stepAccum to avoid
      // cross-stream synchronization overhead (thrust::device uses default stream).
      auto policy = thrust::cuda::par.on(stream);
      size_t rows = buffers->accum.rows;
      for (size_t j = 0; j < 4; j++) {
        size_t col = buffers->accum.cols - 4 + j;
        Fp* itBegin = buffers->accum.buf + col * rows;
        Fp* itEnd = buffers->accum.buf + col * rows + lastCycle;
        thrust::inclusive_scan(policy, itBegin, itEnd, itBegin);
      }

      finalizeAccum<<<cfg.grid, cfg.block, 0, stream>>>(ctx.get_ctx(), lastCycle);
      // No sync needed: all GPU work is ordered on the persistent stream,
      // HostAccumContext destructor uses only cudaFreeAsync (non-blocking),
      // and the next sync happens at sync_stream() in commit_group.
    }
    auto t2 = std::chrono::steady_clock::now();
    fprintf(stderr, "      [ffi_accum] ctx_setup: %.1fms, kernels: %.1fms\n",
            std::chrono::duration<double, std::milli>(t1 - t0).count(),
            std::chrono::duration<double, std::milli>(t2 - t1).count());

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
