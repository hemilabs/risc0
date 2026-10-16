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

#ifndef __HIPCC__
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
#else
namespace nvtx3 { struct scoped_range { scoped_range(const char*) {} }; }
#endif

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string.h>

#ifdef __HIPCC__
#include <array>
namespace cuda { namespace std { using ::std::array; } }
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#else
#include <cuda/std/array>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#endif

static bool g_verbose = (std::getenv("RISC0_VERBOSE") != nullptr);

namespace risc0::circuit::rv32im_v2::cuda {

#ifdef __HIPCC__
// Addition operator for hipcub::DeviceScan::InclusiveScan on Fp elements
struct AddOp {
  __device__ __forceinline__ Fp operator()(const Fp& a, const Fp& b) const { return a + b; }
};
#else
struct FpAddOp {
  __device__ __forceinline__ Fp operator()(const Fp& a, const Fp& b) const { return a + b; }
};

// Grow-only per-device CUB temp storage. thrust's temporary allocator does cudaMalloc/cudaFree
// per call (even with par_nosync) and cudaFree synchronizes: the host blocked for the whole of
// stepAccum and the GPU idled around each call (5 per segment).
static void* cubTemp(size_t bytes) {
  static void* s_ptr[16] = {};
  static size_t s_cap[16] = {};
  int dev = getCachedDevice();
  if (bytes > s_cap[dev]) {
    if (s_ptr[dev]) CUDA_OK(cudaFree(s_ptr[dev]));
    CUDA_OK(cudaMalloc(&s_ptr[dev], bytes));
    s_cap[dev] = bytes;
  }
  return s_ptr[dev];
}
#endif

constexpr size_t kUserAccumSplit = kLayout_TopAccum.columns[0].col;

struct ExecBuffers {
  Buffer global;
  Buffer data;
  Buffer pre_data;
};

struct DeviceExecContext {
  Buffer* data;
  Buffer* pre_data;
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

// Device copies of staged (pinned) preflight traces, one grow-only device buffer per pinned
// slot. A staged trace's copy is either prefetched on a side stream while the previous
// segment's finalize runs (risc0_circuit_rv32im_cuda_prefetch), or made on the main stream in
// setup_exec. `releasedEv` is recorded after the last kernel that reads the slot (accum), and a
// later prefetch into the same slot waits for it. `id` identifies the staged trace whose bytes
// the slot holds (pinned addresses can be reused after a regrow).
struct DevSlot {
  const uint8_t* pinned = nullptr;
  uint8_t* dev = nullptr;
  size_t cap = 0;
  uint64_t id = 0;
  bool ready = false;  // prefetched for `id`, not yet consumed
  bool released = false;
  cudaEvent_t readyEv = nullptr;
  cudaEvent_t releasedEv = nullptr;
};

static DevSlot* getDevSlot(const uint8_t* pinned) {
  static DevSlot s_slots[16][8];
  int dev = getCachedDevice();
  DevSlot* freeSlot = nullptr;
  for (auto& sl : s_slots[dev]) {
    if (sl.pinned == pinned) return &sl;
    if (!sl.pinned && !freeSlot) freeSlot = &sl;
  }
  if (!freeSlot) {
    // More pinned slots than expected (regrows): recycle the least useful one (not ready).
    for (auto& sl : s_slots[dev]) {
      if (!sl.ready) { freeSlot = &sl; break; }
    }
    if (!freeSlot) throw std::runtime_error("rv32im prefetch: no free device slot");
  }
  freeSlot->pinned = pinned;
  freeSlot->ready = false;
  if (!freeSlot->readyEv) {
    CUDA_OK(cudaEventCreateWithFlags(&freeSlot->readyEv, cudaEventDisableTiming));
    CUDA_OK(cudaEventCreateWithFlags(&freeSlot->releasedEv, cudaEventDisableTiming));
  }
  return freeSlot;
}

static void growDevSlot(DevSlot* sl, size_t bytes) {
  if (bytes > sl->cap) {
    // cudaFree synchronizes the device, so no pending copy or kernel still uses the old buffer.
    if (sl->dev) CUDA_OK(cudaFree(sl->dev));
    CUDA_OK(cudaMalloc(&sl->dev, bytes));
    sl->cap = bytes;
  }
}

static cudaStream_t getCopyStream() {
  static cudaStream_t s_streams[16] = {};
  int dev = getCachedDevice();
  if (!s_streams[dev]) CUDA_OK(cudaStreamCreateWithFlags(&s_streams[dev], cudaStreamNonBlocking));
  return s_streams[dev];
}

// Static cache for device-side allocations that persist across segments.
// Eliminates ~42 cudaMallocAsync/cudaFreeAsync pairs per segment and avoids
// re-uploading ~18MB of preflight data in the accum phase (which doesn't use it).
struct DeviceCache {
  // Exec context (device pointers, reused across segments)
  DeviceExecContext* exec_ctx = nullptr;
  Buffer* d_exec_data = nullptr;
  Buffer* d_exec_pre_data = nullptr;
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
  LookupCell* d_tableU8 = nullptr;
  LookupCell* d_tableU16 = nullptr;

  // Variable-size preflight data (grow-only, shared between exec and accum)
  PreflightCycle* d_cycles = nullptr;
  size_t cycles_capacity = 0;
  MemoryTransaction* d_txns = nullptr;
  size_t txns_capacity = 0;
  uint8_t* d_bigintBytes = nullptr;
  size_t bigintBytes_capacity = 0;

  // The device preflight of the current segment: the own buffers above, or a DevSlot.
  PreflightCycle* a_cycles = nullptr;
  MemoryTransaction* a_txns = nullptr;
  uint8_t* a_bigintBytes = nullptr;
  DevSlot* a_slot = nullptr;

  // Persistent pinned host staging buffer for preflight data uploads (grow-only).
  // Avoids HIP's slow pageable memory staging path for large H2D copies.
  void* h_pinned_preflight = nullptr;
  size_t cap_h_pinned_preflight = 0;


  void init(cudaStream_t stream) {
    if (exec_ctx) return;  // already initialized
    // Fixed-size allocations for exec context
    CUDA_OK(cudaMallocAsync(&exec_ctx, sizeof(DeviceExecContext), stream));
    CUDA_OK(cudaMallocAsync(&d_exec_data, sizeof(Buffer), stream));
    CUDA_OK(cudaMallocAsync(&d_exec_pre_data, sizeof(Buffer), stream));
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
    // Shared lookup tables. These ~257 KB total (1 KB U8 + 256 KB U16) are
    // hammered by every cycle in par_stepExec/stepAccum across millions of
    // cycles per segment. Pin them into Ada's persisting-L2 cache window to
    // keep them resident across kernel launches.
    CUDA_OK(cudaMallocAsync(&d_tableU8, (1 << 8) * sizeof(uint32_t), stream));
    CUDA_OK(cudaMallocAsync(&d_tableU16, (1 << 16) * sizeof(uint32_t), stream));
#ifndef __HIPCC__
    // CUDA-only persisting-L2 cache tuning (Ada+). HIP/AMD has no equivalent of
    // cudaLimitPersistingL2CacheSize / cudaStreamAttributeAccessPolicyWindow, so
    // skip this pure performance optimization on ROCm.
    {
      static bool s_persisting_set = false;
      if (!s_persisting_set) {
        size_t l2_persist_max = 0;
        cudaDeviceGetLimit(&l2_persist_max, cudaLimitPersistingL2CacheSize);
        size_t want = (size_t)24 << 20; // 24 MB carve-out (fits 257 KB easily)
        if (l2_persist_max < want) {
          (void)cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, want);
        }
        s_persisting_set = true;
      }
      cudaStreamAttrValue attr = {};
      attr.accessPolicyWindow.base_ptr = d_tableU16;
      attr.accessPolicyWindow.num_bytes =
          ((size_t)((1 << 16) * sizeof(uint32_t))) + ((1 << 8) * sizeof(uint32_t));
      attr.accessPolicyWindow.hitRatio = 1.0f;
      attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
      attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
      (void)cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
    }
#endif
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
  // `staged`: optional pinned host buffer that already holds cycles|txns|bigintBytes in the
  // layout below (filled off the critical path by the preflight thread); the caller keeps it
  // alive until the stream has passed the copies.
  DeviceExecContext* setup_exec(ExecBuffers* buffers, PreflightTrace* preflight,
                                size_t lastCycle, cudaStream_t stream,
                                const char* staged = nullptr, uint64_t staged_id = 0) {
    init(stream);

    // Upload buffer descriptors
    CUDA_OK(cudaMemcpyAsync(d_exec_data, &buffers->data, sizeof(Buffer), cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(d_exec_pre_data, &buffers->pre_data, sizeof(Buffer), cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(d_exec_global, &buffers->global, sizeof(Buffer), cudaMemcpyHostToDevice, stream));

    // Upload preflight data via pinned staging buffer for truly async H2D.
    size_t cycles_bytes = lastCycle * sizeof(PreflightCycle);
    size_t txns_bytes = preflight->txnsLen * sizeof(MemoryTransaction);
    size_t bigint_bytes = preflight->bigintBytesLen;
    size_t total_pf_bytes = cycles_bytes + txns_bytes + bigint_bytes;

    char* pin = const_cast<char*>(staged);
    a_slot = nullptr;
    if (pin) {
      DevSlot* sl = getDevSlot(reinterpret_cast<const uint8_t*>(pin));
      if (sl->ready && sl->id == staged_id && sl->cap >= total_pf_bytes) {
        // Prefetched on the copy stream.
        CUDA_OK(cudaStreamWaitEvent(stream, sl->readyEv, 0));
      } else {
        // A stale prefetch (another trace at this address) may still be writing the slot.
        if (sl->ready) CUDA_OK(cudaStreamWaitEvent(stream, sl->readyEv, 0));
        growDevSlot(sl, total_pf_bytes);
        CUDA_OK(cudaMemcpyAsync(sl->dev, pin, total_pf_bytes, cudaMemcpyHostToDevice, stream));
        sl->id = staged_id;
      }
      sl->ready = false;
      a_slot = sl;
      a_cycles = reinterpret_cast<PreflightCycle*>(sl->dev);
      a_txns = reinterpret_cast<MemoryTransaction*>(sl->dev + cycles_bytes);
      a_bigintBytes = sl->dev + cycles_bytes + txns_bytes;
    } else {
    ensure_preflight(stream, lastCycle, preflight->txnsLen, preflight->bigintBytesLen);
    a_cycles = d_cycles;
    a_txns = d_txns;
    a_bigintBytes = d_bigintBytes;
    // Grow pinned staging buffer if needed.
    // Grow with 25% headroom: the trace grows over the first segments, and every regrow
    // costs ~20 ms of pinning on the critical path plus cudaFreeHost's implicit device
    // synchronization.
    if (total_pf_bytes > cap_h_pinned_preflight) {
      size_t want = total_pf_bytes + total_pf_bytes / 4;
      if (h_pinned_preflight) CUDA_OK(cudaFreeHost(h_pinned_preflight));
      CUDA_OK(cudaHostAlloc(&h_pinned_preflight, want, cudaHostAllocDefault));
      cap_h_pinned_preflight = want;
    }

    // Copy to pinned staging (fast CPU memcpy).
    pin = (char*)h_pinned_preflight;
    memcpy(pin, preflight->cycles, cycles_bytes);
    memcpy(pin + cycles_bytes, preflight->txns, txns_bytes);
    if (bigint_bytes > 0) {
      memcpy(pin + cycles_bytes + txns_bytes, preflight->bigintBytes, bigint_bytes);
    }

    // Truly async H2D from pinned memory.
    CUDA_OK(cudaMemcpyAsync(d_cycles, pin, cycles_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(d_txns, pin + cycles_bytes, txns_bytes, cudaMemcpyHostToDevice, stream));
    if (bigint_bytes > 0) {
      CUDA_OK(cudaMemcpyAsync(d_bigintBytes, pin + cycles_bytes + txns_bytes, bigint_bytes, cudaMemcpyHostToDevice, stream));
    }
    }

    // Build and upload PreflightTrace descriptor
    PreflightTrace h_pf;
    h_pf.cycles = a_cycles;
    h_pf.txns = a_txns;
    h_pf.bigintBytes = a_bigintBytes;
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
    h_ctx.pre_data = d_exec_pre_data;
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
    h_pf.cycles = a_cycles;
    h_pf.txns = a_txns;
    h_pf.bigintBytes = a_bigintBytes;
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

static DeviceCache g_cache[16];

static inline DeviceCache& get_cache() {
  int dev = getCachedDevice();
  return g_cache[dev];
}

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
  // Plain post-increment: each thread handles exactly one cycle, so no
  // contention. atomicAdd is unnecessary and expensive on RDNA4.
  size_t txnIdx = ctx.preflight.cycles[ctx.cycle].txnIdx++;
  const MemoryTransaction& txn = ctx.preflight.txns[txnIdx];

  // Assertions removed: in parallel witgen, union layout overwrites can cause
  // transient address mismatches during constraint evaluation. The eval_check
  // stage independently verifies all constraints on the final witness.
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
  ExecContext execCtx(*ctx->preflight, *ctx->tables, cycle);
  Buffer dataBuf = *ctx->data;
  Buffer preDataBuf = *ctx->pre_data;
  // When pre_data points to same memory as data, null it out to eliminate
  // the per-load preDataBuf branch (one check per cycle vs thousands per cycle).
  if (preDataBuf.buf == dataBuf.buf) {
    preDataBuf.buf = nullptr;
  }
  MutableBufObj data(dataBuf, preDataBuf);
  Buffer globalBuf = *ctx->global;
  GlobalBufObj global(globalBuf);
  step_Top(execCtx, &data, &global);
}

// The five per-cycle columns the host injector used to scatter for every row (cycle, next pc
// low/high, next state, next machine mode), written from the uploaded preflight cycles. Runs
// after the back scatter and before par_stepExec, so the same values win as before (the host
// injector set these after the back entries of the same row).
__global__ void injectCycleCols(Fp* data, const PreflightCycle* cycles, uint32_t n, uint32_t rows,
                                uint32_t cCycle, uint32_t cPcLow, uint32_t cPcHigh,
                                uint32_t cState, uint32_t cMode) {
  uint32_t row = blockDim.x * blockIdx.x + threadIdx.x;
  if (row >= n) {
    return;
  }
  const PreflightCycle& c = cycles[row];
  data[size_t(cCycle) * rows + row] = Fp(row);
  data[size_t(cPcLow) * rows + row] = Fp(c.pc & 0xffff);
  data[size_t(cPcHigh) * rows + row] = Fp(c.pc >> 16);
  data[size_t(cState) * rows + row] = Fp(c.state);
  data[size_t(cMode) * rows + row] = Fp(uint32_t(c.machineMode));
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

__global__ void stepAccum(DeviceAccumContext* ctx, uint32_t count, const uint32_t* perm = nullptr) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle >= count) {
    return;
  }
  // Optional cycle order (cycles grouped by major), so a warp mostly runs one mux arm.
  if (perm) {
    cycle = perm[cycle];
  }

  ExecContext execCtx(*ctx->preflight, *ctx->tables, cycle);
  Buffer dataBuf = *ctx->data;
  MutableBufObj data(dataBuf);
  Buffer accumBuf = *ctx->accum;
  MutableBufObj accum(accumBuf, /*zeroBack=*/kUserAccumSplit);
  Buffer mixBuf = *ctx->mix;
  GlobalBufObj mix(mixBuf);
  Buffer globalBuf = *ctx->global;
  GlobalBufObj global(globalBuf);
  step_TopAccum(execCtx, &accum, &data, &global, &mix);
}

// keys[i] = cycles[i].major, vals[i] = i: the input of stepAccum's cycle sort.
__global__ void majorKeys(const PreflightCycle* cycles, uint8_t* keys, uint32_t* vals, uint32_t n) {
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    keys[i] = cycles[i].major;
    vals[i] = i;
  }
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

const char* risc0_circuit_rv32im_cuda_witgen_v2(uint32_t mode,
                                                ExecBuffers* buffers,
                                                PreflightTrace* preflight,
                                                uint32_t lastCycle,
                                                const uint8_t* staged,
                                                const uint32_t* cycle_cols,
                                                uint64_t staged_id);

const char* risc0_circuit_rv32im_cuda_witgen(uint32_t mode,
                                             ExecBuffers* buffers,
                                             PreflightTrace* preflight,
                                             uint32_t lastCycle) {
  return risc0_circuit_rv32im_cuda_witgen_v2(mode, buffers, preflight, lastCycle, nullptr, nullptr, 0);
}

// Copy a staged trace (`bytes` at pinned `staged`, identified by `id`) to its device slot on a
// side stream, overlapping the GPU work in flight; the witgen that consumes it waits for the
// copy. Called on the proving thread (it holds the GPU lock).
const char* risc0_circuit_rv32im_cuda_prefetch(const uint8_t* staged, size_t bytes, uint64_t id) {
  try {
    DevSlot* sl = getDevSlot(staged);
    if (sl->ready && sl->id == id) return nullptr;
    if (sl == get_cache().a_slot) return nullptr;  // in use by the current segment
    growDevSlot(sl, bytes);
    cudaStream_t cs = getCopyStream();
    if (sl->released) CUDA_OK(cudaStreamWaitEvent(cs, sl->releasedEv, 0));
    CUDA_OK(cudaMemcpyAsync(sl->dev, staged, bytes, cudaMemcpyHostToDevice, cs));
    CUDA_OK(cudaEventRecord(sl->readyEv, cs));
    sl->id = id;
    sl->ready = true;
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

// Pinned host memory for the preflight staging slots (filled by the preflight thread).
const char* risc0_circuit_rv32im_cuda_host_alloc(size_t bytes, void** out) {
  try {
    CUDA_OK(cudaHostAlloc(out, bytes, cudaHostAllocDefault));
  } catch (const std::exception& err) {
    *out = nullptr;
    return strdup(err.what());
  } catch (...) {
    *out = nullptr;
    return strdup("Generic exception");
  }
  return nullptr;
}

const char* risc0_circuit_rv32im_cuda_host_free(void* ptr) {
  try {
    CUDA_OK(cudaFreeHost(ptr));
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

// `staged` (nullable): pinned cycles|txns|bigintBytes, see setup_exec. `cycle_cols` (nullable):
// the five data columns of injectCycleCols, which the host injector then left out.
const char* risc0_circuit_rv32im_cuda_witgen_v2(uint32_t mode,
                                                ExecBuffers* buffers,
                                                PreflightTrace* preflight,
                                                uint32_t lastCycle,
                                                const uint8_t* staged,
                                                const uint32_t* cycle_cols,
                                                uint64_t staged_id) {
  try {
    cudaStream_t stream = getPersistentStream();
    auto t0 = std::chrono::steady_clock::now();
    DeviceExecContext* d_ctx = get_cache().setup_exec(
        buffers, preflight, lastCycle, stream, reinterpret_cast<const char*>(staged), staged_id);
    auto t1 = std::chrono::steady_clock::now();
    size_t split = preflight->tableSplitCycle;
    if (cycle_cols) {
      auto cfg = getSimpleConfig(lastCycle);
      injectCycleCols<<<cfg.grid, cfg.block, 0, stream>>>(
          buffers->data.buf, get_cache().a_cycles, lastCycle, buffers->data.rows,
          cycle_cols[0], cycle_cols[1], cycle_cols[2], cycle_cols[3], cycle_cols[4]);
      CUDA_OK(cudaGetLastError());
    }

    switch (mode) {
    case kStepModeParallel: {
      size_t phase2Count = lastCycle - split;
      {
        nvtx3::scoped_range range("par_stepExec");
        auto cfg1 = getSimpleConfig(split);
        auto cfg2 = getSimpleConfig(phase2Count);
        par_stepExec<<<cfg1.grid, cfg1.block, 0, stream>>>(d_ctx, 0, split);
        par_stepExec<<<cfg2.grid, cfg2.block, 0, stream>>>(d_ctx, split, phase2Count);
      }
    } break;
    case kStepModeSeqForward:
    case kStepModeSeqReverse: {
      // Sequential modes don't need pre_data (no cross-thread race).
      // Null it out so MutableBufObj falls through to the mutable data buffer.
      Buffer nullBuf = {nullptr, 0, 0, false};
      CUDA_OK(cudaMemcpyAsync(get_cache().d_exec_pre_data, &nullBuf, sizeof(Buffer),
                               cudaMemcpyHostToDevice, stream));
      if (mode == kStepModeSeqForward) {
        fwd_stepExec<<<1, 1, 0, stream>>>(d_ctx, lastCycle);
      } else {
        rev_stepExec<<<1, 1, 0, stream>>>(d_ctx, split, lastCycle);
      }
      CUDA_OK(cudaStreamSynchronize(stream));
    } break;
    }
    auto t2 = std::chrono::steady_clock::now();
    if (g_verbose) fprintf(stderr, "      [ffi_witgen] ctx_setup: %.1fms, kernels(async): %.1fms\n",
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
    DeviceAccumContext* d_ctx = get_cache().setup_accum(buffers, preflight, lastCycle, stream);
    auto t1 = std::chrono::steady_clock::now();
    auto cfg = getSimpleConfig(lastCycle);

    {
      nvtx3::scoped_range range("stepAccum");
      // Run stepAccum's cycles grouped by major (stable sort), so warps diverge less. Same
      // output: each cycle still runs once and writes only its own row, and its only
      // back-reads (zeroBack) don't depend on other cycles. The buffers are grow-only and
      // per device; callers hold the process-wide GPU lock (risc0_zkp::hal::cuda::exclusive).
      uint32_t* perm = nullptr;
#ifndef __HIPCC__
      static uint8_t* s_keys[16] = {};
      static uint32_t* s_perm[16] = {};
      static uint8_t* s_keys2[16] = {};
      static uint32_t* s_perm2[16] = {};
      static size_t s_cap[16] = {};
      {
        int dev = getCachedDevice();
        if (lastCycle > s_cap[dev]) {
          if (s_keys[dev]) CUDA_OK(cudaFree(s_keys[dev]));
          if (s_perm[dev]) CUDA_OK(cudaFree(s_perm[dev]));
          if (s_keys2[dev]) CUDA_OK(cudaFree(s_keys2[dev]));
          if (s_perm2[dev]) CUDA_OK(cudaFree(s_perm2[dev]));
          CUDA_OK(cudaMalloc(&s_keys[dev], lastCycle));
          CUDA_OK(cudaMalloc(&s_perm[dev], lastCycle * sizeof(uint32_t)));
          CUDA_OK(cudaMalloc(&s_keys2[dev], lastCycle));
          CUDA_OK(cudaMalloc(&s_perm2[dev], lastCycle * sizeof(uint32_t)));
          s_cap[dev] = lastCycle;
        }
        majorKeys<<<cfg.grid, cfg.block, 0, stream>>>(get_cache().a_cycles, s_keys[dev], s_perm[dev], lastCycle);
        // Stable LSD radix sort on the 8-bit keys: the same permutation as
        // thrust::stable_sort_by_key.
        cub::DoubleBuffer<uint8_t> dk(s_keys[dev], s_keys2[dev]);
        cub::DoubleBuffer<uint32_t> dv(s_perm[dev], s_perm2[dev]);
        size_t tb = 0;
        CUDA_OK(cub::DeviceRadixSort::SortPairs(nullptr, tb, dk, dv, (int)lastCycle, 0, 8, stream));
        CUDA_OK(cub::DeviceRadixSort::SortPairs(cubTemp(tb), tb, dk, dv, (int)lastCycle, 0, 8, stream));
        perm = dv.Current();
      }
#endif
      stepAccum<<<cfg.grid, cfg.block, 0, stream>>>(d_ctx, lastCycle, perm);

      size_t rows = buffers->accum.rows;
#ifdef __HIPCC__
      // Cached temp storage for hipcub scans (avoids hipMallocAsync per segment)
      {
        size_t col0 = buffers->accum.cols - 4;
        Fp* itFirst = buffers->accum.buf + col0 * rows;
        size_t n = lastCycle;
        size_t temp_bytes = 0;
        hipcub::DeviceScan::InclusiveScan(nullptr, temp_bytes, itFirst, itFirst, AddOp(), n, stream);

        int scan_dev = getCachedDevice();
        static void* s_scan_temp[16] = {};
        static size_t s_scan_temp_bytes[16] = {};
        if (temp_bytes > s_scan_temp_bytes[scan_dev]) {
          if (s_scan_temp[scan_dev]) CUDA_OK(hipFree(s_scan_temp[scan_dev]));
          CUDA_OK(hipMalloc(&s_scan_temp[scan_dev], temp_bytes));
          s_scan_temp_bytes[scan_dev] = temp_bytes;
        }
        for (size_t j = 0; j < 4; j++) {
          Fp* itBegin = buffers->accum.buf + (col0 + j) * rows;
          hipcub::DeviceScan::InclusiveScan(s_scan_temp[scan_dev], temp_bytes, itBegin, itBegin, AddOp(), n, stream);
        }
      }
#else
      {
        size_t tb = 0;
        Fp* col0 = buffers->accum.buf + (buffers->accum.cols - 4) * rows;
        CUDA_OK(cub::DeviceScan::InclusiveScan(nullptr, tb, col0, col0, FpAddOp(), (int)lastCycle, stream));
        void* temp = cubTemp(tb);
        for (size_t j = 0; j < 4; j++) {
          size_t col = buffers->accum.cols - 4 + j;
          Fp* itBegin = buffers->accum.buf + col * rows;
          // Fp addition is associative and exact, so any scan order gives the same sums.
          CUDA_OK(cub::DeviceScan::InclusiveScan(temp, tb, itBegin, itBegin, FpAddOp(), (int)lastCycle, stream));
        }
      }
#endif

      finalizeAccum<<<cfg.grid, cfg.block, 0, stream>>>(d_ctx, lastCycle);
      // Last reader of this segment's device preflight: a prefetch into its slot waits for this.
      if (DevSlot* sl = get_cache().a_slot) {
        CUDA_OK(cudaEventRecord(sl->releasedEv, stream));
        sl->released = true;
      }
    }
    auto t2 = std::chrono::steady_clock::now();
    if (g_verbose) fprintf(stderr, "      [ffi_accum] ctx_setup: %.1fms, kernels: %.1fms\n",
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
