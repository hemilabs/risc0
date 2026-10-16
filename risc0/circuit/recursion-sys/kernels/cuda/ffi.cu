// Copyright 2025 RISC Zero, Inc.
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

#include "context.h"
#include "cuda.h"
#include "fp.h"
#include "fpext.h"
#include "kernels.h"
#include "thp_pinned.h"

#ifndef __HIPCC__
#include "vendor/nvtx3/nvtx3.hpp"
#else
namespace nvtx3 { struct scoped_range { scoped_range(const char*) {} }; }
#endif

#include <cstring>
#ifdef __HIPCC__
#include <array>
namespace cuda { namespace std { using ::std::array; } }
#else
#include <cuda/std/array>
#endif
#ifndef __HIPCC__
#include <cuda_runtime.h>
#endif
#include <exception>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#ifdef _WIN32
#include <mutex>
#include <vector>
#endif
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <vector>

constexpr size_t kStepModeParallel = 0;
constexpr size_t kStepModeSeqForward = 1;
constexpr size_t kStepModeSeqReverse = 2;

namespace sppark {
void calcPrefixProducts(void* d_inout, uint32_t count);
} // namespace sppark

__device__ void nextStepExec(ExecContext* ctx, uint32_t cycle, uint32_t count) {
  if (cycle == 0 || ctx->isParSafeExec(cycle)) {
    Fp* args[3]{ctx->buffers.ctrl, ctx->buffers.global, ctx->buffers.data};
    // printf("step_exec(%u)\n", cycle);
    step_exec(ctx, ctx->totalCycles, cycle++, args[0], args[1], args[2], nullptr, nullptr);
    while (cycle < count && !ctx->isParSafeExec(cycle)) {
      // printf("  step_exec(%u)\n", cycle);
      step_exec(ctx, ctx->totalCycles, cycle++, args[0], args[1], args[2], nullptr, nullptr);
    }
  }
}

__global__ void parStepExec(ExecContext* ctx) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t count = ctx->trace->numCycles;
  // isParSafe == 2 marks the head of a chain whose rows the host already computed.
  if (cycle < count && ctx->trace->cycles[cycle].isParSafe != 2) {
    nextStepExec(ctx, cycle, count);
  }
}

// Rows of sequential chains executed on the host (see risc0_circuit_recursion_cpu_chains).
struct HostChains {
  const uint32_t* heads;    // chain head cycles
  uint32_t numHeads;
  const uint32_t* cycles;   // one entry per chain row
  const uint32_t* rows;     // numRows * dataCols raw Fp, row-major
  const uint32_t* womIdx;   // numRows
  const uint32_t* womRows;  // numRows * kMaxWomRowsPerCycle * 5
  uint32_t numRows;
  uint32_t dataCols;
};

__global__ void markHostChains(ExecContext* ctx, const uint32_t* heads, uint32_t n) {
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) ctx->trace->cycles[heads[i]].isParSafe = 2;
}

__global__ void scatterHostChains(ExecContext* ctx, const uint32_t* cycles, const uint32_t* rows,
                                  const uint32_t* womIdx, const WomArgumentRow* womRows,
                                  uint32_t n, uint32_t dataCols) {
  uint32_t t = blockDim.x * blockIdx.x + threadIdx.x;
  if (t >= n * dataCols) return;
  uint32_t i = t / dataCols, k = t % dataCols, c = cycles[i];
  reinterpret_cast<uint32_t*>(ctx->buffers.data)[(size_t)k * ctx->totalCycles + c] = rows[t];
  if (k == 0) ctx->womIndex[c] = womIdx[i];
  if (k < kMaxWomRowsPerCycle)
    ctx->womRows[(size_t)c * kMaxWomRowsPerCycle + k] = womRows[(size_t)i * kMaxWomRowsPerCycle + k];
}

__global__ void fwdStepExec(ExecContext* ctx) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t count = ctx->trace->numCycles;
  if (cycle == 0) {
    Fp* args[3]{ctx->buffers.ctrl, ctx->buffers.global, ctx->buffers.data};
    for (uint32_t cycle = 0; cycle < count; cycle++) {
      // printf("step_exec(%u)\n", cycle);
      step_exec(ctx, ctx->totalCycles, cycle, args[0], args[1], args[2], nullptr, nullptr);
    }
  }
}

__global__ void revStepExec(ExecContext* ctx) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t count = ctx->trace->numCycles;
  if (cycle == count - 1) {
    Fp* args[3]{ctx->buffers.ctrl, ctx->buffers.global, ctx->buffers.data};
    for (uint32_t i = 0; i < count; i++) {
      uint32_t cycle = count - i - 1;
      nextStepExec(ctx, cycle, count);
    }
  }
}

__global__ void injectWomBacks(ExecContext* ctx) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t count = ctx->trace->numCycles;
  if (cycle < 1 || cycle >= count) {
    return;
  }

  Fp* data = ctx->buffers.data;
  uint32_t totalCycles = ctx->totalCycles;
  uint32_t idx = ctx->womIndex[cycle];
  if (idx) {
    const WomArgumentRow& prev = ctx->womRows[idx - 1];
    data[0 * totalCycles + cycle - 1] = prev.addr;
    data[1 * totalCycles + cycle - 1] = prev.value.elems[0];
    data[2 * totalCycles + cycle - 1] = prev.value.elems[1];
    data[3 * totalCycles + cycle - 1] = prev.value.elems[2];
    data[4 * totalCycles + cycle - 1] = prev.value.elems[3];
  } else {
    data[0 * totalCycles + cycle - 1] = 0;
    data[1 * totalCycles + cycle - 1] = 0;
    data[2 * totalCycles + cycle - 1] = 0;
    data[3 * totalCycles + cycle - 1] = 0;
    data[4 * totalCycles + cycle - 1] = 0;
  }
}

__global__ void parStepVerifyWom(ExecContext* ctx) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t count = ctx->trace->numCycles;
  if (cycle < count) {
    Fp* args[3]{ctx->buffers.ctrl, ctx->buffers.global, ctx->buffers.data};
    step_verify_mem(ctx, ctx->totalCycles, cycle, args[0], args[1], args[2], nullptr, nullptr);
  }
}

__global__ void fwdStepVerifyWom(ExecContext* ctx) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t count = ctx->trace->numCycles;
  if (cycle == 0) {
    Fp* args[3]{ctx->buffers.ctrl, ctx->buffers.global, ctx->buffers.data};
    for (uint32_t cycle = 0; cycle < count; cycle++) {
      // printf("step_verify_mem: %u\n", cycle);
      step_verify_mem(ctx, ctx->totalCycles, cycle, args[0], args[1], args[2], nullptr, nullptr);
    }
  }
}

#ifndef __HIPCC__
void risc0_recursion_launch_accum_inl(int verify, unsigned grid, unsigned block, cudaStream_t stream,
                                      AccumContext* ctx);
// accum_inl.cu: the accum steps force-inlined into the kernels, and eval_check split into
// standalone kernels (ffi_supra.cu): on unless RISC0_CUDA_REC_KERNELS=0.
static bool accumInl() {
  static const bool on = [] {
    const char* e = getenv("RISC0_CUDA_REC_KERNELS");
    return !(e && e[0] == '0');
  }();
  return on;
}
#endif

__global__ void parStepComputeAccum(AccumContext* ctx) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle < ctx->workCycles) {
    step_compute_accum(ctx,
                       ctx->totalCycles,
                       cycle,
                       ctx->buffers.ctrl,
                       ctx->buffers.global,
                       ctx->buffers.data,
                       ctx->buffers.mix,
                       ctx->buffers.accum);
  }
}

__global__ void parStepVerifyAccum(AccumContext* ctx) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle < ctx->workCycles) {
    step_verify_accum(ctx,
                      ctx->totalCycles,
                      cycle,
                      ctx->buffers.ctrl,
                      ctx->buffers.global,
                      ctx->buffers.data,
                      ctx->buffers.mix,
                      ctx->buffers.accum);
  }
}

#ifdef _WIN32
// Thread exit for the per-thread GPU buffer caches below, on Windows.
//
// MSVC constructs namespace-scope thread_locals on every thread in the process, and
// destroys them in the thread-detach callback, under the loader lock, where CUDA must
// not be called: a thread that never touched the GPU deadlocked there, initializing a
// CUDA context for its `cudaFree(nullptr)`. (GCC builds thread_locals on first use and
// destroys them outside any loader lock, so elsewhere they are freed as before.) Here a
// cache is never freed at thread exit: its buffers, if it has any, are parked for the
// next thread on the same device, which is host-side bookkeeping alone.
template <typename State> class ParkedBuffers {
  std::mutex lock;
  std::vector<State> parked;

public:
  static ParkedBuffers& get() {
    static ParkedBuffers pool;
    return pool;
  }

  void park(const State& state) {
    std::lock_guard<std::mutex> guard(lock);
    parked.push_back(state);
  }

  // Moves a parked state for `device` into `out`, if there is one.
  void take(int device, State& out) {
    std::lock_guard<std::mutex> guard(lock);
    for (size_t i = 0; i < parked.size(); i++) {
      if (parked[i].device == device) {
        out = parked[i];
        parked.erase(parked.begin() + i);
        return;
      }
    }
  }
};
#endif

struct ExecBuffersState {
  ExecContext* ctx = nullptr;
  PreflightTrace* trace = nullptr;
  WomArgumentRow* womRows = nullptr;
  uint32_t* womIndex = nullptr;
  FpExt* wom = nullptr;
  PreflightCycle* cycles = nullptr;
  FpExt* iops = nullptr;
  uint32_t maxCycles = 0;
  uint32_t maxWoms = 0;
  uint32_t maxIops = 0;
  int device = -1;
};

// Cached GPU buffers for HostExecContext to avoid repeated alloc/free overhead.
// Pre-allocated to the max size seen and reused across calls.
struct CachedExecBuffers : ExecBuffersState {
  void ensure(uint32_t numCycles, uint32_t numWoms, uint32_t numIops) {
#ifdef _WIN32
    if (!ctx) {
      CUDA_OK(cudaGetDevice(&device));
      ParkedBuffers<ExecBuffersState>::get().take(device, *this);
    }
#endif
    if (!ctx) {
      CUDA_OK(cudaMallocManaged(&ctx, sizeof(ExecContext)));
      CUDA_OK(cudaMallocManaged(&trace, sizeof(PreflightTrace)));
    }
    if (numCycles > maxCycles) {
      cudaFree(womRows);
      cudaFree(womIndex);
      cudaFree(cycles);
      CUDA_OK(cudaMalloc(&womRows, numCycles * kMaxWomRowsPerCycle * sizeof(WomArgumentRow)));
      CUDA_OK(cudaMalloc(&womIndex, numCycles * sizeof(uint32_t)));
      CUDA_OK(cudaMalloc(&cycles, numCycles * sizeof(PreflightCycle)));
      maxCycles = numCycles;
    }
    if (numWoms > maxWoms) {
      cudaFree(wom);
      CUDA_OK(cudaMalloc(&wom, numWoms * sizeof(FpExt)));
      maxWoms = numWoms;
    }
    if (numIops > maxIops) {
      cudaFree(iops);
      CUDA_OK(cudaMalloc(&iops, numIops * sizeof(FpExt)));
      maxIops = numIops;
    }
  }

  ~CachedExecBuffers() {
#ifdef _WIN32
    if (ctx) {
      ParkedBuffers<ExecBuffersState>::get().park(*this);
    }
#else
    cudaFree(iops);
    cudaFree(cycles);
    cudaFree(wom);
    cudaFree(womIndex);
    cudaFree(womRows);
    cudaFree(trace);
    cudaFree(ctx);
#endif
  }
};

static thread_local CachedExecBuffers g_execCache;

struct HostExecContext {
  ExecContext* ctx;
  CudaStream stream;
  LaunchConfig cfg;

  HostExecContext(ExecBuffers* buffers, PreflightTrace* trace, size_t totalCycles)
      : cfg(getSimpleConfig(trace->numCycles)) {
    g_execCache.ensure(trace->numCycles, trace->numWoms, trace->numIops);

    ctx = g_execCache.ctx;
    ctx->buffers.ctrl = buffers->ctrl;
    ctx->buffers.data = buffers->data;
    ctx->buffers.global = buffers->global;
    ctx->totalCycles = totalCycles;

    ctx->trace = g_execCache.trace;
    ctx->trace->numWoms = trace->numWoms;
    ctx->trace->numCycles = trace->numCycles;
    ctx->trace->numIops = trace->numIops;

    ctx->trace->wom = g_execCache.wom;
    CUDA_OK(cudaMemcpy(
        ctx->trace->wom, trace->wom, trace->numWoms * sizeof(FpExt), cudaMemcpyHostToDevice));

    ctx->trace->cycles = g_execCache.cycles;
    CUDA_OK(cudaMemcpy(ctx->trace->cycles,
                       trace->cycles,
                       trace->numCycles * sizeof(PreflightCycle),
                       cudaMemcpyHostToDevice));

    ctx->trace->iops = g_execCache.iops;
    CUDA_OK(cudaMemcpy(
        ctx->trace->iops, trace->iops, trace->numIops * sizeof(FpExt), cudaMemcpyHostToDevice));

    ctx->womRows = g_execCache.womRows;
    CUDA_OK(cudaMemset(ctx->womRows,
                       kInvalidPattern,
                       trace->numCycles * kMaxWomRowsPerCycle * sizeof(WomArgumentRow)));

    ctx->womIndex = g_execCache.womIndex;
    CUDA_OK(cudaMemset(ctx->womIndex, 0, trace->numCycles * sizeof(uint32_t)));
  }

  ~HostExecContext() {
    // Buffers are owned by g_execCache, not freed here
  }

  // Upload the host-executed chains, write their rows, and mark their heads to be skipped.
  void applyHostChains(const HostChains* hc) {
    nvtx3::scoped_range range("hostChains");
    static_assert(sizeof(WomArgumentRow) == 20, "WomArgumentRow layout");
    if (hc->dataCols != 128) throw std::runtime_error("host chains: unexpected data width");
    static thread_local uint32_t* dBuf = nullptr;
    static thread_local size_t dCap = 0;
    size_t nWords = hc->numHeads + (size_t)hc->numRows * (2 + hc->dataCols + kMaxWomRowsPerCycle * 5);
    if (nWords > dCap) {
      if (dBuf) cudaFree(dBuf);
      CUDA_OK(cudaMalloc(&dBuf, nWords * 4));
      dCap = nWords;
    }
    uint32_t* dHeads = dBuf;
    uint32_t* dCycles = dHeads + hc->numHeads;
    uint32_t* dRows = dCycles + hc->numRows;
    uint32_t* dWomIdx = dRows + (size_t)hc->numRows * hc->dataCols;
    uint32_t* dWomRows = dWomIdx + hc->numRows;
    CUDA_OK(cudaMemcpyAsync(dHeads, hc->heads, hc->numHeads * 4, cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(dCycles, hc->cycles, hc->numRows * 4, cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(dRows, hc->rows, (size_t)hc->numRows * hc->dataCols * 4, cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(dWomIdx, hc->womIdx, hc->numRows * 4, cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(dWomRows, hc->womRows, (size_t)hc->numRows * kMaxWomRowsPerCycle * 20, cudaMemcpyHostToDevice, stream));
    markHostChains<<<(hc->numHeads + 255) / 256, 256, 0, stream>>>(ctx, dHeads, hc->numHeads);
    uint32_t n = hc->numRows * hc->dataCols;
    scatterHostChains<<<(n + 255) / 256, 256, 0, stream>>>(
        ctx, dCycles, dRows, dWomIdx, reinterpret_cast<const WomArgumentRow*>(dWomRows), hc->numRows, hc->dataCols);
    // Ordered before doStepExec on `stream`; the host buffers are pageable (staged at enqueue)
    // and the caller keeps them alive until this witgen call returns (it syncs).
  }

  void doStepExec(uint32_t mode) {
    nvtx3::scoped_range range("stepExec");
    switch (mode) {
    case kStepModeParallel: {
      parStepExec<<<cfg.grid, cfg.block, 0, stream>>>(ctx);
    } break;
    case kStepModeSeqForward: {
      fwdStepExec<<<cfg.grid, cfg.block, 0, stream>>>(ctx);
    } break;
    case kStepModeSeqReverse: {
      revStepExec<<<cfg.grid, cfg.block, 0, stream>>>(ctx);
    } break;
    }
    CUDA_OK(cudaStreamSynchronize(stream));
  }

  void verifyWom(uint32_t mode) {
    nvtx3::scoped_range range("verifyWom");
    uint32_t numCycles = ctx->trace->numCycles;

    {
      nvtx3::scoped_range range("sortWom");
      thrust::sort(thrust::device, ctx->womRows, ctx->womRows + numCycles * kMaxWomRowsPerCycle);
    }

    {
      nvtx3::scoped_range range("scan");
      thrust::exclusive_scan(
          thrust::device, ctx->womIndex, ctx->womIndex + numCycles, ctx->womIndex);
    }

    {
      nvtx3::scoped_range range("injectWomBacks");
      injectWomBacks<<<cfg.grid, cfg.block, 0, stream>>>(ctx);
      // No intermediate sync — the next kernel runs on the same stream and
      // is naturally ordered after this one.
    }

    {
      nvtx3::scoped_range range("stepVerifyWom");
      parStepVerifyWom<<<cfg.grid, cfg.block, 0, stream>>>(ctx);
      CUDA_OK(cudaStreamSynchronize(stream));
    }
  }
};

struct AccumBuffersState {
  AccumContext* ctx = nullptr;
  FpExt* accum = nullptr;
  uint32_t maxWorkCycles = 0;
  int device = -1;
};

// Cached GPU buffers for HostAccumContext
struct CachedAccumBuffers : AccumBuffersState {
  void ensure(uint32_t workCycles) {
#ifdef _WIN32
    if (!ctx) {
      CUDA_OK(cudaGetDevice(&device));
      ParkedBuffers<AccumBuffersState>::get().take(device, *this);
    }
#endif
    if (!ctx) {
      CUDA_OK(cudaMallocManaged(&ctx, sizeof(AccumContext)));
    }
    if (workCycles > maxWorkCycles) {
      cudaFree(accum);
      CUDA_OK(cudaMalloc(&accum, workCycles * sizeof(FpExt)));
      maxWorkCycles = workCycles;
    }
  }

  ~CachedAccumBuffers() {
#ifdef _WIN32
    if (ctx) {
      ParkedBuffers<AccumBuffersState>::get().park(*this);
    }
#else
    cudaFree(accum);
    cudaFree(ctx);
#endif
  }
};

static thread_local CachedAccumBuffers g_accumCache;

__global__ void fillOneExt(FpExt* p, uint32_t n) {
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    p[i] = FpExt(1);
  }
}

struct HostAccumContext {
  AccumContext* ctx;
  CudaStream stream;
  LaunchConfig cfg;

  HostAccumContext(AccumBuffers* buffers, size_t workCycles, size_t totalCycles)
      : cfg(getSimpleConfig(workCycles)) {
    g_accumCache.ensure(workCycles);

    ctx = g_accumCache.ctx;
    ctx->buffers.ctrl = buffers->ctrl;
    ctx->buffers.global = buffers->global;
    ctx->buffers.data = buffers->data;
    ctx->buffers.mix = buffers->mix;
    ctx->buffers.accum = buffers->accum;
    ctx->totalCycles = totalCycles;
    ctx->workCycles = workCycles;

    ctx->accum = g_accumCache.accum;
    // Fill with FpExt(1) on the device instead of uploading a host vector of ones.
    fillOneExt<<<(workCycles + 255) / 256, 256, 0, stream>>>(ctx->accum, workCycles);
    CUDA_OK(cudaGetLastError());
  }

  ~HostAccumContext() {
    // Buffers owned by g_accumCache, not freed here
  }

  void computeAccum() {
    nvtx3::scoped_range range("computeAccum");
#ifndef __HIPCC__
    if (accumInl()) {
      risc0_recursion_launch_accum_inl(0, cfg.grid.x, cfg.block.x, stream, ctx);
    } else
#endif
    parStepComputeAccum<<<cfg.grid, cfg.block, 0, stream>>>(ctx);
    CUDA_OK(cudaStreamSynchronize(stream));
  }

  void calcPrefixProducts() {
    nvtx3::scoped_range range("calcPrefixProducts");
    sppark::calcPrefixProducts(ctx->accum, ctx->workCycles);
    CUDA_OK(cudaStreamSynchronize(stream));
  }

  void verifyAccum() {
    nvtx3::scoped_range range("verifyAccum");
    CUDA_OK(cudaDeviceSynchronize());
#ifndef __HIPCC__
    if (accumInl()) {
      risc0_recursion_launch_accum_inl(1, cfg.grid.x, cfg.block.x, stream, ctx);
    } else
#endif
    parStepVerifyAccum<<<cfg.grid, cfg.block, 0, stream>>>(ctx);
    CUDA_OK(cudaStreamSynchronize(stream));
  }
};

extern "C" {

const char* risc0_circuit_recursion_cuda_witgen_chains(uint32_t mode,
                                                       ExecBuffers* buffers,
                                                       PreflightTrace* trace,
                                                       uint32_t totalCycles,
                                                       const HostChains* hostChains);

const char* risc0_circuit_recursion_cuda_witgen(uint32_t mode,
                                                ExecBuffers* buffers,
                                                PreflightTrace* trace,
                                                uint32_t totalCycles) {
  return risc0_circuit_recursion_cuda_witgen_chains(mode, buffers, trace, totalCycles, nullptr);
}

const char* risc0_circuit_recursion_cuda_witgen_chains(uint32_t mode,
                                                       ExecBuffers* buffers,
                                                       PreflightTrace* trace,
                                                       uint32_t totalCycles,
                                                       const HostChains* hostChains) {
  try {
    // Sync all GPU work before accessing buffers written by external streams
    // (e.g., Rust HAL's persistent stream). hipMemcpy doesn't reliably
    // synchronize across all streams on ROCm.
    CUDA_OK(cudaDeviceSynchronize());
    HostExecContext ctx(buffers, trace, totalCycles);
    if (hostChains && hostChains->numHeads && mode == kStepModeParallel) {
      ctx.applyHostChains(hostChains);
    }
    ctx.doStepExec(mode);
    ctx.verifyWom(mode);
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

// Load the recursion witgen/accum kernels (CUDA lazy loading) off the GPU thread, without
// launching anything: cudaFuncGetAttributes forces the module load.
const char* risc0_circuit_recursion_cuda_warmup() {
  try {
    cudaFuncAttributes attr;
    CUDA_OK(cudaFuncGetAttributes(&attr, parStepExec));
    CUDA_OK(cudaFuncGetAttributes(&attr, injectWomBacks));
    CUDA_OK(cudaFuncGetAttributes(&attr, parStepVerifyWom));
    CUDA_OK(cudaFuncGetAttributes(&attr, markHostChains));
    CUDA_OK(cudaFuncGetAttributes(&attr, scatterHostChains));
    CUDA_OK(cudaFuncGetAttributes(&attr, parStepComputeAccum));
    CUDA_OK(cudaFuncGetAttributes(&attr, parStepVerifyAccum));
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

// Page-locked host memory for the recursion witgen inputs (prepared off the GPU thread), backed
// by huge pages (ThpPinned): cudaHostAlloc pins 4 KB pages while holding the driver lock, which
// stalls the proving thread's CUDA calls.
static std::mutex g_host_bufs_mu;
static std::unordered_map<void*, ThpPinned>& hostBufs() {
  static std::unordered_map<void*, ThpPinned> m;
  return m;
}

const char* risc0_circuit_recursion_cuda_host_alloc(size_t bytes, void** out) {
  *out = nullptr;
  try {
    ThpPinned p;
    p.alloc(bytes);
    std::lock_guard<std::mutex> lk(g_host_bufs_mu);
    hostBufs()[p.ptr] = p;
    *out = p.ptr;
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

const char* risc0_circuit_recursion_cuda_host_free(void* ptr) {
  try {
    ThpPinned p;
    {
      std::lock_guard<std::mutex> lk(g_host_bufs_mu);
      auto it = hostBufs().find(ptr);
      if (it == hostBufs().end()) throw std::runtime_error("host_free: unknown pointer");
      p = it->second;
      hostBufs().erase(it);
    }
    p.free_pinned(); // synchronizes the device first, like cudaFreeHost
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

const char* risc0_circuit_recursion_cuda_accum(AccumBuffers* buffers,
                                               uint32_t workCycles,
                                               uint32_t totalCycles) {
  try {
    // Sync all GPU work before accessing buffers written by external streams
    CUDA_OK(cudaDeviceSynchronize());
    HostAccumContext ctx(buffers, workCycles, totalCycles);
    ctx.computeAccum();
    ctx.calcPrefixProducts();
    ctx.verifyAccum();
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

} // extern "C"
