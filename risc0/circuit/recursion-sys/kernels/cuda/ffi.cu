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

#ifndef __HIPCC__
#include "vendor/nvtx3/nvtx3.hpp"
#else
namespace nvtx3 { struct scoped_range { scoped_range(const char*) {} }; }
#endif

#include <chrono>
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
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#ifdef __HIPCC__
#include <thrust/system/hip/execution_policy.h>
#else
#include <thrust/system/cuda/execution_policy.h>
#endif
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
  if (cycle < count) {
    nextStepExec(ctx, cycle, count);
  }
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

// Cached GPU buffers for HostExecContext to avoid repeated alloc/free overhead.
// Pre-allocated to the max size seen and reused across calls.
struct CachedExecBuffers {
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

  void ensure(uint32_t numCycles, uint32_t numWoms, uint32_t numIops) {
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
    cudaFree(iops);
    cudaFree(cycles);
    cudaFree(wom);
    cudaFree(womIndex);
    cudaFree(womRows);
    cudaFree(trace);
    cudaFree(ctx);
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
    CUDA_OK(cudaMemcpyAsync(
        ctx->trace->wom, trace->wom, trace->numWoms * sizeof(FpExt), cudaMemcpyHostToDevice, stream));

    ctx->trace->cycles = g_execCache.cycles;
    CUDA_OK(cudaMemcpyAsync(ctx->trace->cycles,
                       trace->cycles,
                       trace->numCycles * sizeof(PreflightCycle),
                       cudaMemcpyHostToDevice, stream));

    ctx->trace->iops = g_execCache.iops;
    CUDA_OK(cudaMemcpyAsync(
        ctx->trace->iops, trace->iops, trace->numIops * sizeof(FpExt), cudaMemcpyHostToDevice, stream));

    ctx->womRows = g_execCache.womRows;
    CUDA_OK(cudaMemsetAsync(ctx->womRows,
                       kInvalidPattern,
                       trace->numCycles * kMaxWomRowsPerCycle * sizeof(WomArgumentRow), stream));

    ctx->womIndex = g_execCache.womIndex;
    CUDA_OK(cudaMemsetAsync(ctx->womIndex, 0, trace->numCycles * sizeof(uint32_t), stream));
  }

  ~HostExecContext() {
    // Buffers are owned by g_execCache, not freed here
  }

  void doStepExec(uint32_t mode) {
    nvtx3::scoped_range range("stepExec");
    auto t0 = std::chrono::high_resolution_clock::now();
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
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr, "      [rec_witgen] step_exec: %.1fms\n", ms);
  }

  void verifyWom(uint32_t mode) {
    nvtx3::scoped_range range("verifyWom");
    uint32_t numCycles = ctx->trace->numCycles;

#ifdef __HIPCC__
    auto par = thrust::hip::par.on(stream);
#else
    auto par = thrust::cuda::par.on(stream);
#endif

    auto t0 = std::chrono::high_resolution_clock::now();
    {
      nvtx3::scoped_range range("sortWom");
      thrust::sort(par, ctx->womRows, ctx->womRows + numCycles * kMaxWomRowsPerCycle);
    }
    CUDA_OK(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::high_resolution_clock::now();

    {
      nvtx3::scoped_range range("scan");
      thrust::exclusive_scan(
          par, ctx->womIndex, ctx->womIndex + numCycles, ctx->womIndex);
    }
    CUDA_OK(cudaStreamSynchronize(stream));
    auto t2 = std::chrono::high_resolution_clock::now();

    {
      nvtx3::scoped_range range("injectWomBacks");
      injectWomBacks<<<cfg.grid, cfg.block, 0, stream>>>(ctx);
      CUDA_OK(cudaStreamSynchronize(stream));
    }
    auto t3 = std::chrono::high_resolution_clock::now();

    {
      nvtx3::scoped_range range("stepVerifyWom");
      parStepVerifyWom<<<cfg.grid, cfg.block, 0, stream>>>(ctx);
      CUDA_OK(cudaStreamSynchronize(stream));
    }
    auto t4 = std::chrono::high_resolution_clock::now();

    fprintf(stderr, "      [rec_witgen] sort=%.1fms scan=%.1fms inject=%.1fms verify=%.1fms\n",
            std::chrono::duration<double, std::milli>(t1 - t0).count(),
            std::chrono::duration<double, std::milli>(t2 - t1).count(),
            std::chrono::duration<double, std::milli>(t3 - t2).count(),
            std::chrono::duration<double, std::milli>(t4 - t3).count());
  }
};

// Cached GPU buffers for HostAccumContext
struct CachedAccumBuffers {
  AccumContext* ctx = nullptr;
  FpExt* accum = nullptr;
  uint32_t maxWorkCycles = 0;

  void ensure(uint32_t workCycles) {
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
    cudaFree(accum);
    cudaFree(ctx);
  }
};

static thread_local CachedAccumBuffers g_accumCache;

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
    std::vector<FpExt> accumInit(workCycles, FpExt(1));
    CUDA_OK(cudaMemcpyAsync(
        ctx->accum, accumInit.data(), workCycles * sizeof(FpExt), cudaMemcpyHostToDevice, stream));
  }

  ~HostAccumContext() {
    // Buffers owned by g_accumCache, not freed here
  }

  void computeAccum() {
    nvtx3::scoped_range range("computeAccum");
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
    parStepVerifyAccum<<<cfg.grid, cfg.block, 0, stream>>>(ctx);
    CUDA_OK(cudaStreamSynchronize(stream));
  }
};

extern "C" {

const char* risc0_circuit_recursion_cuda_witgen(uint32_t mode,
                                                ExecBuffers* buffers,
                                                PreflightTrace* trace,
                                                uint32_t totalCycles) {
  try {
    HostExecContext ctx(buffers, trace, totalCycles);
    ctx.doStepExec(mode);
    ctx.verifyWom(mode);
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

const char* risc0_circuit_recursion_cuda_accum(AccumBuffers* buffers,
                                               uint32_t workCycles,
                                               uint32_t totalCycles) {
  try {
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
