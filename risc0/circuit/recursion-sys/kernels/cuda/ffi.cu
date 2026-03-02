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

#include "vendor/nvtx3/nvtx3.hpp"

#include <cstring>
#include <cuda_runtime.h>
#include <exception>
#include <thrust/execution_policy.h>
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

// Static cache for device-side allocations that persist across recursion proofs.
// Eliminates ~774 cudaMalloc/cudaFree + ~86 cudaStreamCreate/Destroy per proving run.
struct RecursionDeviceCache {
  cudaStream_t stream = nullptr;

  // Managed-memory context structs (CPU writes fields, GPU reads via page migration)
  ExecContext* exec_ctx = nullptr;
  PreflightTrace* exec_trace = nullptr;
  AccumContext* accum_ctx = nullptr;

  // Grow-only device buffers for preflight data
  FpExt* d_wom = nullptr;
  size_t wom_capacity = 0;
  PreflightCycle* d_cycles = nullptr;
  size_t cycles_capacity = 0;
  FpExt* d_iops = nullptr;
  size_t iops_capacity = 0;

  // Grow-only device buffers for WOM tracking
  WomArgumentRow* d_womRows = nullptr;
  size_t womRows_capacity = 0;
  uint32_t* d_womIndex = nullptr;
  size_t womIndex_capacity = 0;

  // Grow-only device buffer for accum
  FpExt* d_accum = nullptr;
  size_t accum_capacity = 0;

  // Cached host buffer for accum initialization
  std::vector<FpExt> accumInit;

  void init() {
    if (exec_ctx)
      return;
    CUDA_OK(cudaStreamCreate(&stream));
    CUDA_OK(cudaMallocManaged(&exec_ctx, sizeof(ExecContext)));
    CUDA_OK(cudaMallocManaged(&exec_trace, sizeof(PreflightTrace)));
    CUDA_OK(cudaMallocManaged(&accum_ctx, sizeof(AccumContext)));
  }

  void ensure_buffers(size_t numWoms, size_t numCycles, size_t numIops) {
    if (numWoms > wom_capacity) {
      if (d_wom)
        cudaFree(d_wom);
      CUDA_OK(cudaMalloc(&d_wom, numWoms * sizeof(FpExt)));
      wom_capacity = numWoms;
    }
    if (numCycles > cycles_capacity) {
      if (d_cycles)
        cudaFree(d_cycles);
      CUDA_OK(cudaMalloc(&d_cycles, numCycles * sizeof(PreflightCycle)));
      cycles_capacity = numCycles;
    }
    if (numIops > iops_capacity) {
      if (d_iops)
        cudaFree(d_iops);
      CUDA_OK(cudaMalloc(&d_iops, numIops * sizeof(FpExt)));
      iops_capacity = numIops;
    }
    if (numCycles > womRows_capacity) {
      if (d_womRows)
        cudaFree(d_womRows);
      CUDA_OK(cudaMalloc(&d_womRows, numCycles * kMaxWomRowsPerCycle * sizeof(WomArgumentRow)));
      womRows_capacity = numCycles;
    }
    if (numCycles > womIndex_capacity) {
      if (d_womIndex)
        cudaFree(d_womIndex);
      CUDA_OK(cudaMalloc(&d_womIndex, numCycles * sizeof(uint32_t)));
      womIndex_capacity = numCycles;
    }
  }

  void setup_exec(ExecBuffers* buffers, PreflightTrace* trace, size_t totalCycles) {
    init();
    ensure_buffers(trace->numWoms, trace->numCycles, trace->numIops);

    cudaStream_t s = stream;

    // Upload preflight data (async)
    CUDA_OK(cudaMemcpyAsync(
        d_wom, trace->wom, trace->numWoms * sizeof(FpExt), cudaMemcpyHostToDevice, s));
    CUDA_OK(cudaMemcpyAsync(d_cycles,
                             trace->cycles,
                             trace->numCycles * sizeof(PreflightCycle),
                             cudaMemcpyHostToDevice,
                             s));
    CUDA_OK(cudaMemcpyAsync(
        d_iops, trace->iops, trace->numIops * sizeof(FpExt), cudaMemcpyHostToDevice, s));

    // Reset WOM data (async)
    CUDA_OK(cudaMemsetAsync(
        d_womRows, kInvalidPattern, trace->numCycles * kMaxWomRowsPerCycle * sizeof(WomArgumentRow), s));
    CUDA_OK(cudaMemsetAsync(d_womIndex, 0, trace->numCycles * sizeof(uint32_t), s));

    // Update managed-memory structs (CPU writes, safe after cudaDeviceSynchronize)
    exec_trace->wom = d_wom;
    exec_trace->cycles = d_cycles;
    exec_trace->iops = d_iops;
    exec_trace->numWoms = trace->numWoms;
    exec_trace->numCycles = trace->numCycles;
    exec_trace->numIops = trace->numIops;

    exec_ctx->buffers.ctrl = buffers->ctrl;
    exec_ctx->buffers.data = buffers->data;
    exec_ctx->buffers.global = buffers->global;
    exec_ctx->trace = exec_trace;
    exec_ctx->totalCycles = totalCycles;
    exec_ctx->womRows = d_womRows;
    exec_ctx->womIndex = d_womIndex;
  }

  void setup_accum(AccumBuffers* buffers, size_t workCycles, size_t totalCycles) {
    init();
    if (workCycles > accum_capacity) {
      if (d_accum)
        cudaFree(d_accum);
      CUDA_OK(cudaMalloc(&d_accum, workCycles * sizeof(FpExt)));
      accum_capacity = workCycles;
    }

    // Initialize accum to FpExt(1) using cached host buffer
    if (accumInit.size() < workCycles) {
      accumInit.resize(workCycles, FpExt(1));
    }
    CUDA_OK(cudaMemcpyAsync(
        d_accum, accumInit.data(), workCycles * sizeof(FpExt), cudaMemcpyHostToDevice, stream));

    // Update managed-memory struct (CPU writes, safe after cudaDeviceSynchronize)
    accum_ctx->buffers.ctrl = buffers->ctrl;
    accum_ctx->buffers.global = buffers->global;
    accum_ctx->buffers.data = buffers->data;
    accum_ctx->buffers.mix = buffers->mix;
    accum_ctx->buffers.accum = buffers->accum;
    accum_ctx->totalCycles = totalCycles;
    accum_ctx->workCycles = workCycles;
    accum_ctx->accum = d_accum;
  }
};

static RecursionDeviceCache g_cache;

extern "C" {

const char* risc0_circuit_recursion_cuda_witgen(uint32_t mode,
                                                ExecBuffers* buffers,
                                                PreflightTrace* trace,
                                                uint32_t totalCycles) {
  try {
    // Full device sync to ensure all prior GPU work is visible.
    CUDA_OK(cudaDeviceSynchronize());
    g_cache.setup_exec(buffers, trace, totalCycles);

    ExecContext* ctx = g_cache.exec_ctx;
    cudaStream_t s = g_cache.stream;
    LaunchConfig cfg = getSimpleConfig(trace->numCycles);

    {
      nvtx3::scoped_range range("stepExec");
      switch (mode) {
      case kStepModeParallel:
        parStepExec<<<cfg.grid, cfg.block, 0, s>>>(ctx);
        break;
      case kStepModeSeqForward:
        fwdStepExec<<<cfg.grid, cfg.block, 0, s>>>(ctx);
        break;
      case kStepModeSeqReverse:
        revStepExec<<<cfg.grid, cfg.block, 0, s>>>(ctx);
        break;
      }
      CUDA_OK(cudaStreamSynchronize(s));
    }

    {
      nvtx3::scoped_range range("verifyWom");
      uint32_t numCycles = trace->numCycles;

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
        injectWomBacks<<<cfg.grid, cfg.block, 0, s>>>(ctx);
        CUDA_OK(cudaStreamSynchronize(s));
      }

      {
        nvtx3::scoped_range range("stepVerifyWom");
        parStepVerifyWom<<<cfg.grid, cfg.block, 0, s>>>(ctx);
        CUDA_OK(cudaStreamSynchronize(s));
      }
    }
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

const char* risc0_circuit_recursion_cuda_accum(AccumBuffers* buffers,
                                               uint32_t workCycles,
                                               uint32_t totalCycles) {
  try {
    // Full device sync to ensure all prior GPU work is visible.
    CUDA_OK(cudaDeviceSynchronize());
    g_cache.setup_accum(buffers, workCycles, totalCycles);

    AccumContext* ctx = g_cache.accum_ctx;
    cudaStream_t s = g_cache.stream;
    LaunchConfig cfg = getSimpleConfig(workCycles);

    {
      nvtx3::scoped_range range("computeAccum");
      parStepComputeAccum<<<cfg.grid, cfg.block, 0, s>>>(ctx);
      CUDA_OK(cudaStreamSynchronize(s));
    }

    {
      nvtx3::scoped_range range("calcPrefixProducts");
      sppark::calcPrefixProducts(ctx->accum, ctx->workCycles);
      CUDA_OK(cudaStreamSynchronize(s));
    }

    {
      nvtx3::scoped_range range("verifyAccum");
      parStepVerifyAccum<<<cfg.grid, cfg.block, 0, s>>>(ctx);
      CUDA_OK(cudaStreamSynchronize(s));
    }
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

} // extern "C"
