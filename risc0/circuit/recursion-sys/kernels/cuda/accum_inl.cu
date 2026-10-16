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

// CUDA only: parStepComputeAccum / parStepVerifyAccum with the generated step function
// force-inlined into the kernel (step_*_inl.cuh are step_compute_accum.cu /
// step_verify_accum.cu with the function made static __forceinline__, written by
// accum_inl_gen.py). Built in the unrestricted-register group (no -maxrregcount=96):
// cross-TU calls at 96 registers spilled ~14 KB per thread.
#ifndef __HIPCC__
#include "cuda.h"
#include "step_compute_accum_inl.cuh"
#include "step_verify_accum_inl.cuh"

__global__ void __launch_bounds__(256) parStepComputeAccumInl(AccumContext* ctx) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle < ctx->workCycles) {
    step_compute_accum_inl(ctx, ctx->totalCycles, cycle, ctx->buffers.ctrl, ctx->buffers.global,
                           ctx->buffers.data, ctx->buffers.mix, ctx->buffers.accum);
  }
}

__global__ void __launch_bounds__(256) parStepVerifyAccumInl(AccumContext* ctx) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle < ctx->workCycles) {
    step_verify_accum_inl(ctx, ctx->totalCycles, cycle, ctx->buffers.ctrl, ctx->buffers.global,
                          ctx->buffers.data, ctx->buffers.mix, ctx->buffers.accum);
  }
}

void risc0_recursion_launch_accum_inl(int verify, unsigned grid, unsigned block, cudaStream_t stream,
                                      AccumContext* ctx) {
  if (verify) {
    parStepVerifyAccumInl<<<grid, block, 0, stream>>>(ctx);
  } else {
    parStepComputeAccumInl<<<grid, block, 0, stream>>>(ctx);
  }
}
#endif
