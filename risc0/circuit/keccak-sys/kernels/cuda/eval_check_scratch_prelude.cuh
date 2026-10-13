// Included first in the CUDA keccak eval_check translation unit that build.rs generates (see
// `write_scratch_eval_check` there and eval_check_scratch_kernel.cuh).
//
// The generated constraint code keeps its intermediate values in one array, `poly_fp`'s
// `Fp x13[8237]` (33 KB), and passes it down a chain of 57 functions as `Fp* argN`. build.rs
// rewrites those to `ScratchRef`, so the array lives in a global buffer instead of the stack.

#pragma once

#include "supra/fp.h"

#include <cstdint>

namespace risc0::circuit::keccak::cuda {

/// One thread's slice of the scratch buffer. Element `i` of thread `t` is at `base[i * stride +
/// t]`, `stride` being the number of threads launched: the 32 threads of a warp touching the
/// same element touch 32 consecutive words, as they would in local memory.
struct ScratchRef {
  Fp* base;
  uint32_t stride;

  __device__ __forceinline__ Fp& operator[](uint32_t i) const {
    return base[size_t(i) * stride];
  }
};

/// The scratch buffer of the launch in progress, and its thread count.
__constant__ Fp* scratch_base;
__constant__ uint32_t scratch_stride;

__device__ __forceinline__ ScratchRef scratch_for_this_thread() {
  return ScratchRef{scratch_base + (blockIdx.x * blockDim.x + threadIdx.x), scratch_stride};
}

} // namespace risc0::circuit::keccak::cuda
