// Included last in the CUDA keccak eval_check translation unit that build.rs generates, after the
// rewritten constraint code, which defines `kKeccakScratchElems` and `poly_mix`. The eval_check
// kernel and its launcher for CUDA; ROCm still builds the ones in ffi_supra.cu.
//
// Why CUDA builds it this way: CUDA reserves a kernel's per-thread stack for every thread the GPU
// can hold at once (stack x 1536 threads per SM x SMs), whatever the grid size. Compiled with
// --device-c like the other kernels, eval_check could not inline the generated chain and needed
// ~100 KB of stack per thread: a 13 GB reservation on an RTX 5080 and 20 GB on an RTX 4090, and
// its launch failed with out-of-memory on both. The launch error went unchecked, so the check
// polynomial stayed zero and every keccak proof failed its own verification. As one translation
// unit the chain inlines, and with the scratch array in global memory (see
// eval_check_scratch_prelude.cuh) the stack holds only spills. The scratch buffer is sized to the
// threads launched: a grid-stride loop over the cycles, with no more threads than the GPU runs at
// once.

#pragma once

#include "cuda.h"

#include <algorithm>
#include <exception>

namespace risc0::circuit::keccak::cuda {

constexpr uint32_t kEvalCheckBlock = 256;

__global__ void __launch_bounds__(kEvalCheckBlock) eval_check(Fp* check,
                                                              const Fp* ctrl,
                                                              const Fp* data,
                                                              const Fp* accum,
                                                              const Fp* mix,
                                                              const Fp* out,
                                                              const Fp rou,
                                                              uint32_t po2,
                                                              uint32_t domain) {
  for (uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x; cycle < domain;
       cycle += blockDim.x * gridDim.x) {
    FpExt tot = poly_fp(cycle, domain, ctrl, out, data, mix, accum);
    Fp x = pow(rou, cycle);
    Fp y = pow(Fp(3) * x, 1 << po2);
    FpExt ret = tot * inv(y - Fp(1));
    check[domain * 0 + cycle] = ret[0];
    check[domain * 1 + cycle] = ret[1];
    check[domain * 2 + cycle] = ret[2];
    check[domain * 3 + cycle] = ret[3];
  }
}

/// How many blocks to launch: as many as the GPU runs at once, fewer if the domain needs fewer,
/// and fewer still if their scratch would take more than a quarter of the free device memory.
inline uint32_t eval_check_blocks(uint32_t domain) {
  int device = getCachedDevice();
  int sms = 0;
  int per_sm = 0;
  CUDA_OK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, eval_check, kEvalCheckBlock, 0));
  size_t blocks = size_t(std::max(sms, 1)) * size_t(std::max(per_sm, 1));
  blocks = std::min(blocks, (size_t(domain) + kEvalCheckBlock - 1) / kEvalCheckBlock);
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  CUDA_OK(cudaMemGetInfo(&free_bytes, &total_bytes));
  size_t block_bytes = size_t(kEvalCheckBlock) * kKeccakScratchElems * sizeof(Fp);
  blocks = std::min(blocks, std::max<size_t>(free_bytes / 4 / block_bytes, 1));
  return uint32_t(blocks);
}

} // namespace risc0::circuit::keccak::cuda

extern "C" {

using namespace risc0::circuit::keccak::cuda;

const char* risc0_circuit_keccak_cuda_eval_check(Fp* check,
                                                 const Fp* ctrl,
                                                 const Fp* data,
                                                 const Fp* accum,
                                                 const Fp* mix,
                                                 const Fp* out,
                                                 const Fp& rou,
                                                 uint32_t po2,
                                                 uint32_t domain,
                                                 const FpExt* poly_mix_pows) {
  try {
    if (domain == 0) {
      return nullptr;
    }
    cudaStream_t stream = getPersistentStream();
    uint32_t blocks = eval_check_blocks(domain);
    uint32_t threads = blocks * kEvalCheckBlock;
    Fp* scratch = nullptr;
    CUDA_OK(cudaMallocAsync(&scratch, size_t(threads) * kKeccakScratchElems * sizeof(Fp), stream));
    CUDA_OK(cudaMemcpyToSymbolAsync(
        scratch_base, &scratch, sizeof(scratch), 0, cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyToSymbolAsync(
        scratch_stride, &threads, sizeof(threads), 0, cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyToSymbolAsync(
        poly_mix, poly_mix_pows, sizeof(poly_mix), 0, cudaMemcpyHostToDevice, stream));
    eval_check<<<blocks, kEvalCheckBlock, 0, stream>>>(
        check, ctrl, data, accum, mix, out, rou, po2, domain);
    CUDA_OK(cudaGetLastError());
    // Stream-ordered: the buffer goes back to the pool once the kernel is done with it.
    CUDA_OK(cudaFreeAsync(scratch, stream));
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

} // extern "C"
