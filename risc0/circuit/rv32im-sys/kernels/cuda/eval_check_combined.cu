// Combined compilation of all eval_check device functions + kernel into a single
// translation unit, compiled WITHOUT -dc (separate compilation). This allows
// NVCC to inline the 20 device functions in the poly_fp call chain, eliminating
// 19 cross-TU function calls with full ABI overhead (save/restore all registers,
// pass parameters through local memory → ~9KB local memory per thread).

#include "eval_check_0.cu"
#include "eval_check_1.cu"
#include "eval_check_2.cu"
#include "eval_check_3.cu"

// --- eval_check kernel + host wrappers (moved from ffi_supra.cu) ---

#include "cuda.h"

namespace risc0::circuit::rv32im_v2::cuda {

// poly_mix is defined in eval_check.cuh (included by eval_check_0.cu)

#ifndef EVAL_CHECK_THREADS
#define EVAL_CHECK_THREADS 256
#endif

__launch_bounds__(EVAL_CHECK_THREADS, 1)
__global__ void eval_check(Fp* check,
                           const Fp* ctrl,
                           const Fp* data,
                           const Fp* accum,
                           const Fp* mix,
                           const Fp* out,
                           const Fp rou,
                           uint32_t po2,
                           uint32_t domain) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle < domain) {
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

} // namespace risc0::circuit::rv32im_v2::cuda

using namespace risc0::circuit::rv32im_v2::cuda;

extern "C" {

const char* risc0_circuit_rv32im_cuda_eval_check(Fp* check,
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
    cudaStream_t stream = getPersistentStream();
    const int block = EVAL_CHECK_THREADS;
    const int grid = (domain + block - 1) / block;
    // Maximize L1 cache for local memory spills
    static bool cacheConfigSet = false;
    if (!cacheConfigSet) {
      cudaFuncSetCacheConfig((const void*)eval_check, cudaFuncCachePreferL1);
      cacheConfigSet = true;
    }
    // Use async copy on our stream to avoid implicit device-wide sync
    CUDA_OK(cudaMemcpyToSymbolAsync(poly_mix, poly_mix_pows, sizeof(poly_mix),
                                     0, cudaMemcpyHostToDevice, stream));
    (void)cudaGetLastError(); // consume any stale async errors
    eval_check<<<grid, block, 0, stream>>>(
        check, ctrl, data, accum, mix, out, rou, po2, domain);
    CUDA_OK(cudaGetLastError());
    // No sync: subsequent CUDA operations on the same persistent stream will
    // automatically wait for eval_check to complete. This allows the CPU to do
    // work (e.g., freeing memory) while the GPU computes eval_check.
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

const char* risc0_circuit_rv32im_cuda_warmup_eval_check() {
  try {
    cudaStream_t stream = getPersistentStream();
    // Launch eval_check with domain=0 to trigger kernel binary loading.
    eval_check<<<1, 1, 0, stream>>>(nullptr, nullptr, nullptr, nullptr,
                                     nullptr, nullptr, Fp(0), 0, 0);
    CUDA_OK(cudaStreamSynchronize(stream));
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

} // extern "C"
