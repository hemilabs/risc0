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

#ifdef __HIPCC__
// RDNA4: 128 threads reduces scratch working set → better cache reuse
#ifndef EVAL_CHECK_THREADS
#define EVAL_CHECK_THREADS 128
#endif
#else
#ifndef EVAL_CHECK_THREADS
#define EVAL_CHECK_THREADS 256
#endif
#endif

// Diagnostic kernel: write poly_mix[0..3] and data samples to check[0..15]
// Called with <<<1,1>>> before eval_check to verify constant memory
__global__ void eval_check_diag(Fp* check,
                                const Fp* ctrl,
                                const Fp* data,
                                const Fp* accum,
                                const Fp* mix,
                                const Fp* out,
                                uint32_t domain) {
  // Write poly_mix first 4 values (each is FpExt = 4 Fp)
  // check[0..3] = poly_mix[0] components
  for (int i = 0; i < 4; i++) check[i] = poly_mix[0][i];
  // check[4..7] = poly_mix[1] components
  for (int i = 0; i < 4; i++) check[4+i] = poly_mix[1][i];
  // check[8] = ctrl[0], check[9] = data[0], check[10] = accum[0]
  check[8] = ctrl[0];
  check[9] = data[0];
  check[10] = accum[0];
  check[11] = mix[0];
  check[12] = out[0];
  // check[13] = domain as Fp
  check[13] = Fp(domain);
  // check[14..15] = poly_mix[457] components (last)
  check[14] = poly_mix[457][0];
  check[15] = poly_mix[457][1];
}

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
#ifdef __HIPCC__
    Fp x = rou ^ (unsigned)cycle;
    Fp y = (Fp(3) * x) ^ (unsigned)(1 << po2);
#else
    Fp x = pow(rou, cycle);
    Fp y = pow(Fp(3) * x, 1 << po2);
#endif
    FpExt ret = tot * inv(y - Fp(1));
    check[domain * 0 + cycle] = ret[0];
    check[domain * 1 + cycle] = ret[1];
    check[domain * 2 + cycle] = ret[2];
    check[domain * 3 + cycle] = ret[3];
  }
}

} // namespace risc0::circuit::rv32im_v2::cuda

using namespace risc0::circuit::rv32im_v2::cuda;

// Secondary stream for eval_check pipelining.
// eval_check runs on this separate stream so that subsequent segments'
// main phase (witgen, commit, accum) can overlap with it on the persistent stream.
// eval_check uses only ~6% of GPU wave slots (128 out of 2048), so 94% of
// compute resources remain available for concurrent work.
static cudaStream_t getEvalCheckStream() {
  static cudaStream_t stream = nullptr;
  if (!stream) {
    CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
  return stream;
}

static cudaEvent_t getEvalCheckEvent() {
  static cudaEvent_t event = nullptr;
  if (!event) {
    CUDA_OK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  }
  return event;
}

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
    // Use separate stream for eval_check to enable pipelining:
    // eval_check(N) runs concurrently with main phase of segment N+1.
    cudaStream_t stream = getEvalCheckStream();

    // Make eval_check stream wait for persistent stream's prior work to complete.
    // This ensures that the input buffers (ctrl, data, accum, mix, out) written
    // by the main phase on the persistent stream are visible to eval_check.
    {
      cudaEvent_t event = getEvalCheckEvent();
      CUDA_OK(cudaEventRecord(event, getPersistentStream()));
      CUDA_OK(cudaStreamWaitEvent(stream, event, 0));
    }

    const int block = EVAL_CHECK_THREADS;
    const int grid = (domain + block - 1) / block;
    // Maximize L1 cache for local memory spills
    static bool cacheConfigSet = false;
    if (!cacheConfigSet) {
      cudaFuncSetCacheConfig((const void*)eval_check, cudaFuncCachePreferL1);
      cacheConfigSet = true;
    }
    // Copy poly_mix_pows to __constant__ memory on eval_check stream
#ifdef __HIPCC__
    // HIP: hipMemcpyToSymbol doesn't work reliably without -fgpu-rdc.
    // Use hipGetSymbolAddress + hipMemcpyAsync instead.
    {
      void* dev_ptr = nullptr;
      CUDA_OK(hipGetSymbolAddress(&dev_ptr, HIP_SYMBOL(poly_mix)));
      CUDA_OK(hipMemcpyAsync(dev_ptr, poly_mix_pows, sizeof(poly_mix),
                              hipMemcpyHostToDevice, stream));
    }
#else
    CUDA_OK(cudaMemcpyToSymbolAsync(poly_mix, poly_mix_pows, sizeof(poly_mix),
                                     0, cudaMemcpyHostToDevice, stream));
#endif
    (void)cudaGetLastError(); // consume any stale async errors

    eval_check<<<grid, block, 0, stream>>>(
        check, ctrl, data, accum, mix, out, rou, po2, domain);
    CUDA_OK(cudaGetLastError());
    // No sync: eval_check runs asynchronously on its own stream.
    // Call risc0_circuit_rv32im_cuda_eval_check_sync() or
    // risc0_circuit_rv32im_cuda_eval_check_dep() to synchronize.
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

// Wait for eval_check to complete before reading check_poly.
// Uses cudaDeviceSynchronize() because eval_check runs on its own stream
// (getEvalCheckStream) while downstream operations (iNTT via sppark, Merkle
// via risc0-sys) run on separate streams. cudaDeviceSynchronize() ensures
// ALL streams are drained.
const char* risc0_circuit_rv32im_cuda_eval_check_dep() {
  try {
    CUDA_OK(cudaDeviceSynchronize());
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

const char* risc0_circuit_rv32im_cuda_warmup_eval_check() {
  try {
    cudaStream_t stream = getEvalCheckStream();
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
