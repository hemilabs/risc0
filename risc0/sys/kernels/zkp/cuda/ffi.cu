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
#include "fp.h"
#include "fpext.h"
#include "kernels.h"

#include "vendor/nvtx3/nvtx3.hpp"

#include <cstdint>
#include <exception>

extern "C" {

// Trigger CUDA module loading for risc0-zkp kernels (eltwise, sha, etc.)
// by launching a dummy kernel. All kernels in this compilation unit share
// the same CUDA module, so loading one loads all.
const char* risc0_zkp_cuda_warmup() {
  try {
    cudaStream_t stream = getPersistentStream();
    // batch_bit_reverse with count=0 exits immediately (bounds check).
    batch_bit_reverse<<<1, 1, 0, stream>>>(nullptr, 0, 0);
    CUDA_OK(cudaStreamSynchronize(stream));
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("warmup failed");
  }
  return nullptr;
}

// Sync the persistent risc0 stream. Must be called before sppark operations
// that read from buffers written by risc0 kernels (different CUDA stream).
const char* risc0_zkp_cuda_sync_stream() {
  try {
    CUDA_OK(cudaStreamSynchronize(getPersistentStream()));
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

// Async H2D copy: uses persistent stream to avoid device-wide sync from hipMemcpy.
const char* risc0_zkp_cuda_memcpy_h2d(void* dst, const void* src, size_t size) {
  try {
    cudaStream_t stream = getPersistentStream();
    CUDA_OK(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream));
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

// Async fill: uses persistent stream to avoid device-wide sync from hipMemsetD32.
const char* risc0_zkp_cuda_fill_u32(uint32_t* buf, uint32_t value, uint32_t count) {
  try {
    cudaStream_t stream = getPersistentStream();
    if (value == 0) {
      CUDA_OK(cudaMemsetAsync(buf, 0, (size_t)count * 4, stream));
    } else {
#ifdef __HIPCC__
      CUDA_OK(hipMemsetD32Async(buf, value, count, stream));
#else
      // cuMemsetD32Async returns CUresult (driver API) but CUDA_OK expects
      // cudaError_t (runtime API). Cast through the compatible error code.
      CUresult res = cuMemsetD32Async((CUdeviceptr)buf, value, count, stream);
      if (res != CUDA_SUCCESS) {
        throw std::runtime_error(fmt("cuMemsetD32Async failed: %d", (int)res));
      }
#endif
    }
  } catch (const std::exception& err) {
    return strdup(err.what());
  }
  return nullptr;
}

const char* risc0_zkp_cuda_eltwise_add_fp(Fp* out, const Fp* x, const Fp* y, uint32_t count) {
  return launchKernel(eltwise_add_fp, count, 0, out, x, y, count);
}

const char* risc0_zkp_cuda_eltwise_mul_factor_fp(Fp* io, Fp factor, uint32_t count) {
  return launchKernel(eltwise_mul_factor_fp, count, 0, io, factor, count);
}

const char* risc0_zkp_cuda_eltwise_copy_fp(Fp* out, const Fp* in, const uint32_t count) {
  return launchKernel(eltwise_copy_fp, count, 0, out, in, count);
}

const char* risc0_zkp_cuda_eltwise_copy_fp_region(Fp* into,
                                                  const Fp* from,
                                                  const uint32_t fromRows,
                                                  const uint32_t fromCols,
                                                  const uint32_t fromOffset,
                                                  const uint32_t fromStride,
                                                  const uint32_t intoOffset,
                                                  const uint32_t intoStride) {
  return launchKernel(eltwise_copy_fp_region,
                      fromRows,
                      0,
                      into,
                      from,
                      fromRows,
                      fromCols,
                      fromOffset,
                      fromStride,
                      intoOffset,
                      intoStride);
}

const char* risc0_zkp_cuda_eltwise_sum_fpext(Fp* out,
                                             const FpExt* in,
                                             const uint32_t to_add,
                                             const uint32_t count) {
  return launchKernel(eltwise_sum_fpext, count, 0, out, in, to_add, count);
}

const char* risc0_zkp_cuda_eltwise_zeroize_fp(Fp* elems, const uint32_t count) {
  return launchKernel(eltwise_zeroize_fp, count, 0, elems);
}

const char* risc0_zkp_cuda_eltwise_zeroize_fpext(FpExt* elems, const uint32_t count) {
  return launchKernel(eltwise_zeroize_fpext, count, 0, elems);
}

const char* risc0_zkp_cuda_fri_fold(Fp* out, const Fp* in, const FpExt* mix, const uint32_t count) {
  return launchKernel(fri_fold, count, 0, out, in, mix, count);
}

const char* risc0_zkp_cuda_mix_poly_coeffs(FpExt* out,
                                           const Fp* in,
                                           const uint32_t* combos,
                                           const FpExt* mixStart,
                                           const FpExt* mix,
                                           const uint32_t inputSize,
                                           const uint32_t count) {
  return launchKernel(mix_poly_coeffs, count, inputSize * sizeof(FpExt),
                      out, in, combos, mixStart, mix, inputSize, count);
}

const char* risc0_zkp_cuda_batch_bit_reverse(Fp* io, const uint32_t nBits, const uint32_t count) {
  return launchKernel(batch_bit_reverse, count, 0, io, nBits, count);
}

const char* risc0_zkp_cuda_batch_evaluate_any(FpExt* out,
                                              const Fp* coeffs,
                                              const uint32_t* which,
                                              const FpExt* xs,
                                              uint32_t shared_size,
                                              const uint32_t count,
                                              const uint32_t deg) {
  return launchKernel(batch_evaluate_any, count, shared_size, out, coeffs, which, xs, deg);
}

const char* risc0_zkp_cuda_gather_sample(
    Fp* dst, const Fp* src, const uint32_t idx, const uint32_t size, const uint32_t stride) {
  return launchKernel(gather_sample, size, 0, dst, src, idx, size, stride);
}

const char* risc0_zkp_cuda_scatter(Fp* into,
                                   const uint32_t* index,
                                   const uint32_t* offsets,
                                   const Fp* values,
                                   const uint32_t count) {
  return launchKernel(scatter, count, 0, into, index, offsets, values, count);
}

const char* risc0_zkp_cuda_scatter_bits(Fp* into,
                                        const uint32_t* data,
                                        const uint32_t cycles,
                                        const uint32_t count) {
  return launchKernel(scatter_bits, count, 0, into, data, cycles, count);
}

// Scatter from host memory using persistent device + pinned host buffers.
// Pinned host buffers avoid HIP's slow pageable memory staging path.
const char* risc0_zkp_cuda_scatter_from_host(Fp* into,
                                             const uint32_t* h_index,
                                             uint32_t index_count,
                                             const uint32_t* h_offsets,
                                             uint32_t offsets_count,
                                             const Fp* h_values,
                                             uint32_t values_count,
                                             uint32_t count) {
  try {
    cudaStream_t stream = getPersistentStream();
    int dev = getCachedDevice();

    // Per-device persistent device buffers (grow-only, reused across segments).
    static uint32_t* d_index[16] = {};
    static uint32_t* d_offsets[16] = {};
    static Fp* d_values[16] = {};
    static size_t cap_d_index[16] = {}, cap_d_offsets[16] = {}, cap_d_values[16] = {};

    // Per-device persistent pinned host staging buffers (grow-only).
    static void* h_pinned[16] = {};
    static size_t cap_h_pinned[16] = {};

    size_t index_bytes = index_count * sizeof(uint32_t);
    size_t offsets_bytes = offsets_count * sizeof(uint32_t);
    size_t values_bytes = values_count * sizeof(Fp);
    size_t total_bytes = index_bytes + offsets_bytes + values_bytes;

    // Grow device buffers if needed.
    if (index_bytes > cap_d_index[dev]) {
      if (d_index[dev]) CUDA_OK(cudaFree(d_index[dev]));
      CUDA_OK(cudaMalloc(&d_index[dev], index_bytes));
      cap_d_index[dev] = index_bytes;
    }
    if (offsets_bytes > cap_d_offsets[dev]) {
      if (d_offsets[dev]) CUDA_OK(cudaFree(d_offsets[dev]));
      CUDA_OK(cudaMalloc(&d_offsets[dev], offsets_bytes));
      cap_d_offsets[dev] = offsets_bytes;
    }
    if (values_bytes > cap_d_values[dev]) {
      if (d_values[dev]) CUDA_OK(cudaFree(d_values[dev]));
      CUDA_OK(cudaMalloc(&d_values[dev], values_bytes));
      cap_d_values[dev] = values_bytes;
    }

    // Grow pinned host staging buffer if needed.
    if (total_bytes > cap_h_pinned[dev]) {
      if (h_pinned[dev]) CUDA_OK(cudaFreeHost(h_pinned[dev]));
      CUDA_OK(cudaHostAlloc(&h_pinned[dev], total_bytes, cudaHostAllocDefault));
      cap_h_pinned[dev] = total_bytes;
    }

    // Copy to pinned staging buffer (fast CPU memcpy).
    char* pin = (char*)h_pinned[dev];
    memcpy(pin, h_index, index_bytes);
    memcpy(pin + index_bytes, h_offsets, offsets_bytes);
    memcpy(pin + index_bytes + offsets_bytes, h_values, values_bytes);

    // Truly async H2D transfers from pinned memory.
    CUDA_OK(cudaMemcpyAsync(d_index[dev], pin, index_bytes,
                            cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(d_offsets[dev], pin + index_bytes, offsets_bytes,
                            cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(d_values[dev], pin + index_bytes + offsets_bytes, values_bytes,
                            cudaMemcpyHostToDevice, stream));

    // Launch scatter kernel on same stream (waits for DMA implicitly).
    LaunchConfig cfg = getCachedSimpleConfig(count);
    scatter<<<cfg.grid, cfg.block, 0, stream>>>(
        into, d_index[dev], d_offsets[dev], d_values[dev], count);

  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

// Scatter bits from host memory using persistent device + pinned host buffer.
const char* risc0_zkp_cuda_scatter_bits_from_host(Fp* into,
                                                  const uint32_t* h_data,
                                                  uint32_t triplet_count,
                                                  uint32_t cycles) {
  try {
    cudaStream_t stream = getPersistentStream();
    int dev = getCachedDevice();

    // Per-device persistent device buffer (grow-only, reused across segments).
    static uint32_t* d_bitdata[16] = {};
    static size_t cap_d_bitdata[16] = {};

    // Per-device persistent pinned host staging buffer (grow-only).
    static void* h_pinned_bits[16] = {};
    static size_t cap_h_pinned_bits[16] = {};

    size_t data_bytes = (size_t)triplet_count * 3 * sizeof(uint32_t);

    if (data_bytes > cap_d_bitdata[dev]) {
      if (d_bitdata[dev]) CUDA_OK(cudaFree(d_bitdata[dev]));
      CUDA_OK(cudaMalloc(&d_bitdata[dev], data_bytes));
      cap_d_bitdata[dev] = data_bytes;
    }

    // Grow pinned host staging buffer if needed.
    if (data_bytes > cap_h_pinned_bits[dev]) {
      if (h_pinned_bits[dev]) CUDA_OK(cudaFreeHost(h_pinned_bits[dev]));
      CUDA_OK(cudaHostAlloc(&h_pinned_bits[dev], data_bytes, cudaHostAllocDefault));
      cap_h_pinned_bits[dev] = data_bytes;
    }

    // Copy to pinned staging and do truly async H2D.
    memcpy(h_pinned_bits[dev], h_data, data_bytes);
    CUDA_OK(cudaMemcpyAsync(d_bitdata[dev], h_pinned_bits[dev], data_bytes,
                            cudaMemcpyHostToDevice, stream));

    // Launch scatter_bits kernel on same stream.
    LaunchConfig cfg = getCachedSimpleConfig(triplet_count);
    scatter_bits<<<cfg.grid, cfg.block, 0, stream>>>(
        into, d_bitdata[dev], cycles, triplet_count);

  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

const char*
risc0_zkp_cuda_sha_rows(ShaDigest* output, const Fp* matrix, uint32_t rowSize, uint32_t colSize) {
  return launchKernel(sha_rows, rowSize, 0, output, matrix, rowSize, colSize);
}

const char* risc0_zkp_cuda_sha_fold(ShaDigest* output, const ShaDigest* input, uint32_t count) {
  return launchKernel(sha_fold, count, 0, output, input, count);
}

const char* risc0_zkp_cuda_gather_digests(
    uint32_t* dst, const uint32_t* src, const uint32_t* indices, uint32_t count) {
  return launchKernel(gather_digests, count, 0, dst, src, indices, count);
}

const char* risc0_zkp_cuda_combos_prepare(FpExt* combos,
                                          const FpExt* coeffU,
                                          const uint32_t comboCount,
                                          const uint32_t cycles,
                                          const uint32_t regsCount,
                                          const uint32_t* regSizes,
                                          const uint32_t* regComboIds,
                                          const uint32_t checkSize,
                                          const FpExt* mix) {

  try {
    cudaStream_t stream = getPersistentStream();
    combos_prepare<<<1, 1, 0, stream>>>(
        combos, coeffU, regsCount, regSizes, regComboIds, cycles, mix, checkSize, comboCount);
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

} // extern "C"
