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

// Scatter from host memory using persistent device buffers.
// Reuses grow-only device buffers across segments to avoid per-call allocation.
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

    // Persistent device buffers (grow-only, reused across segments).
    static uint32_t* d_index = nullptr;
    static uint32_t* d_offsets = nullptr;
    static Fp* d_values = nullptr;
    static size_t cap_d_index = 0, cap_d_offsets = 0, cap_d_values = 0;

    size_t index_bytes = index_count * sizeof(uint32_t);
    size_t offsets_bytes = offsets_count * sizeof(uint32_t);
    size_t values_bytes = values_count * sizeof(Fp);

    // Grow device buffers if needed.
    if (index_bytes > cap_d_index) {
      if (d_index) CUDA_OK(cudaFree(d_index));
      CUDA_OK(cudaMalloc(&d_index, index_bytes));
      cap_d_index = index_bytes;
    }
    if (offsets_bytes > cap_d_offsets) {
      if (d_offsets) CUDA_OK(cudaFree(d_offsets));
      CUDA_OK(cudaMalloc(&d_offsets, offsets_bytes));
      cap_d_offsets = offsets_bytes;
    }
    if (values_bytes > cap_d_values) {
      if (d_values) CUDA_OK(cudaFree(d_values));
      CUDA_OK(cudaMalloc(&d_values, values_bytes));
      cap_d_values = values_bytes;
    }

    // Async H2D transfers (CUDA driver stages pageable memory internally).
    CUDA_OK(cudaMemcpyAsync(d_index, h_index, index_bytes,
                            cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(d_offsets, h_offsets, offsets_bytes,
                            cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyAsync(d_values, h_values, values_bytes,
                            cudaMemcpyHostToDevice, stream));

    // Launch scatter kernel on same stream (waits for DMA implicitly).
    LaunchConfig cfg = getCachedSimpleConfig(count);
    scatter<<<cfg.grid, cfg.block, 0, stream>>>(
        into, d_index, d_offsets, d_values, count);

  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

// Scatter bits from host memory using persistent device buffer.
const char* risc0_zkp_cuda_scatter_bits_from_host(Fp* into,
                                                  const uint32_t* h_data,
                                                  uint32_t triplet_count,
                                                  uint32_t cycles) {
  try {
    cudaStream_t stream = getPersistentStream();

    // Persistent device buffer (grow-only, reused across segments).
    static uint32_t* d_bitdata = nullptr;
    static size_t cap_d_bitdata = 0;

    size_t data_bytes = (size_t)triplet_count * 3 * sizeof(uint32_t);

    if (data_bytes > cap_d_bitdata) {
      if (d_bitdata) CUDA_OK(cudaFree(d_bitdata));
      CUDA_OK(cudaMalloc(&d_bitdata, data_bytes));
      cap_d_bitdata = data_bytes;
    }

    // Async H2D transfer (CUDA driver stages pageable memory internally).
    CUDA_OK(cudaMemcpyAsync(d_bitdata, h_data, data_bytes,
                            cudaMemcpyHostToDevice, stream));

    // Launch scatter_bits kernel on same stream.
    LaunchConfig cfg = getCachedSimpleConfig(triplet_count);
    scatter_bits<<<cfg.grid, cfg.block, 0, stream>>>(
        into, d_bitdata, cycles, triplet_count);

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
