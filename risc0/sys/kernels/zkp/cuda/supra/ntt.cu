#include "ff/baby_bear.hpp"
#include "ntt/ntt.cuh"

// Expose protected bit_rev from NTT base class
struct NTT_helper : public NTT {
    using NTT::bit_rev;
};

// Batched LDE expand kernel: write all output elements in one coalesced pass.
// For each output position, either copy the corresponding input element
// (if position is a multiple of blowup) or write zero.
template<class fr_t>
__launch_bounds__(256)
__global__ void batch_lde_expand_kernel(
    fr_t* __restrict__ d_out,
    const fr_t* __restrict__ d_in,
    uint32_t domain_size,
    uint32_t lg_ext_size,
    uint32_t lg_blowup,
    uint32_t poly_count)
{
    uint32_t ext_size = 1u << lg_ext_size;
    uint32_t ext_mask = ext_size - 1;
    uint32_t blowup_mask = (1u << lg_blowup) - 1;
    uint64_t total = (uint64_t)ext_size * poly_count;

    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < total;
         i += (uint64_t)gridDim.x * blockDim.x)
    {
        uint32_t pos_in_ext = (uint32_t)i & ext_mask;
        uint32_t col = (uint32_t)(i >> lg_ext_size);
        fr_t val;
        if ((pos_in_ext & blowup_mask) == 0) {
            uint32_t src_idx = pos_in_ext >> lg_blowup;
            val = d_in[(uint64_t)col * domain_size + src_idx];
        } else {
            val = fr_t(0);
        }
        d_out[i] = val;
    }
}

static inline int cached_sm_count() {
    int device = 0;
    CUDA_OK(cudaGetDevice(&device));
    static int counts[16] = {};
    if (counts[device] == 0) {
        CUDA_OK(cudaDeviceGetAttribute(&counts[device], cudaDevAttrMultiProcessorCount, device));
    }
    return counts[device];
}

static inline void launch_batch_expand(const gpu_t& gpu,
                                       fr_t* d_out, fr_t* d_in,
                                       uint32_t domain_size,
                                       uint32_t lg_domain_size,
                                       uint32_t lg_blowup,
                                       uint32_t poly_count)
{
    uint32_t lg_ext = lg_domain_size + lg_blowup;
    uint32_t ext_domain_size = 1u << lg_ext;
    uint64_t total = (uint64_t)ext_domain_size * poly_count;

    uint32_t block_sz = 256;
    uint32_t max_blocks = (uint32_t)cached_sm_count() * 16;
    uint32_t need_blocks = (uint32_t)((total + block_sz - 1) / block_sz);
    uint32_t grid_sz = need_blocks < max_blocks ? need_blocks : max_blocks;

    batch_lde_expand_kernel<fr_t><<<grid_sz, block_sz, 0, (cudaStream_t)gpu>>>(
        d_out, d_in, domain_size, lg_ext, lg_blowup, poly_count);
}

extern "C" RustError::by_value sppark_init() {
  // Always call select_gpu(-1) to ensure the CUDA context is retained
  // on the calling thread for its current device.
  (void)select_gpu(-1);

  int device = 0;
  cudaGetDevice(&device);
  static bool initialized[16] = {};
  if (initialized[device])
    return RustError{cudaSuccess};

  // Use lg_domain_size=16 (64K elements) to exercise all NTT kernel variants
  // (CT_NTT<8,true/false>, GS_NTT<8,true/false>, batch_bit_reverse,
  // LDE_distribute_powers). The small size (256KB) runs in ~1ms but avoids
  // ~5-10ms of first-launch stalls during the actual proof.
  uint32_t lg_domain_size = 16;
  uint32_t domain_size = 1U << lg_domain_size;

  std::vector<fr_t> inout(domain_size, fr_t(0));
  inout[0] = fr_t(1);
  inout[1] = fr_t(1);

  const gpu_t& gpu = select_gpu(-1);

  try {
    NTT::Base(gpu,
              &inout[0],
              lg_domain_size,
              NTT::InputOutputOrder::NR,
              NTT::Direction::forward,
              NTT::Type::standard);
    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  } catch (...) {
    return RustError(cudaErrorUnknown, "Generic exception");
  }

  initialized[device] = true;
  return RustError{cudaSuccess};
}

extern "C" RustError::by_value sppark_batch_expand(
    fr_t* d_out, fr_t* d_in, uint32_t lg_domain_size, uint32_t lg_blowup, uint32_t poly_count) {
  if (lg_domain_size == 0)
    return RustError{cudaSuccess};

  uint32_t domain_size = 1U << lg_domain_size;

  const gpu_t& gpu = select_gpu(-1);

  try {
    launch_batch_expand(gpu, d_out, d_in, domain_size, lg_domain_size, lg_blowup, poly_count);
    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  } catch (...) {
    return RustError(cudaErrorUnknown, "Generic exception");
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
sppark_batch_NTT(fr_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count) {
  if (lg_domain_size == 0)
    return RustError{cudaSuccess};

  uint32_t domain_size = 1U << lg_domain_size;

  const gpu_t& gpu = select_gpu(-1);

  try {
    // Process each polynomial sequentially on the same stream
    for (uint32_t i = 0; i < poly_count; i++) {
      NTT::Base_dev_ptr(gpu, &d_inout[(uint64_t)i * domain_size],
                        lg_domain_size,
                        NTT::InputOutputOrder::RN,
                        NTT::Direction::forward,
                        NTT::Type::standard);
    }

    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  } catch (...) {
    return RustError(cudaErrorUnknown, "Generic exception");
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
sppark_batch_iNTT(fr_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count) {
  if (lg_domain_size == 0)
    return RustError{cudaSuccess};

  uint32_t domain_size = 1U << lg_domain_size;

  const gpu_t& gpu = select_gpu(-1);

  try {
    for (uint32_t i = 0; i < poly_count; i++) {
      NTT::Base_dev_ptr(gpu, &d_inout[(uint64_t)i * domain_size],
                        lg_domain_size,
                        NTT::InputOutputOrder::NR,
                        NTT::Direction::inverse,
                        NTT::Type::standard);
    }

    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  } catch (...) {
    return RustError(cudaErrorUnknown, "Generic exception");
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
sppark_batch_zk_shift(fr_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count) {
  if (lg_domain_size == 0)
    return RustError{cudaSuccess};

  uint32_t domain_size = 1U << lg_domain_size;

  const gpu_t& gpu = select_gpu(-1);

  try {
    // Apply ZK shift per polynomial
    for (uint32_t i = 0; i < poly_count; i++) {
      NTT::LDE_powers(gpu, &d_inout[(uint64_t)i * domain_size], lg_domain_size);
    }

    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  } catch (...) {
    return RustError(cudaErrorUnknown, "Generic exception");
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
sppark_batch_iNTT_zk_shift(fr_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count) {
  if (lg_domain_size == 0)
    return RustError{cudaSuccess};

  uint32_t domain_size = 1U << lg_domain_size;

  const gpu_t& gpu = select_gpu(-1);

  try {
    // iNTT + ZK shift per polynomial
    for (uint32_t i = 0; i < poly_count; i++) {
      NTT::Base_dev_ptr(gpu, &d_inout[(uint64_t)i * domain_size],
                        lg_domain_size,
                        NTT::InputOutputOrder::NR,
                        NTT::Direction::inverse,
                        NTT::Type::standard);
      NTT::LDE_powers(gpu, &d_inout[(uint64_t)i * domain_size], lg_domain_size);
    }

    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  } catch (...) {
    return RustError(cudaErrorUnknown, "Generic exception");
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
sppark_batch_expand_NTT(fr_t* d_out, fr_t* d_in,
                        uint32_t lg_domain_size, uint32_t lg_blowup, uint32_t poly_count) {
  if (lg_domain_size == 0)
    return RustError{cudaSuccess};

  uint32_t domain_size = 1U << lg_domain_size;
  uint32_t ext_domain_size = domain_size << lg_blowup;
  uint32_t lg_ext = lg_domain_size + lg_blowup;

  const gpu_t& gpu = select_gpu(-1);

  try {
    // Batched expand: single kernel for all columns
    launch_batch_expand(gpu, d_out, d_in, domain_size, lg_domain_size, lg_blowup, poly_count);

    // Forward NTT per polynomial on the extended domain
    for (uint32_t i = 0; i < poly_count; i++) {
      NTT::Base_dev_ptr(gpu, &d_out[(uint64_t)i * ext_domain_size],
                        lg_ext,
                        NTT::InputOutputOrder::RN,
                        NTT::Direction::forward,
                        NTT::Type::standard);
    }

    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  } catch (...) {
    return RustError(cudaErrorUnknown, "Generic exception");
  }

  return RustError{cudaSuccess};
}

// Batched bit-reverse permutation on the sppark stream.
// Uses sppark's optimized bit_rev (shared memory + Z_COUNT blocking) per polynomial.
extern "C" RustError::by_value
sppark_batch_bit_reverse(fr_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count) {
  if (lg_domain_size == 0 || poly_count == 0)
    return RustError{cudaSuccess};

  uint64_t domain_size = 1ULL << lg_domain_size;

  const gpu_t& gpu = select_gpu(-1);

  try {
    for (uint32_t i = 0; i < poly_count; i++) {
      NTT_helper::bit_rev(&d_inout[i * domain_size], &d_inout[i * domain_size],
                          lg_domain_size, gpu);
    }

    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  } catch (...) {
    return RustError(cudaErrorUnknown, "Generic exception");
  }

  return RustError{cudaSuccess};
}
