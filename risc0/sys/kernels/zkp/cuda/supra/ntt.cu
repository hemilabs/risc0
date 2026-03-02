#include "ff/baby_bear.hpp"
#include "ntt/ntt.cuh"
#include <mutex>

// Batched LDE expand kernel: write all output elements in one coalesced pass.
// For each output position, either copy the corresponding input element
// (if position is a multiple of blowup) or write zero.
template<class fr_t>
__launch_bounds__(256)
__global__ void batch_lde_expand_kernel(
    fr_t* __restrict__ d_out,
    const fr_t* __restrict__ d_in,
    uint32_t domain_size,
    uint32_t ext_size,
    uint32_t blowup,
    uint32_t poly_count)
{
    uint64_t total = (uint64_t)ext_size * poly_count;

    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < total;
         i += (uint64_t)gridDim.x * blockDim.x)
    {
        uint32_t pos_in_ext = (uint32_t)(i % ext_size);
        uint32_t col = (uint32_t)(i / ext_size);
        fr_t val;
        if ((pos_in_ext % blowup) == 0) {
            uint32_t src_idx = pos_in_ext / blowup;
            val = d_in[(uint64_t)col * domain_size + src_idx];
        } else {
            val = fr_t(0);
        }
        d_out[i] = val;
    }
}

static inline int cached_sm_count() {
    // Benign race: worst case two threads both compute the same value.
    static int count = 0;
    if (count == 0) {
        int device;
        CUDA_OK(cudaGetDevice(&device));
        int c;
        CUDA_OK(cudaDeviceGetAttribute(&c, cudaDevAttrMultiProcessorCount, device));
        count = c;
    }
    return count;
}

static inline void launch_batch_expand(const gpu_t& gpu,
                                       fr_t* d_out, fr_t* d_in,
                                       uint32_t domain_size,
                                       uint32_t lg_domain_size,
                                       uint32_t lg_blowup,
                                       uint32_t poly_count)
{
    uint32_t ext_domain_size = domain_size << lg_blowup;
    uint32_t blowup = 1u << lg_blowup;
    uint64_t total = (uint64_t)ext_domain_size * poly_count;

    uint32_t block_sz = 256;
    uint32_t max_blocks = (uint32_t)cached_sm_count() * 16;
    uint32_t need_blocks = (uint32_t)((total + block_sz - 1) / block_sz);
    uint32_t grid_sz = need_blocks < max_blocks ? need_blocks : max_blocks;

    batch_lde_expand_kernel<fr_t><<<grid_sz, block_sz, 0, (cudaStream_t)gpu>>>(
        d_out, d_in, domain_size, ext_domain_size, blowup, poly_count);
}

extern "C" RustError::by_value sppark_init() {
  // Always call select_gpu() to ensure the primary CUDA context is retained
  // on the calling thread. This is critical when cuda_warmup() runs on a
  // background thread first (setting initialized=true), then the main thread
  // calls sppark_init() and needs the context to be current for cust/DeviceBuffer.
  (void)select_gpu();

  // Thread-safe initialization using std::call_once
  static std::once_flag init_flag;
  static RustError init_result{cudaSuccess};

  std::call_once(init_flag, []() {
    // Use lg_domain_size=16 (64K elements) to exercise all NTT kernel variants
    // (CT_NTT<8,true/false>, GS_NTT<8,true/false>, batch_bit_reverse,
    // LDE_distribute_powers). The small size (256KB) runs in ~1ms but avoids
    // ~5-10ms of first-launch stalls during the actual proof.
    uint32_t lg_domain_size = 16;
    uint32_t domain_size = 1U << lg_domain_size;

    std::vector<fr_t> inout(domain_size, fr_t(0));
    inout[0] = fr_t(1);
    inout[1] = fr_t(1);

    const gpu_t& gpu = select_gpu();

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
      init_result = RustError{e.code(), e.what()};
    } catch (...) {
      init_result = RustError(cudaErrorUnknown, "Generic exception");
    }
  });

  return init_result;
}

extern "C" RustError::by_value sppark_batch_expand(
    fr_t* d_out, fr_t* d_in, uint32_t lg_domain_size, uint32_t lg_blowup, uint32_t poly_count) {
  if (lg_domain_size == 0)
    return RustError{cudaSuccess};

  uint32_t domain_size = 1U << lg_domain_size;

  const gpu_t& gpu = select_gpu();

  try {
    launch_batch_expand(gpu, d_out, d_in, domain_size, lg_domain_size, lg_blowup, poly_count);
    CUDA_OK(cudaGetLastError());
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

  const gpu_t& gpu = select_gpu();

  try {
    // Single batched NTT call processes all columns in parallel
    NTT::Base_dev_ptr_batch(gpu,
                            d_inout,
                            lg_domain_size,
                            NTT::InputOutputOrder::RN,
                            NTT::Direction::forward,
                            NTT::Type::standard,
                            poly_count, domain_size);

    CUDA_OK(cudaGetLastError());
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

  const gpu_t& gpu = select_gpu();

  try {
    // Single batched iNTT call processes all columns in parallel
    NTT::Base_dev_ptr_batch(gpu,
                            d_inout,
                            lg_domain_size,
                            NTT::InputOutputOrder::NR,
                            NTT::Direction::inverse,
                            NTT::Type::standard,
                            poly_count, domain_size);

    CUDA_OK(cudaGetLastError());
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

  const gpu_t& gpu = select_gpu();

  try {
    // Single batched kernel for all columns instead of per-column launches
    NTT::LDE_powers_batch(gpu, d_inout, lg_domain_size, poly_count, domain_size);

    CUDA_OK(cudaGetLastError());
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

  const gpu_t& gpu = select_gpu();

  try {
    // Single batched iNTT call processes all columns in parallel
    NTT::Base_dev_ptr_batch(gpu,
                            d_inout,
                            lg_domain_size,
                            NTT::InputOutputOrder::NR,
                            NTT::Direction::inverse,
                            NTT::Type::standard,
                            poly_count, domain_size);
    // Single batched ZK shift for all columns
    NTT::LDE_powers_batch(gpu, d_inout, lg_domain_size, poly_count, domain_size);

    CUDA_OK(cudaGetLastError());
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

  const gpu_t& gpu = select_gpu();

  try {
    // Batched expand: single kernel for all columns
    launch_batch_expand(gpu, d_out, d_in, domain_size, lg_domain_size, lg_blowup, poly_count);

    // Single batched forward NTT for all columns
    NTT::Base_dev_ptr_batch(gpu,
                            d_out,
                            lg_ext,
                            NTT::InputOutputOrder::RN,
                            NTT::Direction::forward,
                            NTT::Type::standard,
                            poly_count, ext_domain_size);

    CUDA_OK(cudaGetLastError());
    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  } catch (...) {
    return RustError(cudaErrorUnknown, "Generic exception");
  }

  return RustError{cudaSuccess};
}
