#include <ff/alt_bn128.hpp>
#include <ff/baby_bear.hpp>
#include <util/gpu_t.cuh>
#include <util/rusterror.h>

#include <polynomial/div_by_x_minus_z.cuh>
#include <polynomial/prefix_op.cuh>

#include "poseidon2.cuh"
#include "poseidon254.cuh"

// Workaround: cudaGetDeviceProperties returns multiProcessorCount=1
// in some VM/passthrough setups, while cudaDeviceGetAttribute returns
// the correct value. Cache the correct SM count for use in cooperative
// kernel launches.
static int get_real_sm_count() {
    static int sm_count = 0;
    if (sm_count == 0) {
        int device;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    }
    return sm_count;
}

extern "C" RustError::by_value sppark_poseidon2_init() {
  static bool initialized = false;
  if (initialized)
    return RustError{cudaSuccess};

  const gpu_t& gpu = select_gpu();
  try {
    // Allocate tiny device buffer and run 1-hash poseidon2_fold to trigger
    // CUDA module loading for poseidon2 kernels (~5-10ms first-use overhead).
    void* d_buf = nullptr;
    CUDA_OK(cudaMalloc(&d_buf, 2048));
    CUDA_OK(cudaMemset(d_buf, 0, 2048));
    _poseidon2_fold<<<1, 1, 0, gpu>>>(
        (poseidon_out_t*)d_buf,
        (const poseidon_in_t*)((char*)d_buf + 1024),
        1);
    CUDA_OK(cudaGetLastError());
    gpu.sync();
    CUDA_OK(cudaFree(d_buf));
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }
  initialized = true;
  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
sppark_poseidon2_fold(poseidon_out_t* d_out, const poseidon_in_t* d_in, size_t num_hashes) {
  const gpu_t& gpu = select_gpu();

  size_t block_size = num_hashes < 256 ? num_hashes : 256;
  size_t num_blocks = num_hashes < 256 ? 1 : num_hashes / 256;

  try {
    (void)cudaGetLastError(); // consume any stale async errors
    _poseidon2_fold<<<num_blocks, block_size, 0, gpu>>>(d_out, d_in, num_hashes);

    CUDA_OK(cudaGetLastError());

    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
sppark_poseidon2_fold_tree(poseidon_out_t* nodes, uint32_t layers) {
  const gpu_t& gpu = select_gpu();

  try {
    (void)cudaGetLastError(); // consume any stale async errors
    for (int i = layers - 1; i >= 0; i--) {
      uint32_t layer_size = 1u << i;
      size_t block_size = layer_size < 256 ? layer_size : 256;
      size_t num_blocks = layer_size < 256 ? 1 : layer_size / 256;

      _poseidon2_fold<<<num_blocks, block_size, 0, gpu>>>(
          nodes + layer_size,
          (const poseidon_in_t*)(nodes + 2 * layer_size),
          layer_size);
      CUDA_OK(cudaGetLastError());
    }
    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
sppark_poseidon2_rows(poseidon_out_t* d_out, const fr_t* d_in, uint32_t count, uint32_t col_size) {
  const gpu_t& gpu = select_gpu();

  size_t block_size = count < 256 ? count : 256;
  size_t num_blocks = (count + block_size - 1) / block_size;

  try {
    (void)cudaGetLastError(); // consume any stale async errors
    _poseidon2_rows<<<num_blocks, block_size, 0, gpu>>>(d_out, d_in, count, col_size);

    CUDA_OK(cudaGetLastError());

    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }

  return RustError{cudaSuccess};
}

static void compute_grid_block_size(size_t total_count, size_t& block_size, size_t& num_blocks) {
  size_t min_block_size = 4 * WARP_SZ;

  if (total_count < (block_size * num_blocks)) {
    size_t count_per_block = total_count / num_blocks;

    if (count_per_block > min_block_size) {
      block_size = ((count_per_block + min_block_size - 1) / min_block_size) * min_block_size;
      num_blocks = (total_count + block_size - 1) / block_size;
    } else {
      block_size = min_block_size;
      num_blocks = (total_count + min_block_size - 1) / min_block_size;
    }
  } else {
    size_t base_iter = (total_count + (num_blocks * block_size) - 1) / (num_blocks * block_size);
    size_t out_block_size = block_size;

    for (size_t cur_block_size = block_size - min_block_size; cur_block_size >= min_block_size;
         cur_block_size -= min_block_size) {
      size_t cur_iter =
          (total_count + (num_blocks * cur_block_size) - 1) / (num_blocks * cur_block_size);

      if (cur_iter != base_iter)
        break;
      out_block_size = cur_block_size;
    }

    block_size = out_block_size;
  }
}

extern "C" RustError::by_value
sppark_poseidon254_fold(alt_bn128::fr_t* d_out, const alt_bn128::fr_t* d_in, size_t num_hashes) {
  const gpu_t& gpu = select_gpu();

  size_t block_size = 512;
  size_t num_blocks = gpu.sm_count();

  compute_grid_block_size(num_hashes, block_size, num_blocks);

  try {
    (void)cudaGetLastError(); // consume any stale async errors
    _poseidon254_fold<<<num_blocks, block_size, 0, gpu>>>(d_out, d_in, num_hashes);

    CUDA_OK(cudaGetLastError());

    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
sppark_poseidon254_fold_tree(alt_bn128::fr_t* nodes, uint32_t layers) {
  const gpu_t& gpu = select_gpu();

  size_t block_size = 512;
  size_t num_blocks = gpu.sm_count();

  try {
    (void)cudaGetLastError(); // consume any stale async errors
    for (int i = layers - 1; i >= 0; i--) {
      uint32_t layer_size = 1u << i;
      size_t bs = block_size;
      size_t nb = num_blocks;
      compute_grid_block_size(layer_size, bs, nb);

      _poseidon254_fold<<<nb, bs, 0, gpu>>>(
          nodes + layer_size,
          nodes + 2 * layer_size,
          layer_size);
      CUDA_OK(cudaGetLastError());
    }
    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
sppark_poseidon254_rows(alt_bn128::fr_t* d_out, const fr_t* d_in, size_t count, uint32_t col_size) {
  const gpu_t& gpu = select_gpu();

  size_t block_size = 512;
  size_t num_blocks = gpu.sm_count();

  compute_grid_block_size(count, block_size, num_blocks);

  try {
    (void)cudaGetLastError(); // consume any stale async errors
    _poseidon254_rows<<<num_blocks, block_size, 0, gpu>>>(d_out, d_in, count, col_size);

    CUDA_OK(cudaGetLastError());

    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value sppark_prefix_product(fr4_t d_inout[/*count*/], uint32_t count) {
  const gpu_t& gpu = select_gpu();

  try {
    prefix_op<Multiply<fr4_t>>(d_inout, count, gpu);
    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
supra_poly_divide(fr4_t d_inout[/*len*/], size_t len, fr4_t* remainder, const fr4_t& pow) {
  const gpu_t& gpu = select_gpu();

  try {
    div_by_x_minus_z<true>(d_inout, len, pow, gpu, get_real_sm_count());
    gpu.DtoH(remainder, &d_inout[len - 1], 1);
    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
supra_poly_divide_batch(fr4_t d_inout[/*len*/], size_t len,
                        fr4_t* remainders, const fr4_t pows[],
                        uint32_t num_divides) {
  const gpu_t& gpu = select_gpu();
  int sm_count = get_real_sm_count();

  try {
    for (uint32_t d = 0; d < num_divides; d++) {
      div_by_x_minus_z<true>(d_inout, len, pows[d], gpu, sm_count);
      gpu.DtoH(&remainders[d], &d_inout[len - 1], 1);
    }
    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }

  return RustError{cudaSuccess};
}
