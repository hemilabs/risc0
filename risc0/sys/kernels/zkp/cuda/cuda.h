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

#pragma once

// CUDA 13.1's CCCL v2 headers (thrust, cub) need driver API types (CUcontext,
// CUdevice, etc.) from the system <cuda.h>. Forward to it first so those types
// are defined before any CCCL header is processed.
#include_next <cuda.h>

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>

template <typename... Types> inline std::string fmt(const char* fmt, Types... args) {
  size_t len = std::snprintf(nullptr, 0, fmt, args...);
  std::string ret(++len, '\0');
  std::snprintf(&ret.front(), len, fmt, args...);
  ret.resize(--len);
  return ret;
}

#define CUDA_OK(expr)                                                                              \
  do {                                                                                             \
    cudaError_t code = expr;                                                                       \
    if (code != cudaSuccess) {                                                                     \
      auto file = std::strstr(__FILE__, "sppark");                                                 \
      auto msg = fmt("%s@%s:%d failed: \"%s\"",                                                    \
                     #expr,                                                                        \
                     file ? file : __FILE__,                                                       \
                     __LINE__,                                                                     \
                     cudaGetErrorString(code));                                                    \
      throw std::runtime_error{msg};                                                               \
    }                                                                                              \
  } while (0)

class CudaStream {
private:
  cudaStream_t stream;

public:
  CudaStream() { cudaStreamCreate(&stream); }
  ~CudaStream() { cudaStreamDestroy(stream); }

  inline operator cudaStream_t() const { return stream; }
};

struct LaunchConfig {
  dim3 grid;
  dim3 block;
  size_t shared;

  LaunchConfig(dim3 grid, dim3 block, size_t shared = 0)
      : grid(grid), block(block), shared(shared) {}
  LaunchConfig(int grid, int block, size_t shared = 0) : grid(grid), block(block), shared(shared) {}
};

inline cudaStream_t getPersistentStream() {
  static cudaStream_t stream = nullptr;
  if (!stream) {
    CUDA_OK(cudaStreamCreate(&stream));
  }
  return stream;
}

inline LaunchConfig getCachedSimpleConfig(uint32_t count) {
  static int block = 0;
  if (block == 0) {
    int device;
    CUDA_OK(cudaGetDevice(&device));
    int maxThreads;
    CUDA_OK(cudaDeviceGetAttribute(&maxThreads, cudaDevAttrMaxThreadsPerBlock, device));
    block = maxThreads / 4;
  }
  int grid = (count + block - 1) / block;
  return LaunchConfig{grid, block, 0};
}

// Backward-compat alias used by circuit ffi files
inline LaunchConfig getSimpleConfig(uint32_t count) {
  return getCachedSimpleConfig(count);
}

template <typename... ExpTypes, typename... ActTypes>
const char* launchKernel(void (*kernel)(ExpTypes...),
                         uint32_t count,
                         uint32_t shared_size,
                         ActTypes&&... args) {
  try {
    cudaStream_t stream = getPersistentStream();
    LaunchConfig cfg = getCachedSimpleConfig(count);
    cudaLaunchConfig_t config;
    config.attrs = nullptr;
    config.numAttrs = 0;
    config.gridDim = cfg.grid;
    config.blockDim = cfg.block;
    config.dynamicSmemBytes = shared_size;
    config.stream = stream;
    CUDA_OK(cudaLaunchKernelEx(&config, kernel, std::forward<ActTypes>(args)...));
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}
