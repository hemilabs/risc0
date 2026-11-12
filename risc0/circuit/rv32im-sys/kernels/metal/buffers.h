// Copyright 2026 RISC Zero, Inc.
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

/// \file
/// Column-major trace buffer view (port of kernels/cuda/buffers.h; Intel precedent
/// intel/buffers.h). CUDA's `Buffer` is also the FFI struct (`RawBuffer`); on Metal it is only a
/// thread-local view that a kernel builds from a `device Fp*` binding and the row/column counts
/// in its params, so the FFI-only `checked` flag is gone and rows/cols are 32-bit (see
/// `rv32im_index_t` in cuda_compat.h).
///
/// Access ordering: CUDA's HIP build puts compiler barriers around each access because witgen
/// threads read back rows that neighbouring threads write (SHA2 back 68). MSL has no compiler
/// barrier. The default plan (4.2) avoids the race with a separate `pre_data` buffer; defining
/// RV32IM_ORDERED_ROWS makes every access a volatile 32-bit access instead, which the compiler
/// may neither reorder against other volatile accesses nor merge.

#include "cuda_compat.h"
#include "rv32im_fp.h"

namespace rv32im_metal {

struct Buffer {
  device Fp* buf;
  uint32_t rows;
  uint32_t cols;

#ifdef RV32IM_ORDERED_ROWS
  void set(uint32_t row, uint32_t col, Fp val) const {
    ((volatile device uint32_t*)buf)[rv32im_cell(row, col, rows)] = val.asRaw();
  }
  Fp get(uint32_t row, uint32_t col) const {
    return Fp::fromRaw(((volatile device uint32_t*)buf)[rv32im_cell(row, col, rows)]);
  }
#else
  void set(uint32_t row, uint32_t col, Fp val) const { buf[rv32im_cell(row, col, rows)] = val; }
  Fp get(uint32_t row, uint32_t col) const { return buf[rv32im_cell(row, col, rows)]; }
#endif
};

/// A Buffer with no storage (CUDA: `Buffer{nullptr, 0, 0, false}`).
inline Buffer nullBuffer() {
  Buffer b;
  b.buf = nullptr;
  b.rows = 0;
  b.cols = 0;
  return b;
}

inline Buffer makeBuffer(device Fp* buf, uint32_t rows, uint32_t cols) {
  Buffer b;
  b.buf = buf;
  b.rows = rows;
  b.cols = cols;
  return b;
}

} // namespace rv32im_metal
