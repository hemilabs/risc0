// Intel SYCL version of buffers.h — matches CUDA layout, no error checking.
#pragma once
#include "fp.h"

namespace risc0::circuit::rv32im_v2::intel {

struct Buffer {
  Fp* buf;
  size_t rows;
  size_t cols;
  bool checked;  // retained for FFI layout compatibility

  inline void set(size_t row, size_t col, Fp val) {
    buf[col * rows + row] = val;
  }

  inline Fp get(size_t row, size_t col) {
    return buf[col * rows + row];
  }
};

} // namespace risc0::circuit::rv32im_v2::intel
