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
/// rv32im-specific CUDA compatibility on top of build_kernel's `cuda2metal.h` (Metal port plan
/// 2.3, item R1). Everything generic (qualifiers, barriers, shuffles, atomics, `assert`,
/// `printf`, `cuda::std::array`) lives in `cuda2metal.h`; this header adds only what the rv32im
/// sources (witgen_tu.cu, accum_tu.cu, steps.cu, eval_check*.cu and the zirgen .inc files) use
/// beyond that, plus the rv32im fault bits and layout-assert helpers shared by the other headers
/// in this directory.

#include "cuda2metal.h"

#include <metal_stdlib>

// ---------------------------------------------------------------------------------------------
// std::initializer_list
// ---------------------------------------------------------------------------------------------
// The generated step functions pass `std::initializer_list<Val>{...}` (empty or with values) to
// `extern_log`, which is a no-op on the GPU (witgen.h). MSL has no <initializer_list>: this
// stand-in accepts and drops any brace list, so those calls compile and cost nothing.
namespace std {
template <typename T> struct initializer_list {
  constexpr initializer_list() {}
  template <typename... A> constexpr initializer_list(A...) {}
};
} // namespace std

// ---------------------------------------------------------------------------------------------
// Index arithmetic
// ---------------------------------------------------------------------------------------------
// Column-major trace buffers are indexed `col * rows + row`. CUDA computes that in size_t (64
// bits); Apple GPUs emulate 64-bit integer multiplies (D0: ~3.8x a 32-bit one), so the Metal
// headers index in 32 bits. That is exact while rows * cols < 2^32 (data at po2 22 is
// 211 * 2^22 < 2^30); the host asserts it before every dispatch. Define RV32IM_WIDE_INDEX to
// index in 64 bits instead.
#ifdef RV32IM_WIDE_INDEX
typedef ulong rv32im_index_t;
#else
typedef uint rv32im_index_t;
#endif

/// Index of (row, col) in a column-major buffer of `rows` rows.
inline rv32im_index_t rv32im_cell(uint row, uint col, uint rows) {
  return rv32im_index_t(col) * rv32im_index_t(rows) + rv32im_index_t(row);
}

// ---------------------------------------------------------------------------------------------
// Fault bits (plan 2.9): a kernel accumulates them in a thread-local `uint` and ORs it once into
// its `device atomic_uint* fault` buffer at the end (`r0_fault_flush`).
// ---------------------------------------------------------------------------------------------
#define RV32IM_FAULT_LOOKUP_TABLE 0x1u  // lookup into a table other than 0, 8, 16
#define RV32IM_FAULT_LOOKUP_INDEX 0x2u  // u8/u16 lookup index out of range (clamped)
#define RV32IM_FAULT_TXN_INDEX 0x4u     // memory transaction index out of range (clamped)
#define RV32IM_FAULT_BIGINT_INDEX 0x8u  // bigint byte index out of range (clamped)

// ---------------------------------------------------------------------------------------------
// Layout asserts for structs shared with Rust (`offsetof` is not a macro in MSL).
// ---------------------------------------------------------------------------------------------
#define RV32IM_ASSERT_SIZE(T, N) static_assert(sizeof(T) == (N), "sizeof(" #T ") != " #N)
#define RV32IM_ASSERT_OFFSET(T, F, N)                                                           \
  static_assert(__builtin_offsetof(T, F) == (N), "offsetof(" #T ", " #F ") != " #N)
