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

// scatter_preflight: 1:1 port of kernels/cuda/ffi.cu `scatter_preflight` (one thread per
// ScatterInfo), plus the plan 2.9 fault flag: an info that would read or write out of bounds
// (or has a `bits` the shift cannot express) is skipped and sets `*fault`, instead of corrupting
// memory as the CUDA kernel (no checks) or panicking as the CPU path (`assert!`) would.
//
// Bindings (mirrored by `SCATTER_*` in keccak/src/prove/hal/metal/mod.rs):
//   buffer(0) into   device Fp*                  witness `data`, column-major: into[col * rows + row]
//   buffer(1) infos  const device ScatterInfo*   `count` infos
//   buffer(2) from   const device uint*          the preflight's packed words, `from_len` of them
//   buffer(3) fault  device atomic_uint*         ORed with 1 by any thread that rejected its info
//   buffer(4) p      constant ScatterParams&     set_bytes

#include <metal_stdlib>

#include "fp.h"

using namespace metal;

// Must match risc0_circuit_keccak_sys::ScatterInfo (repr(C), 16 bytes, asserted there too).
struct ScatterInfo {
  uint32_t offset;
  uint32_t row;
  uint16_t col;
  uint16_t count;
  uint32_t bits;
};
static_assert(sizeof(ScatterInfo) == 16, "ScatterInfo must be 16 bytes (see keccak-sys/src/lib.rs)");

// Must match `ScatterParams` in keccak/src/prove/hal/metal/mod.rs.
struct ScatterParams {
  uint32_t rows;     // rows of `into` (the cycle count)
  uint32_t cols;     // columns of `into`
  uint32_t count;    // number of infos
  uint32_t from_len; // number of words in `from`
};
static_assert(sizeof(ScatterParams) == 16, "ScatterParams must be 16 bytes");

[[max_total_threads_per_threadgroup(256)]]
kernel void scatter_preflight(device Fp* into [[buffer(0)]],
                              const device ScatterInfo* infos [[buffer(1)]],
                              const device uint32_t* from [[buffer(2)]],
                              device atomic_uint* fault [[buffer(3)]],
                              constant ScatterParams& p [[buffer(4)]],
                              uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count) {
    return;
  }

  const ScatterInfo info = infos[gid];
  // CUDA computes `32 / bits` and `1 << bits` unchecked: bits == 0 divides by zero and bits >= 32
  // shifts out of range. Every info the preflight writes has 0 < bits < 32.
  if (info.bits == 0 || info.bits >= 32 || info.row >= p.rows ||
      uint32_t(info.col) + uint32_t(info.count) > p.cols) {
    atomic_fetch_or_explicit(fault, 1u, memory_order_relaxed);
    return;
  }
  uint32_t innerCount = 32 / info.bits;
  uint32_t mask = (1u << info.bits) - 1;
  // The last word read is offset + (count - 1) / innerCount.
  if (info.count != 0 && (info.offset >= p.from_len ||
                          (uint32_t(info.count) - 1) / innerCount >= p.from_len - info.offset)) {
    atomic_fetch_or_explicit(fault, 1u, memory_order_relaxed);
    return;
  }
  for (uint32_t i = 0; i < info.count; i++) {
    uint32_t word = from[info.offset + (i / innerCount)];
    uint32_t j = i % innerCount;
    uint32_t val = (word >> (j * info.bits)) & mask;
    uint32_t col = info.col + i;
    // col < cols and row < rows, so this is below rows * cols (asserted < 2^32 on the host).
    into[col * p.rows + info.row] = Fp(val);
  }
}
