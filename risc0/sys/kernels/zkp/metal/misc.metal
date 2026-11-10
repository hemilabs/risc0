// Copyright 2025 RISC Zero, Inc.
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

// Small kernels of the zkp HAL (item Z1), 1:1 ports of
//   eltwise_copy_fp_region  sys/kernels/zkp/cuda/eltwise.cu:41
//   scatter                 sys/kernels/zkp/cuda/kernels.cu:116
//   scatter_bits            sys/kernels/zkp/cuda/kernels.cu:131
//   combos_sub              sys/kernels/zkp/cuda/combos.cu:54
// plus fill_u32 (plan 2.5: cudaMemsetAsync of a non byte-repeating word).
// All are 1-D, launched over `count` threads like CUDA's launchKernel(count).
// Untrusted indices (offsets, rows/cols) are validated on the host before the launch
// (zkp/src/hal/metal/small.rs), which panics exactly where the CPU HAL would index out of
// bounds, so these kernels need no fault flag.
//
// Buffer indices (mirrored in small.rs):
//   eltwise_copy_fp_region 0 into, 1 from, 2 CopyRegionParams (set_bytes)
//   scatter                0 into, 1 index, 2 offsets, 3 values, 4 count
//   scatter_bits           0 into, 1 data, 2 cycles, 3 count
//   combos_sub             0 combos, 1 deltas, 2 offsets, 3 count
//   fill_u32               0 out, 1 value, 2 count

#include "fp.h"
#include "fpext.h"

using namespace metal;

struct CopyRegionParams {
  uint32_t from_rows;
  uint32_t from_cols;
  uint32_t from_offset;
  uint32_t from_stride;
  uint32_t into_offset;
  uint32_t into_stride;
  uint32_t _pad0;
  uint32_t _pad1;
};
static_assert(sizeof(CopyRegionParams) == 32, "CopyRegionParams layout");

// One thread per row; each copies `from_cols` consecutive elements.
kernel void eltwise_copy_fp_region(device Fp* into,
                                   const device Fp* from,
                                   constant CopyRegionParams& p,
                                   uint row [[thread_position_in_grid]]) {
  if (row < p.from_rows) {
    for (uint32_t col = 0; col < p.from_cols; col++) {
      into[p.into_offset + row * p.into_stride + col] = from[p.from_offset + row * p.from_stride + col];
    }
  }
}

// One thread per cycle: into[offsets[i]] = values[i] for i in index[gid]..index[gid + 1].
kernel void scatter(device Fp* into,
                    const device uint32_t* index,
                    const device uint32_t* offsets,
                    const device Fp* values,
                    constant uint32_t& count,
                    uint gid [[thread_position_in_grid]]) {
  if (gid < count) {
    uint32_t end = index[gid + 1];
    for (uint32_t idx = index[gid]; idx < end; idx++) {
      into[offsets[idx]] = values[idx];
    }
  }
}

// One thread per packed (row, base_col, value) triple; writes the 32 bits of value to
// into[(base_col + i) * cycles + row].
kernel void scatter_bits(device Fp* into,
                         const device uint32_t* data,
                         constant uint32_t& cycles,
                         constant uint32_t& count,
                         uint gid [[thread_position_in_grid]]) {
  if (gid < count) {
    uint32_t row = data[gid * 3];
    uint32_t base = data[gid * 3 + 1];
    uint32_t val = data[gid * 3 + 2];
    uint32_t cyc = cycles;
    for (uint32_t i = 0; i < 32; i++) {
      into[(base + i) * cyc + row] = Fp((val >> i) & 1);
    }
  }
}

// combos[offsets[k]] -= deltas[k] (offsets are unique; deltas precomputed on the host).
kernel void combos_sub(device FpExt* combos,
                       const device FpExt* deltas,
                       const device uint32_t* offsets,
                       constant uint32_t& count,
                       uint gid [[thread_position_in_grid]]) {
  if (gid < count) {
    FpExt d = deltas[gid];
    uint32_t off = offsets[gid];
    FpExt c = combos[off];
    combos[off] = c - d;
  }
}

kernel void fill_u32(device uint32_t* out,
                     constant uint32_t& value,
                     constant uint32_t& count,
                     uint gid [[thread_position_in_grid]]) {
  if (gid < count) {
    out[gid] = value;
  }
}
