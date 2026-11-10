// Copyright 2022 Risc0, Inc.
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

#include "fp.h"
#include "fpext.h"

using namespace metal;

// Elementwise kernels (ports of sys/kernels/zkp/cuda/eltwise.cu). 1-D, dispatched with
// MetalHal::launch (dispatch_threads, so the grid is exact; kernels that take a `count` still
// bounds-check as plan 2.4 asks).
//
// Buffer indices:
//   eltwise_add_fp        0 out, 1 in1, 2 in2
//   eltwise_mul_factor_fp 0 io, 1 factor (set_bytes Fp; no Hal caller since Z7, kept as the
//                         KernelArg::Bytes plumbing kernel of hal::metal::tests)
//   eltwise_copy_fp       0 out, 1 in
//   eltwise_sum_fpext     0 out, 1 in, 2 to_add, 3 count          (CUDA argument order)
//   eltwise_zeroize_fp    0 elems, 1 count

kernel void eltwise_add_fp(device Fp* out,
                           const device Fp* in1,
                           const device Fp* in2,
                           uint gid [[thread_position_in_grid]]) {
  out[gid] = in1[gid] + in2[gid];
}

kernel void eltwise_mul_factor_fp(device Fp* io,
                                  constant Fp& factor,
                                  uint gid [[thread_position_in_grid]]) {
  Fp f = factor;
  io[gid] = io[gid] * f;
}

kernel void eltwise_copy_fp(device Fp* out,
                            const device Fp* in,
                            uint gid [[thread_position_in_grid]]) {
  out[gid] = in[gid];
}

// eltwise.cu:57 eltwise_sum_fpext(out, in, to_add, count).
kernel void eltwise_sum_fpext(device Fp* out,
                              const device FpExt* in,
                              constant uint32_t& to_add,
                              constant uint32_t& count,
                              uint gid [[thread_position_in_grid]]) {
  if (gid >= count) {
    return;
  }
  FpExt tot;
  for (uint32_t i = 0; i < to_add; i++) {
    tot += in[count * i + gid];
  }
  out[gid + 0 * count] = tot.elems[0];
  out[gid + 1 * count] = tot.elems[1];
  out[gid + 2 * count] = tot.elems[2];
  out[gid + 3 * count] = tot.elems[3];
}

// eltwise.cu:72: write only the INVALID cells (a read-mostly pass with the same result).
kernel void eltwise_zeroize_fp(device Fp* elems,
                               constant uint32_t& count,
                               uint gid [[thread_position_in_grid]]) {
  if (gid >= count) {
    return;
  }
  Fp val = elems[gid];
  if (val.asRaw() == Fp::INVALID) {
    elems[gid] = Fp(0);
  }
}
