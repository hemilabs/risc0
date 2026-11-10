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

// Shared definitions of the MSM kernels (msm_sort.metal, msm_reduce.metal; item G6). The host
// mirror is `MsmParams` in src/metal/msm.rs (same field order, 64 bytes).
#pragma once

#include <metal_stdlib>

// One parameter block for every MSM dispatch (bound with set_bytes).
struct MsmParams {
  uint32_t n;          // scalars sorted (the witness length for A/B1/B2/C)
  uint32_t wbits;      // window width c (1..16)
  uint32_t nwins;      // ceil(256 / c)
  uint32_t nbuckets;   // 2^c (bucket 0 is never used)
  uint32_t piece;      // T: at most T points are summed by one bucket thread (a "piece")
  uint32_t max_pieces; // per-window stride of the piece arrays (bound on pieces per window)
  uint32_t seg;        // S: pieces per segment_reduce thread
  uint32_t max_segs;   // per-window stride of the partial array (ceil(max_pieces / S))
  uint32_t first;      // scalar i uses point i - first; scalars below `first` are skipped
  uint32_t npoints;    // points available (n - first <= npoints, checked on the host)
  uint32_t w0;         // first window of this window group (buckets / segment_reduce)
  uint32_t wcount;     // windows in this group
  uint32_t _pad[4];
};

// The c-bit digit of window w of a canonical little-endian 256-bit scalar (8 x u32). Bits above
// 255 read as zero; the windows may straddle a limb boundary.
inline uint32_t msm_digit(const device uint32_t* s, uint32_t w, uint32_t c) {
  uint32_t bit = w * c;
  uint32_t lo = bit >> 5;
  uint32_t sh = bit & 31;
  uint32_t v = s[lo] >> sh;
  if (sh + c > 32 && lo + 1 < 8)
    v |= s[lo + 1] << (32 - sh);
  return v & ((1u << c) - 1);
}
