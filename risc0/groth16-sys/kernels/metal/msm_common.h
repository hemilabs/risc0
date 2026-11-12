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
  uint32_t booth;      // 1: signed (Booth) digits, msm_digit_booth (item G9)
  uint32_t _pad[3];
};

// Sorted entries of a Booth sort carry the digit's sign in the top bit of the scalar index.
constant uint32_t MSM_NEG = 0x80000000u;

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

// `nb` (<= 32) bits of a little-endian 256-bit scalar from bit `bit`; bits above 255 read 0.
inline uint32_t msm_bits(const device uint32_t* s, uint32_t bit, uint32_t nb) {
  uint32_t lo = bit >> 5;
  uint32_t sh = bit & 31;
  if (lo >= 8)
    return 0;
  uint32_t v = s[lo] >> sh;
  if (sh + nb > 32 && lo + 1 < 8)
    v |= s[lo + 1] << (32 - sh);
  return nb >= 32 ? v : v & ((1u << nb) - 1);
}

// The signed (Booth) c-bit digit of window w (item G9), c <= 17:
//   d_w = sum_{i<c} b_{wc+i} 2^i + b_{wc-1} - b_{wc+c-1} 2^c, with b_{-1} = 0,
// in [-2^(c-1), 2^(c-1)]. sum_{w<W} d_w 2^(wc) = s whenever s < 2^(Wc - 1) (the host picks W).
// Returns |d| and sets `neg` for d < 0.
inline uint32_t msm_digit_booth(const device uint32_t* s, uint32_t w, uint32_t c,
                                thread bool& neg) {
  // The c + 1 bits from bit wc - 1 (for w = 0, bits 0..c-1 shifted up, bit -1 = 0).
  uint32_t v = w == 0 ? (msm_bits(s, 0, c) << 1) : msm_bits(s, w * c - 1, c + 1);
  uint32_t mag = (v >> 1) + (v & 1);
  bool top = ((v >> c) & 1) != 0;
  neg = top && mag != (1u << c);
  // d = mag - top 2^c: for top, |d| = 2^c - mag (0 when mag = 2^c, i.e. all ones: d = 0).
  return top ? (1u << c) - mag : mag;
}
