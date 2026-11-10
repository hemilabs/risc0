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

// Fused multi-root polynomial division for combos_divide: a transliteration of
// sys/kernels/zkp/cuda/supra/mdiv.cuh (the default, non-MD_COAL path; MD_NTH = 256).
//
// One root z on a slice a[0..n): S_k = sum_{j>=k} a[j] z^(j-k) (suffix Horner); the output is
// a'[k] = S_{k+1} (k < n-1), a'[n-1] = S_0 (the remainder), i.e. exactly sppark
// div_by_x_minus_z<rotate=true>. Tiles of T = 256 * E elements. Per root, mdiv_scan (one
// threadgroup per slice) turns the tile aggregates into tile carries, and mdiv_apply produces
// each tile's outputs and, if another root follows, the next root's aggregates. The remainder of
// root r (produced by tile 0) is installed at a[n-1] by the next launch (scan or final).
//
// Differences from CUDA, all forced by Metal or needed for shapes CUDA does not support:
// * No pointers in the job struct: CUDA passes `MdJobs` (with device pointers) as a
//   __grid_constant__. Here every job holds element offsets into two explicitly bound buffers
//   (the combos buffer and one scratch buffer), so the buffers are bound with set_buffer and
//   hazard-tracked like any other argument; no use_resource is needed (plan 2.0 "explicit
//   bindings"; D0 showed pointer structs without use_resource silently lose writes).
// * E (elements per thread) is a template parameter: 8 as on CUDA, 16 instantiated for
//   measurement at po2 22 (see mdiv.rs).
// * mdiv_scan takes K tiles per thread (K = ntiles / NT, 1 on every CUDA shape) so that 2048 or
//   4096 tiles (po2 22/23 at E = 8) fit one 1024-thread threadgroup, and pads with zero tiles when
//   ntiles < 32 (po2 < 16 at E = 8), so one 32-thread threadgroup handles small slices.
//
// Buffer indices (mirrored in zkp/src/hal/metal/mdiv.rs):
//   0: device FpExt* a        combos buffer (job offsets are element indices into it)
//   1: device FpExt* s        scratch: per-job aggregates and carries, then all remainders
//   2: constant MdJobs& J     set_bytes
//   (mdiv_final) 3: constant uint& njobs
//
// The warp-level steps assume 32-wide SIMD groups made of consecutive thread indices (Apple GPUs,
// 1-D threadgroups whose size is a multiple of 32). Lanes read across the SIMD-group boundary by
// simd_shuffle_down/up return the reading lane's own value (D0 probe); the csel()s discard them,
// as on CUDA.

#include <metal_stdlib>

#include "fp.h"
#include "fpext.h"

using namespace metal;

#define MD_NTH 256
#define MD_MAXJ 8
#define MD_MAXK 4
#define MD_NONE 0xffffffffu

struct MdJob {
  uint a;        // slice offset in buffer 0 (n elements)
  uint agg;      // per-tile aggregates, offset in buffer 1 (ntiles)
  uint carry;    // per-tile carries, offset in buffer 1 (ntiles)
  uint rem;      // remainder of this root, offset in buffer 1
  uint prev_rem; // remainder of the previous root of this slice (buffer 1), or MD_NONE
  uint has_next;
  uint _pad0;
  uint _pad1;
  uint z[4];     // raw (Montgomery) words of the root
  uint znext[4]; // and of the next root (= z if none)
};

struct MdJobs {
  MdJob j[MD_MAXJ];
  uint n;      // slice length
  uint ntiles; // n / T
  uint k;      // tiles per mdiv_scan thread
  uint t;      // tile size T = MD_NTH * E
};

static_assert(sizeof(MdJob) == 64, "MdJob layout is mirrored in mdiv.rs");
static_assert(sizeof(MdJobs) == 528, "MdJobs layout is mirrored in mdiv.rs");

inline FpExt md_ext(constant uint* w) {
  return FpExt(Fp::fromRaw(w[0]), Fp::fromRaw(w[1]), Fp::fromRaw(w[2]), Fp::fromRaw(w[3]));
}

inline FpExt md_shfl_down(FpExt v, ushort d) {
  return FpExt(Fp::fromRaw(simd_shuffle_down(v.elems[0].asRaw(), d)),
               Fp::fromRaw(simd_shuffle_down(v.elems[1].asRaw(), d)),
               Fp::fromRaw(simd_shuffle_down(v.elems[2].asRaw(), d)),
               Fp::fromRaw(simd_shuffle_down(v.elems[3].asRaw(), d)));
}

inline FpExt md_shfl_up(FpExt v, ushort d) {
  return FpExt(Fp::fromRaw(simd_shuffle_up(v.elems[0].asRaw(), d)),
               Fp::fromRaw(simd_shuffle_up(v.elems[1].asRaw(), d)),
               Fp::fromRaw(simd_shuffle_up(v.elems[2].asRaw(), d)),
               Fp::fromRaw(simd_shuffle_up(v.elems[3].asRaw(), d)));
}

inline FpExt md_csel(FpExt a, FpExt b, bool sel_a) {
  return sel_a ? a : b;
}

// Inclusive suffix scan over the threadgroup: returns V_t = sum_{u>=t} v_u * w^(u-t), w given.
// Also returns (via *next) V_{t+1} (zero for the last thread). NT threads, NT % 32 == 0,
// NT <= 1024. Threadgroup scratch: sh[NT/32 + 33]. (mdiv.cuh md_block_suffix_scan.)
template <uint NT>
inline FpExt md_block_suffix_scan(FpExt v, FpExt w, threadgroup FpExt* sh, thread FpExt* next,
                                  uint t) {
  const uint lane = t & 31, warp = t >> 5;
  constexpr uint NW = NT / 32;
  // SIMD-level Kogge-Stone; wp = w^d
  FpExt wp = w;
  for (ushort d = 1; d < 32; d <<= 1) {
    FpExt o = md_shfl_down(v, d);
    FpExt c = v + o * wp;
    v = md_csel(c, v, lane + d < 32);
    wp = wp * wp;
  }
  // wp = w^32. lane powers: pw[l] = w^(l+1)
  threadgroup FpExt* pw = sh + NW + 1;
  {
    FpExt p = w;
    for (ushort d = 1; d < 32; d <<= 1) {
      FpExt o = md_shfl_up(p, d);
      FpExt c = p * o;
      p = md_csel(c, p, lane >= d);
    }
    if (warp == 0) {
      pw[lane] = p;
    }
  }
  if (lane == 0) {
    sh[warp] = v; // warp total
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    FpExt x;
    sh[NW] = x;
    for (int k = NW - 1; k >= 0; k--) {
      x = sh[k] + wp * x; // suffix over warps
      sh[k] = x;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  FpExt xa = sh[warp + 1]; // carry from warps above (V at 32(warp+1))
  FpExt pl = pw[31 - lane];
  FpExt r = v + pl * xa;
  FpExt nx = md_shfl_down(r, 1);
  if (lane == 31) {
    nx = xa;
  }
  *next = nx;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return r;
}

// Root-0 tile aggregates: agg[i] = sum_{k in tile i} a[k] z^(k - i*T). Grid (ntiles, njobs) x 256.
template <uint E>
[[max_total_threads_per_threadgroup(MD_NTH)]] kernel void
mdiv_agg0(device FpExt* a [[buffer(0)]],
          device FpExt* s [[buffer(1)]],
          constant MdJobs& J [[buffer(2)]],
          uint3 threadIdx [[thread_position_in_threadgroup]],
          uint3 blockIdx [[threadgroup_position_in_grid]]) {
  threadgroup FpExt sh[MD_NTH / 32 + 33];
  constant MdJob& jb = J.j[blockIdx.y];
  const uint T = MD_NTH * E;
  const uint b = jb.a + blockIdx.x * T + threadIdx.x * E;
  FpExt z = md_ext(jb.z);
  FpExt p = a[b + E - 1];
  for (int j = E - 2; j >= 0; j--) {
    p = a[b + j] + z * p;
  }
  FpExt zE = pow(z, E);
  FpExt nx;
  FpExt v = md_block_suffix_scan<MD_NTH>(p, zE, sh, &nx, threadIdx.x);
  if (threadIdx.x == 0) {
    s[jb.agg + blockIdx.x] = v;
  }
}

// Per slice (threadgroup x), NT threads, K = J.k tiles per thread: carries
// C_i = sum_{u>i} agg_u z^(T(u-i-1)). Also installs the previous root's remainder at a[n-1] and
// in the top tile's aggregate. Tiles >= ntiles (only when ntiles < NT) are zero.
template <uint NT>
[[max_total_threads_per_threadgroup(NT)]] kernel void
mdiv_scan(device FpExt* a [[buffer(0)]],
          device FpExt* s [[buffer(1)]],
          constant MdJobs& J [[buffer(2)]],
          uint3 threadIdx [[thread_position_in_threadgroup]],
          uint3 blockIdx [[threadgroup_position_in_grid]]) {
  threadgroup FpExt sh[NT / 32 + 33];
  constant MdJob& jb = J.j[blockIdx.x];
  const uint t = threadIdx.x;
  const uint K = J.k, ntiles = J.ntiles, T = J.t;
  FpExt z = md_ext(jb.z);
  FpExt zT = pow(z, T);
  FpExt v[MD_MAXK];
  for (uint k = 0; k < K; k++) {
    const uint u = t * K + k;
    v[k] = u < ntiles ? FpExt(s[jb.agg + u]) : FpExt();
    if (jb.prev_rem != MD_NONE && u == ntiles - 1) {
      FpExt r = s[jb.prev_rem];
      a[jb.a + J.n - 1] = r;
      v[k] = v[k] + r * pow(z, T - 1);
    }
  }
  // Thread-local Horner over this thread's K tiles (K == 1: P = v[0]).
  FpExt p = v[K - 1];
  for (int k = K - 2; k >= 0; k--) {
    p = v[k] + zT * p;
  }
  FpExt nx;
  md_block_suffix_scan<NT>(p, pow(zT, K), sh, &nx, t);
  // nx is the carry of this thread's top tile; walk down its tiles.
  FpExt c = nx;
  for (int k = K - 1; k >= 0; k--) {
    const uint u = t * K + k;
    if (u < ntiles) {
      s[jb.carry + u] = c;
    }
    c = v[k] + zT * c;
  }
}

// Apply one root on each tile; optionally compute the next root's tile aggregates.
// Grid (ntiles, njobs) x 256.
template <uint E>
[[max_total_threads_per_threadgroup(MD_NTH)]] kernel void
mdiv_apply(device FpExt* a [[buffer(0)]],
           device FpExt* s [[buffer(1)]],
           constant MdJobs& J [[buffer(2)]],
           uint3 threadIdx [[thread_position_in_threadgroup]],
           uint3 blockIdx [[threadgroup_position_in_grid]],
           uint3 gridDim [[threadgroups_per_grid]]) {
  threadgroup FpExt sh[MD_NTH / 32 + 33];
  constant MdJob& jb = J.j[blockIdx.y];
  const uint T = MD_NTH * E;
  const uint tile = blockIdx.x, ntiles = gridDim.x;
  const uint b = jb.a + tile * T + threadIdx.x * E;
  FpExt z = md_ext(jb.z);
  FpExt x[E];
  for (uint j = 0; j < E; j++) {
    x[j] = a[b + j];
  }
  // thread-local Horner from the top
  FpExt p = x[E - 1];
  for (int j = E - 2; j >= 0; j--) {
    p = x[j] + z * p;
  }
  FpExt zE = pow(z, E);
  FpExt C = s[jb.carry + tile];
  if (threadIdx.x == MD_NTH - 1) {
    p = p + zE * C;
  }
  FpExt nx;
  md_block_suffix_scan<MD_NTH>(p, zE, sh, &nx, threadIdx.x);
  // S at this thread's top boundary
  FpExt sv = (threadIdx.x == MD_NTH - 1) ? C : nx;
  // outputs: out[b+j] = S_{b+j+1}
  FpExt o[E];
  for (int j = E - 1; j >= 0; j--) {
    o[j] = sv;
    sv = x[j] + z * sv;
  }
  // sv = S_b
  const bool top = (tile == ntiles - 1) && (threadIdx.x == MD_NTH - 1);
  for (uint j = 0; j < E; j++) {
    if (!(top && j == E - 1)) {
      a[b + j] = o[j];
    }
  }
  if (tile == 0 && threadIdx.x == 0) {
    s[jb.rem] = sv;
  }
  if (jb.has_next) {
    FpExt zn = md_ext(jb.znext);
    if (top) {
      o[E - 1] = FpExt();
    }
    FpExt q = o[E - 1];
    for (int j = E - 2; j >= 0; j--) {
      q = o[j] + zn * q;
    }
    FpExt znE = pow(zn, E);
    FpExt v = md_block_suffix_scan<MD_NTH>(q, znE, sh, &nx, threadIdx.x);
    if (threadIdx.x == 0) {
      s[jb.agg + tile] = v;
    }
  }
}

// Install the last remainder of each slice at a[n-1]. One threadgroup of 32 threads.
kernel void mdiv_final(device FpExt* a [[buffer(0)]],
                       device FpExt* s [[buffer(1)]],
                       constant MdJobs& J [[buffer(2)]],
                       constant uint& njobs [[buffer(3)]],
                       uint t [[thread_position_in_grid]]) {
  if (t < njobs && t < MD_MAXJ) {
    a[J.j[t].a + J.n - 1] = FpExt(s[J.j[t].prev_rem]);
  }
}

template [[host_name("mdiv_agg0_e8")]] kernel void
mdiv_agg0<8>(device FpExt*, device FpExt*, constant MdJobs&, uint3, uint3);
template [[host_name("mdiv_agg0_e16")]] kernel void
mdiv_agg0<16>(device FpExt*, device FpExt*, constant MdJobs&, uint3, uint3);
template [[host_name("mdiv_apply_e8")]] kernel void
mdiv_apply<8>(device FpExt*, device FpExt*, constant MdJobs&, uint3, uint3, uint3);
template [[host_name("mdiv_apply_e16")]] kernel void
mdiv_apply<16>(device FpExt*, device FpExt*, constant MdJobs&, uint3, uint3, uint3);
template [[host_name("mdiv_scan_32")]] kernel void
mdiv_scan<32>(device FpExt*, device FpExt*, constant MdJobs&, uint3, uint3);
template [[host_name("mdiv_scan_64")]] kernel void
mdiv_scan<64>(device FpExt*, device FpExt*, constant MdJobs&, uint3, uint3);
template [[host_name("mdiv_scan_128")]] kernel void
mdiv_scan<128>(device FpExt*, device FpExt*, constant MdJobs&, uint3, uint3);
template [[host_name("mdiv_scan_256")]] kernel void
mdiv_scan<256>(device FpExt*, device FpExt*, constant MdJobs&, uint3, uint3);
template [[host_name("mdiv_scan_512")]] kernel void
mdiv_scan<512>(device FpExt*, device FpExt*, constant MdJobs&, uint3, uint3);
template [[host_name("mdiv_scan_1024")]] kernel void
mdiv_scan<1024>(device FpExt*, device FpExt*, constant MdJobs&, uint3, uint3);
