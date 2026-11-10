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

// Metal port of the batched narrow CT/GS NTT passes of `supra/ntt_x.cuh` (themselves extended
// copies of sppark 0.1.14 `_CT_NTT`/`_GS_NTT`), sppark's `get_intermediate_roots`,
// `coalesced_load/store`, `transpose`, the tiled batched bit reversal `bit_rev_z_batched<64>`,
// a batched naive bit reversal (sppark `bit_rev_permutation`, swap form) for n < 4096, and
// `nttx_zk_shift_batched` (plan 4.1 row Z2).
//
// Arithmetic is the same field arithmetic in the same order as the CUDA kernels; twiddle tables
// are built on the CPU (zkp/src/hal/metal/ntt.rs) with sppark's NTTParameters layout.
//
// Buffer bindings (mirrored by the `CT_ARGS` / `GS_ARGS` / ... comments in ntt.rs):
//   ct_x_*  : 0 in, 1 out, 2 partial_twiddles[WINDOW_NUM][WINDOW_SIZE],
//             3 plus_one_twiddles[MAX_LG_DOMAIN_SIZE-15][1024], 4 radix6 twiddles,
//             5 radixX twiddles, 6 NttXParams; threadgroup(0) = Z * 2^(radix-1) words
//   gs_x_*  : as ct_x_*, plus 7 zk lo[1024], 8 zk hi[2^(lg-10)] (post-multiply, last pass)
//   bit_rev_z_batched_64        : 0 io, 1 BitRevParams; threadgroup(0) = 64*64 words
//   bit_rev_permutation_batched : 0 io, 1 BitRevParams
//   nttx_zk_shift_batched       : 0 io, 1 lo, 2 hi, 3 BitRevParams

#include <metal_stdlib>

#include "fp.h"

using namespace metal;

// sppark ntt/parameters.cuh with FEATURE_BABY_BEAR.
#define MAX_LG_DOMAIN_SIZE 27
#define LG_WINDOW_SIZE ((MAX_LG_DOMAIN_SIZE + 4) / 5)
#define WINDOW_SIZE (1 << LG_WINDOW_SIZE)
#define WINDOW_NUM ((MAX_LG_DOMAIN_SIZE + LG_WINDOW_SIZE - 1) / LG_WINDOW_SIZE)

static_assert(LG_WINDOW_SIZE == 6, "sppark baby bear window");
static_assert(WINDOW_NUM == 5, "sppark baby bear window count");

// Mirrored by `NttXParams` in ntt.rs (48 bytes).
struct NttXParams {
  uint radix;
  uint lg;
  uint stage;
  uint iterations;
  uint in_stride;   // column stride of `in` (elements)
  uint out_stride;  // column stride of `out`
  uint lg_expand;   // CT stage 0 only: synthesise the zero-padded LDE input on load
  uint skip_expand; // CT stage 0 only: skip the first lg_expand (trivial) levels
  uint is_intt;
  uint dsi;  // raw (Montgomery) domain_size_inverse[lg]
  uint post; // GS last pass only: multiply by lo[k & 1023] * hi[k >> 10]
  uint _pad;
};

// Mirrored by `BitRevParams` in ntt.rs (16 bytes).
struct BitRevParams {
  uint lg;
  uint cols;
  uint stride;
  uint _pad;
};

// sppark bit_rev. CUDA's `__brev(i) >> 32` is 0 for nbits == 0 (PTX clamps the shift); MSL's
// shift by 32 is undefined, so that case is explicit.
inline uint bit_rev(uint i, uint nbits) {
  return nbits == 0 ? 0u : (reverse_bits(i) >> (32 - nbits));
}

inline Fp shfl_bfly(Fp x, uint laneMask) {
  return Fp::fromRaw(simd_shuffle_xor(x.asRaw(), ushort(laneMask)));
}

inline Fp csel(Fp a, Fp b, bool sel_a) {
  return sel_a ? a : b;
}

// sppark ntt/kernels.cu get_intermediate_roots; roots = partial_twiddles[WINDOW_NUM][WINDOW_SIZE].
inline void get_intermediate_roots(
    thread Fp& root0, thread Fp& root1, uint idx0, uint idx1, const device Fp* roots) {
  int win = (WINDOW_NUM - 1) * LG_WINDOW_SIZE;
  int off = (WINDOW_NUM - 1);
  uint idxo = idx0 | idx1;
  uint mask = (1u << win) - 1;

  root0 = roots[off * WINDOW_SIZE + (idx0 >> win)];
  root1 = roots[off * WINDOW_SIZE + (idx1 >> win)];
  while (off-- && (idxo & mask)) {
    win -= LG_WINDOW_SIZE;
    mask >>= LG_WINDOW_SIZE;
    root0 = root0 * roots[off * WINDOW_SIZE + ((idx0 >> win) % WINDOW_SIZE)];
    root1 = root1 * roots[off * WINDOW_SIZE + ((idx1 >> win) % WINDOW_SIZE)];
  }
}

template <int Z>
inline void coalesced_load(thread Fp* r, const device Fp* inout, uint idx, uint stage, uint tx) {
  const uint x = tx & (Z - 1);
  idx &= ~(uint(Z - 1) << stage);
  idx += x;
  for (int z = 0; z < Z; z++, idx += 1u << stage) {
    r[z] = inout[idx];
  }
}

// Threadgroup scratch viewed as xchg[blockDim][Z] (raw words).
template <int Z> inline void transpose(thread Fp* r, threadgroup uint* xchg, uint tx) {
  const uint x = tx & (Z - 1);
  const uint y = tx & ~(Z - 1);
  for (int z = 0; z < Z; z++) {
    xchg[(y + z) * Z + x] = r[z].asRaw();
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (int z = 0; z < Z; z++) {
    r[z] = Fp::fromRaw(xchg[(y + x) * Z + z]);
  }
}

template <int Z>
inline void coalesced_store(device Fp* inout, uint idx, thread const Fp* r, uint stage, uint tx) {
  const uint x = tx & (Z - 1);
  idx &= ~(uint(Z - 1) << stage);
  idx += x;
  for (int z = 0; z < Z; z++, idx += 1u << stage) {
    inout[idx] = r[z];
  }
}

// Twiddles applied before a pass that does not start at stage 0 (shared by CT and GS).
template <int Z>
inline void apply_intermediate_twiddles(thread Fp* r0,
                                        thread Fp* r1,
                                        uint tiz,
                                        uint diff_mask,
                                        uint thread_ntt_pos,
                                        uint pstage,
                                        const device Fp* partial,
                                        const device Fp* plus_one) {
  uint thread_ntt_idx = (tiz & diff_mask) * 2;
  uint nbits = MAX_LG_DOMAIN_SIZE - pstage;
  uint ridx0 = bit_rev(thread_ntt_idx, nbits);
  uint root_idx0 = ridx0 * thread_ntt_pos;
  uint root_idx1 = root_idx0 + (thread_ntt_pos << (nbits - 1));

  Fp first_root, second_root;
  get_intermediate_roots(first_root, second_root, root_idx0, root_idx1, partial);
  r0[0] = r0[0] * first_root;
  r1[0] = r1[0] * second_root;

  if (Z > 1) {
    uint off = nbits >= 10 ? (nbits - 10) : 0;
    uint scale = nbits >= 10 ? 0 : (10 - nbits);

    thread_ntt_idx <<= scale;
    Fp first_root_z = plus_one[off * 1024 + thread_ntt_idx];
    Fp second_root_z = plus_one[off * 1024 + thread_ntt_idx + (1u << scale)];

    for (int z = 1; z < Z; z++) {
      first_root = first_root * first_root_z;
      second_root = second_root * second_root_z;
      r0[z] = r0[z] * first_root;
      r1[z] = r1[z] * second_root;
    }
  }
}

// supra ntt_x.cuh _CT_X (forward DIT pass, RN). blockIdx.y selects the column.
template <int Z, bool COALESCED>
[[max_total_threads_per_threadgroup(512)]] kernel void
ct_x(const device Fp* in0 [[buffer(0)]],
     device Fp* out0 [[buffer(1)]],
     const device Fp* partial [[buffer(2)]],
     const device Fp* plus_one [[buffer(3)]],
     const device Fp* radix6 [[buffer(4)]],
     const device Fp* radixX [[buffer(5)]],
     constant NttXParams& p [[buffer(6)]],
     threadgroup uint* sh [[threadgroup(0)]],
     uint3 threadIdx [[thread_position_in_threadgroup]],
     uint3 blockIdx [[threadgroup_position_in_grid]],
     uint3 blockDim [[threads_per_threadgroup]]) {
  const uint radix = p.radix;
  const uint lg_domain_size = p.lg;
  const uint stage = p.stage;
  const uint iterations = p.iterations;
  const uint tx = threadIdx.x;

  const device Fp* d_in = in0 + ulong(blockIdx.y) * p.in_stride;
  device Fp* d_out = out0 + ulong(blockIdx.y) * p.out_stride;

  uint tid = tx + blockDim.x * blockIdx.x;

  const uint diff_mask = (1u << (iterations - 1)) - 1;
  const uint inp_mask = (1u << stage) - 1;
  const uint out_mask = (1u << (stage + iterations - 1)) - 1;

  const uint tiz = (tid & ~diff_mask) * Z + (tid & diff_mask);
  const uint thread_ntt_pos = (tiz >> (iterations - 1)) & inp_mask;

  uint idx0 = (tiz & ~out_mask) | ((tiz << stage) & out_mask);
  idx0 = idx0 * 2 + thread_ntt_pos;
  uint idx1 = idx0 + (1u << stage);

  Fp r0[Z];
  Fp r1[Z];

  const bool skip = p.lg_expand != 0 && p.skip_expand != 0;
  if (COALESCED) {
    coalesced_load<Z>(r0, d_in, idx0, stage + 1, tx);
    coalesced_load<Z>(r1, d_in, idx1, stage + 1, tx);
    transpose<Z>(r0, sh, tx);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    transpose<Z>(r1, sh, tx);
  } else {
    uint z_shift = inp_mask == 0 ? iterations : 0;
    if (skip) {
      // the first lg_expand DIT levels on [c,0,..,0] produce [c,c,..,c]: replicate, skip them
      for (int z = 0; z < Z; z++) {
        Fp v = d_in[(idx0 + (uint(z) << z_shift)) >> p.lg_expand];
        r0[z] = v;
        r1[z] = v;
      }
    } else if (p.lg_expand != 0) {
      const uint emask = (1u << p.lg_expand) - 1;
      for (int z = 0; z < Z; z++) {
        uint i0 = idx0 + (uint(z) << z_shift);
        uint i1 = idx1 + (uint(z) << z_shift);
        r0[z] = Fp(0);
        r1[z] = Fp(0);
        if ((i0 & emask) == 0) {
          r0[z] = d_in[i0 >> p.lg_expand];
        }
        if ((i1 & emask) == 0) {
          r1[z] = d_in[i1 >> p.lg_expand];
        }
      }
    } else {
      for (int z = 0; z < Z; z++) {
        r0[z] = d_in[idx0 + (uint(z) << z_shift)];
        r1[z] = d_in[idx1 + (uint(z) << z_shift)];
      }
    }
  }

  if (stage != 0) {
    apply_intermediate_twiddles<Z>(
        r0, r1, tiz, diff_mask, thread_ntt_pos, stage, partial, plus_one);
  }

  const uint s0 = skip ? p.lg_expand : 1u;
  if (s0 == 1) {
    for (int z = 0; z < Z; z++) {
      Fp t = r1[z];
      r1[z] = r0[z] - t;
      r0[z] = r0[z] + t;
    }
  }

  for (uint s = s0; s < min(iterations, 6u); s++) {
    uint laneMask = 1u << (s - 1);
    uint thrdMask = (1u << s) - 1;
    uint rank = tx & thrdMask;
    bool pos = rank < laneMask;

    Fp root = radix6[rank << (6 - (s + 1))];

    for (int z = 0; z < Z; z++) {
      Fp t = csel(r1[z], r0[z], pos);
      t = shfl_bfly(t, laneMask);
      r0[z] = csel(r0[z], t, pos);
      r1[z] = csel(t, r1[z], pos);
      t = root * r1[z];
      r1[z] = r0[z] - t;
      r0[z] = r0[z] + t;
    }
  }

  for (uint s = 6; s < iterations; s++) {
    uint laneMask = 1u << (s - 1);
    uint thrdMask = (1u << s) - 1;
    uint rank = tx & thrdMask;
    bool pos = rank < laneMask;

    Fp root = radixX[rank << (radix - (s + 1))];

    for (int z = 0; z < Z; z++) {
      Fp t = csel(r1[z], r0[z], pos);
      sh[tx * Z + z] = t.asRaw();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int z = 0; z < Z; z++) {
      Fp t = Fp::fromRaw(sh[(tx ^ laneMask) * Z + z]);
      r0[z] = csel(r0[z], t, pos);
      r1[z] = csel(t, r1[z], pos);
      t = root * r1[z];
      r1[z] = r0[z] - t;
      r0[z] = t + r0[z];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (p.is_intt != 0 && (stage + iterations) == lg_domain_size) {
    const Fp dsi = Fp::fromRaw(p.dsi);
    for (int z = 0; z < Z; z++) {
      r0[z] = r0[z] * dsi;
      r1[z] = r1[z] * dsi;
    }
  }

  uint mask = ((1u << iterations) - 1) << stage;
  uint rotw = idx0 & mask;
  rotw = (rotw >> 1) | (rotw << (iterations - 1));
  idx0 = (idx0 & ~mask) | (rotw & mask);
  rotw = idx1 & mask;
  rotw = (rotw >> 1) | (rotw << (iterations - 1));
  idx1 = (idx1 & ~mask) | (rotw & mask);

  // V3 barrier fix: WAR on the exchange buffer before the transpose.
  if (COALESCED && iterations > 6) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (COALESCED) {
    transpose<Z>(r0, sh, tx);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    transpose<Z>(r1, sh, tx);
    coalesced_store<Z>(d_out, idx0, r0, stage, tx);
    coalesced_store<Z>(d_out, idx1, r1, stage, tx);
  } else {
    uint z_shift = inp_mask == 0 ? iterations : 0;
    for (int z = 0; z < Z; z++) {
      d_out[idx0 + (uint(z) << z_shift)] = r0[z];
      d_out[idx1 + (uint(z) << z_shift)] = r1[z];
    }
  }
}

// supra ntt_x.cuh _GS_X (inverse DIF pass, NR). blockIdx.y selects the column. The optional
// post-multiply on the last pass fuses the zk shift (LDE_distribute_powers, bit-reversed).
// The CUDA kernel's `nat` output (natural-order copy) is not ported: no host path uses it.
template <int Z, bool COALESCED>
[[max_total_threads_per_threadgroup(512)]] kernel void
gs_x(const device Fp* in0 [[buffer(0)]],
     device Fp* out0 [[buffer(1)]],
     const device Fp* partial [[buffer(2)]],
     const device Fp* plus_one [[buffer(3)]],
     const device Fp* radix6 [[buffer(4)]],
     const device Fp* radixX [[buffer(5)]],
     constant NttXParams& p [[buffer(6)]],
     const device Fp* mul_lo [[buffer(7)]],
     const device Fp* mul_hi [[buffer(8)]],
     threadgroup uint* sh [[threadgroup(0)]],
     uint3 threadIdx [[thread_position_in_threadgroup]],
     uint3 blockIdx [[threadgroup_position_in_grid]],
     uint3 blockDim [[threads_per_threadgroup]]) {
  const uint radix = p.radix;
  const uint lg_domain_size = p.lg;
  const uint stage = p.stage;
  const uint iterations = p.iterations;
  const uint tx = threadIdx.x;

  const device Fp* d_in = in0 + ulong(blockIdx.y) * p.in_stride;
  device Fp* d_out = out0 + ulong(blockIdx.y) * p.out_stride;

  uint tid = tx + blockDim.x * blockIdx.x;

  const uint diff_mask = (1u << (iterations - 1)) - 1;
  const uint inp_mask = (1u << (stage - 1)) - 1;
  const uint out_mask = (1u << (stage - iterations)) - 1;

  const uint tiz = (tid & ~diff_mask) * Z + (tid & diff_mask);

  uint idx0 = (tiz & ~inp_mask) * 2;
  idx0 += (tiz << (stage - iterations)) & inp_mask;
  idx0 += (tiz >> (iterations - 1)) & out_mask;
  uint idx1 = idx0 + (1u << (stage - 1));

  Fp r0[Z];
  Fp r1[Z];

  if (COALESCED) {
    coalesced_load<Z>(r0, d_in, idx0, stage - iterations, tx);
    coalesced_load<Z>(r1, d_in, idx1, stage - iterations, tx);
    transpose<Z>(r0, sh, tx);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    transpose<Z>(r1, sh, tx);
  } else {
    uint z_shift = out_mask == 0 ? iterations : 0;
    for (int z = 0; z < Z; z++) {
      r0[z] = d_in[idx0 + (uint(z) << z_shift)];
      r1[z] = d_in[idx1 + (uint(z) << z_shift)];
    }
  }

  for (uint s = iterations; --s >= 6;) {
    uint laneMask = 1u << (s - 1);
    uint thrdMask = (1u << s) - 1;
    uint rank = tx & thrdMask;
    bool pos = rank < laneMask;

    Fp root = radixX[rank << (radix - (s + 1))];

    for (int z = 0; z < Z; z++) {
      Fp t = r0[z] - r1[z];
      r0[z] = r0[z] + r1[z];
      r1[z] = t * root;
      t = csel(r1[z], r0[z], pos);
      sh[tx * Z + z] = t.asRaw();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int z = 0; z < Z; z++) {
      Fp t = Fp::fromRaw(sh[(tx ^ laneMask) * Z + z]);
      r0[z] = csel(r0[z], t, pos);
      r1[z] = csel(t, r1[z], pos);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint s = min(iterations, 6u); --s >= 1;) {
    uint laneMask = 1u << (s - 1);
    uint thrdMask = (1u << s) - 1;
    uint rank = tx & thrdMask;
    bool pos = rank < laneMask;

    Fp root = radix6[rank << (6 - (s + 1))];

    for (int z = 0; z < Z; z++) {
      Fp t = r0[z] - r1[z];
      r0[z] = r0[z] + r1[z];
      r1[z] = t * root;
      t = csel(r1[z], r0[z], pos);
      t = shfl_bfly(t, laneMask);
      r0[z] = csel(r0[z], t, pos);
      r1[z] = csel(t, r1[z], pos);
    }
  }

  for (int z = 0; z < Z; z++) {
    Fp t = r0[z] - r1[z];
    r0[z] = r0[z] + r1[z];
    r1[z] = t;
  }

  if (stage - iterations != 0) {
    uint thread_ntt_pos = (tiz & inp_mask) >> (iterations - 1);
    apply_intermediate_twiddles<Z>(
        r0, r1, tiz, diff_mask, thread_ntt_pos, stage - iterations, partial, plus_one);
  }

  if (p.is_intt != 0 && stage == iterations) {
    const Fp dsi = Fp::fromRaw(p.dsi);
    for (int z = 0; z < Z; z++) {
      r0[z] = r0[z] * dsi;
      r1[z] = r1[z] * dsi;
    }
  }

  uint mask = ((1u << iterations) - 1) << (stage - iterations);
  uint rotw = idx0 & mask;
  rotw = (rotw << 1) | (rotw >> (iterations - 1));
  idx0 = (idx0 & ~mask) | (rotw & mask);
  rotw = idx1 & mask;
  rotw = (rotw << 1) | (rotw >> (iterations - 1));
  idx1 = (idx1 & ~mask) | (rotw & mask);

  // V3 barrier fix: WAR on the exchange buffer before the transpose.
  if (COALESCED && iterations > 6) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (COALESCED) {
    transpose<Z>(r0, sh, tx);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    transpose<Z>(r1, sh, tx);
    coalesced_store<Z>(d_out, idx0, r0, stage - iterations + 1, tx);
    coalesced_store<Z>(d_out, idx1, r1, stage - iterations + 1, tx);
  } else {
    uint z_shift = out_mask == 0 ? iterations : 0;
    const bool post = p.post != 0 && stage == iterations;
    for (int z = 0; z < Z; z++) {
      uint i0 = idx0 + (uint(z) << z_shift);
      uint i1 = idx1 + (uint(z) << z_shift);
      Fp v0 = r0[z];
      Fp v1 = r1[z];
      if (post) {
        uint k0 = bit_rev(i0, lg_domain_size);
        uint k1 = bit_rev(i1, lg_domain_size);
        v0 = v0 * (mul_lo[k0 & 1023] * mul_hi[k0 >> 10]);
        v1 = v1 * (mul_lo[k1 & 1023] * mul_hi[k1 >> 10]);
      }
      d_out[i0] = v0;
      d_out[i1] = v1;
    }
  }
}

#define CT_X_INSTANCE(NAME, Z, C)                                                                  \
  template [[host_name(NAME)]] kernel void ct_x<Z, C>(const device Fp*,                            \
                                                      device Fp*,                                  \
                                                      const device Fp*,                            \
                                                      const device Fp*,                            \
                                                      const device Fp*,                            \
                                                      const device Fp*,                            \
                                                      constant NttXParams&,                        \
                                                      threadgroup uint*,                           \
                                                      uint3,                                       \
                                                      uint3,                                       \
                                                      uint3);
#define GS_X_INSTANCE(NAME, Z, C)                                                                  \
  template [[host_name(NAME)]] kernel void gs_x<Z, C>(const device Fp*,                            \
                                                      device Fp*,                                  \
                                                      const device Fp*,                            \
                                                      const device Fp*,                            \
                                                      const device Fp*,                            \
                                                      const device Fp*,                            \
                                                      constant NttXParams&,                        \
                                                      const device Fp*,                            \
                                                      const device Fp*,                            \
                                                      threadgroup uint*,                           \
                                                      uint3,                                       \
                                                      uint3,                                       \
                                                      uint3);

// X_launcher instantiations: <1>, <Z_COUNT = 256/8/sizeof(fr_t) = 8>, <8, coalesced>.
CT_X_INSTANCE("ct_x_1", 1, false)
CT_X_INSTANCE("ct_x_8", 8, false)
CT_X_INSTANCE("ct_x_8_coalesced", 8, true)
GS_X_INSTANCE("gs_x_1", 1, false)
GS_X_INSTANCE("gs_x_8", 8, false)
GS_X_INSTANCE("gs_x_8_coalesced", 8, true)

// supra ntt_x.cuh bit_rev_z_batched<64> (in place: out == in), launched as
// grid (n / 64 / 64, cols) x 64 threads with 64 * 64 words of threadgroup memory. With Z_COUNT
// (64) > warpSize the CUDA do/while runs its body once, which is what this does.
[[max_total_threads_per_threadgroup(64)]] kernel void
bit_rev_z_batched_64(device uint* io0 [[buffer(0)]],
                     constant BitRevParams& p [[buffer(1)]],
                     threadgroup uint* xchg [[threadgroup(0)]],
                     uint3 threadIdx [[thread_position_in_threadgroup]],
                     uint3 blockIdx [[threadgroup_position_in_grid]],
                     uint3 blockDim [[threads_per_threadgroup]]) {
  constexpr uint Z_COUNT = 64;
  constexpr uint LG_Z_COUNT = 6;
  const uint lg_domain_size = p.lg;
  device uint* io = io0 + ulong(blockIdx.y) * p.stride;

  const uint tx = threadIdx.x;
  const uint gid = tx / Z_COUNT;
  const uint idx = tx % Z_COUNT;
  const uint rev = bit_rev(idx, LG_Z_COUNT);

  const uint step = 1u << (lg_domain_size - LG_Z_COUNT);
  const uint tid = tx + blockDim.x * blockIdx.x;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Uniform per threadgroup: blockDim.x == Z_COUNT, so group_idx == blockIdx.x.
  const uint group_idx = tid >> LG_Z_COUNT;
  const uint group_rev = bit_rev(group_idx, lg_domain_size - 2 * LG_Z_COUNT);
  if (group_idx > group_rev) {
    return;
  }

  const uint base_idx = group_idx * Z_COUNT + idx;
  const uint base_rev = group_rev * Z_COUNT + idx;
  threadgroup uint* x = xchg + gid * Z_COUNT * Z_COUNT;

  uint regs[Z_COUNT];
  for (uint i = 0; i < Z_COUNT; i++) {
    regs[i] = io[i * step + base_idx];
    x[i * Z_COUNT + rev] = regs[i];
    if (group_idx != group_rev) {
      regs[i] = io[i * step + base_rev];
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = 0; i < Z_COUNT; i++) {
    io[i * step + base_rev] = x[rev * Z_COUNT + i];
  }

  if (group_idx == group_rev) {
    return;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = 0; i < Z_COUNT; i++) {
    x[i * Z_COUNT + rev] = regs[i];
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = 0; i < Z_COUNT; i++) {
    io[i * step + base_idx] = x[rev * Z_COUNT + i];
  }
}

// sppark bit_rev_permutation (in place, swap form), batched over columns: grid (n, cols).
kernel void bit_rev_permutation_batched(device uint* io0 [[buffer(0)]],
                                        constant BitRevParams& p [[buffer(1)]],
                                        uint2 gid [[thread_position_in_grid]]) {
  const uint n = 1u << p.lg;
  if (gid.x >= n || gid.y >= p.cols) {
    return;
  }
  device uint* io = io0 + ulong(gid.y) * p.stride;
  const uint idx = gid.x;
  const uint rev = bit_rev(idx, p.lg);
  if (idx < rev) {
    uint t0 = io[idx];
    uint t1 = io[rev];
    io[idx] = t1;
    io[rev] = t0;
  }
}

// supra ntt_x.cuh nttx_zk_shift_batched: zk shift of bit-reversed data, grid (n, cols).
kernel void nttx_zk_shift_batched(device Fp* io0 [[buffer(0)]],
                                  const device Fp* lo [[buffer(1)]],
                                  const device Fp* hi [[buffer(2)]],
                                  constant BitRevParams& p [[buffer(3)]],
                                  uint2 gid [[thread_position_in_grid]]) {
  const uint n = 1u << p.lg;
  if (gid.x >= n || gid.y >= p.cols) {
    return;
  }
  device Fp* io = io0 + ulong(gid.y) * p.stride;
  const uint k = bit_rev(gid.x, p.lg);
  io[gid.x] = io[gid.x] * (lo[k & 1023] * hi[k >> 10]);
}
