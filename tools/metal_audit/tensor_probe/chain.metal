// N1 tensor probe, part 2: the RDNA4 WMMA chain rewrite (kernels/cuda/evalfast/wmma_hip.py) in
// Metal form, against the scalar lazy chain eval_check runs today, on the same data, bit-exact.
//
// One thread = one point. Each thread runs NCH (= `iters`) chains of L links:
//   chain ch: S = base; S += c_l * pm[k_l] (l < L) lazily (rk_link / rk_link_nf, EF_WIDE form);
//   result rk_fin(S) (REDC to canonical). out[point] = canonical sum of the chain results.
//   c_l = ef_lmul(d0, d1) of two trace words loaded per link (as the generated chains compute a
//   product of loaded columns per link); with -DCH_NOLOAD the words come from an ALU hash instead.
// Tensor form (ch_mm_*): per block of BL links each thread stores its c_l raw words as 4*BL bytes
// (point-major row) in its simdgroup's threadgroup staging; one matmul2d (execution_simdgroup)
//   D[(j,b)][p] = sum_{l,a} Q[k_l][(j,b)][a] * byte_a(c_l[p]),  Q[k][(j,b)][a] = byte_b(pm[k][j]*256^a mod P)
// (uint8 x uint8 -> int32, M = 16, N = 32 points, K = 4*BL, right operand transposed); D goes to
// threadgroup memory; each thread combines T_j = sum_b D[(j,b)] << 8b into its 64-bit lazy
// accumulator (fold of the high word). rk_fin is the same, so the results are bit-exact.
// Table buffer (buffer 2, from tables.py): pm rows [NK][4] u32 at 0, Q tiles at QOFF bytes:
// tile (ch, block) = 16 x (4*BL) bytes row-major at QOFF + ((ch*NB + block) * 16 * 4*BL).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant uint P = 0x78000001u, MINV = 0x88000001u;  // P^-1 mod 2^32 (Fp::M)
#ifndef L
#define L 32
#endif
#define NK 1255u
#define QOFF (NK * 16u)
#define DMASK ((1u << 22) - 1u)   // trace window (words)

inline uint lmul(uint x, uint y) {  // ef_lmul: [0, 2P)
  uint lo = x * y, hi = mulhi(x, y), m = lo * MINV; return hi - mulhi(m, P) + P;
}
inline uint canon(uint x) { return min(x, x - P); }  // chain terms are canonical Fp (< P)
inline uint kof(uint ch, uint l) { return ((ch * L + l) * 7u) % NK; }
inline uint cval(device const uint* data, uint gid, uint ch, uint l) {
#ifdef CH_NOLOAD
  uint h = (gid * 0x9e3779b9u) ^ ((ch * L + l) * 0x85ebca6bu); h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12;
  return canon(lmul(h & 0x3fffffffu, (h >> 2) & 0x3fffffffu));
#else
  uint i = ((ch * L + l) * 4099u + gid) & DMASK;
  return canon(lmul(data[i] & 0x3fffffffu, data[(i + 1048576u) & DMASK] & 0x3fffffffu));
#endif
}
inline uint fin1(uint lo, uint hi) { uint m = lo * MINV; uint t = hi - mulhi(m, P); return min(t, t + P); }
inline uint addp(uint a, uint b) { uint s = a + b; return min(s, s - P); }

kernel void ch_scalar(device uint* out [[buffer(0)]], device const uint* data [[buffer(1)]],
                      device const uint* tab [[buffer(2)]], constant uint& nch [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
  uint r0 = 0, r1 = 0, r2 = 0, r3 = 0;
  for (uint ch = 0; ch < nch; ch++) {
    ulong s0 = ulong(ch + 1u) << 32, s1 = ulong(ch + 2u) << 32, s2 = ulong(ch + 3u) << 32, s3 = ulong(ch + 4u) << 32;
    #pragma unroll
    for (uint l = 0; l < L; l++) {
      uint c = cval(data, gid, ch, l);
      uint4 pm = reinterpret_cast<device const uint4*>(tab)[kof(ch, l)];
      s0 += ulong(c) * pm.x; s1 += ulong(c) * pm.y; s2 += ulong(c) * pm.z; s3 += ulong(c) * pm.w;
      if (l & 1u) {  // rk_link (fold) after rk_link_nf, as the generator alternates them
#define FOLD(s) { uint h = uint(s >> 32); h = min(h, h - P); s = (ulong(h) << 32) | uint(s); }
        FOLD(s0) FOLD(s1) FOLD(s2) FOLD(s3)
      }
    }
    FOLD(s0) FOLD(s1) FOLD(s2) FOLD(s3)
    r0 = addp(r0, fin1(uint(s0), uint(s0 >> 32))); r1 = addp(r1, fin1(uint(s1), uint(s1 >> 32)));
    r2 = addp(r2, fin1(uint(s2), uint(s2 >> 32))); r3 = addp(r3, fin1(uint(s3), uint(s3 >> 32)));
  }
  reinterpret_cast<device uint4*>(out)[gid] = uint4(r0, r1, r2, r3);
}

template <uint BL>
inline void ch_mm_body(device uint* out, device const uint* data, device const uint* tab, uint nch, uint gid,
                       uint lane, threadgroup uint* stg, threadgroup int* dst) {
  constexpr uint K = 4u * BL, NB = L / BL;
  constexpr auto desc = matmul2d_descriptor(16, 32, K, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  using EQ = extents<int32_t, K, 16>;    // Q: 16 x K row-major
  using EB = extents<int32_t, K, 32>;    // staging: 32 points x K (B transposed)
  using ED = extents<int32_t, 32, 16>;   // D: 16 x 32
  tensor<threadgroup uint8_t, EB, tensor_inline> tB((threadgroup uint8_t*)stg, EB());
  tensor<threadgroup int32_t, ED, tensor_inline> tD(dst, ED());
  uint r0 = 0, r1 = 0, r2 = 0, r3 = 0;
  for (uint ch = 0; ch < nch; ch++) {
    ulong s0 = ulong(ch + 1u) << 32, s1 = ulong(ch + 2u) << 32, s2 = ulong(ch + 3u) << 32, s3 = ulong(ch + 4u) << 32;
    for (uint bi = 0; bi < NB; bi++) {
      #pragma unroll
      for (uint l = 0; l < BL; l++) stg[lane * BL + l] = cval(data, gid, ch, bi * BL + l);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      device uint8_t* q = (device uint8_t*)tab + QOFF + (ch * NB + bi) * 16u * K;
      tensor<device uint8_t, EQ, tensor_inline> tQ(q, EQ());
      auto cD = op.template get_destination_cooperative_tensor<decltype(tQ), decltype(tB), int32_t>();
      op.run(tQ, tB, cD);
      cD.store(tD);
      simdgroup_barrier(mem_flags::mem_threadgroup);
#define COMB(s, j) { ulong t = ulong(uint(dst[(4u*j+0u)*32u + lane])) + (ulong(uint(dst[(4u*j+1u)*32u + lane])) << 8) \
                            + (ulong(uint(dst[(4u*j+2u)*32u + lane])) << 16) + (ulong(uint(dst[(4u*j+3u)*32u + lane])) << 24); \
                     s += t; FOLD(s) }
      COMB(s0, 0u) COMB(s1, 1u) COMB(s2, 2u) COMB(s3, 3u)
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    r0 = addp(r0, fin1(uint(s0), uint(s0 >> 32))); r1 = addp(r1, fin1(uint(s1), uint(s1 >> 32)));
    r2 = addp(r2, fin1(uint(s2), uint(s2 >> 32))); r3 = addp(r3, fin1(uint(s3), uint(s3 >> 32)));
  }
  reinterpret_cast<device uint4*>(out)[gid] = uint4(r0, r1, r2, r3);
}

#define CHMM(BL)                                                                                     \
  kernel void ch_mm_##BL(device uint* out [[buffer(0)]], device const uint* data [[buffer(1)]],       \
                         device const uint* tab [[buffer(2)]], constant uint& nch [[buffer(3)]],      \
                         uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]],\
                         uint sg [[simdgroup_index_in_threadgroup]]) {                                \
    threadgroup uint stg[4 * 32 * BL];                                                                 \
    threadgroup int dst[4 * 16 * 32];                                                                  \
    ch_mm_body<BL>(out, data, tab, nch, gid, lane, stg + sg * 32 * BL, dst + sg * 16 * 32);            \
  }
CHMM(8)
CHMM(16)
CHMM(32)

// The c_l work alone (loads + lmul, summed into the accumulators without the pm products): the
// part both forms share; chain arithmetic = form - ch_none.
kernel void ch_none(device uint* out [[buffer(0)]], device const uint* data [[buffer(1)]],
                    device const uint* tab [[buffer(2)]], constant uint& nch [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
  uint r0 = 0;
  for (uint ch = 0; ch < nch; ch++) {
    uint s = ch;
    #pragma unroll
    for (uint l = 0; l < L; l++) s = addp(s, cval(data, gid, ch, l));
    r0 = addp(r0, s);
  }
  reinterpret_cast<device uint4*>(out)[gid] = uint4(r0, 0u, 0u, 0u);
}

// One matmul per threadgroup (execution_simdgroups<4>, N = 128 points), BL = 16.
kernel void ch_mm4_16(device uint* out [[buffer(0)]], device const uint* data [[buffer(1)]],
                      device const uint* tab [[buffer(2)]], constant uint& nch [[buffer(3)]],
                      uint gid [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint BL = 16, K = 4u * BL, NB = L / BL;
  threadgroup uint stg[128 * BL];
  threadgroup int dst[16 * 128];
  constexpr auto desc = matmul2d_descriptor(16, 128, K, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<4>> op;
  using EQ = extents<int32_t, K, 16>; using EB = extents<int32_t, K, 128>; using ED = extents<int32_t, 128, 16>;
  tensor<threadgroup uint8_t, EB, tensor_inline> tB((threadgroup uint8_t*)stg, EB());
  tensor<threadgroup int32_t, ED, tensor_inline> tD(dst, ED());
  uint r0 = 0, r1 = 0, r2 = 0, r3 = 0;
  for (uint ch = 0; ch < nch; ch++) {
    ulong s0 = ulong(ch + 1u) << 32, s1 = ulong(ch + 2u) << 32, s2 = ulong(ch + 3u) << 32, s3 = ulong(ch + 4u) << 32;
    for (uint bi = 0; bi < NB; bi++) {
      #pragma unroll
      for (uint l = 0; l < BL; l++) stg[tid * BL + l] = cval(data, gid, ch, bi * BL + l);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      device uint8_t* q = (device uint8_t*)tab + QOFF + (ch * NB + bi) * 16u * K;
      tensor<device uint8_t, EQ, tensor_inline> tQ(q, EQ());
      auto cD = op.template get_destination_cooperative_tensor<decltype(tQ), decltype(tB), int32_t>();
      op.run(tQ, tB, cD);
      cD.store(tD);
      threadgroup_barrier(mem_flags::mem_threadgroup);
#define COMB4(s, j) { ulong t = ulong(uint(dst[(4u*j+0u)*128u + tid])) + (ulong(uint(dst[(4u*j+1u)*128u + tid])) << 8) \
                             + (ulong(uint(dst[(4u*j+2u)*128u + tid])) << 16) + (ulong(uint(dst[(4u*j+3u)*128u + tid])) << 24); \
                      s += t; FOLD(s) }
      COMB4(s0, 0u) COMB4(s1, 1u) COMB4(s2, 2u) COMB4(s3, 3u)
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    r0 = addp(r0, fin1(uint(s0), uint(s0 >> 32))); r1 = addp(r1, fin1(uint(s1), uint(s1 >> 32)));
    r2 = addp(r2, fin1(uint(s2), uint(s2 >> 32))); r3 = addp(r3, fin1(uint(s3), uint(s3 >> 32)));
  }
  reinterpret_cast<device uint4*>(out)[gid] = uint4(r0, r1, r2, r3);
}
