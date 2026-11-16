// N1 tensor probe, part 1: raw throughput of MPP matmul2d (Metal 4 tensor ops) against the scalar
// ALU forms eval_check uses. Every kernel loops `iters` times over the same operands so the
// operands stay in threadgroup memory / cache; the result is stored so nothing is dead.
// Threadgroups always have 4 simdgroups (128 threads).
//   mm_<src>_<types>_<M>x<N>x<K>_<sg>: src tg (operands in threadgroup memory) or dev (device
//     memory, a 64 KB window per threadgroup, cache resident); sg 1 = execution_simdgroup (each
//     simdgroup runs its own tile), 4 = execution_simdgroups<4> (one tile per threadgroup).
//     Destination: a cooperative tensor accumulated in place (multiply_accumulate).
//     MACs per dispatch = M*N*K * iters * ntg * (sg == 1 ? 4 : 1).
//   s_*: scalar loops, 8 independent chains per thread, ops per dispatch = 8 * iters * threads.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

template <typename TA, typename TB, typename TC, int M, int N, int K, int SG, bool DEV>
inline void mm_body(device TC* out, device const TA* ga, device const TB* gb, uint iters, uint tid, uint sgid,
                    uint tgid, threadgroup TA* sA, threadgroup TB* sB) {
  constexpr int TILES = SG == 1 ? 4 : 1;
  const uint tile = SG == 1 ? sgid : 0;
  // fill the threadgroup operands from device memory
  for (uint i = tid; i < TILES * M * K; i += 128) sA[i] = ga[i];
  for (uint i = tid; i < TILES * K * N; i += 128) sB[i] = gb[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  constexpr auto desc = matmul2d_descriptor(M, N, K, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  using Scope = metal::conditional_t<SG == 1, execution_simdgroup, execution_simdgroups<4>>;
  matmul2d<desc, Scope> op;
  // A: M x K (extents K, M); B: K x N (extents N, K); C: M x N (extents N, M)
  using EA = extents<int32_t, K, M>; using EB = extents<int32_t, N, K>;
  if constexpr (DEV) {
    const uint woff = (tgid & 63u) * 0u;  // all threadgroups share one window (cache resident)
    device TA* pa = (device TA*)(ga + woff + tile * M * K);
    device TB* pb = (device TB*)(gb + woff + tile * K * N);
    tensor<device TA, EA, tensor_inline> tA(pa, EA());
    tensor<device TB, EB, tensor_inline> tB(pb, EB());
    auto cT = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), TC>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
    for (uint it = 0; it < iters; it++) op.run(tA, tB, cT);
    threadgroup TC* sC = (threadgroup TC*)sA;  // unused; store to device
    (void)sC;
    using EC = extents<int32_t, N, M>;
    tensor<device TC, EC, tensor_inline> tC(out + (tgid * TILES + tile) * M * N, EC());
    cT.store(tC);
  } else {
    tensor<threadgroup TA, EA, tensor_inline> tA(sA + tile * M * K, EA());
    tensor<threadgroup TB, EB, tensor_inline> tB(sB + tile * K * N, EB());
    auto cT = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), TC>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
    for (uint it = 0; it < iters; it++) op.run(tA, tB, cT);
    using EC = extents<int32_t, N, M>;
    tensor<device TC, EC, tensor_inline> tC(out + (tgid * TILES + tile) * M * N, EC());
    cT.store(tC);
  }
}

#define MM(src, DEV, tn, TA, TB, TC, M, N, K, SG)                                                     \
  kernel void mm_##src##_##tn##_##M##x##N##x##K##_##SG(                                               \
      device TC* out [[buffer(0)]], device const TA* ga [[buffer(1)]], device const TB* gb [[buffer(2)]],\
      constant uint& iters [[buffer(3)]], uint tid [[thread_index_in_threadgroup]],                    \
      uint sgid [[simdgroup_index_in_threadgroup]], uint tgid [[threadgroup_position_in_grid]]) {      \
    threadgroup TA sA[(SG == 1 ? 4 : 1) * M * K];                                                      \
    threadgroup TB sB[(SG == 1 ? 4 : 1) * K * N];                                                      \
    mm_body<TA, TB, TC, M, N, K, SG, DEV>(out, ga, gb, iters, tid, sgid, tgid, sA, sB);               \
  }

#define SHAPES(src, DEV, tn, TA, TB, TC)  \
  MM(src, DEV, tn, TA, TB, TC, 16, 16, 16, 1) \
  MM(src, DEV, tn, TA, TB, TC, 16, 32, 64, 1) \
  MM(src, DEV, tn, TA, TB, TC, 32, 32, 32, 1) \
  MM(src, DEV, tn, TA, TB, TC, 32, 32, 32, 4) \
  MM(src, DEV, tn, TA, TB, TC, 64, 32, 32, 4) \
  MM(src, DEV, tn, TA, TB, TC, 64, 64, 64, 4)

SHAPES(tg, false, u8, uint8_t, uint8_t, int32_t)
SHAPES(tg, false, i8, int8_t, int8_t, int32_t)
SHAPES(tg, false, h, half, half, float)
SHAPES(dev, true, u8, uint8_t, uint8_t, int32_t)
SHAPES(dev, true, h, half, half, float)
// the chain shape (16 output rows, 32 points per simdgroup, 32 / 64 links per block)
MM(tg, false, u8, uint8_t, uint8_t, int32_t, 16, 32, 128, 1)
MM(dev, true, u8, uint8_t, uint8_t, int32_t, 16, 32, 256, 1)

// ---- scalar baselines: 8 independent chains per thread ----
constant uint P = 0x78000001u, MINV = 0x88000001u;  // Baby Bear, P^-1 mod 2^32 (rv32im Fp::M)
#define SCALAR(name, init, step, fin)                                                              \
  kernel void name(device uint* out [[buffer(0)]], device const uint* ga [[buffer(1)]],               \
                   constant uint& iters [[buffer(3)]], uint gid [[thread_position_in_grid]]) {       \
    uint y = ga[gid & 1023u] | 1u, z = ga[(gid + 7u) & 1023u]; (void)z;                                        \
    init                                                                                               \
    for (uint it = 0; it < iters; it++) { step }                                                     \
    out[gid] = fin;                                                                                    \
  }
#define X8(e) e(0) e(1) e(2) e(3) e(4) e(5) e(6) e(7)
#define DECL(i) uint x##i = gid * 8u + i;
#define DECL64(i) ulong a##i = gid * 8u + i; uint x##i = gid + i;
#define MUL(i) x##i = x##i * y + z;
#define MULHI(i) x##i = mulhi(x##i, y) + z;
#define WMAC(i) a##i += ulong(x##i) * ulong(y); x##i += z;
// one link of the lazy chain as eval_check does it (EF_WIDE rk_link: 64-bit madd + fold of hi)
#define LINK(i) { ulong t = a##i + ulong(x##i) * ulong(y); uint h = uint(t >> 32); h = min(h, h - P); a##i = (ulong(h) << 32) | uint(t); x##i += z; }
// ef_lmul: Montgomery product, lazy [0, 2P)
#define MONT(i) { uint lo = x##i * y, hi = mulhi(x##i, y); uint m = lo * MINV; x##i = hi - mulhi(m, P) + P; }
#define FMA(i) f##i = fma(f##i, fy, fz);
#define DECLF(i) float f##i = float(gid + i);
#define SUM(i) ^ x##i
#define SUMA(i) ^ uint(a##i) ^ uint(a##i >> 32)
#define SUMF(i) ^ as_type<uint>(f##i)
SCALAR(s_mul32, X8(DECL), X8(MUL), 0u X8(SUM))
SCALAR(s_mulhi, X8(DECL), X8(MULHI), 0u X8(SUM))
SCALAR(s_wmac, X8(DECL64), X8(WMAC), 0u X8(SUMA))
SCALAR(s_link, X8(DECL64), X8(LINK), 0u X8(SUMA))
SCALAR(s_mont, X8(DECL), X8(MONT), 0u X8(SUM))
kernel void s_fma(device uint* out [[buffer(0)]], device const uint* ga [[buffer(1)]],
                  constant uint& iters [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  float fy = as_type<float>((ga[gid & 1023u] & 0x007fffffu) | 0x3f000000u), fz = 0.25f;
  X8(DECLF)
  for (uint it = 0; it < iters; it++) { X8(FMA) }
  out[gid] = 0u X8(SUMF);
}
