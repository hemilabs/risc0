// PROBE simd_shuffle: out-of-range lane behaviour of simd_shuffle_down/up/simd_shuffle.
// EXPECT: ACCEPT
// Runner `simd-shuffle` dispatches one 32-thread group (full SIMD group) and one 24-thread
// group (partial SIMD group); each lane's own value is 1000 + lane.
// out layout per thread: [down(lane+delta), up(lane-delta), shuffle(idx = lane + 40), xor 16]
#include <metal_stdlib>
using namespace metal;

kernel void simd_shuffle_probe(device uint* out [[buffer(0)]],
                               constant uint& delta [[buffer(1)]],
                               uint lid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
  uint v = 1000u + lane;
  uint dn = simd_shuffle_down(v, ushort(delta));
  uint up = simd_shuffle_up(v, ushort(delta));
  uint sh = simd_shuffle(v, ushort(lane + 40u));
  uint xr = simd_shuffle_xor(v, ushort(16));
  out[lid * 4 + 0] = dn;
  out[lid * 4 + 1] = up;
  out[lid * 4 + 2] = sh;
  out[lid * 4 + 3] = xr;
}
