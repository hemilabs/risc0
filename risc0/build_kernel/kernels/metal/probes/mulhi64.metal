// PROBE mulhi64: mulhi on ulong (plan 8.5: accepted).
// EXPECT: ACCEPT
#include <metal_stdlib>
using namespace metal;

kernel void mulhi64(device ulong* o [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  o[gid] = mulhi(o[gid], 0x9e3779b97f4a7c15ul) + o[gid] * 3ul;
}
