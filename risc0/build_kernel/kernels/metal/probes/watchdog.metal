// PROBE watchdog: one thread spinning for a host-chosen number of iterations
// (calibrated by the runner to ~60 s). Does macOS kill the command buffer?
// EXPECT: ACCEPT
#include <metal_stdlib>
using namespace metal;

kernel void spin(device uint* out [[buffer(0)]],
                 constant uint2& iters [[buffer(1)]],  // (low, high) 32-bit halves, total = high * 2^32 + low
                 uint gid [[thread_position_in_grid]]) {
  uint x = gid + 0x9e3779b9u;
  for (uint h = 0; h <= iters.y; h++) {
    uint n = (h == iters.y) ? iters.x : 0xffffffffu;
    for (uint i = 0; i < n; i++) {
      x = x * 1664525u + 1013904223u;
    }
  }
  out[gid] = x;
}
