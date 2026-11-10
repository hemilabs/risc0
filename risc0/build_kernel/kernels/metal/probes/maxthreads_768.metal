// PROBE maxthreads_768: [[max_total_threads_per_threadgroup(768)]] (non power of two) placed
// BEFORE `kernel void` (plan 2.4). A plain kernel is included for comparison.
// EXPECT: ACCEPT
// Runtime (runner `maxthreads`): report PSO maxTotalThreadsPerThreadgroup for both and dispatch
// 768-thread groups.
#include <metal_stdlib>
using namespace metal;

[[max_total_threads_per_threadgroup(768)]]
kernel void k768(device uint* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  out[gid] = gid;
}

kernel void kdefault(device uint* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  out[gid] = gid;
}
