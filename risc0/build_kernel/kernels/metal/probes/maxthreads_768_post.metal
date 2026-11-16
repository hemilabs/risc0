// PROBE maxthreads_768_post: the attribute placed after the parameter list (the form the plan
// originally showed). Recorded so translators never emit it.
// EXPECT: REJECT
// Expected diagnostic: 'max_total_threads_per_threadgroup' attribute cannot be applied to types
#include <metal_stdlib>
using namespace metal;

kernel void k768(device uint* out [[buffer(0)]], uint gid [[thread_position_in_grid]])
    [[max_total_threads_per_threadgroup(768)]] {
  out[gid] = gid;
}
