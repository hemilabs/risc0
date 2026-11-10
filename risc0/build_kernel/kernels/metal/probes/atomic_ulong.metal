// PROBE atomic_ulong: 64-bit device atomic fetch_add (plan 8.5: rejected).
// EXPECT: REJECT
#include <metal_stdlib>
using namespace metal;

kernel void atomic64(device atomic_ulong* c [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  atomic_fetch_add_explicit(c, 1ul, memory_order_relaxed);
}
