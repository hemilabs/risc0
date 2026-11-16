// PROBE dispatch: kernels used by the runtime-only probes (empty-dispatch cost, maximum
// dispatches per command buffer, counter sampling, new_buffer_with_bytes_no_copy).
// EXPECT: ACCEPT
#include <metal_stdlib>
using namespace metal;

kernel void empty_kernel(uint gid [[thread_position_in_grid]]) {}

// One atomic increment per dispatch; the host checks the count equals the dispatch count.
kernel void count_kernel(device atomic_uint* counter [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  if (gid == 0) atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
}

kernel void inc_kernel(device uint* data [[buffer(0)]], constant uint& n [[buffer(1)]],
                       uint gid [[thread_position_in_grid]]) {
  if (gid < n) data[gid] = data[gid] * 2u + 1u;
}
