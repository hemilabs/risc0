// PROBE assert_release: is assert() a no-op in MSL (with only <metal_stdlib> included)?
// EXPECT: ACCEPT
// Compile-time: tests/metal_probes.rs preprocesses this file (-E) and checks that the assert
// below expands to `((void) 0)`. Runtime (runner `assert`): the assert always fails, yet the
// store after it must happen (out[gid] == gid + 7).
#include <metal_stdlib>
using namespace metal;

kernel void assert_release(device uint* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  assert(gid == 0xffffffffu && "always false");
  out[gid] = gid + 7u;
}
