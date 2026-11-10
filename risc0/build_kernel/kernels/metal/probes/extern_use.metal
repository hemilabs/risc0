// PROBE extern (part 1 of 2): a kernel calling a function declared here and defined in
// extern_def.metal (a different .air). Linked with `xcrun metallib extern_use.air extern_def.air`.
// EXPECT: ACCEPT
// Linking extern_use.air alone must fail ("Undefined symbol"), checked by tests/metal_probes.rs.
#include <metal_stdlib>
using namespace metal;

uint ext_fn(uint x);

kernel void extern_call(device uint* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  out[gid] = ext_fn(gid);
}
