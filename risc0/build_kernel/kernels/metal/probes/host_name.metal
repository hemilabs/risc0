// PROBE host_name: explicit template kernel instantiations named with [[host_name]].
// EXPECT: ACCEPT
// Runtime (runner `host-name`): Library::get_function("ct_x_8_coalesced") / ("ct_x_1_plain")
// must resolve and produce out[gid] = gid * R + C.
#include <metal_stdlib>
using namespace metal;

template <uint R, bool C>
kernel void ct_x(device uint* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  out[gid] = gid * R + (C ? 1u : 0u);
}

template [[host_name("ct_x_8_coalesced")]] kernel void ct_x<8, true>(device uint*, uint);
template [[host_name("ct_x_1_plain")]] kernel void ct_x<1, false>(device uint*, uint);
