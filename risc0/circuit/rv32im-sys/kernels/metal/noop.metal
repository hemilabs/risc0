// Placeholder kernel: lets this crate's `metal` feature build and embed a metallib before the
// unit's first Metal kernel lands (docs/metal-port/PLAN.md 8.2). The unit's first kernel item
// deletes this file and removes `noop` from METAL_KERNEL_NAMES in src/lib.rs.

#include <metal_stdlib>

using namespace metal;

kernel void noop(device uint* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  if (gid == 0) {
    out[0] = 0;
  }
}
