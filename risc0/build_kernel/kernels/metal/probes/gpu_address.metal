// PROBE gpu_address: a POD struct of `device` pointers passed with set_bytes (values taken from
// MTLBuffer.gpuAddress on the host) and dereferenced in the kernel. This is the zkp K5 `MdJobs`
// pattern (plan 2.0 "Kernel context passing").
// EXPECT: ACCEPT
// Runtime (runner `gpu-address`): job.ptrs[gid & 3][gid >> 2] = gid + 1 for gid < 4 * n.
#include <metal_stdlib>
using namespace metal;

struct Jobs {
  device uint* ptrs[4];  // 4 x 8 bytes, host side: [u64; 4]
  uint n;                // per-pointer element count
  uint _pad;
};

kernel void gpu_address(constant Jobs& job [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= 4 * job.n) return;
  job.ptrs[gid & 3][gid >> 2] = gid + 1;
}
