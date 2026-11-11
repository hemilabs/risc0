// Apple GPU miscompile / execution bug: a private-array element written in the last iteration of a
// non-unrolled loop that calls a noinline function reads back as its initial value (77).
#include <metal_stdlib>
using namespace metal;

// Stores v to device memory. Must not be inlined.
__attribute__((noinline)) static void put(device uint* row, uint col, uint v) { row[col] = v; }

struct Elem { uint v; Elem() : v(77u) {} };   // every element starts as 77
struct A3 { Elem e[3]; };

kernel void test_kernel(device uint* scratch [[buffer(0)]], device uint* out [[buffer(1)]],
                        constant uint4& prm [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= prm.x) return;
  device uint* row = scratch + gid * prm.z;
  A3 x;
#pragma nounroll
  for (uint i = 0; i < 3; i++) {
    uint v = (gid + i + 1) * 2654435761u;
    put(row, i, v);
    x.e[i].v = v;
  }
  // x.e[i].v must equal (gid + i + 1) * 2654435761u, the value put() also stored in row[i].
  uint code = 0;
  for (uint i = 0; i < 3; i++)
    if (x.e[i].v != (gid + i + 1) * 2654435761u) code |= (1u << i);
  out[gid] = code ? (code | (min(x.e[2].v, 255u) << 8)) : 0;
}
