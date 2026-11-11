// Apple GPU: a thread-private array element stored in the last iteration of a non-unrolled loop that
// contains a function call reads back stale (its previous value) in the statement right after the loop.
// The kernel is free of undefined behaviour. VARIANT selects the bug case, controls and probes (see README).
#include <metal_stdlib>
using namespace metal;
#ifndef STRIDE
#define STRIDE 32   // words per thread in out; set by the host
#endif
#ifndef VARIANT
#define VARIANT 0
#endif

// Any real (non-inlined) call triggers it. This callee touches no memory at all.
#if VARIANT == 3
__attribute__((always_inline))
#else
__attribute__((noinline))
#endif
static uint mix(uint v) { return (v * 2246822519u) ^ (v >> 13); }

kernel void test_kernel(device uint* out [[buffer(0)]], constant uint2& prm [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
  const uint n = prm.x, zero = prm.y;          // zero == 0 at run time (used for a dynamic index)
  (void)zero;
  if (gid >= n) return;

  uint x[4];
#if VARIANT == 4
  for (uint i = 0; i < 4; i++) x[i] = 100u + i;                          // index-loop init
#else
  for (thread uint* p = x; p != x + 4; ++p) *p = 100u + (uint)(p - x);  // x = {100, 101, 102, 103}
#endif
#if VARIANT == 8
  for (uint i = 0; i < 4; i++) x[i] = 200u + i;                          // second init: x = {200..203}
#endif

  uint acc = 0;
#if VARIANT != 2
#pragma nounroll
#endif
  for (uint i = 0; i < 4; i++) {
    uint v = (gid + i + 1u) * 2654435761u;
#if VARIANT == 1
    x[i] = v;                     // store before the call
    acc ^= mix(v);
#else
    acc ^= mix(v);
    x[i] = v;                     // store after the call (the bug case)
#endif
  }

#if VARIANT == 6
  threadgroup_barrier(mem_flags::mem_none);
#elif VARIANT == 9
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_thread);
#elif VARIANT == 10
  simdgroup_barrier(mem_flags::mem_none);
#elif VARIANT == 11
  acc ^= mix(acc);                // another call between the loop and the read
#endif

  // First read, right after the loop: x[i] must equal (gid + i + 1) * 2654435761u.
  uint bad = 0;
  for (uint i = 0; i < 4; i++) {
#if VARIANT == 5
    uint got = x[(i + zero) & 3];   // dynamic index
#else
    uint got = x[i];
#endif
    if (got != (gid + i + 1u) * 2654435761u) bad |= 1u << i;
  }
  uint stale = x[3];
  // Second, later read through a data-dependent index (VARIANT 7 and 8 only).
  uint bad2 = 0;
#if VARIANT == 7 || VARIANT == 8
  uint z = (bad * 2654435761u + stale) & zero;   // == 0, but depends on the first read
  for (uint i = 0; i < 4; i++) if (x[(i + z) & 3] != (gid + i + 1u) * 2654435761u) bad2 |= 1u << i;
#endif
  out[gid * STRIDE + 0] = bad;
  out[gid * STRIDE + 1] = stale;
  out[gid * STRIDE + 2] = bad2;
  out[gid * STRIDE + 3] = acc;
}
