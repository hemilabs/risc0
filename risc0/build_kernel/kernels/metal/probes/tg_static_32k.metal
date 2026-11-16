// PROBE tg_static_32k: static threadgroup array of exactly 32768 bytes.
// EXPECT: ACCEPT
// Runtime (runner `tg-static 32k`): PSO creation + a 256-thread dispatch that checks every word.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint WORDS = 8192;  // 32 KB

kernel void tg_static_32k(device uint* out [[buffer(0)]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint tpg [[threads_per_threadgroup]]) {
  threadgroup uint buf[WORDS];
  for (uint i = lid; i < WORDS; i += tpg) buf[i] = i * 3u + 1u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint bad = 0;
  for (uint i = lid; i < WORDS; i += tpg) bad += (buf[(i + 1) % WORDS] != ((i + 1) % WORDS) * 3u + 1u);
  out[lid] = bad;
}
