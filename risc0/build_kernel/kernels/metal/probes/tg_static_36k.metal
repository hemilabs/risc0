// PROBE tg_static_36k: static threadgroup array of 36864 bytes (above the 32 KB device limit).
// EXPECT: ACCEPT
// Runtime (runner `tg-static 36k`): does PSO creation fail? if not, a 256-thread dispatch that checks every word.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint WORDS = 9216;  // 36 KB

kernel void tg_static_36k(device uint* out [[buffer(0)]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint tpg [[threads_per_threadgroup]]) {
  threadgroup uint buf[WORDS];
  for (uint i = lid; i < WORDS; i += tpg) buf[i] = i * 3u + 1u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint bad = 0;
  for (uint i = lid; i < WORDS; i += tpg) bad += (buf[(i + 1) % WORDS] != ((i + 1) % WORDS) * 3u + 1u);
  out[lid] = bad;
}
