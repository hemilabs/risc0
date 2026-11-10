// PROBE tg_dynamic: threadgroup memory sized from the host with
// set_threadgroup_memory_length(0, bytes) (plan 2.1 "threadgroup" rule).
// EXPECT: ACCEPT
// Runtime (runner `tg-dynamic <bytes>`): fill `words = bytes / 4` words, barrier, verify;
// out[lid] = number of mismatching words seen by thread lid.
#include <metal_stdlib>
using namespace metal;

kernel void tg_dynamic(device uint* out [[buffer(0)]],
                       constant uint& words [[buffer(1)]],
                       threadgroup uint* buf [[threadgroup(0)]],
                       uint lid [[thread_index_in_threadgroup]],
                       uint tpg [[threads_per_threadgroup]]) {
  for (uint i = lid; i < words; i += tpg) buf[i] = i ^ 0x5a5a5a5au;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint bad = 0;
  for (uint i = lid; i < words; i += tpg) {
    uint j = (i + 7) % words;
    bad += (buf[j] != (j ^ 0x5a5a5a5au));
  }
  out[lid] = bad;
}
