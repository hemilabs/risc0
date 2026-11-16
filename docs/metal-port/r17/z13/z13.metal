#include <metal_stdlib>
using namespace metal;
// Z13 shape: a private uint v[16] (kept in private memory by one dynamically indexed read),
// fully inlined and unrolled; the coset twist is applied element-wise inside `if (bz != 0)`.
// The bz == 0 blocks need no twist (identity), so skipping the branch is correct by construction.
// CONTROL: the twist is unconditional, as an identity for bz == 0 (t = st = Montgomery one).
// W7-T1 fix round 1: the first version's CONTROL (and CPU reference) set t = ONE but kept st, i.e.
// multiplied the bz == 0 blocks by st^m; that is a different function, which made the repro look
// "wrong" in exactly the blocks that skip the branch. See README.md.
constant uint P = 0x78000001u;
constant uint MI = 0x77ffffffu;   // -P^-1 mod 2^32
constant uint ONE = 0x0ffffffeu;  // 2^32 mod P (Montgomery one)
static inline __attribute__((always_inline)) uint mul(uint a, uint b) {
  ulong o = ulong(a) * ulong(b);
  uint red = uint(o) * MI;
  o += ulong(red) * ulong(P);
  uint r = uint(o >> 32);
  return r >= P ? r - P : r;
}
static inline __attribute__((always_inline)) uint add(uint a, uint b) { uint r = a + b; return r >= P ? r - P : r; }
static inline __attribute__((always_inline)) uint sub(uint a, uint b) { return a >= b ? a - b : a + P - b; }
static inline __attribute__((always_inline)) void bfly(thread uint (&v)[16], constant uint* w) {
#pragma unroll
  for (uint s = 0; s < 4; s++) {
    uint h = 8u >> s;
#pragma unroll
    for (uint i = 0; i < 16; i++) {
      if ((i & h) == 0) {
        uint a = v[i], b = mul(v[i + h], w[s * 16 + i]);
        v[i] = add(a, b); v[i + h] = sub(a, b);
      }
    }
  }
}
kernel void z13_shape(device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]],
                      constant uint4& prm [[buffer(2)]], constant uint* w [[buffer(3)]],
                      device const uint* tw [[buffer(4)]], device const uint* stw [[buffer(5)]],
                      threadgroup uint* sh [[threadgroup(0)]],
                      uint3 tg [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
                      uint3 tpg [[threads_per_threadgroup]]) {
  uint n = prm.x, bz = tg.y, T = tpg.x;
  uint col = tg.x * T + lid;
  uint v[16];
#pragma unroll
  for (uint m = 0; m < 16; m++) v[m] = in[(bz * 16 + m) * n + col];
  uint t = mul(prm.y, bz + col + 1u), st = prm.z;
#ifdef TABLE
  // coset factors from device memory, identity for bz == 0 (tw[col] = stw[0] = ONE)
  t = tw[bz * n + col]; st = stw[bz];
#endif
#ifdef CONTROL
  if (bz == 0) { t = ONE; st = ONE; }
#pragma unroll
  for (uint m = 0; m < 16; m++) { v[m] = mul(v[m], t); t = mul(t, st); }
#elif defined(SINGLE)
  if (bz != 0) v[0] = mul(v[0], t);
#elif defined(GUARD_DIV)
  if ((col & 1u) != 0) {
#pragma unroll
    for (uint m = 0; m < 16; m++) { v[m] = mul(v[m], t); t = mul(t, st); }
  }
#elif defined(TRIP)
  uint cnt = bz != 0 ? 16u : 0u;
  for (uint m = 0; m < cnt; m++) { v[m] = mul(v[m], t); t = mul(t, st); }
#elif defined(SPILL)
  uint u[16];
#pragma unroll
  for (uint m = 0; m < 16; m++) u[m] = v[(m * 7u + lid) & 15u];
  if (bz != 0) {
#pragma unroll
    for (uint m = 0; m < 16; m++) { v[m] = mul(v[m], t); u[m] = mul(u[m], st); t = mul(t, st); }
  }
#pragma unroll
  for (uint m = 0; m < 16; m++) v[m] = add(v[m], u[(m + lid) & 15u]);
#elif defined(NOUNROLL)
  if (bz != 0) {
#pragma unroll 1
    for (uint m = 0; m < 16; m++) { v[m] = mul(v[m], t); t = mul(t, st); }
  }
#else
  if (bz != 0) {
#pragma unroll
    for (uint m = 0; m < 16; m++) { v[m] = mul(v[m], t); t = mul(t, st); }
  }
#endif
  bfly(v, w);
#pragma unroll
#ifdef NODYN
  for (uint m = 0; m < 16; m++) sh[m * T + lid] = v[m];
#else
  for (uint m = 0; m < 16; m++) sh[m * T + lid] = v[(m + lid) & 15u];
#endif
  threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
  for (uint m = 0; m < 16; m++) v[m] = sh[m * T + ((lid + m) % T)];
  bfly(v, w);
#pragma unroll
  for (uint m = 0; m < 16; m++) out[(bz * 16 + m) * n + col] = v[m];
}
