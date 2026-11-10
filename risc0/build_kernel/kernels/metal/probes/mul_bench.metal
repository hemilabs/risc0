// PROBE mul_bench: 32-bit vs 64-bit integer multiply / mulhi throughput.
// EXPECT: ACCEPT
// Each thread runs `iters` iterations of 4 independent dependent chains (4 ops per iteration
// per chain type). Runner `mul-bench` reports ops/s from GPU start/end time of one dispatch.
#include <metal_stdlib>
using namespace metal;

kernel void mul32_lo(device uint* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
  uint a = gid | 1u, b = gid * 3u + 5u, c = gid * 7u + 11u, d = gid * 13u + 17u;
  const uint m = 0x9e3779b1u;
  for (uint i = 0; i < iters; i++) { a = a * m + i; b = b * m + i; c = c * m + i; d = d * m + i; }
  out[gid] = a ^ b ^ c ^ d;
}

kernel void mul32_hi(device uint* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
  uint a = gid | 1u, b = gid * 3u + 5u, c = gid * 7u + 11u, d = gid * 13u + 17u;
  const uint m = 0x9e3779b1u;
  for (uint i = 0; i < iters; i++) {
    a = mulhi(a, m) + i; b = mulhi(b, m) + i; c = mulhi(c, m) + i; d = mulhi(d, m) + i;
  }
  out[gid] = a ^ b ^ c ^ d;
}

kernel void mul64_lo(device ulong* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
  ulong a = gid | 1ul, b = gid * 3ul + 5ul, c = gid * 7ul + 11ul, d = gid * 13ul + 17ul;
  const ulong m = 0x9e3779b97f4a7c15ul;
  for (uint i = 0; i < iters; i++) { a = a * m + i; b = b * m + i; c = c * m + i; d = d * m + i; }
  out[gid] = a ^ b ^ c ^ d;
}

kernel void mul64_hi(device ulong* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
  ulong a = gid | 1ul, b = gid * 3ul + 5ul, c = gid * 7ul + 11ul, d = gid * 13ul + 17ul;
  const ulong m = 0x9e3779b97f4a7c15ul;
  for (uint i = 0; i < iters; i++) {
    a = mulhi(a, m) + i; b = mulhi(b, m) + i; c = mulhi(c, m) + i; d = mulhi(d, m) + i;
  }
  out[gid] = a ^ b ^ c ^ d;
}

// 32x32 -> 64 widening multiply (the Fp Montgomery shape) for reference.
kernel void mul32_wide(device ulong* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
                       uint gid [[thread_position_in_grid]]) {
  uint a = gid | 1u, b = gid * 3u + 5u, c = gid * 7u + 11u, d = gid * 13u + 17u;
  const uint m = 0x9e3779b1u;
  ulong acc = 0;
  for (uint i = 0; i < iters; i++) {
    ulong pa = ulong(a) * m, pb = ulong(b) * m, pc = ulong(c) * m, pd = ulong(d) * m;
    a = uint(pa >> 32) + i; b = uint(pb >> 32) + i; c = uint(pc >> 32) + i; d = uint(pd >> 32) + i;
    acc += pa ^ pb ^ pc ^ pd;
  }
  out[gid] = acc ^ a ^ b ^ c ^ d;
}
