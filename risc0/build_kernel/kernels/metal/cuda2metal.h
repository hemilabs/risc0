// Copyright 2026 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

/// \file
/// CUDA spellings mapped onto MSL, for hand ports and the cuda2metal translators (Metal port plan
/// section 2.3). Only what aliases cleanly lives here; kernel signatures, index attributes,
/// `extern __shared__`, `__constant__` globals, `__launch_bounds__` (emit
/// `[[max_total_threads_per_threadgroup(N)]]` before `kernel`), template kernel instantiation
/// (`[[host_name("...")]]`), lambdas, structured bindings and inheritance must be rewritten.
///
/// All warp-level helpers assume a 32-wide SIMD-group (Apple GPUs; `WARP_SZ`).

#include <metal_stdlib>
// Pull in <metal_assert> now, so that the `assert` redefinition below is not undone by a later
// include of it.
#include <metal_assert>

// ---------------------------------------------------------------------------------------------
// Function / variable qualifiers
// ---------------------------------------------------------------------------------------------

// `__device__ __forceinline__` and `__device__ inline` expand to a repeated `inline`, which is
// legal but warned about; silence that for the translation unit.
#pragma clang diagnostic ignored "-Wduplicate-decl-specifier"

#define __device__ inline
#define __host__ inline
#define __forceinline__ inline
#define __noinline__ __attribute__((noinline))
#define __restrict__

// ---------------------------------------------------------------------------------------------
// Barriers
// ---------------------------------------------------------------------------------------------

/// CUDA `__syncthreads()`: a threadgroup execution barrier ordering threadgroup memory.
#define __syncthreads() metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup)

/// `__syncthreads()` where the threads exchange data through device memory. Orders device memory
/// and, like CUDA's `__syncthreads()`, threadgroup memory too.
#define __syncthreads_device()                                                                  \
  metal::threadgroup_barrier(metal::mem_flags::mem_device | metal::mem_flags::mem_threadgroup)

/// CUDA `__syncwarp([mask])`: a SIMD-group barrier (the mask is ignored: all 32 lanes).
#define __syncwarp(...) metal::simdgroup_barrier(metal::mem_flags::mem_threadgroup)

// ---------------------------------------------------------------------------------------------
// Warp size and SIMD-group shuffles / votes
// ---------------------------------------------------------------------------------------------

#ifndef WARP_SZ
#define WARP_SZ 32
#endif
#define warpSize 32

// The `_sync` forms take the CUDA member mask first and ignore it (every lane of the SIMD-group
// participates). There is deliberately no `width` overload: a sub-warp width changes the
// semantics, so a port passing one fails to compile instead of silently using width 32.
template <typename T> inline T __shfl_down_sync(uint mask, T v, uint delta) {
  return metal::simd_shuffle_down(v, ushort(delta));
}
template <typename T> inline T __shfl_up_sync(uint mask, T v, uint delta) {
  return metal::simd_shuffle_up(v, ushort(delta));
}
template <typename T> inline T __shfl_xor_sync(uint mask, T v, uint laneMask) {
  return metal::simd_shuffle_xor(v, ushort(laneMask));
}
template <typename T> inline T __shfl_sync(uint mask, T v, uint srcLane) {
  return metal::simd_shuffle(v, ushort(srcLane));
}
template <typename T> inline T __shfl_down(T v, uint delta) {
  return metal::simd_shuffle_down(v, ushort(delta));
}
template <typename T> inline T __shfl_up(T v, uint delta) {
  return metal::simd_shuffle_up(v, ushort(delta));
}
template <typename T> inline T __shfl_xor(T v, uint laneMask) {
  return metal::simd_shuffle_xor(v, ushort(laneMask));
}
template <typename T> inline T __shfl(T v, uint srcLane) {
  return metal::simd_shuffle(v, ushort(srcLane));
}

// 64-bit values shuffle as two 32-bit halves (MSL shuffles only 32-bit and narrower scalars).
#define R0_C2M_SHFL64(NAME, MSL)                                                                \
  inline ulong NAME(uint mask, ulong v, uint arg) {                                             \
    uint lo = MSL(uint(v), ushort(arg));                                                        \
    uint hi = MSL(uint(v >> 32), ushort(arg));                                                  \
    return (ulong(hi) << 32) | ulong(lo);                                                       \
  }                                                                                             \
  inline long NAME(uint mask, long v, uint arg) { return long(NAME(mask, ulong(v), arg)); }
R0_C2M_SHFL64(__shfl_down_sync, metal::simd_shuffle_down)
R0_C2M_SHFL64(__shfl_up_sync, metal::simd_shuffle_up)
R0_C2M_SHFL64(__shfl_xor_sync, metal::simd_shuffle_xor)
R0_C2M_SHFL64(__shfl_sync, metal::simd_shuffle)
#undef R0_C2M_SHFL64

/// CUDA `__ballot_sync(mask, pred)`: bit i set iff lane i's predicate holds (32 lanes).
/// `simd_ballot` returns a `simd_vote`, which only converts to an integer via `vote_t`.
#define __ballot_sync(mask, pred) ((uint)(metal::simd_vote::vote_t)metal::simd_ballot(bool(pred)))

// ---------------------------------------------------------------------------------------------
// Integer intrinsics
// ---------------------------------------------------------------------------------------------

inline int __popc(uint x) {
  return int(metal::popcount(x));
}
inline int __popcll(ulong x) {
  return int(metal::popcount(x));
}
inline int __clz(uint x) {
  return int(metal::clz(x));
}
inline int __clzll(ulong x) {
  return int(metal::clz(x));
}
inline uint __brev(uint x) {
  return metal::reverse_bits(x);
}
/// 1-based index of the least significant set bit, 0 for 0 (CUDA semantics).
inline int __ffs(uint x) {
  return x == 0 ? 0 : int(metal::ctz(x)) + 1;
}
inline uint __umulhi(uint a, uint b) {
  return metal::mulhi(a, b);
}
inline ulong __umul64hi(ulong a, ulong b) {
  return metal::mulhi(a, b);
}

// ---------------------------------------------------------------------------------------------
// Loads
// ---------------------------------------------------------------------------------------------

#define __ldg(p) (*(p))
#define __ldcg(p) (*(p))

// ---------------------------------------------------------------------------------------------
// 32-bit atomics (relaxed, like CUDA's). Overloads take atomic pointers or plain 32-bit integer
// pointers in device or threadgroup memory and return the previous value.
// ---------------------------------------------------------------------------------------------

#define R0_C2M_ATOMICS(AS, T, AT)                                                               \
  inline T atomicAdd(AS AT* p, T v) {                                                           \
    return metal::atomic_fetch_add_explicit(p, v, metal::memory_order_relaxed);                 \
  }                                                                                             \
  inline T atomicAdd(AS T* p, T v) { return atomicAdd((AS AT*)p, v); }                          \
  inline T atomicOr(AS AT* p, T v) {                                                            \
    return metal::atomic_fetch_or_explicit(p, v, metal::memory_order_relaxed);                  \
  }                                                                                             \
  inline T atomicOr(AS T* p, T v) { return atomicOr((AS AT*)p, v); }                            \
  /* Strong CAS from MSL's weak one: retry spurious failures (expected still == cmp). */        \
  inline T atomicCAS(AS AT* p, T cmp, T val) {                                                  \
    T expected = cmp;                                                                           \
    while (!metal::atomic_compare_exchange_weak_explicit(                                       \
               p, &expected, val, metal::memory_order_relaxed, metal::memory_order_relaxed) &&  \
           expected == cmp) {                                                                   \
    }                                                                                           \
    return expected;                                                                            \
  }                                                                                             \
  inline T atomicCAS(AS T* p, T cmp, T val) { return atomicCAS((AS AT*)p, cmp, val); }

R0_C2M_ATOMICS(device, uint, metal::atomic_uint)
R0_C2M_ATOMICS(device, int, metal::atomic_int)
R0_C2M_ATOMICS(threadgroup, uint, metal::atomic_uint)
R0_C2M_ATOMICS(threadgroup, int, metal::atomic_int)
#undef R0_C2M_ATOMICS

// ---------------------------------------------------------------------------------------------
// Compiler hints and debugging
// ---------------------------------------------------------------------------------------------

#define asm(...)
#define __asm__(...)
#define __builtin_assume(x) ((void)0)
#define __builtin_elementwise_min(a, b) metal::min(a, b)

/// sppark's optimization barrier `opq(x)`: the identity.
template <typename T> inline T opq(T x) {
  return x;
}

/// `assert` compiles to nothing (as `<metal_assert>` already does). A kernel that must report a
/// violated invariant uses `R0_FAULT` instead: `flag` is a thread-local `uint` lvalue (for
/// example `ctx.bad`) that becomes 1 when `cond` is false; at kernel end the kernel calls
/// `r0_fault_flush(fault, flag)` once, which ORs it into the `device atomic_uint* fault` buffer
/// the host reads after `sync()` (plan sections 2.0 and 2.9).
#undef assert
#define assert(x) ((void)0)
#define R0_FAULT(flag, cond) ((flag) |= (cond) ? 0u : 1u)

inline void r0_fault_flush(device metal::atomic_uint* fault, uint flag) {
  if (flag != 0) {
    metal::atomic_fetch_or_explicit(fault, flag, metal::memory_order_relaxed);
  }
}

#define printf(...) ((void)0)

// ---------------------------------------------------------------------------------------------
// cuda::std::array
// ---------------------------------------------------------------------------------------------

namespace r0 {

/// A minimal `std::array` (aggregate, so brace initialization works) for thread-local use.
template <typename T, size_t N> struct array {
  T elems[N];

  thread T& operator[](size_t i) thread { return elems[i]; }
  const thread T& operator[](size_t i) const thread { return elems[i]; }
  device T& operator[](size_t i) device { return elems[i]; }
  const device T& operator[](size_t i) const device { return elems[i]; }
  threadgroup T& operator[](size_t i) threadgroup { return elems[i]; }
  const threadgroup T& operator[](size_t i) const threadgroup { return elems[i]; }
  const constant T& operator[](size_t i) const constant { return elems[i]; }

  static constexpr size_t size() { return N; }
  thread T* data() thread { return elems; }
  const thread T* data() const thread { return elems; }
};

} // namespace r0

namespace cuda {
namespace std {
using r0::array;
} // namespace std
} // namespace cuda
