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

// Included first by every generated keccak eval_check stage file ($OUT_DIR/metal/eval_stage_<k>.metal,
// written by keccak-sys/build_metal_eval.rs from the CUDA constraint code). The Metal counterpart of
// kernels/intel/prelude.h and kernels/cuda/eval_check_scratch_prelude.cuh: the scratch array in a
// device buffer, the polynomial mix powers and the carry between stages passed in an `EvalCtx`
// instead of `__constant__` globals, and the stage kernels' shared signature.
//
// Bindings of every eval_stage_<k> kernel (mirrored by EVAL_BUF_* in $OUT_DIR/keccak_metal_eval.rs):
//   buffer(0) check     device Fp*              check polynomial, check[domain * i + cycle]; last stage only
//   buffer(1) ctrl      const device Fp*        \
//   buffer(2) out       const device Fp*         |
//   buffer(3) data      const device Fp*         | poly_fp's inputs, column-major (c * domain + cycle)
//   buffer(4) mix       const device Fp*         |
//   buffer(5) accum     const device Fp*        /
//   buffer(6) poly_mix  const device FpExt*     kNumPolyMixPows powers of the mix (34.8 KB)
//   buffer(7) scratch   device Fp*              EVAL_SCRATCH_ELEMS * stride words
//   buffer(8) carry     device uint32_t*        EVAL_CARRY_WORDS * stride words
//   buffer(9) p         constant EvalParams&    set_bytes
// Dispatch `p.count` threads (thread t runs cycle p.first + t); every stage of a chunk, in order,
// before the next chunk reuses scratch and carry.

#pragma once

#include <metal_stdlib>

#include "fp.h"
#include "fpext.h"

using namespace metal;

// Threadgroup size limit of the stage kernels (the HAL may dispatch smaller groups).
#ifndef R0_KECCAK_EVAL_TG
#define R0_KECCAK_EVAL_TG 256
#endif

// How the generated functions are emitted. By default the compiler decides (with one function a
// stage, the default, each is inlined into its kernel); -DR0_KECCAK_EVAL_NOINLINE keeps them calls,
// as the Intel build does, for stage counts that put several large functions in one kernel.
#ifdef R0_KECCAK_EVAL_NOINLINE
#define R0_KECCAK_EVAL_FN static __attribute__((noinline))
#else
#define R0_KECCAK_EVAL_FN static inline
#endif

/// Per-dispatch parameters (set_bytes; 32 bytes, EVAL_PARAMS_SIZE on the Rust side).
struct EvalParams {
  /// Threads in this dispatch (cycles of this chunk); must be <= stride.
  uint32_t count;
  /// The chunk's first cycle.
  uint32_t first;
  /// Number of cycles (`size` in the constraint code); a power of two.
  uint32_t domain;
  /// Scratch/carry stride in words: the chunk capacity the buffers were sized for
  /// (<= EVAL_MAX_STRIDE).
  uint32_t stride;
  /// The vanishing quotient by cycle mod 4, as raw (Montgomery) Fp words, which is how the Rust
  /// side stores them. Read by the last stage only.
  uint32_t quot[4];
};
static_assert(sizeof(EvalParams) == 32, "EvalParams must be 32 bytes (EVAL_PARAMS_SIZE)");

namespace keccak_eval {

/// One thread's slice of the scratch buffer, which holds poly_fp's `Fp xN[M]` (8237 elements,
/// 33 KB a thread: far too large for thread memory). Element `i` of thread `t` is at
/// `scratch[i * stride + t]` (`base` already points at `scratch + t`), so the threads of a SIMD
/// group touching the same element touch consecutive words. 32-bit index math: the host keeps
/// `EVAL_SCRATCH_ELEMS * stride <= 2^32` (EVAL_MAX_STRIDE; any power of two up to 2^18).
struct ScratchRef {
  device Fp* base;
  uint32_t stride;

  device Fp& operator[](uint32_t i) const { return base[i * stride]; }
};

/// What the constraint functions take in place of CUDA's `__constant__` globals.
struct EvalCtx {
  const device FpExt* poly_mix;
  ScratchRef scratch;
  /// This thread's carry: word `w` at `carry[w * stride]`.
  device uint32_t* carry;
  uint32_t stride;
};

inline EvalCtx make_eval_ctx(const device FpExt* poly_mix,
                             device Fp* scratch,
                             device uint32_t* carry,
                             uint32_t stride,
                             uint32_t t) {
  EvalCtx ectx;
  ectx.poly_mix = poly_mix;
  ectx.scratch.base = scratch + t;
  ectx.scratch.stride = stride;
  ectx.carry = carry + t;
  ectx.stride = stride;
  return ectx;
}

// The constraint code is one chain of tail calls (poly_fp -> keccak_56 -> ... -> keccak_0), cut by
// build_metal_eval.rs into stages, one kernel each. At a cut, the call into the next stage's first
// function becomes `carry_store`s of its FpExt arguments (slot = their order among its FpExt
// parameters); that stage's kernel `carry_load`s them in the same order. Its other arguments are
// the scratch (already in device memory) and the kernel's own input buffers.

inline void carry_store(EvalCtx ectx, uint32_t slot, FpExt v) {
  for (uint32_t i = 0; i < 4; i++) {
    ectx.carry[(4 * slot + i) * ectx.stride] = v.elems[i].asRaw();
  }
}

inline FpExt carry_load(EvalCtx ectx, uint32_t slot) {
  FpExt v;
  for (uint32_t i = 0; i < 4; i++) {
    v.elems[i] = Fp::fromRaw(ectx.carry[(4 * slot + i) * ectx.stride]);
  }
  return v;
}

// K7 lazy multiply-accumulate (build_metal_eval.rs `lazy_plan`): a chain of
// `acc = acc + poly_mix[N] * x` keeps, per FpExt limb, a 64-bit sum T of raw (Montgomery) products,
// T == value * 2^32 (mod P) in raw terms, and Montgomery-reduces it once at the chain's end. The
// translator bounds T statically and folds it before it could overflow. Every helper is
// always_inline and straight-line on scalars (R17: no out-of-line call, no aggregate written
// element by element after a call).

/// Start of a chain from a reduced value: raw << 32 (< P * 2^32).
__attribute__((always_inline)) inline uint64_t lazy_start(Fp a) {
  return uint64_t(a.asRaw()) << 32;
}

/// The 64-bit product of two raw words (< (P - 1)^2).
__attribute__((always_inline)) inline uint64_t lazy_prod(Fp a, Fp b) {
  return uint64_t(a.asRaw()) * uint64_t(b.asRaw());
}

/// T -> T' < P * 2^32 with T' == T (mod P): the high word reduced mod P (< 2^32 < 3P).
__attribute__((always_inline)) inline uint64_t lazy_fold(uint64_t t) {
  uint32_t h = uint32_t(t >> 32);
  h = h >= Fp::P ? h - Fp::P : h;
  h = h >= Fp::P ? h - Fp::P : h;
  return (uint64_t(h) << 32) | (t & 0xffffffffull);
}

/// Montgomery reduction of any 64-bit T: the canonical Fp whose raw word is T / 2^32 mod P (as
/// fp.h's `mul` reduces its product).
__attribute__((always_inline)) inline Fp lazy_redc(uint64_t t) {
  uint64_t o64 = lazy_fold(t);
  uint32_t low = -uint32_t(o64);
  uint32_t red = Fp::M * low;
  o64 += uint64_t(red) * uint64_t(Fp::P);
  uint32_t ret = uint32_t(o64 >> 32);
  return Fp::fromRaw(ret >= Fp::P ? ret - Fp::P : ret);
}

/// The FpExt a chain ends in.
__attribute__((always_inline)) inline FpExt lazy_ext(uint64_t t0, uint64_t t1, uint64_t t2,
                                                     uint64_t t3) {
  return FpExt(lazy_redc(t0), lazy_redc(t1), lazy_redc(t2), lazy_redc(t3));
}

/// The last stage: check = tot * quot(cycle), as kernels/intel/kernels.h `launch_eval_stage`.
inline void write_check(device Fp* check, constant EvalParams& p, uint32_t cycle, FpExt tot) {
  Fp quot = Fp::fromRaw(p.quot[cycle & 3]);
  for (uint32_t i = 0; i < 4; i++) {
    check[p.domain * i + cycle] = tot.elems[i] * quot;
  }
}

} // namespace keccak_eval

// The shared stage kernel signature and prologue (see the bindings above).
#define R0_KECCAK_EVAL_KERNEL_ATTRS [[max_total_threads_per_threadgroup(R0_KECCAK_EVAL_TG)]]

#define R0_KECCAK_EVAL_KERNEL_PARAMS                                                               \
  device Fp *check [[buffer(0)]], const device Fp *ctrl [[buffer(1)]],                             \
      const device Fp *out [[buffer(2)]], const device Fp *data [[buffer(3)]],                     \
      const device Fp *mix [[buffer(4)]], const device Fp *accum [[buffer(5)]],                    \
      const device FpExt *poly_mix [[buffer(6)]], device Fp *scratch [[buffer(7)]],                \
      device uint32_t *carry [[buffer(8)]], constant EvalParams &p [[buffer(9)]],                  \
      uint32_t t [[thread_position_in_grid]]

#define R0_KECCAK_EVAL_KERNEL_PROLOGUE                                                             \
  if (t >= p.count) {                                                                              \
    return;                                                                                        \
  }                                                                                                \
  uint32_t cycle = p.first + t;                                                                    \
  keccak_eval::EvalCtx ectx = keccak_eval::make_eval_ctx(poly_mix, scratch, carry, p.stride, t);
