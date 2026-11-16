#!/usr/bin/env python3
# Copyright 2026 RISC Zero, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The Metal (MSL) form of the evalfast eval_check: regen.py --metal applies convert() to the
# chain's output after remat_lu.py (HIP_LAST, the same branch point as the ROCm kernel: no
# mksm2.py shared-memory slab words and no mksub.py splits) and writes eval_check_ef.metal plus
# ef_tables.rs into the build's OUT_DIR (docs/metal-port/PLAN.md 4.2, item R12). Every rewrite
# must apply exactly as often as pinned (COUNTS), so a generator change that moves one fails here.
#
#  - No inline asm: the C++ (#else) forms of the slab accesses (EF_SLAB_ASM) and of the 64-bit
#    multiply-add (EF_MADW_ASM).
#  - EfCtx: the chain functions (rv32im_v2_N, rk_accblk, poly_fp) take one `thread const EfCtx&`
#    instead of the trace / slab pointers and the unused poly_mix pointer. EfCtx holds the four
#    `const device` trace pointers (accum, data, out, mix), the `constant` poly_mix tables (pm,
#    pmn: CUDA's __constant__ ef_pm_raw / ef_pmn_raw, here one host-written buffer) and the
#    `thread` slab. Pointer identities come from args_hip.py's analysis (kernel args[] + call
#    sites through local aliases); a pointer parameter without a proven identity fails here.
#    Each function re-declares its former pointer parameters as locals from ctx, so the bodies
#    are unchanged.
#  - The slab is the kernel's 324-word `thread` array (RK_NWORDS); cycle / steps are 32-bit.
#  - EF_WIDE (64-bit lazy accumulators, CUDA's choice) is the default; -DEF_NO_WIDE selects the
#    32-bit lo/hi carry forms for the A/B (R15).
#  - poly_mix table (R15e): the table is a `const device` buffer (EF_TAB; on Apple GPUs ~20 %
#    faster than `constant` for this kernel, docs/metal-port/notes/R15e.md) and the NBETA-folded
#    rows (pmn) are computed from the pm row (-11 x by additions, ef_nbeta) instead of loaded,
#    which halves the table loads. -DEF_TAB_CONSTANT / -DEF_PMN_LOAD select R12's forms one by
#    one; -DEF_R12_FORM selects both and names the split kernel ef_eval_split_r12 (no
#    ef_combine), which the build compiles as a second TU (eval_check_ef_r12.metal) for the
#    runtime A/B switch RISC0_METAL_RV32IM_EVAL=evalfast_r12. The prelude's EfCtx and helpers
#    are in an anonymous namespace, so the TUs share no definition that differs.
#  - Table row loads (R18): every poly_mix row is read as half of a 32-byte `ulong4` load of
#    the aligned row pair (EF_ROW / ef_row_pair) instead of a 16-byte `uint4` load. Same words,
#    same results; on the M5 Max the split kernel takes ~55 % less time (docs/metal-port/notes/
#    R18.md). -DEF_ROW_U4 keeps the uint4 loads; -DEF_R15E_FORM selects R15e's whole form
#    (uint4 loads, split kernel ef_eval_split_r15e, no ef_combine), which the build compiles as a
#    third TU (eval_check_ef_r15e.metal) for RISC0_METAL_RV32IM_EVAL=evalfast_r15e. The pair
#    load needs the table bound at a 32-byte aligned address (the host binds it at offset 0).
#  - Domain function constant (R20): every chain function's row stride `rk_s` is
#    EF_FC_STEPS(steps), and the split kernel's own `domain` uses go through it too. By default
#    that is the runtime value (the plain kernel is unchanged). -DEF_FC_FORM makes it the
#    function constant `ef_fc_domain` (index 0) and names the split kernel ef_eval_split_fc (no
#    ef_combine); the build compiles that as a fourth TU (eval_check_ef_fc.metal) and the host
#    creates one PSO of it per job domain with MTLFunctionConstantValues (constant folding only:
#    the trace offsets `k * rk_s` and the row mask become immediates). A separate kernel rather
#    than an optional constant in ef_eval_split, because Metal refuses a PSO of a function with
#    function constants that was not created through newFunctionWithName:constantValues:.
#  - Flat chain (R22): the CUDA chain is a 21-deep nest of noinline functions, each ending in a
#    call of the next (rv32im_v2_19 -> 18 -> ... -> 3 under poly_fp, 2 -> 1 -> 0 under
#    rk_accblk); Apple's back end does no sibling-call optimisation, so every frame stays live
#    and each thread's hot stack spans all of them (~17 KB). By default the chain is flat: each
#    link takes (cycle, steps, ctx) only, reads its Fp arguments from a `thread` array of the
#    kernel (EfCtx::targs, EF_TARGS words) at entry, and instead of calling the next link writes
#    that link's arguments there and returns; poly_fp / rk_accblk call the links one after the
#    other. Same arithmetic in the same order, so bit-exact; one chain frame is live at a time
#    (native stack ~4 KB instead of ~19 KB), and the split kernel is ~7 % faster on the M5 Max
#    (docs/metal-port/notes/R22.md). -DEF_NEST_FORM keeps the nested calls (the TU's text is
#    then the pre-R22 one); the R12 and R15e forms always nest. The build compiles the nested
#    plain and specialized kernels as ef_eval_split_nest / ef_eval_split_fc_nest (TUs
#    eval_check_ef_nest.metal, eval_check_ef_fc_nest.metal) for RISC0_METAL_RV32IM_EVAL_FLAT=0.
#  - Trace loads issued ahead (R23, the Metal form of the ROCm post-pass hoist_hip.py): in every
#    chain function the body is cut into windows of HOIST_WINDOW statement lines, and each
#    distinct trace word (pointer, column, row offset) first read in window j is read into a
#    raw uint32 (EF_HL) at the start of window j - 1; its uses read that value (EF_HV). Same
#    words, same arithmetic in the same order: bit-exact. The split kernel is ~3.5 % faster on
#    the M5 Max (fc and plain, po2 18 and 20; docs/metal-port/notes/R23.md). -DEF_NO_HOIST
#    expands EF_HL to nothing and EF_HV to the load at its use (the R22 text token for token);
#    the nested, R12 and R15e forms never hoist. The build compiles the unhoisted flat plain and
#    specialized kernels as ef_eval_split_nohoist / ef_eval_split_fc_nohoist (TUs
#    eval_check_ef_nohoist.metal, eval_check_ef_fc_nohoist.metal) for
#    RISC0_METAL_RV32IM_EVAL_HOIST=0.
#  - Address spaces: helper reference parameters are `thread`; program-scope constants are
#    `constant constexpr`.
#  - ef_prep / ef_prep_n (one GPU thread on CUDA) are not ported: the host builds the 627-entry
#    poly_mix table, its NBETA-folded copy and the 4 vanishing quotients (PLAN 4.2 "replace").
#    rk_hoist (the hoisted out/mix terms) and the rk_qa / rk_qb product indices become Rust in
#    ef_tables.rs (straight-line BabyBear Elem / ExtElem code, same operation order), which
#    risc0-circuit-rv32im-sys includes.
#  - ef_eval_split / ef_combine become Metal kernels with explicit [[buffer(n)]] bindings
#    (ABI in the kernel comment below); the CUDA host code is dropped.
#
# usage: to_metal.py <in.cu> <out.metal> <out.rs>
import os
import re
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import args_hip  # noqa: E402

# The trace / slab identities (args_hip._analyze) -> EfCtx member.
TRACE = ('accum', 'data', 'out', 'mix')
CTX_HELPERS = ('rk_cl_acc', 'rk_cl0', 'rk_link_k', 'rk_link_nf_k')

# Rewrite counts (a change in the generated circuit or a CUDA pass moves one of these).
COUNTS = {
    'risc0_ns': 210,
    'device_inline_static': 36,
    'ctx_helper_calls': 5357,
    'ctx_helper_defs': 6,
    'thread_refs': 33,
    'chain_sigs': 106,
    'chain_calls': 20,
    'ptr_locals': 5,
    'args_uses': 0,
    'mask': 22,
    'fc_steps': 22,
    'flat_fn_sigs': 100,
    'flat_links': 20,
    'flat_roots': 2,
    'hoist_fns': 22,
    'hoist_loads': 3844,
}
# R23: statement lines per hoist window (hoist_loads()); loads first used in window j are issued at the
# start of window j - 1. ef_variant_bench at po2 20 / 18: windows 4-12 within ~2 ms of each
# other (6 best), 16 and 32 lose part of the gain, everything at the top is +22 % (R23 note).
HOIST_WINDOW = 6

PRELUDE = r'''// GENERATED by kernels/cuda/evalfast/regen.py --metal (to_metal.py) -- do not edit.
//
// rv32im eval_check, the Metal form of the evalfast kernel (docs/metal-port/PLAN.md 4.2, R12):
// the chain's output after remat_lu.py (as the ROCm kernel eval_check_ef_hip.cu) converted by
// evalfast/to_metal.py: no inline asm (the C++ forms of the slab accesses and of the 64-bit
// multiply-adds), the chain functions take one `thread const EfCtx&` (trace pointers, poly_mix
// tables, slab) instead of pointer arguments, a 324-word `thread` slab, 32-bit cycle/steps.
// Bit-exact with the baseline eval_check for canonical inputs (every Fp < P), which is what the
// prover passes; the lazy reductions rely on it.
#include <metal_stdlib>
#include "fp.h"
#include "fpext.h"
// The generated chains leave values and pointer locals unused.
#pragma clang diagnostic ignored "-Wunused-variable"
// EF_WIDE: 64-bit lazy accumulators (CUDA's default); -DEF_NO_WIDE: 32-bit lo/hi carries.
#if !defined(EF_WIDE) && !defined(EF_NO_WIDE)
#define EF_WIDE 1
#endif
#ifndef EF_THREADS
#define EF_THREADS 128
#endif
#ifndef EF_SPLIT_SHIFT
#define EF_SPLIT_SHIFT 1
#endif
#define EF_NOINLINE __attribute__((noinline))
#define EF_NPC 458u
#define EF_NPM (EF_NPC + 107u + 62u)
// poly_mix table forms (R15e): EF_TAB is the table's address space (default `const device`,
// -DEF_TAB_CONSTANT: `constant`); EF_PMN_ROW(ctx, b, k) is the NBETA-folded row k (default
// computed from the pm row b, -DEF_PMN_LOAD: loaded from the table's second half).
// -DEF_R12_FORM: both of R12's forms, split kernel named ef_eval_split_r12, no ef_combine.
#ifdef EF_R12_FORM
#define EF_TAB_CONSTANT 1
#define EF_PMN_LOAD 1
#define EF_ROW_U4 1
#endif
// R18: -DEF_R15E_FORM is R15e's form (16-byte uint4 row loads, kernel ef_eval_split_r15e).
#ifdef EF_R15E_FORM
#define EF_ROW_U4 1
#endif
// R22: the flat chain is the default; -DEF_NEST_FORM keeps the nested chain calls. R12's and
// R15e's forms (revert kernels of older items) always nest.
#if (defined(EF_R12_FORM) || defined(EF_R15E_FORM)) && !defined(EF_NEST_FORM)
#define EF_NEST_FORM 1
#endif
#ifdef EF_TAB_CONSTANT
#define EF_TAB constant
#else
#define EF_TAB const device
#endif
#ifdef EF_PMN_LOAD
#define EF_PMN_ROW(ctx, b, k) ((ctx).pmn[(k)])
#else
#define EF_PMN_ROW(ctx, b, k) (ef_nbeta(b))
#endif
// R20: -DEF_FC_FORM: the evaluation domain is function constant 0 (`ef_fc_domain`, set per job
// po2 by the host, eval_check.rs) and the split kernel is ef_eval_split_fc; otherwise
// EF_FC_STEPS(s) is the runtime value s.
#if defined(EF_FC_FORM) && (defined(EF_R12_FORM) || defined(EF_R15E_FORM))
#error "EF_FC_FORM is a form of the default kernel"
#endif
#ifdef EF_FC_FORM
constant uint ef_fc_domain [[function_constant(0)]];
#define EF_FC_STEPS(s) (ef_fc_domain)
#else
#define EF_FC_STEPS(s) ((uint32_t)(s))
#endif
// R23: trace loads issued ahead of their use (to_metal.py hoist_loads()). EF_HL(n, p, k, j) reads the
// raw word of column k of p at row offset j into n, EF_HV(n, p, k, j) is that word as an Fp.
// -DEF_NO_HOIST (implied by the nested, R12 and R15e forms) keeps every load at its use: EF_HL
// is empty and EF_HV the R22 load expression.
#if defined(EF_NEST_FORM) && !defined(EF_NO_HOIST)
#define EF_NO_HOIST 1
#endif
#ifdef EF_NO_HOIST
#define EF_HL(n, p, k, j)
#define EF_HV(n, p, k, j) EF_LD(p, (uint32_t)(k * rk_s + ((rk_c - j) & rk_m)))
#else
#define EF_HL(n, p, k, j) const uint32_t n = reinterpret_cast<const device uint32_t*>(p)[(uint32_t)(k * rk_s + ((rk_c - j) & rk_m))];
#define EF_HV(n, p, k, j) (Fp::fromRaw(n))
#endif
namespace {
// Per-thread context of the chain functions (to_metal.py): the trace columns (read-only for the
// whole kernel, which the rematerialisation relies on), the poly_mix table (EF_NPM FpExt as raw
// words, then its NBETA-folded copy) and the per-row slab.
struct EfCtx {
  const device Fp* accum;
  const device Fp* data;
  const device Fp* out;
  const device Fp* mix;
  EF_TAB uint4* pm;
  EF_TAB uint4* pmn;
  thread Fp* slab;
#ifndef EF_NEST_FORM
  // R22: the flat chain's link arguments (EF_TARGS words of the kernel's `thread` array).
  thread Fp* targs;
#endif
};
#define EF_LD(p, i) (Fp::fromRaw(reinterpret_cast<const device uint32_t*>(p)[(i)]))
#define EF_PM(i) (ef_pm_at(ctx.pm, (i)))
// Row i of the table (R18): the half of the 32-byte aligned row pair (2 (i / 2), 2 (i / 2) + 1)
// that holds it, read with one `ulong4` load (the chain's indices are compile-time constants,
// so the select is resolved at compile time). Same words as `tab[i]`; the 32-byte loads are
// much cheaper than 16-byte `uint4` loads on Apple GPUs for this kernel (R18). -DEF_ROW_U4: the
// plain uint4 load (R15e).
#ifdef EF_ROW_U4
#define EF_ROW(p, i) ((p)[(i)])
#else
inline uint4 ef_row_pair(EF_TAB uint4* p, uint32_t i) {
  const ulong4 v = reinterpret_cast<EF_TAB ulong4*>(p)[i >> 1];
  return (i & 1u) ? uint4(as_type<uint2>(v.z), as_type<uint2>(v.w)) : uint4(as_type<uint2>(v.x), as_type<uint2>(v.y));
}
#define EF_ROW(p, i) (ef_row_pair((p), (i)))
#endif
inline FpExt ef_pm_at(EF_TAB uint4* pm, uint32_t i) {
  const uint4 v = EF_ROW(pm, i);
  return FpExt(Fp::fromRaw(v.x), Fp::fromRaw(v.y), Fp::fromRaw(v.z), Fp::fromRaw(v.w));
}
// (P - 11) * x for a canonical x, canonical: -(8x + 2x + x) with Fp's reducing add / sub. The
// result is the unique canonical value, so it equals the host's `x * (P - 11)` bit for bit.
inline uint32_t ef_m11(uint32_t x) {
  const Fp a = Fp::fromRaw(x), a2 = a + a, a4 = a2 + a2, a8 = a4 + a4;
  return (Fp(0) - (a8 + a2 + a)).asRaw();
}
// The NBETA-folded row of the pm row b: (b.y, b.z, b.w) * (P - 11), 0 (the table's pmn half).
inline uint4 ef_nbeta(uint4 b) {
  return uint4(ef_m11(b.y), ef_m11(b.z), ef_m11(b.w), 0u);
}
} // namespace
'''

SLAB_CUDA = '''#ifdef EF_SLAB_ASM
__device__ __forceinline__ uint32_t ef_ldl(const void* p, uint32_t w) { uint32_t v; uint32_t a = (uint32_t)__cvta_generic_to_local(p) + 4u * w; asm volatile("ld.local.u32 %0, [%1];" : "=r"(v) : "r"(a)); return v; }
__device__ __forceinline__ void ef_stl(void* p, uint32_t w, uint32_t v) { uint32_t a = (uint32_t)__cvta_generic_to_local(p) + 4u * w; asm volatile("st.local.u32 [%0], %1;" :: "r"(a), "r"(v) : "memory"); }
#define RK_LD1(p, w) (static_cast<const ::risc0::Fp>(::risc0::Fp::fromRaw(ef_ldl((p), (w)))))
#define RK_ST1(p, w, v) ef_stl((p), (w), ::risc0::Fp(v).asRaw())
#else
#define RK_LD1(p, w) (static_cast<const ::risc0::Fp>(::risc0::Fp::fromRaw(reinterpret_cast<const uint32_t*>(p)[(size_t)(w) * RK_STRIDE])))
#define RK_ST1(p, w, v) (reinterpret_cast<uint32_t*>(p)[(size_t)(w) * RK_STRIDE] = ::risc0::Fp(v).asRaw())
#endif
'''
SLAB_METAL = '''// The slab is the kernel's `thread` array (no EF_SLAB_ASM on Metal).
#define RK_LD1(p, w) (static_cast<const ::risc0::Fp>(::risc0::Fp::fromRaw(reinterpret_cast<const thread uint32_t*>(p)[(uint32_t)(w) * RK_STRIDE])))
#define RK_ST1(p, w, v) (reinterpret_cast<thread uint32_t*>(p)[(uint32_t)(w) * RK_STRIDE] = ::risc0::Fp(v).asRaw())
'''
MADW_CUDA = '''#ifdef EF_MADW_ASM
  uint64_t d; asm("mad.wide.u32 %0, %1, %2, %3;" : "=l"(d) : "r"(a), "r"(b), "l"(c)); return d;
#else
  return c + uint64_t(a) * uint64_t(b);
#endif
'''
MADW_METAL = '''  return c + uint64_t(a) * uint64_t(b);
'''

KERNELS = r'''
static_assert(EF_NPM == EF_NPC + rv32im_metal_ef::RK_QN + rv32im_metal_ef::RK_HN, "poly_mix table size");
static_assert(EF_SPLIT_SHIFT == 1, "ef_eval_split has two phases: main (even groups) and LogUp (odd)");

// Table buffer (host-built, R12 eval_check.rs): uint4 tab[2 * EF_NPM + 1]:
//   [0, EF_NPM)          pm: the poly_mix table (EF_NPC powers, RK_QN products, RK_HN hoisted
//                        out/mix terms) as raw FpExt words (CUDA ef_pm_raw);
//   [EF_NPM, 2 EF_NPM)   pmn: (pm.y, pm.z, pm.w) * (P - 11), 0 (CUDA ef_pmn_raw);
//   2 EF_NPM             the 4 vanishing quotients 1/((3 rou^c)^(2^po2) - 1), c = cycle & 3, raw
//                        (CUDA ef_qtab).
//
// ef_eval_split: buffers 0 check (device Fp*, 4*domain), 1 data, 2 accum, 3 mix, 4 out
// (const device Fp*), 5 tab (EF_TAB uint4*), 6 scratch (device Fp*, 4*domain: the LogUp
// partials), 7 domain (constant uint&); tab must be 32-byte aligned (R18 row-pair loads).
// Threadgroups of EF_THREADS threads; group 2r evaluates the main chain (poly_fp) of rows
// [r*EF_THREADS, (r+1)*EF_THREADS) into check, group 2r+1 the LogUp chain (rk_accblk) of the
// same rows into scratch.
#if defined(EF_R12_FORM)
#define EF_SPLIT_KERNEL ef_eval_split_r12
#elif defined(EF_R15E_FORM)
#define EF_SPLIT_KERNEL ef_eval_split_r15e
#elif defined(EF_FC_FORM) && defined(EF_NEST_FORM)
#define EF_SPLIT_KERNEL ef_eval_split_fc_nest
#elif defined(EF_FC_FORM) && defined(EF_NO_HOIST)
#define EF_SPLIT_KERNEL ef_eval_split_fc_nohoist
#elif defined(EF_FC_FORM)
#define EF_SPLIT_KERNEL ef_eval_split_fc
#elif defined(EF_NEST_FORM)
#define EF_SPLIT_KERNEL ef_eval_split_nest
#elif defined(EF_NO_HOIST)
#define EF_SPLIT_KERNEL ef_eval_split_nohoist
#else
#define EF_SPLIT_KERNEL ef_eval_split
#endif
[[max_total_threads_per_threadgroup(EF_THREADS)]]
kernel void EF_SPLIT_KERNEL(device Fp* check            [[buffer(0)]],
                          const device Fp* data       [[buffer(1)]],
                          const device Fp* accum      [[buffer(2)]],
                          const device Fp* mix        [[buffer(3)]],
                          const device Fp* out        [[buffer(4)]],
                          EF_TAB uint4* tab           [[buffer(5)]],
                          device Fp* scratch          [[buffer(6)]],
                          constant uint& domain       [[buffer(7)]],
                          uint tid                    [[thread_position_in_threadgroup]],
                          uint bid                    [[threadgroup_position_in_grid]],
                          uint bdim                   [[threads_per_threadgroup]]) {
  const uint32_t rb = bid >> EF_SPLIT_SHIFT, ph = bid & ((1u << EF_SPLIT_SHIFT) - 1u);
  const uint32_t cycle = bdim * rb + tid;
  const uint32_t dom = EF_FC_STEPS(domain);
  if (cycle >= dom) return;
  uint32_t lslab[RK_NWORDS];
  EfCtx ctx;
  ctx.accum = accum;
  ctx.data = data;
  ctx.out = out;
  ctx.mix = mix;
  ctx.pm = tab;
  ctx.pmn = tab + EF_NPM;
  ctx.slab = reinterpret_cast<thread Fp*>(lslab);
#ifndef EF_NEST_FORM
  uint32_t ltargs[EF_TARGS];
  ctx.targs = reinterpret_cast<thread Fp*>(ltargs);
#endif
  FpExt t;
  device Fp* dst;
  if (ph == 0) {
    t = rv32im_metal_ef::poly_fp(cycle, dom, ctx);
    dst = check;
  } else {
    t = rv32im_metal_ef::rk_accblk(cycle, dom, ctx);
    dst = scratch;
  }
  for (uint32_t i = 0; i < 4; i++) dst[i * dom + cycle] = t.elems[i];
}

#if !defined(EF_NEST_FORM) && !defined(EF_FC_FORM) && !defined(EF_NO_HOIST)
// ef_combine: buffers 0 check (device Fp*), 1 scratch (const device Fp*), 2 tab (constant
// uint4*), 3 domain (constant uint&); one thread per row: check = (check + scratch) * quotient.
[[max_total_threads_per_threadgroup(256)]]
kernel void ef_combine(device Fp* check          [[buffer(0)]],
                       const device Fp* scratch  [[buffer(1)]],
                       constant uint4* tab       [[buffer(2)]],
                       constant uint& domain     [[buffer(3)]],
                       uint cycle                [[thread_position_in_grid]]) {
  if (cycle >= domain) return;
  const uint4 q = tab[2 * EF_NPM];
  const uint32_t c = cycle & 3u;
  const Fp quot = Fp::fromRaw(c == 0 ? q.x : c == 1 ? q.y : c == 2 ? q.z : q.w);
  for (uint32_t i = 0; i < 4; i++) {
    const Fp a = Fp::fromRaw(reinterpret_cast<const device uint32_t*>(check)[i * domain + cycle]);
    const Fp b = Fp::fromRaw(reinterpret_cast<const device uint32_t*>(scratch)[i * domain + cycle]);
    check[i * domain + cycle] = (a + b) * quot;
  }
}
#endif
'''

RUST_HEADER = '''// GENERATED by kernels/cuda/evalfast/regen.py --metal (to_metal.py) -- do not edit.
//
// Host half of the Metal evalfast eval_check (R12): the CUDA ef_prep kernel's table inputs, so
// the host can build the poly_mix table (EF_NPM entries) that eval_check_ef.metal reads.
// `rk_hoist` is the generated circuit's rk_hoist (same statements, same order) over BabyBear.
'''


class Pins:
    def __init__(self):
        self.counts = {}

    def sub(self, s, old, new, key, count=None):
        n = s.count(old)
        want = COUNTS[key] if count is None else count
        if n != want:
            sys.exit('to_metal: %s: expected %d of %r, found %d' % (key, want, old[:80], n))
        self.counts[key] = self.counts.get(key, 0) + n
        return s.replace(old, new)

    def resub(self, s, pat, new, key, count=None, flags=0):
        s2, n = re.subn(pat, new, s, flags=flags)
        want = COUNTS[key] if count is None else count
        if n != want:
            sys.exit('to_metal: %s: expected %d of %r, found %d' % (key, want, pat[:80], n))
        self.counts[key] = self.counts.get(key, 0) + n
        return s2


def _rust_hoist(body):
    """rk_hoist's statements as Rust over risc0_core's BabyBear Elem / ExtElem."""
    types = {}
    out = []

    def lit(n):
        return 'Elem::new(%su32)' % n

    def term(t):
        t = t.strip()
        m = re.fullmatch(r'FpExt\(Fp\((\d+)u\),Fp\((\d+)u\),Fp\((\d+)u\),Fp\((\d+)u\)\)', t)
        if m:
            return 'ExtElem::new(%s)' % ', '.join(lit(g) for g in m.groups()), 'e'
        m = re.fullmatch(r'Fp\((\d+)u\)', t)
        if m:
            return lit(m.group(1)), 'f'
        if t == 'FpExt(0)':
            return '<ExtElem as risc0_core::field::Elem>::ZERO', 'e'
        m = re.fullmatch(r'(out|mix|poly_mix)\[(\d+)\]', t)
        if m:
            return '%s[%s]' % m.groups(), 'e' if m.group(1) == 'poly_mix' else 'f'
        m = re.fullmatch(r'h(\d+)', t)
        if m:
            return t, types[t]
        sys.exit('to_metal: rk_hoist: unknown term %r' % t)

    def expr(e):
        e = e.strip()
        # left-associative over + - (top level); * binds tighter
        parts = re.split(r' ([+-]) ', e)
        if len(parts) > 1:
            acc, ty = expr_mul(parts[0])
            for op, rhs in zip(parts[1::2], parts[2::2]):
                r, rty = expr_mul(rhs)
                acc = '(%s %s %s)' % (acc, op, r)
                ty = 'e' if 'e' in (ty, rty) else 'f'
            return acc, ty
        return expr_mul(e)

    def expr_mul(e):
        parts = e.split(' * ')
        acc, ty = term(parts[0])
        for rhs in parts[1:]:
            r, rty = term(rhs)
            acc = '(%s * %s)' % (acc, r)
            ty = 'e' if 'e' in (ty, rty) else 'f'
        return acc, ty

    for line in body.strip().split('\n'):
        line = line.strip()
        m = re.fullmatch(r'(Fp|FpExt) (h\d+) = (.*);', line)
        if m:
            want = 'e' if m.group(1) == 'FpExt' else 'f'
            e, ty = expr(m.group(3))
            if ty != want:
                sys.exit('to_metal: rk_hoist: %s is %s but its expression is %s' % (m.group(2), want, ty))
            types[m.group(2)] = want
            out.append('    let %s = %s;' % (m.group(2), e))
            continue
        m = re.fullmatch(r'tab\[(\d+)\] = (FpExt\((h\d+)\)|h\d+);', line)
        if m:
            if m.group(3):
                if types[m.group(3)] != 'f':
                    sys.exit('to_metal: rk_hoist: FpExt(%s) of an FpExt' % m.group(3))
                out.append('    tab[%s] = ExtElem::from(%s);' % (m.group(1), m.group(3)))
            else:
                if types[m.group(2)] != 'e':
                    sys.exit('to_metal: rk_hoist: tab[] = Fp %s' % m.group(2))
                out.append('    tab[%s] = %s;' % (m.group(1), m.group(2)))
            continue
        sys.exit('to_metal: rk_hoist: unknown statement %r' % line)
    return out


LINK_SIG = re.compile(r'^(EF_NOINLINE )?FpExt (rv32im_v2_\d+)\(uint32_t cycle, uint32_t steps, '
                      r'thread const EfCtx& ctx((?:, Fp \w+)*)\)( \{|;)$', re.M)
LINK_CALL = re.compile(r'^  auto (x\d+) = (rv32im_v2_\d+)\(cycle, steps, ctx((?:, [\w.\[\]]+)*)\);\n'
                       r'  return \1;\n', re.M)
ROOT_DEFS = {
    'poly_fp': 'FpExt poly_fp(uint32_t cycle, uint32_t steps, thread const EfCtx& ctx) {\n',
    'rk_accblk': 'EF_NOINLINE FpExt rk_accblk(uint32_t cycle, uint32_t steps, thread const EfCtx& ctx) {\n',
}


def flatten(core, p):
    """R22: the flat chain (module comment). Emits both forms under #ifndef EF_NEST_FORM, so the
    nested form's text is the input's."""
    # Which function ends in a call of which link (each link is called once, at the end of its
    # caller): the chains from the two roots.
    starts = sorted([(m.start(), m.group(2)) for m in LINK_SIG.finditer(core) if m.group(4) == ' {'] +
                    [(core.index(d), r) for r, d in ROOT_DEFS.items()])
    nxt = {}
    for m in LINK_CALL.finditer(core):
        owner = max(x for x in starts if x[0] < m.start())[1]
        if owner in nxt:
            sys.exit('to_metal: flat: %s calls two links' % owner)
        nxt[owner] = m.group(2)
    chains = {}
    for r in ROOT_DEFS:
        chains[r] = [r]
        while chains[r][-1] in nxt:
            chains[r].append(nxt[chains[r][-1]])
    links = [f for c in chains.values() for f in c[1:]]
    if len(set(links)) != len(links) or len(links) != len(starts) - len(ROOT_DEFS):
        sys.exit('to_metal: flat: the links do not form two chains: %s' % chains)
    head = {c[1]: c for c in chains.values()}
    void = {f for c in chains.values() for f in c[1:-1]}
    nargs = [0]
    nsig = [0]

    def sig(m):
        params = [q.strip()[len('Fp '):] for q in m.group(3).split(',') if q.strip()]
        nargs[0] = max(nargs[0], len(params))
        nsig[0] += 1
        f, decl = m.group(2), m.group(4) == ';'
        flat_ret = 'void' if f in void else 'FpExt'
        flat = '%s%s %s(uint32_t cycle, uint32_t steps, thread const EfCtx& ctx)%s' % (
            m.group(1) or '', flat_ret, f, m.group(4))
        if not decl:
            flat += ''.join('\n  const Fp %s = ctx.targs[%d];' % (q, i) for i, q in enumerate(params))
        return '#ifdef EF_NEST_FORM\n%s\n#else\n%s\n#endif' % (m.group(0), flat)

    core = LINK_SIG.sub(sig, core)
    if nsig[0] != COUNTS['flat_fn_sigs']:
        sys.exit('to_metal: flat_fn_sigs: expected %d, found %d' % (COUNTS['flat_fn_sigs'], nsig[0]))
    p.counts['flat_fn_sigs'] = nsig[0]
    nlink, nroot = [0], [0]

    def call(m):
        args = [a.strip() for a in m.group(3).split(',') if a.strip()]
        flat = ''.join('  ctx.targs[%d] = %s;\n' % (i, a) for i, a in enumerate(args))
        g = m.group(2)
        nlink[0] += 1
        if g in head:
            nroot[0] += 1
            c = head[g]
            flat += ''.join('  %s(cycle, steps, ctx);\n' % f for f in c[1:-1])
            flat += '  return %s(cycle, steps, ctx);\n' % c[-1]
        else:
            flat += '  return;\n'
        return '#ifdef EF_NEST_FORM\n%s#else\n%s#endif\n' % (m.group(0), flat)

    core = LINK_CALL.sub(call, core)
    if nlink[0] != COUNTS['flat_links'] or nroot[0] != COUNTS['flat_roots']:
        sys.exit('to_metal: flat_links / flat_roots: expected %d / %d, found %d / %d'
                 % (COUNTS['flat_links'], COUNTS['flat_roots'], nlink[0], nroot[0]))
    p.counts['flat_links'] = nlink[0]
    p.counts['flat_roots'] = nroot[0]
    # Link arguments: the largest link's Fp count.
    ns = 'namespace rv32im_metal_ef {\n'
    core = core.replace(ns, '#define EF_TARGS %du\n%s' % (nargs[0], ns), 1)
    return core


HOIST_LD = re.compile(r'EF_LD\((\w+(?:\.\w+)?), \(uint32_t\)\((\d+u) \* rk_s \+ \(\(rk_c - (\d+u)\) & rk_m\)\)\)')
HOIST_BODY = 'const uint32_t rk_s = EF_FC_STEPS(steps)'


def hoist_loads(core, window=HOIST_WINDOW):
    """R23: trace loads issued one window ahead (module comment; the Metal form of hoist_hip.py).
    A chain function's body runs from its rk_s line to its closing '}' line (shared by the flat
    and nested signatures); declarations go before a top-level statement line that follows a
    complete one."""
    lines = core.split('\n')
    out = []
    nfn = nld = 0
    i = 0

    def name(key):
        return 'efl_%s_%s_%s' % (re.sub(r'\W', '', key[0]), key[1][:-1], key[2][:-1])

    def stmt_start(body, h, inpp):
        b = body[h]
        return (b.startswith('  ') and not b.startswith('   ') and not inpp[h]
                and (h == 0 or body[h - 1].rstrip().endswith((';', '{'))))
    while i < len(lines):
        if HOIST_BODY not in lines[i]:
            out.append(lines[i])
            i += 1
            continue
        end = lines.index('}', i + 1)
        body = lines[i + 1:end]
        # lines inside a conditional block (R22's #ifdef EF_NEST_FORM call sites) take no
        # declarations: both forms must see every one
        inpp, depth = [], 0
        for b in body:
            if b.startswith('#if'):
                depth += 1
            elif b.startswith('#endif'):
                depth -= 1
            inpp.append(depth > 0 or b.startswith('#'))
        first = {}
        for k, b in enumerate(body):
            for m in HOIST_LD.finditer(b):
                if inpp[k]:
                    sys.exit('to_metal: hoist: a trace load in a conditional block: %r' % b)
                first.setdefault(m.groups(), k)
        decl = {}
        for key, k in first.items():
            h = max(0, (k // window - 1) * window)
            while h > 0 and not stmt_start(body, h, inpp):
                h -= 1
            decl.setdefault(h, []).append('  EF_HL(%s, %s, %s, %s)' % ((name(key),) + key))
        out.append(lines[i])
        for k, b in enumerate(body):
            out.extend(decl.get(k, []))
            out.append(HOIST_LD.sub(lambda m: 'EF_HV(%s, %s, %s, %s)' % ((name(m.groups()),) + m.groups()), b))
        out.append('}')
        nfn += 1
        nld += len(first)
        i = end + 1
    if nfn != COUNTS['hoist_fns'] or nld != COUNTS['hoist_loads']:
        sys.exit('to_metal: hoist: expected %d functions / %d loads, found %d / %d'
                 % (COUNTS['hoist_fns'], COUNTS['hoist_loads'], nfn, nld))
    out = '\n'.join(out)
    if HOIST_LD.search(out):
        sys.exit('to_metal: hoist: a trace load outside the chain bodies')
    return out


def convert(s, stats=None):
    p = Pins()
    # ---- identities of the pointer parameters (before any rewrite) ----
    defs, ident = args_hip._analyze(s)
    kern = s[s.index('__global__ void ef_eval_split('):]
    roots = re.search(r'Fp\* args\[5\] = \{(.*?)\};', kern).group(1).split(', ')
    if roots != ['accum', 'data', 'out', 'mix', 'slab']:
        sys.exit('to_metal: kernel args[] changed: %s' % roots)

    # ---- host tables -> Rust ----
    qa = re.search(r'__device__ const uint16_t rk_qa\[\] = \{([\d,]+)\};\n', s)
    qb = re.search(r'__device__ const uint16_t rk_qb\[\] = \{([\d,]+)\};\n', s)
    qn = int(re.search(r'constexpr uint32_t RK_QN = (\d+);', s).group(1))
    hn = int(re.search(r'constexpr uint32_t RK_HN = (\d+);', s).group(1))
    npc = int(re.search(r'#define EF_NPC (\d+)u', s).group(1))
    qa_v, qb_v = qa.group(1).split(','), qb.group(1).split(',')
    if len(qa_v) != qn or len(qb_v) != qn:
        sys.exit('to_metal: rk_qa/rk_qb have %d/%d entries, RK_QN %d' % (len(qa_v), len(qb_v), qn))
    s = s.replace(qa.group(0), '').replace(qb.group(0), '')
    hm = re.search(r'__device__ EF_NOINLINE void rk_hoist\(const FpExt\* poly_mix, const Fp\* out, '
                   r'const Fp\* mix, FpExt\* tab\) \{\n(.*?)\n\}\n', s, re.S)
    hoist = _rust_hoist(hm.group(1))
    ntab = len(re.findall(r'tab\[\d+\] = ', hm.group(1)))
    if ntab != hn:
        sys.exit('to_metal: rk_hoist writes %d entries, RK_HN %d' % (ntab, hn))
    s = s.replace(hm.group(0), '')
    rust = [RUST_HEADER,
            'use risc0_core::field::baby_bear::{Elem, ExtElem};',
            '',
            '/// poly_mix powers in the table (`EF_NPC`).',
            'pub const EF_NPC: usize = %d;' % npc,
            '/// Products of two powers (`RK_QN`): `tab[EF_NPC + i] = pm[RK_QA[i]] * pm[RK_QB[i]]`.',
            'pub const RK_QN: usize = %d;' % qn,
            '/// Hoisted out/mix terms (`RK_HN`), written by [rk_hoist].',
            'pub const RK_HN: usize = %d;' % hn,
            '/// Table entries (`EF_NPM`).',
            'pub const EF_NPM: usize = EF_NPC + RK_QN + RK_HN;',
            'pub const RK_QA: [u16; RK_QN] = [%s];' % ', '.join(qa_v),
            'pub const RK_QB: [u16; RK_QN] = [%s];' % ', '.join(qb_v),
            '',
            '/// `tab[0..RK_HN]` from the poly_mix table so far (`poly_mix`, the first EF_NPC + RK_QN',
            '/// entries), the circuit outputs and the mix values.',
            '#[allow(clippy::all, unused_parens, unused_variables)]',
            'pub fn rk_hoist(poly_mix: &[ExtElem], out: &[Elem], mix: &[Elem], tab: &mut [ExtElem]) {',
            '    assert!(tab.len() >= RK_HN);']
    rust += hoist
    rust += ['}', '']

    # ---- cut the CUDA head and the host tail ----
    ns_open = 'namespace risc0::circuit::rv32im_v2 {\n'
    ns_close = '} // namespace\n} // namespace risc0::circuit::rv32im_v2\n'
    head_end = s.index('#define EF_PM(i) ')
    head_end = s.index('\n', head_end) + 1
    core = s[head_end:s.index(ns_close) + len(ns_close)]
    if not core.startswith(ns_open):
        sys.exit('to_metal: unexpected text after EF_PM: %r' % core[:80])
    core = p.sub(core, ns_open, 'namespace rv32im_metal_ef {\n', 'ns', 1)
    core = p.sub(core, ns_close, '} // namespace\n} // namespace rv32im_metal_ef\n', 'ns', 1)

    # ---- no asm ----
    core = p.sub(core, SLAB_CUDA, SLAB_METAL, 'slab', 1)
    core = p.sub(core, MADW_CUDA, MADW_METAL, 'madw', 1)
    # ---- poly_mix tables through ctx ----
    core = p.sub(core, 'reinterpret_cast<const uint4*>(ef_pm_raw)[k], n = reinterpret_cast<const uint4*>(ef_pmn_raw)[k];',
                 'EF_ROW(ctx.pm, k), n = EF_PMN_ROW(ctx, b, k);', 'pm_rows', 2)
    # ---- program-scope constants ----
    core = p.sub(core, 'constexpr size_t kInvRate = 4;\n', '', 'kinvrate', 1)
    core = p.resub(core, r'^constexpr uint32_t (RK_QN|RK_HN) = ', r'constant constexpr uint32_t \1 = ', 'rkn', 2,
                   flags=re.M)

    # ---- helpers that read the tables get ctx ----
    core = p.resub(core, r'\b(%s)\(' % '|'.join(CTX_HELPERS), r'\1(ctx, ', 'ctx_helper_calls')
    core = p.resub(core, r'^((?:template <typename A> )?__device__ __forceinline__ static \w+ (?:%s))\(ctx, '
                   % '|'.join(CTX_HELPERS), r'\1(thread const EfCtx& ctx, ', 'ctx_helper_defs', flags=re.M)

    # ---- helper reference parameters -> thread ----
    def thread_refs(m):
        line = m.group(0)
        sig_end = line.index(')')
        sig, rest = line[:sig_end], line[sig_end:]
        sig, n = re.subn(r'(?<![\w&])((?:const )?\w+)& ', r'thread \1& ', sig)
        thread_refs.n += n
        return sig + rest
    thread_refs.n = 0
    core = re.sub(r'^(?:template <typename A> )?__device__ __forceinline__ static .*$', thread_refs, core, flags=re.M)
    if thread_refs.n != COUNTS['thread_refs']:
        sys.exit('to_metal: thread_refs: expected %d, found %d' % (COUNTS['thread_refs'], thread_refs.n))
    p.counts['thread_refs'] = thread_refs.n

    # ---- chain functions: ctx instead of pointers, 32-bit cycle/steps ----
    drop, local = {}, {}
    for f, ps in defs.items():
        names = [args_hip._pname(q) for q in ps]
        drop[f], local[f] = set(), []
        for i, (q, n) in enumerate(zip(ps, names)):
            if i < 3:
                continue  # cycle, steps, poly_mix
            if '*' not in q:
                continue
            if f == 'poly_fp':
                if q != 'Fp** args':
                    sys.exit('to_metal: poly_fp parameter %r' % q)
                drop[f].add(i)
                continue
            a = ident.get((f, n))
            if a is None:
                sys.exit('to_metal: %s: no identity for pointer parameter %r' % (f, q))
            drop[f].add(i)
            if q.startswith('Fp* ') and a in TRACE:
                local[f].append('  const device Fp* %s = ctx.%s;' % (n, a))
            elif q.startswith('Fp* ') and a == 'slab':
                local[f].append('  thread Fp* %s = ctx.slab;' % n)
            elif q.startswith('FpExt* ') and a == 'slab':
                local[f].append('  thread FpExt* %s = reinterpret_cast<thread FpExt*>(ctx.slab);' % n)
            else:
                sys.exit('to_metal: %s: parameter %r is %s' % (f, q, a))
    head3 = 'size_t cycle, size_t steps, FpExt* poly_mix'
    nsig = [0]

    def sig(m):
        f = m.group(2)
        ps = [q.strip() for q in m.group(3).split(',')]
        if ', '.join(ps[:3]) != head3:
            sys.exit('to_metal: %s signature starts %r' % (f, ps[:3]))
        nsig[0] += 1
        keep = ['uint32_t cycle', 'uint32_t steps', 'thread const EfCtx& ctx']
        keep += [q for i, q in enumerate(ps) if i >= 3 and i not in drop[f]]
        tail = m.group(4)
        if tail == ' {':
            tail += ''.join('\n' + l for l in local[f])
        pre = m.group(1).replace('__device__ ', '')
        return '%sFpExt %s(%s)%s' % (pre, f, ', '.join(keep), tail)

    core = args_hip.FNSIG.sub(sig, core)
    if nsig[0] != COUNTS['chain_sigs']:
        sys.exit('to_metal: chain_sigs: expected %d, found %d' % (COUNTS['chain_sigs'], nsig[0]))
    p.counts['chain_sigs'] = nsig[0]
    ncall = [0]

    def call(m):
        g = m.group(2)
        a = m.group(3).split(', ')
        if a[:3] != ['cycle', 'steps', 'poly_mix']:
            sys.exit('to_metal: call of %s with %r' % (g, a[:3]))
        ncall[0] += 1
        keep = ['cycle', 'steps', 'ctx'] + [x for i, x in enumerate(a) if i >= 3 and i not in drop[g]]
        return '%s%s(%s);' % (m.group(1), g, ', '.join(keep))

    core = args_hip.CALL.sub(call, core)
    if ncall[0] != COUNTS['chain_calls']:
        sys.exit('to_metal: chain_calls: expected %d, found %d' % (COUNTS['chain_calls'], ncall[0]))
    p.counts['chain_calls'] = ncall[0]

    # pointer locals (aliases) in the bodies: address space by identity
    nloc = [0]

    def ptr_local(m):
        ty, name, rhs = m.group(1), m.group(2), m.group(3)
        fs = [f for (f, n), _ in ident.items() if n == name]
        ids = {ident[(f, name)] for f in fs}
        if len(ids) != 1:
            sys.exit('to_metal: pointer local %s has identities %s' % (name, ids))
        a = ids.pop()
        nloc[0] += 1
        rhs = rhs.strip()
        if rhs == 'reinterpret_cast<Fp*>(args[4])':
            rhs = 'ctx.slab'
        elif rhs == 'reinterpret_cast<FpExt*>(args[4])':
            rhs = 'reinterpret_cast<thread FpExt*>(ctx.slab)'
        elif not re.fullmatch(r'\w+', rhs):
            sys.exit('to_metal: pointer local %s = %r' % (name, rhs))
        if a == 'slab':
            return 'thread %s* %s = %s;' % (ty, name, rhs)
        if ty != 'Fp' or a not in TRACE:
            sys.exit('to_metal: pointer local %s %s* is %s' % (name, ty, a))
        return 'const device Fp* %s = %s;' % (name, rhs)

    core = re.sub(r'(?<![\w*])(?<!thread )(?<!device )(Fp|FpExt)\* (\w+) = ([^;]+);', ptr_local, core)
    if nloc[0] != COUNTS['ptr_locals']:
        sys.exit('to_metal: ptr_locals: expected %d, found %d' % (COUNTS['ptr_locals'], nloc[0]))
    p.counts['ptr_locals'] = nloc[0]
    core = p.sub(core, 'slot-major global slab (args[4])', 'slot-major slab (EfCtx::slab, a thread array)', 'slab_comment', 1)
    # poly_fp's remaining args[k] (trace columns)
    for k, a in enumerate(roots[:4]):
        core = core.replace('args[%d]' % k, 'ctx.%s' % a)
    left = len(re.findall(r'\bargs\[', core))
    if left != COUNTS['args_uses']:
        sys.exit('to_metal: %d args[] uses left' % left)
    core = p.sub(core, 'size_t mask = steps - 1;', 'uint32_t mask = steps - 1;', 'mask')
    core = p.sub(core, 'const uint32_t rk_s = (uint32_t)steps,', 'const uint32_t rk_s = EF_FC_STEPS(steps),',
                 'fc_steps')

    # ---- CUDA spellings ----
    core = p.sub(core, '::risc0::', '', 'risc0_ns')
    core = p.sub(core, '__device__ __forceinline__ static ', 'inline ', 'device_inline_static')
    for bad in ('__device__', '__global__', '__constant__', '__shared__', 'asm(', 'asm volatile', '__ldg',
                'size_t', 'uint64_t(p)', 'args[', 'ef_pm_raw', 'ef_pmn_raw', 'Fp** '):
        if bad in core:
            ln = core[:core.index(bad)].count('\n') + 1
            sys.exit('to_metal: %r left at core line %d' % (bad, ln))
    # every remaining pointer declaration carries an address space
    for m in re.finditer(r'(?<![\w>])(Fp|FpExt|uint32_t|uint4)\* ', core):
        before = core[max(0, m.start() - 16):m.start()]
        if not re.search(r'(thread|device|constant) (const )?$', before) and 'reinterpret_cast<' not in before:
            ln = core[:m.start()].count('\n') + 1
            sys.exit('to_metal: pointer without an address space at core line %d: %r'
                     % (ln, core[m.start() - 40:m.end() + 20]))

    core = flatten(core, p)
    core = hoist_loads(core)

    metal = PRELUDE + core + KERNELS
    bad = [i + 1 for i, l in enumerate(metal.split('\n')) if l != l.rstrip()]
    if bad:
        sys.exit('to_metal: trailing whitespace on %d lines (first %d)' % (len(bad), bad[0]))
    nptr = sum(len(v) for k, v in drop.items())
    msg = ('to_metal: %d signatures and %d calls take EfCtx instead of %d pointer arguments, %d pointer '
           'locals, %d table-helper calls, %d hoisted terms and %d products on the host'
           % (p.counts['chain_sigs'], p.counts['chain_calls'], nptr, p.counts['ptr_locals'],
              p.counts['ctx_helper_calls'], hn, qn))
    if stats is not None:
        stats.append(msg)
    else:
        print(msg)
    return metal, '\n'.join(rust)


if __name__ == '__main__':
    with open(sys.argv[1], newline='') as f:
        text = f.read()
    metal, rust = convert(text)
    with open(sys.argv[2], 'w', newline='') as f:
        f.write(metal)
    with open(sys.argv[3], 'w', newline='') as f:
        f.write(rust)
