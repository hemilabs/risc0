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

"""CUDA -> Metal translator for the recursion circuit's eval_check (Metal port item C3, plan 4.3
"eval mode", errata 8.2).

usage:
  cuda2metal_eval.py -o OUTDIR [--split [--split-lz [--split-rg]]] [--check] [--no-pin]
                     [--compile] [--metallib PATH] [--define NAME=VALUE ...]
  cuda2metal_eval.py --check [--split [--split-lz [--split-rg]]]        # round trip + pins only
  cuda2metal_eval.py --list-kernels [--split [--split-lz [--split-rg]]] # the kernel entry names

Inputs (kernels/cuda, checked in) -> outputs (OUTDIR, never checked in):
  eval_check.cu        -> eval_check.metal        + kernel eval_check (monolith, always)
  eval_check_split.cu  -> eval_check_split.metal  + kernels ec_part_0 .. ec_part_5 (--split)
                       -> eval_check_split_lz.metal + kernels ec_lz_part_0 .. 5 (--split-lz)
  ec_regroup_metal.py  -> eval_check_split_rg.metal + kernels ec_rg_part_0 .. 13 (--split-rg)

Monolith: poly_fp (zirgen output, 12k lines) becomes a `static inline` function whose first
parameter is `const device FpExt* poly_mix` (CUDA reads a `__constant__ FpExt poly_mix[158]`
symbol filled with cudaMemcpyToSymbol; Metal has no writable program-scope memory, so the host
binds the 158 mix powers as buffer 0). The kernel `eval_check` (a generated block) is the port of
ffi_supra.cu's `eval_check` (CUDA branch: `pow(rou, cycle)`).

Split (--split): eval_check_split.cu (eval_check_split.py + ec_lazyredc.py output) after selecting
the CUDA branch of every preprocessor conditional (all HIP/GFX macros undefined: the `#else`
bodies), so the six poly_fp_part<P> are the CUDA ones with the lazy Montgomery `RecLz` sums.
The CUDA template kernel `ec_part<P>` and the host code after it (launch_parts, eval_check_split,
warm_parts, rec_eval_check_split_warmup) are kept as `//c2m- ` comment lines (not compiled), and
the six instantiations are text-expanded into kernels `ec_part_0` .. `ec_part_5` (generated
block) that call `ec_part_body<P>`. The parts run in order on one serial encoder: part P > 0 adds
the sum part P - 1 left in `check`, the last part scales by the quotient.

Split, lazy Fp (--split-lz, item C12): the translated split with every part's straight-line Fp
arithmetic rewritten by ec_lazyfp.py's 'lz2' form (the HIP generator's lazy Fp sums of products:
a sum/difference tree over products stays a raw 64-bit accumulator standing for x * 2^-32 mod P,
folded only when one more term could overflow, and one REDC makes a canonical Fp only where one
is needed). REDC is linear mod P and its output canonical, so for canonical inputs the result is
bit-identical (the canonical-input assumption ec_lazyredc.py documents). The `rlf_*` helpers are
Metal ports of ec_lazyfp.py's HIP ones (`Fp::asRaw`/`Fp::fromRaw` for sppark's raw `*a`/`Fp(r)`);
the kernels are renamed `ec_lz_part_P` so both splits link into one metallib. C12 measured the
six parts at po2 18 (domain 2^20) 58.6 -> 47.8 ms per call on the M5 Max (lz1 49.3 ms, a per-part
mix of lz1/lz2/RecLz-only 47.7 ms: no better than all-lz2; the lazy-REDC split without the RecLz
pass, i.e. plain Fp, 97 ms).

Split, regrouped (--split-rg, item C15): ec_regroup_metal.py runs eval_check_split.py's generator
with a Metal-only grouping (14 parts instead of 6; two leaves split into their terms) and returns
the CUDA-form split source, which then takes exactly the --split and --split-lz path above (branch
selection, rules, round trip, lazy Fp 'lz2', uint4 mix loads); the kernels are renamed
`ec_rg_part_P`. Only the grouping of field sums changes, so the result is bit-identical. C15
measured eval_check at po2 18 44.2 -> 27.8 ms per call on the M5 Max: parts 0 and 5 of the
6-part split compiled to 2.6 / 3.1 KB per-thread stacks, every regrouped part to <= 1.1 KB.

Kernel bindings (all modes; RecEvalParams is defined in the generated prelude, 16 B):
  buffer(0) poly_mix  const device FpExt*  158 mix powers (CUDA __constant__ poly_mix)
  buffer(1) check     device Fp*           4 * domain (column-major, as CUDA)
  buffer(2) ctrl      const device Fp*
  buffer(3) data      const device Fp*
  buffer(4) accum     const device Fp*
  buffer(5) mix       const device Fp*
  buffer(6) out       const device Fp*     (the globals; CUDA `out`)
  buffer(7) params    constant RecEvalParams& { rou (raw Montgomery word, as Rust's Elem),
                                                po2, domain, _pad = 0 }
  Grid::Threads(domain), threadgroup size <= 256 (CUDA __launch_bounds__(256, 1)).

Rewrites (reversible `Rule`s from cuda2metal_common.py, counted and pinned in PINNED; one rule
list for both inputs, a rule that does not apply to an input is pinned at 0):
  include_fp       `#include "supra/fp.h"` -> `#include "fpext.h"` (METAL_INCS, item F1).
  disable          lines with no Metal meaning (CUDA headers, eval_check.cuh, the __constant__
                   poly_mix symbol, the CUDA EC_PART_BOUNDS macro) -> `// cuda2metal (not used): `.
  constexpr        program-scope `constexpr uint32_t|int` -> `constant constexpr ...` (MSL
                   program-scope variables live in `constant`).
  inv_rate         split's `constexpr size_t INV_RATE = 4;` -> `constant constexpr uint32_t ...`
                   (32-bit index math: `(idx - INV_RATE * k) & mask` has the same bits because
                   mask < 2^32). The monolith gets the same definition in its prelude (CUDA: from
                   eval_check.cuh).
  qual.force       `__device__ __forceinline__ ` -> `static inline __attribute__((always_inline)) `.
  qual.device      `__device__ ` (line start) -> `static inline `.
  poly_mix_def     `FpExt poly_fp[_partN](` -> `FpExt poly_fp[_partN](const device FpExt* poly_mix, `.
  poly_mix_call    `return poly_fp_partN(` / `= poly_fp_part(` -> `...(poly_mix, `.
  lz_add_def       `RecLz rec_lz_add(RecLz r,` -> `RecLz rec_lz_add(const device FpExt* poly_mix, RecLz r,`.
  lz_add_call      `rec_lz_add(` -> `rec_lz_add(poly_mix, ` (every RecLz accumulation step).
  lz_mix_ptr       `const uint32_t* m = reinterpret_cast<const uint32_t*>(&poly_mix[k]);` -> device.
  thread_ref       `const Fp& ` / `const FpExt& ` -> `const thread Fp& ` / `const thread FpExt& `.
  thread_u32_ptr   other `[const ]uint32_t*` -> `[const ]thread uint32_t*` (RecLz word views of
                   thread values).
  ptr_param        `const Fp* (ctrl|out|data|mix|accum)` -> `const device Fp* ...`.
  check_param      `(Fp* check` -> `(device Fp* check`.
  body_def         `void ec_part_body(` -> `void ec_part_body(const device FpExt* poly_mix,
                   uint32_t gid, `.
  gid              `uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;` -> `... = gid;`.
  ext_index        `tot[i]` -> `tot.elems[i]` (Metal FpExt has no operator[]).
Constructs that compile unchanged against F1's fp.h/fpext.h/sppark_compat.h: Fp/FpExt arithmetic,
`Fp x = data[k * size + ((idx - INV_RATE * k) & mask)]` (device reads), `poly_mix[k] * x`,
`rou ^ cycle` (sppark_compat.h operator^), uint64_t RecLz accumulators, `min` (metal::min).

`--check` translates in memory, checks the pins, runs the residual scan and the round trip
(strip generated blocks, un-comment `//c2m- ` lines, undo every rule in reverse order, compare
byte for byte with the CUDA source; for --split, with the branch-selected source, and the
selection itself is pinned: conditionals seen, lines kept/dropped). With OUTDIR it also compares
the written outputs. `--compile` compiles every output with `xcrun metal -std=metal3.2 -O2`
(include path: build_kernel/kernels/metal and recursion-sys/kernels/metal only) and, with
--metallib, links them. `--define C3M_PSO_NONCE=N` makes a binary with fresh content (a
never-taken compare on params._pad) to time cold PSO creation past the OS shader cache;
production builds never define it.
"""

import argparse
import os
import re
import sys

sys.dont_write_bytecode = True  # never leave __pycache__ in the source tree (build.rs runs this)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cuda2metal_common as c  # noqa: E402

TG_SIZE = 256  # CUDA: __launch_bounds__(256, 1) (monolith) / kBlock = 256 (split, CUDA branch)
NUM_PARTS = 6  # eval_check_split.cu kNumParts (checked against the source)
RG = 'eval_check_split_rg'  # --split-rg (C15): the regrouped split's translation
NS = 'risc0::circuit::recursion::cuda'
HOST_PREFIX = '//c2m- '

BINDINGS = """\
// buffer(0) poly_mix  const device FpExt*  158 mix powers (CUDA __constant__ poly_mix)
// buffer(1) check     device Fp*           4 * domain, column-major
// buffer(2) ctrl      const device Fp*
// buffer(3) data      const device Fp*
// buffer(4) accum     const device Fp*
// buffer(5) mix       const device Fp*
// buffer(6) out       const device Fp*     globals
// buffer(7) params    constant RecEvalParams& (set_bytes)
"""

SIG = """\
    const device FpExt* poly_mix [[buffer(0)]],
    device Fp* check [[buffer(1)]],
    const device Fp* ctrl [[buffer(2)]],
    const device Fp* data [[buffer(3)]],
    const device Fp* accum [[buffer(4)]],
    const device Fp* mix [[buffer(5)]],
    const device Fp* out [[buffer(6)]],
    constant RecEvalParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]]"""

PARAMS = """\
// Kernel parameters (set_bytes, 16 B; mirrored #[repr(C)] on the Rust side, item C8).
// rou is the raw (Montgomery) word of the root of unity, as Rust's BabyBear Elem stores it.
struct RecEvalParams {
  uint32_t rou;
  uint32_t po2;
  uint32_t domain;
  uint32_t _pad; // 0
};
static_assert(sizeof(RecEvalParams) == 16, "RecEvalParams is 16 bytes");
"""

NONCE = """\
#ifdef C3M_PSO_NONCE
  // Cold-PSO timing only (never defined in builds): a distinct binary per nonce.
  if (params._pad == uint32_t(C3M_PSO_NONCE)) check[0] = Fp(0);
#endif
"""

PRELUDE_EVAL = """\
// Translated from kernels/cuda/eval_check.cu by kernels/cuda/cuda2metal_eval.py (Metal port
// item C3); never edit by hand. Compile with -I build_kernel/kernels/metal.
// poly_fp takes the mix powers as `const device FpExt* poly_mix` (CUDA: __constant__ symbol);
// the kernel eval_check at the end of this file is the port of ffi_supra.cu's eval_check.
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wc++17-extensions" // namespace a::b::c (kept from the CUDA source)
#include <metal_stdlib>
// From eval_check.cuh (CUDA: size_t; 32-bit gives the same `& mask` index bits).
constant constexpr uint32_t INV_RATE = 4;
""" + PARAMS

PRELUDE_SPLIT = """\
// Translated from kernels/cuda/eval_check_split.cu by kernels/cuda/cuda2metal_eval.py (Metal port
// item C3, --split); never edit by hand. Compile with -I build_kernel/kernels/metal.
// The CUDA branch of every #if was selected (HIP/GFX macros undefined); the CUDA host code at
// the end is kept as `//c2m- ` comments; kernels ec_part_0..5 at the end of this file.
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wc++17-extensions" // namespace a::b::c (kept from the CUDA source)
#include <metal_stdlib>
#include "sppark_compat.h" // operator^(Fp, unsigned) for `rou ^ cycle`
using metal::min;
""" + PARAMS

EVAL_KERNEL = (
    "// Kernel eval_check (CUDA ffi_supra.cu eval_check, non-HIP branch): one thread per row of\n"
    "// the evaluation domain; dispatch Grid::Threads(domain), threadgroup size <= %d.\n"
    "%s"
    "[[max_total_threads_per_threadgroup(%d)]]\n"
    "kernel void eval_check(\n%s) {\n"
    "  uint32_t domain = params.domain;\n"
    "  uint32_t cycle = gid;\n"
    "  if (cycle >= domain) return;\n"
    "%s"
    "  FpExt tot = %s::poly_fp(poly_mix, cycle, domain, ctrl, out, data, mix, accum);\n"
    "  Fp rou = Fp::fromRaw(params.rou);\n"
    "  Fp x = pow(rou, cycle);\n"
    "  Fp y = pow(Fp(3) * x, 1u << params.po2);\n"
    "  FpExt ret = tot * inv(y - Fp(1));\n"
    "  check[domain * 0 + cycle] = ret.elems[0];\n"
    "  check[domain * 1 + cycle] = ret.elems[1];\n"
    "  check[domain * 2 + cycle] = ret.elems[2];\n"
    "  check[domain * 3 + cycle] = ret.elems[3];\n"
    "}\n" % (TG_SIZE, BINDINGS, TG_SIZE, SIG, NONCE, NS))


def split_kernels(nparts=NUM_PARTS):
    out = ("// Kernels ec_part_0 .. ec_part_%d: the CUDA `template <int P> __global__ ec_part`\n"
           "// instantiations (eval_check_split.cu, non-HIP branch), text-expanded. Dispatch all of\n"
           "// them in order on one serial encoder, each Grid::Threads(domain), threadgroup size\n"
           "// <= %d: part P > 0 reads the running sum part P - 1 left in `check`.\n%s"
           % (nparts - 1, TG_SIZE, BINDINGS))
    for p in range(nparts):
        out += ("[[max_total_threads_per_threadgroup(%d)]]\n"
                "kernel void ec_part_%d(\n%s) {\n%s"
                "  %s::ec_part_body<%d>(poly_mix, gid, check, ctrl, data, accum, mix, out,\n"
                "      Fp::fromRaw(params.rou), params.po2, params.domain);\n"
                "}\n" % (TG_SIZE, p, SIG, NONCE, NS, p))
    return out


KERNELS = {
    'eval_check': ['eval_check'],
    'eval_check_split': ['ec_part_%d' % p for p in range(NUM_PARTS)],
    'eval_check_split_lz': ['ec_lz_part_%d' % p for p in range(NUM_PARTS)],
    'eval_check_split_lz_u32': ['ec_lz32_part_%d' % p for p in range(NUM_PARTS)],
}


def rg_parts():
    import ec_regroup_metal  # kernels/cuda, next to this script (C15)
    return ec_regroup_metal.NUM_PARTS


KERNELS['eval_check_split_rg_lz'] = ['ec_rg_part_%d' % p for p in range(rg_parts())]


# ---- split, lazy Fp (--split-lz, C12) -------------------------------------------------------

LZ_FORM = 'lz2'
LZ_HELPERS = """\
// ec_lazyfp.py's lazy Fp helpers (HIP originals; Metal: Fp::asRaw / Fp::fromRaw). An accumulator
// x (uint64_t) stands for x * 2^-32 mod P. Canonical inputs assumed (see ec_lazyredc.py).
static inline __attribute__((always_inline)) uint64_t rlf_mul(Fp a, Fp b) { return uint64_t(a.asRaw()) * uint64_t(b.asRaw()); }
static inline __attribute__((always_inline)) uint64_t rlf_c(Fp a) { return uint64_t(a.asRaw()) << 32; }
static inline __attribute__((always_inline)) uint64_t rlf_neg(Fp a) { return uint64_t(0x78000001u - a.asRaw()) << 32; }
static inline __attribute__((always_inline)) uint64_t rlf_mneg(Fp a, Fp b) { return uint64_t(0x78000001u - a.asRaw()) * uint64_t(b.asRaw()); }
// High word below 2P (fold1) or 3P (fold2) reduced below P.
static inline __attribute__((always_inline)) uint64_t rlf_fold1(uint64_t x) {
  uint32_t hi = uint32_t(x >> 32);
  hi = min(hi, hi - 0x78000001u);
  return (uint64_t(hi) << 32) | uint32_t(x);
}
static inline __attribute__((always_inline)) uint64_t rlf_fold2(uint64_t x) {
  uint32_t hi = uint32_t(x >> 32);
  hi = min(hi, hi - 0x78000001u);
  hi = min(hi, hi - 0x78000001u);
  return (uint64_t(hi) << 32) | uint32_t(x);
}
// REDC of x < P * 2^32: canonical x * 2^-32 mod P.
static inline __attribute__((always_inline)) Fp rlf_fin(uint64_t x) {
  uint32_t m = uint32_t(x) * 0x77ffffffu;
  uint64_t u = x + uint64_t(m) * 0x78000001u;
  uint32_t r = uint32_t(u >> 32);
  return Fp::fromRaw(min(r, r - 0x78000001u));
}
"""
# C13: the mix-power read of every RecLz accumulation step (rec_lz_add) as one 16-byte uint4
# load (the 158 powers sit in a HAL-owned buffer at offset 0, FpExt = 4 words, so every power is
# 16-byte aligned) instead of four 4-byte loads. Measured -6.5 % per eval_check call (po2 18);
# a 32-byte ulong4 pair load (K1's form) was +40 %. The four-load form stays in the metallib as
# eval_check_split_lz_u32.metal (kernels ec_lz32_part_0..5, RISC0_METAL_REC_LZ_UINT4=0).
LZ_ADD_U32 = """\
  const device uint32_t* m = reinterpret_cast<const device uint32_t*>(&poly_mix[k]);
  uint32_t cv = rec_raw(c);
  for (int i = 0; i < 4; i++)
    r.a[i] += uint64_t(m[i]) * cv;
"""
LZ_ADD_UINT4 = """\
  // C13: one 16-byte load of the mix power (see cuda2metal_eval.py LZ_ADD_UINT4).
  uint4 m = *(reinterpret_cast<const device uint4*>(poly_mix) + k);
  uint32_t cv = rec_raw(c);
  r.a[0] += uint64_t(m.x) * cv;
  r.a[1] += uint64_t(m.y) * cv;
  r.a[2] += uint64_t(m.z) * cv;
  r.a[3] += uint64_t(m.w) * cv;
"""


def lz_u32(lz_text):
    """eval_check_split_lz_u32.metal: the C12 lazy Fp split (four 4-byte mix loads per
    rec_lz_add), kernels renamed ec_lz32_part_N. Returns (text, counts)."""
    counts = c.Counter()
    text, n = re.subn(r'^kernel void ec_lz_part_(\d+)\(', r'kernel void ec_lz32_part_\1(',
                      lz_text, flags=re.M)
    counts.add('lz32.kernels', n)
    text, n = re.subn(r'^// Lazy Fp form \(--split-lz, C12;',
                      '// Four-load rec_lz_add (C12 form; C13 revert, RISC0_METAL_REC_LZ_UINT4=0).\n'
                      '// Lazy Fp form (--split-lz, C12;', text, flags=re.M)
    counts.add('lz32.header', n)
    counts.add('lz32.u32_add', text.count(LZ_ADD_U32))
    tallies('eval_check_split_lz_u32', text, counts)
    residual_scan('eval_check_split_lz_u32', text)
    return text, counts


LZ_HEAD = re.compile(r'^static inline __attribute__\(\(always_inline\)\) FpExt poly_fp_part(\d+)\('
                     r'const device FpExt\* poly_mix, ', re.M)


def lazy_fp(split_text):
    """eval_check_split_lz.metal from the translated split (see the module doc). Returns
    (text, counts)."""
    import ec_lazyfp  # kernels/cuda (HIP variant generator), next to this script
    counts = c.Counter()
    ns = 'namespace %s {\n' % NS
    if split_text.count(ns) != 1:
        raise c.TranslateError('split-lz: expected one namespace open line')
    text = split_text.replace(ns, ns + c.gen_block(LZ_HELPERS), 1)
    out, pos, vc = [], 0, {}
    for m in LZ_HEAD.finditer(text):
        s = m.start()
        e = text.index('\n}\n', s) + 3
        out.append(text[pos:s])
        out.append(ec_lazyfp.variant(text[s:e], LZ_FORM, vc))
        counts.add('lz.parts')
        pos = e
    out.append(text[pos:])
    text = ''.join(out)
    for k, v in vc.items():
        counts.add('lz.' + k, v)
    text, n = re.subn(r'^kernel void ec_part_(\d+)\(', r'kernel void ec_lz_part_\1(', text,
                      flags=re.M)
    counts.add('lz.kernels', n)
    text, n = re.subn(r'^// Translated from kernels/cuda/eval_check_split\.cu by ',
                      '// Lazy Fp form (--split-lz, C12; ec_lazyfp.py %s on every part) of the split.\n'
                      '// Translated from kernels/cuda/eval_check_split.cu by ' % LZ_FORM, text,
                      flags=re.M)
    counts.add('lz.header', n)
    counts.add('lz.uint4_add', text.count(LZ_ADD_U32))
    text = text.replace(LZ_ADD_U32, LZ_ADD_UINT4)
    tallies('eval_check_split_lz', text, counts)
    residual_scan('eval_check_split_lz', text)
    return text, counts


def rg_lz(lz_text):
    """eval_check_split_rg.metal (C15) from the lazy Fp form of the regrouped split's translation:
    kernels renamed ec_rg_part_N, header noted. Returns (text, counts)."""
    counts = c.Counter()
    text, n = re.subn(r'^kernel void ec_lz_part_(\d+)\(', r'kernel void ec_rg_part_\1(', lz_text,
                      flags=re.M)
    counts.add('rg.kernels', n)
    text, n = re.subn(r'^// Lazy Fp form \(--split-lz, C12;',
                      '// Regrouped split (--split-rg, C15: ec_regroup_metal.py, %d parts; only\n'
                      '// the grouping of field sums differs from eval_check_split.cu).\n'
                      '// Lazy Fp form (--split-lz, C12;' % rg_parts(), text, flags=re.M)
    counts.add('rg.header', n)
    residual_scan('eval_check_split_rg', text)
    return text, counts


# Lines with no Metal meaning, kept as comments (reversible).
DISABLE = [
    r'#include "cuda\.h"', r'#include "eval_check\.cuh"', r'#include <cstdint>',
    r'#include <cstring>', r'#include <stdexcept>', r'__constant__ FpExt poly_mix\[158\];',
    r'#define EC_PART_BOUNDS __launch_bounds__\(kBlock, 1\)',
]
DISABLED = '// cuda2metal (not used): '
INV_RATE_NOTE = '  // cuda2metal: was size_t (same `& mask` index bits)'


def rules():
    return [
        c.Rule('include_fp', r'^#include "supra/fp\.h"$', '#include "fpext.h"',
               r'^#include "fpext\.h"$', '#include "supra/fp.h"'),
        c.Rule('disable', r'^(%s)$' % '|'.join(DISABLE), DISABLED.replace('\\', '') + r'\1',
               r'^%s(.*)$' % re.escape(DISABLED), r'\1'),
        c.Rule('constexpr', r'^constexpr (uint32_t|int) ', r'constant constexpr \1 ',
               r'^constant constexpr (uint32_t|int) ', r'constexpr \1 ',
               collision=r'^constant constexpr '),
        c.Rule('inv_rate', r'^constexpr size_t INV_RATE = (\d+);$',
               r'constant constexpr uint32_t INV_RATE = \1;' + INV_RATE_NOTE,
               r'^constant constexpr uint32_t INV_RATE = (\d+);%s$' % re.escape(INV_RATE_NOTE),
               r'constexpr size_t INV_RATE = \1;'),
        c.Rule('qual.force', r'__device__ __forceinline__ ',
               'static inline __attribute__((always_inline)) ',
               r'static inline __attribute__\(\(always_inline\)\) ',
               '__device__ __forceinline__ '),
        c.Rule('qual.device', r'^__device__ ', 'static inline ', r'^static inline (?!__attr)',
               '__device__ '),
        c.Rule('poly_mix_def', r'\b(FpExt poly_fp(?:_part\d*)?)\(',
               r'\1(const device FpExt* poly_mix, ',
               r'\b(FpExt poly_fp(?:_part\d*)?)\(const device FpExt\* poly_mix, ', r'\1('),
        c.Rule('poly_mix_call', r'(return |= )(poly_fp_part\d*)\(', r'\1\2(poly_mix, ',
               r'(return |= )(poly_fp_part\d*)\(poly_mix, ', r'\1\2('),
        c.Rule('lz_add_def', r'\bRecLz rec_lz_add\(RecLz r,',
               'RecLz rec_lz_add(const device FpExt* poly_mix, RecLz r,',
               r'\bRecLz rec_lz_add\(const device FpExt\* poly_mix, RecLz r,',
               'RecLz rec_lz_add(RecLz r,'),
        c.Rule('lz_add_call', r'(?<!RecLz )\brec_lz_add\((?!const device)', 'rec_lz_add(poly_mix, ',
               r'\brec_lz_add\(poly_mix, ', 'rec_lz_add('),
        c.Rule('lz_mix_ptr',
               r'^(\s*)const uint32_t\* m = reinterpret_cast<const uint32_t\*>\(&poly_mix\[k\]\);$',
               r'\1const device uint32_t* m = reinterpret_cast<const device uint32_t*>(&poly_mix[k]);',
               r'^(\s*)const device uint32_t\* m = reinterpret_cast<const device uint32_t\*>'
               r'\(&poly_mix\[k\]\);$',
               r'\1const uint32_t* m = reinterpret_cast<const uint32_t*>(&poly_mix[k]);',
               collision=r'\bdevice uint32_t\*'),
        c.Rule('thread_ref', r'\bconst (Fp|FpExt)& ', r'const thread \1& ',
               r'\bconst thread (Fp|FpExt)& ', r'const \1& '),
        c.Rule('thread_u32_ptr', r'\b(const )?(?<!device )(?<!thread )uint32_t\*',
               r'\1thread uint32_t*', r'\b(const )?thread uint32_t\*', r'\1uint32_t*'),
        c.Rule('ptr_param', r'\bconst Fp\* (ctrl|out|data|mix|accum)\b', r'const device Fp* \1',
               r'\bconst device Fp\* (ctrl|out|data|mix|accum)\b', r'const Fp* \1'),
        c.Rule('check_param', r'\(Fp\* check\b', '(device Fp* check',
               r'\(device Fp\* check\b', '(Fp* check'),
        c.Rule('body_def', r'\bvoid ec_part_body\(',
               'void ec_part_body(const device FpExt* poly_mix, uint32_t gid, ',
               r'\bvoid ec_part_body\(const device FpExt\* poly_mix, uint32_t gid, ',
               'void ec_part_body('),
        c.Rule('gid', r'^(\s*)uint32_t cycle = blockDim\.x \* blockIdx\.x \+ threadIdx\.x;$',
               r'\1uint32_t cycle = gid;', r'^(\s*)uint32_t cycle = gid;$',
               r'\1uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;'),
        c.Rule('ext_index', r'\btot\[(\d)\]', r'tot.elems[\1]', r'\btot\.elems\[(\d)\]',
               r'tot[\1]'),
    ]


# Rewrite counts and tallies for the checked-in sources. A change means zirgen's (or
# eval_check_split.py's) output changed shape: run with --no-pin, review the output and its
# compile, update. `select.*` describe the branch selection, `tally.*` count constructs that pass
# through unchanged, `host.*` the commented-out CUDA host code.
# 3.1 re-pin (P31-08): BIT_AND_ELEM's canonical decomposition adds two constraints (top_a/top_b)
# to the macro-op sum gated by x320: +12 ops, +2 poly_mix terms. Hence everywhere +2 Fp*mix
# products (fp_mul, lz2_prod), +2 RecLz accumulations (lz_add_call), +2 mix reads (monolith),
# +4 lazy-Fp REDCs (lz2_fin), and the HIP-only bodies of the split sources grew by 24 (split) /
# 28 (regrouped) lines (select.dropped_lines). 3.0 values in the comments.
PINNED = {
    'eval_check': {
        'include_fp': 1, 'disable': 2, 'qual.device': 1, 'poly_mix_def': 1, 'ptr_param': 5,
        'kernels': 1, 'tally.poly_mix_read': 1230, 'tally.data_read': 669,  # 3.0: 1228
        'tally.fp_mul': 4681,  # 3.0: 4679
    },
    'eval_check_split': {
        # 59 directives (#if/#ifdef/#elif/#else/#endif) resolved, 14188 HIP-only lines dropped
        # (3.0: 14164).
        'select.directives': 59, 'select.dropped_lines': 14188,
        'include_fp': 1, 'disable': 6, 'constexpr': 2, 'inv_rate': 1,
        # 6 RecLz helpers + 6 parts + poly_fp_part + ec_part_body.
        'qual.force': 14,
        'poly_mix_def': 7, 'poly_mix_call': 7, 'lz_add_def': 1, 'lz_add_call': 1062,  # 3.0: 1060
        'lz_mix_ptr': 1, 'thread_ref': 3, 'thread_u32_ptr': 5,
        # 40 in the parts/dispatcher/body + 15 in the commented CUDA host code.
        'ptr_param': 55, 'check_param': 2, 'body_def': 1, 'gid': 1, 'ext_index': 4,
        'host.lines': 51, 'kernels': 6, 'tally.poly_mix_read': 172,
        'tally.data_read': 1371, 'tally.fp_mul': 4857,  # 3.0: 4855
    },
    'eval_check_split_lz': {
        # ec_lazyfp.py 'lz2' on the six parts (C12): products made lazy / REDCs emitted.
        # 3.0: lz2_prod 4855, lz2_fin 2557.
        'lz.parts': 6, 'lz.kernels': 6, 'lz.header': 1, 'lz.lz2_prod': 4857, 'lz.lz2_fin': 2561,
        'lz.uint4_add': 1,  # C13: rec_lz_add's mix read as one uint4 load (no &poly_mix[k])
        'tally.poly_mix_read': 171, 'tally.data_read': 1371, 'tally.fp_mul': 0,
    },
    RG: {
        # The regrouped split (C15, ec_regroup_metal.py): 14 parts, the same rules and selection.
        # Fewer HIP-only lines than the CUDA split: its per-part HIP variants are keyed by the
        # 6-part numbering (they never reach Metal either way).
        'select.directives': 59, 'select.dropped_lines': 6610,  # 3.0: 6582
        'include_fp': 1, 'disable': 6, 'constexpr': 2, 'inv_rate': 1,
        # 6 RecLz helpers + 14 parts + poly_fp_part + ec_part_body.
        'qual.force': 22,
        'poly_mix_def': 15, 'poly_mix_call': 15, 'lz_add_def': 1, 'lz_add_call': 1076,  # 3.0: 1074
        'lz_mix_ptr': 1, 'thread_ref': 3, 'thread_u32_ptr': 5,
        # 80 in the parts/dispatcher/body + 15 in the commented CUDA host code.
        'ptr_param': 95, 'check_param': 2, 'body_def': 1, 'gid': 1, 'ext_index': 4,
        'host.lines': 51, 'kernels': 14, 'tally.poly_mix_read': 184,
        'tally.data_read': 1829, 'tally.fp_mul': 5049,  # 3.0: 5047
    },
    'eval_check_split_rg_lz': {
        # The lazy Fp form ('lz2', uint4 mix loads) of RG, kernels ec_rg_part_0..13.
        # 3.0: lz2_prod 5047, lz2_fin 2673.
        'lz.parts': 14, 'lz.kernels': 14, 'lz.header': 1, 'lz.lz2_prod': 5049, 'lz.lz2_fin': 2677,
        'lz.uint4_add': 1, 'rg.kernels': 14, 'rg.header': 1,
        'tally.poly_mix_read': 183, 'tally.data_read': 1829, 'tally.fp_mul': 0,
    },
    'eval_check_split_lz_u32': {
        # The C12 form of eval_check_split_lz (C13's revert, RISC0_METAL_REC_LZ_UINT4=0).
        'lz32.kernels': 6, 'lz32.header': 1, 'lz32.u32_add': 1,
        'tally.poly_mix_read': 172, 'tally.data_read': 1371, 'tally.fp_mul': 0,
    },
}


# ---- split: CUDA branch selection ----------------------------------------------------------

# Every conditional in eval_check_split.cu tests a HIP/GFX macro; all are undefined for the CUDA
# (and Metal) branch. Any other condition is refused, so a new conditional cannot be resolved
# silently.
HIP_MACROS = ('__HIPCC__', '__HIP_DEVICE_COMPILE__', '__GFX11__', '__GFX12__', '__gfx1201__')
COND_RX = re.compile(r'^#(if|ifdef|ifndef|elif|else|endif)\b(.*)$')


def _cond_value(kind, expr):
    expr = expr.strip()
    if kind == 'ifdef' or kind == 'ifndef':
        if expr not in HIP_MACROS:
            raise c.TranslateError('split: unknown macro in #%s %s' % (kind, expr))
        return kind == 'ifndef'
    # `#if` / `#elif`: only defined(<HIP macro>) terms joined by && / || and parentheses.
    toks = re.findall(r'defined\((\w+)\)|(&&|\|\||\(|\))|(\S+)', expr)
    py = []
    for name, op, other in toks:
        if other:
            raise c.TranslateError('split: unsupported #%s condition %r' % (kind, expr))
        if name:
            if name not in HIP_MACROS:
                raise c.TranslateError('split: unknown macro %s in #%s %s' % (name, kind, expr))
            py.append('False')
        else:
            py.append({'&&': 'and', '||': 'or'}.get(op, op))
    return bool(eval(' '.join(py), {'__builtins__': {}}))


def select_cuda_branches(src, counts):
    """Evaluate every #if/#ifdef/#elif/#else/#endif with the HIP macros undefined and keep the
    active lines (the directives themselves are dropped)."""
    out = []
    stack = []  # (parent_active, a_branch_was_taken)
    active = True
    for line in src.splitlines(keepends=True):
        m = COND_RX.match(line.rstrip('\n'))
        if not m:
            if active:
                out.append(line)
            else:
                counts.add('select.dropped_lines')
            continue
        kind, expr = m.group(1), m.group(2)
        counts.add('select.directives')
        if kind in ('if', 'ifdef', 'ifndef'):
            v = _cond_value(kind, expr)
            stack.append((active, v))
            active = active and v
        elif kind == 'elif':
            if not stack:
                raise c.TranslateError('split: #elif without #if')
            par, taken = stack[-1]
            v = (not taken) and _cond_value('if', expr)
            stack[-1] = (par, taken or v)
            active = par and v
        elif kind == 'else':
            par, taken = stack[-1]
            stack[-1] = (par, True)
            active = par and not taken
        else:
            par, _ = stack.pop()
            active = par
    if stack:
        raise c.TranslateError('split: unterminated #if')
    return ''.join(out)


def comment_host_code(text, counts):
    """Comment out (prefix `//c2m- `) the CUDA template kernel and the host code: from the
    `template <int P>` line before `... __global__ void ec_part(` to the end of the file, except
    the namespace's closing line."""
    lines = text.splitlines(keepends=True)
    for i, l in enumerate(lines):
        if l.startswith(HOST_PREFIX.rstrip()):
            raise c.TranslateError('split: input already contains %r lines' % HOST_PREFIX)
    k = [i for i, l in enumerate(lines) if re.match(r'^\S.*__global__ void ec_part\(', l)]
    if len(k) != 1 or lines[k[0] - 1] != 'template <int P>\n':
        raise c.TranslateError('split: expected one `template <int P>` + __global__ ec_part')
    start = k[0] - 1
    ns_close = '} // namespace %s\n' % NS
    if lines.count(ns_close) != 1:
        raise c.TranslateError('split: expected one namespace close line')
    for i in range(start, len(lines)):
        if lines[i] == ns_close:
            continue
        lines[i] = HOST_PREFIX + lines[i]
        counts.add('host.lines')
    return ''.join(lines)


def uncomment_host_code(text):
    return ''.join(l[len(HOST_PREFIX):] if l.startswith(HOST_PREFIX) else l
                   for l in text.splitlines(keepends=True))


# ---- residual scan -------------------------------------------------------------------------

RESIDUALS = [
    (r'__device__|__host__|__global__|__noinline__|__forceinline__|__constant__|__trap|'
     r'__launch_bounds__|__HIP|__GFX|__gfx', 'CUDA/HIP qualifier or macro'),
    (r'^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b', 'preprocessor conditional'),
    (r'\bblockDim\b|\bblockIdx\b|\bthreadIdx\b|\bgridDim\b', 'CUDA index builtin'),
    (r'cudaStream|cudaMemcpy|<<<|\bCUDA_OK\b|extern "C"', 'CUDA host API'),
    (r'<cuda|<thrust|"cuda\.h"|<cstdint>|<cstring>|<stdexcept>|supra/', 'CUDA header'),
    (r'\b(tot|ret)\[\d\]', 'FpExt operator[]'),
    (r'\bsize_t\b', 'size_t (64-bit index math)'),
    (r'\bprintf\b', 'printf'),
]
QUALIFIED = re.compile(r'\b(device|thread|constant|threadgroup) (const )?$')


def code_only(text):
    """`text` with every // comment blanked (positions and line numbers preserved)."""
    return re.sub(r'//[^\n]*', lambda m: ' ' * len(m.group(0)), text)


def residual_scan(name, text):
    # CUDA-only constructs: in the translated text (the generated blocks use #ifdef for the
    # PSO nonce); reported line numbers then count without the generated blocks.
    c.residual_scan(name, code_only(c.strip_gen_blocks(text)), RESIDUALS)
    code = code_only(text)
    # Every pointer/reference to a field or word type must carry an address space.
    for m in re.finditer(r'\b(Fp|FpExt|uint32_t)\s*[*&]', code):
        if not QUALIFIED.search(code[max(0, m.start() - 24):m.start()]):
            raise c.TranslateError('%s: pointer/reference without address space at %s: %r'
                                   % (name, c.where(code, m.start()), code[m.start():m.end()]))
    if re.search(r'^\s*poly_mix\s*\[', code, re.M):
        raise c.TranslateError('%s: poly_mix written' % name)


# ---- translate -----------------------------------------------------------------------------

def tallies(name, out, counts):
    code = code_only(out)
    counts.add('tally.poly_mix_read', len(re.findall(r'\bpoly_mix\[', code)))
    counts.add('tally.data_read', len(re.findall(r'= (data|ctrl|accum)\[', code)))
    counts.add('tally.fp_mul', len(re.findall(r'^\s*Fp x\d+ = x\d+ \* x\d+;$', code, re.M)))


def translate(name, src):
    """Returns (metal_text, counts, base) where `base` is the CUDA text the round trip must
    reproduce (the source, or for the split its branch-selected form)."""
    c.check_no_markers(src, name)
    counts = c.Counter()
    split = name in ('eval_check_split', RG)
    nparts = rg_parts() if name == RG else NUM_PARTS
    if split:
        m = re.search(r'^constexpr int kNumParts = (\d+);$', src, re.M)
        if not m or int(m.group(1)) != nparts:
            raise c.TranslateError('%s: kNumParts != %d' % (name, nparts))
        base = select_cuda_branches(src, counts)
    else:
        base = src
    out = c.apply_rules(base, rules(), counts)
    if split:
        out = comment_host_code(out, counts)
        prelude = PRELUDE_SPLIT
        tail = split_kernels(nparts)
    else:
        prelude = PRELUDE_EVAL
        tail = EVAL_KERNEL
    inc = '#include "fpext.h"\n'
    if out.count(inc) != 1:
        raise c.TranslateError('%s: expected exactly one #include "fpext.h"' % name)
    out = out.replace(inc, c.gen_block(prelude) + inc)
    if not out.endswith('\n'):
        raise c.TranslateError('%s: input does not end with a newline' % name)
    out += '\n' + c.gen_block(tail)
    counts.add('kernels', nparts if split else len(KERNELS[name]))
    tallies(name, out, counts)
    residual_scan(name, out)
    return out, counts, base


def untranslate(name, text):
    """The inverse of translate (minus the generated blocks and the branch selection)."""
    body = c.strip_gen_blocks(text)
    if not body.endswith('\n\n'):
        raise c.TranslateError('%s: unexpected tail before the kernel block' % name)
    body = body[:-1]
    body = uncomment_host_code(body)
    return c.unapply_rules(body, rules())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('-o', '--outdir')
    ap.add_argument('--split', action='store_true',
                    help='also translate eval_check_split.cu (kernels ec_part_0..5)')
    ap.add_argument('--split-lz', action='store_true',
                    help='with --split: also write eval_check_split_lz.metal (kernels '
                    'ec_lz_part_0..5, lazy Fp form, C12; uint4 mix loads, C13) and '
                    'eval_check_split_lz_u32.metal (ec_lz32_part_0..5, four-load form)')
    ap.add_argument('--split-rg', action='store_true',
                    help='with --split-lz: also write eval_check_split_rg.metal (kernels '
                    'ec_rg_part_0..N, the Metal-only regrouped split of ec_regroup_metal.py, C15)')
    ap.add_argument('--check', action='store_true',
                    help='round trip + pins + residuals; compare with OUTDIR files if present')
    ap.add_argument('--no-pin', action='store_true', help='report count mismatches only')
    ap.add_argument('--compile', action='store_true', help='compile each output with xcrun')
    ap.add_argument('--metallib', help='with --compile: link the .air files into this metallib')
    ap.add_argument('--define', action='append', default=[], help='extra -D for --compile')
    ap.add_argument('--list-kernels', action='store_true')
    ap.add_argument('-q', '--quiet', action='store_true')
    a = ap.parse_args(argv)

    if a.split_lz and not a.split:
        ap.error('--split-lz needs --split')
    if a.split_rg and not a.split_lz:
        ap.error('--split-rg needs --split-lz')
    names = ['eval_check'] + (['eval_check_split'] if a.split else [])
    if a.list_kernels:
        names += ['eval_check_split_lz', 'eval_check_split_lz_u32'] if a.split_lz else []
        names += ['eval_check_split_rg_lz'] if a.split_rg else []
        print('\n'.join(k for n in names for k in KERNELS[n]))
        return 0
    if not a.outdir and not a.check:
        ap.error('-o OUTDIR is required unless --check')
    if a.compile and not a.outdir:
        ap.error('--compile needs -o OUTDIR')

    log = (lambda *x: None) if a.quiet else print
    airs = []
    mismatches = []
    for name in names:
        src = c.read(os.path.join(c.CUDA_DIR, name + '.cu'))
        out, counts, base = translate(name, src)
        mismatches += c.check_pins(name, counts, PINNED[name], strict=not a.no_pin)
        if a.check:
            back = untranslate(name, out)
            if back != base:
                i = next((k for k in range(min(len(back), len(base))) if back[k] != base[k]),
                         min(len(back), len(base)))
                raise c.TranslateError('%s: round trip differs from the CUDA source at %s'
                                       % (name, c.where(base, i)))
        outs = [(name, out, counts, a.check)]
        if name == 'eval_check_split' and a.split_lz:
            lz, lz_counts = lazy_fp(out)
            mismatches += c.check_pins('eval_check_split_lz', lz_counts,
                                       PINNED['eval_check_split_lz'], strict=not a.no_pin)
            outs.append(('eval_check_split_lz', lz, lz_counts, False))
            lz32, lz32_counts = lz_u32(lz.replace(LZ_ADD_UINT4, LZ_ADD_U32))
            mismatches += c.check_pins('eval_check_split_lz_u32', lz32_counts,
                                       PINNED['eval_check_split_lz_u32'], strict=not a.no_pin)
            outs.append(('eval_check_split_lz_u32', lz32, lz32_counts, False))
        if name == 'eval_check_split' and a.split_rg:
            import ec_regroup_metal  # kernels/cuda (C15)
            try:
                rg_src, _ = ec_regroup_metal.regrouped(check=True)
            except ec_regroup_metal.RegroupError as e:
                raise c.TranslateError('split-rg: %s' % e)
            rg, rg_counts, rg_base = translate(RG, rg_src)
            mismatches += c.check_pins(RG, rg_counts, PINNED[RG], strict=not a.no_pin)
            if a.check:
                if untranslate(RG, rg) != rg_base:
                    raise c.TranslateError('%s: round trip differs from the regrouped source' % RG)
            rg_lz_text, rg_lz_counts = lazy_fp(rg)
            rg_text, rg_more = rg_lz(rg_lz_text)
            for k, v in rg_more.items():
                rg_lz_counts.add(k, v)
            mismatches += c.check_pins('eval_check_split_rg_lz', rg_lz_counts,
                                       PINNED['eval_check_split_rg_lz'], strict=not a.no_pin)
            outs.append(('eval_check_split_rg', rg_text, rg_lz_counts, a.check))
        for oname, otext, ocounts, rt in outs:
            dst = os.path.join(a.outdir, oname + '.metal') if a.outdir else None
            status = c.write_or_check(dst, otext, a.check) if dst else None
            log('%s: %s%s%s' % (oname, ', '.join('%s=%d' % kv for kv in ocounts.items()),
                                ' round-trip=ok' if rt else '',
                                (' -> %s (%s)' % (dst, status)) if dst else ''))
            if a.compile:
                air = os.path.join(a.outdir, oname + '.air')
                dt, err = c.compile_metal(dst, air, defines=a.define)
                warns = [l for l in err.splitlines() if 'warning:' in l]
                log('  compiled %s in %.1f s, %d warning lines' % (os.path.basename(air), dt,
                                                                      len(warns)))
                for w in warns[:10]:
                    log('    ' + w)
                airs.append(air)
    if a.metallib:
        dt = c.link_metallib(airs, a.metallib)
        log('linked %s in %.1f s' % (a.metallib, dt))
    if mismatches:
        log('pin mismatches (--no-pin):\n  ' + '\n  '.join(mismatches))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except c.TranslateError as e:
        print('cuda2metal_eval.py: error: %s' % e, file=sys.stderr)
        sys.exit(1)
