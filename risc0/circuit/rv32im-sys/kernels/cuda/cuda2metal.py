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

"""CUDA -> Metal Shading Language translator for the rv32im generated kernels (Metal port item R2).

usage:
  cuda2metal.py --selftest [--headers DIR] [--keep DIR]
  cuda2metal.py steps -o OUTDIR [--compile] [--headers DIR] [--check] [--no-pin]
  cuda2metal.py incs -o OUTDIR [--check]
  cuda2metal.py eval -o OUTDIR [--compile] [--check] [--no-pin]
  cuda2metal.py file IN.cu -o OUT.metal [--types types.cuh.inc]
  cuda2metal.py check

`steps` translates steps.cu -> OUTDIR/steps.metal and steps.cuh -> OUTDIR/steps.metal.h and
checks the per-pass rewrite counts against PINNED_STEPS (it fails when the generated input
changes shape; --no-pin only reports). `--compile` also writes the three .metal.inc files
(`incs`) and compiles steps.metal with `xcrun metal`, reporting the time and every residual
error class. `incs` translates defs.cu.inc, types.cuh.inc and layout.cu.inc. `--check` compares
the would-be outputs with the files already in OUTDIR instead of writing them (and creates
nothing). `check` (item R16) is the drift check CI runs without the Metal toolchain: the
checked-in kernels/metal/*.metal.inc must equal their translation, and `steps` and `eval` must
translate with their pinned counts (their outputs live in the build's OUT_DIR and are never
checked in; `eval -o $OUT_DIR/metal --check` compares a build's file).

The translator is a sequence of text passes over the CUDA source. Every pass counts what it
rewrote; anything it cannot translate faithfully raises TranslateError (no silent fallbacks):

  strip          __restrict__, nvtx ranges, empty `asm("")`/asm-comment barriers (an asm with
                 instructions is an error); __launch_bounds__ is consumed by `kernel`.
  include        #include / namespace mapping (namespace risc0::circuit::rv32im_v2::cuda ->
                 rv32im_metal, the namespace of rv32im-sys/kernels/metal/witgen.h).
  accum_guard    steps.cu's `#ifdef __HIPCC__` fence around exec_TopAccum/step_TopAccum
                 (accum_tu_gen.py) -> `#ifndef RV32IM_METAL_STEPS_NO_ACCUM`.
  array          ::cuda::std::array -> r0::array (cuda2metal.h).
  value_type     X::value_type -> the element type of X (from the `using X = array<T, N>`
                 typedefs of types.cuh.inc and the input itself); MSL rejects array::value_type.
  sbind          `auto [a, b] = e;` -> `auto c2m_sbK = e; auto a = c2m_sbK[0]; ...`.
  lambda         `T x = map(a[, b], ([&](..) {..; return e;}));` and the reduce forms ->
                 explicit loops over the arrays (one top-level `return` at the end of the
                 lambda body is required; anything else is an error). With
                 `translate(map_lowering='unroll')` (witgen_tu_gen.py, R17) each site becomes one
                 block per element instead, in element order, values in scalars: the map's array
                 is brace-initialized from them, the reduce's accumulator is threaded through
                 `c2m_accK_0..N` (the element count comes from the array typedef). A loop that
                 writes a default-constructed private array after calling a noinline function is
                 the shape in which the Apple GPU loses the last store (docs/metal-port/notes/
                 R17.md). R17b: each element scalar is declared with its value through a GNU
                 statement expression (`T c2m_rK_j = ({ ...; e; });`), never default-constructed
                 and assigned after the block's calls.
  mux_init       (R17b; not in ALL_PASSES: pass_mux_init(text, counts, zero), run by
                 witgen_tu_gen.py and accum_tu_gen.py after translate) `T x; if (c0) {..; x = v0;}
                 else if .. else {<no value>}` -> `T x = (c0) ? ({..; v0;}) : .. : ({..; Z;});`
                 with Z an explicit all-zero initializer of T (zero_init_fn over
                 types.metal.inc): mux results declared with their value, no constructor loop.
  fpp            `Fp** args` -> `device Fp* args_0, ..., args_{K-1}`; `args[k]` -> `args_k`
                 (K = 1 + the largest literal index used in the definition).
  kernel         `__global__` kernel signature synthesis: [[max_total_threads_per_threadgroup]]
                 before `kernel void`, explicit [[buffer(i)]] bindings (pointers per the policy
                 table, scalars as `constant T&`), the CUDA index quartet as uint3 attributes.
  qual           __device__/__host__/__noinline__/__forceinline__/__constant__ and
                 namespace-scope constexpr data -> MSL qualifiers.
  aspace         the address-space policy table (ASPACE_POLICY): every pointer/reference of a
                 listed type gets its MSL address space; struct-of-pointer contexts are refused.
  fp_zero        the Fp(0) rule: zero assigned to an Fp/Val/FpExt/ExtVal -> T(0).
  assert         `assert(c)` -> `C2M_ASSERT(ctx, c)`, which ORs RV32IM_FAULT_ASSERT into
                 ctx.tables.bad (plan 2.9: the kernel ORs it into its fault buffer once);
                 without a ctx in scope -> C2M_ASSERT_NOCTX(c), a no-op like CUDA release.

A final residual scan refuses output that still contains CUDA-only constructs.

`eval` (item R6) translates the baseline eval_check -- eval_check.cuh, eval_check_0..3.cu (poly_fp
+ the 20 noinline rv32im_v2_* chain functions) and the eval_check kernel of eval_check_kernel.cu
-- into one OUTDIR/eval_check.metal (written only when its text changes; build.rs runs it into
$OUT_DIR/metal), with counts pinned in PINNED_EVAL. Its passes (EVAL_PASSES) add:

  poly_mix       `__constant__ FpExt poly_mix[N];` (host-filled, no initializer) -> a
                 `const FpExt* poly_mix` parameter on every function that uses it directly or
                 through a call, `, poly_mix` at their call sites; the kernel binds it last.
  fpext_index    `e[i]` on a by-value FpExt (CUDA FpExt::operator[]) -> `e.elems[i]`.
  fpp (calls)    an Fp** argument that is a local `Fp* a[N] = {p, q, ...};` is passed as
                 `p, q, ...` (`a[k]` elsewhere -> `(elem k)`, the array goes); the caller's own
                 Fp** parameter is forwarded as `args_0, ..., args_{K-1}`.
  eval_aspace    `const Fp*` -> `const device Fp*` (trace buffers), mutable `Fp*`/`FpExt*` ->
                 `thread` (poly_fp's x33/x34 temporaries passed down the chain).
The chain lives in namespace rv32im_metal_eval; the kernel `eval_check` is at global scope.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RV32IM_SYS = os.path.dirname(os.path.dirname(HERE))
RISC0_ROOT = os.path.dirname(os.path.dirname(RV32IM_SYS))
BK_METAL = os.path.join(RISC0_ROOT, 'build_kernel', 'kernels', 'metal')
R1_METAL = os.path.join(RV32IM_SYS, 'kernels', 'metal')
SELFTEST_DIR = os.path.join(HERE, 'cuda2metal_selftest')
STUBS = os.path.join(SELFTEST_DIR, 'stubs')

CUDA_NS = 'risc0::circuit::rv32im_v2::cuda'
METAL_NS = 'rv32im_metal'
METAL_FLAGS = ['-std=metal3.2', '-O2']

# Rewrite counts of `steps` for the checked-in steps.cu / steps.cuh. A change means the generated
# input changed shape: re-run `cuda2metal.py steps --no-pin`, review the new counts (and the
# compile), then update these numbers.
PINNED_STEPS = {
    'steps.cu': {
        'accum_guard': 2,
        'array': 0,
        'aspace.thread': 210,
        'assert': 134,
        'assert_noctx': 0,
        'fp_zero': 0,
        'fpp': 0,
        'include': 2,
        'kernel': 0,
        'lambda.map': 42,
        'lambda.reduce': 1,
        'qual.device_fn': 210,
        'sbind': 6,
        'strip.asm': 0,
        'strip.launch_bounds': 0,
        'strip.nvtx': 0,
        'strip.restrict': 0,
        'value_type': 86,
    },
    'steps.cuh': {
        'aspace.thread': 210,
        'include': 1,
        'qual.extern_device': 210,
    },
}

ACCUM_GUARD_OPEN = '#ifdef __HIPCC__  // CUDA: stepAccumA in accum_tu.cu (accum_tu_gen.py)'
ACCUM_GUARD_CLOSE = '#endif  // __HIPCC__'

FP_TYPES = ('Fp', 'Val', 'FpExt', 'ExtVal')

# ------------------------------------------------------------------------------------------------
# Address-space policy (plan 2.1). Each row: CUDA base types, which declarators it applies to
# ('*' pointer, '&' reference), the MSL address space, and why. 'REJECT' rows are errors: the
# Metal kernels take explicit buffer bindings and build their thread-local context structs
# themselves (plan 2.0 "Kernel context passing"); a hand-written wrapper (R4/R5) replaces them.
# ------------------------------------------------------------------------------------------------
ASPACE_POLICY = [
    (('ExecContext', 'PreflightTrace', 'LookupTables', 'BufferObj', 'MutableBufObj',
      'GlobalBufObj', 'Buffer'), '*&', 'thread',
     'per-thread context / buffer objects the kernel builds on its stack'),
    (('BoundLayout<>',), '&', 'thread', 'bound layouts are thread values'),
    (('char',), '*', 'constant', 'string literals (eqz/log/assert messages)'),
    (('Fp', 'Val', 'FpExt', 'ExtVal', 'uint32_t', 'uint', 'int32_t', 'PreflightCycle',
      'MemoryTransaction'), '*', 'device',
     'trace / witness / scratch / preflight buffers (plan 2.1: every buffer is device)'),
    (('PreflightCycle', 'MemoryTransaction'), '&', 'device', 'element of a preflight buffer'),
    (('Fp', 'Val', 'FpExt', 'ExtVal'), '&', 'REJECT',
     'an Fp reference may alias a thread local or a buffer element: spell its address space'),
    (('DeviceExecContext', 'DeviceAccumContext', 'DeviceAccumContextA', 'WmsInject'), '*&',
     'REJECT', 'struct of device pointers: Metal kernels take explicit [[buffer(n)]] bindings'),
]

ASPACES = ('device', 'thread', 'constant', 'threadgroup')

# Includes of the CUDA sources and their Metal replacements (None: dropped; the Metal headers
# reach fp.h/fpext.h/cuda2metal.h through witgen.h).
INCLUDE_MAP = {
    '"steps.cuh"': '"steps.metal.h"',
    '"witgen.h"': '"witgen.h"',
    '"eval_check.cuh"': '"eval_check.metal.h"',
    '"supra/fp.h"': None,
    '"cuda.h"': None,
    '<cstdint>': None,
    '<cassert>': None,
    '<cuda_runtime.h>': None,
    '<cuda/std/array>': None,
}

# `__launch_bounds__(MACRO)` -> [[max_total_threads_per_threadgroup(C2M_LB_FIRST(MACRO))]]: the
# first of a `256, 8` expansion (pass_kernel).
LB_PRELUDE = '''\
#ifndef C2M_LB_FIRST
#define C2M_LB_FIRST_(a, ...) a
#define C2M_LB_FIRST(...) C2M_LB_FIRST_(__VA_ARGS__, 0)
#endif
'''

# Mixed Fp +- FpExt: build_kernel's fpext.h has them since R3 (it defines
# C2M_NO_FPEXT_MIXED_ADDSUB / R0_FPEXT_MIXED_ADDSUB), so the R2 shim that lived here is gone.
PRELUDE = '''\
// Fault bit and assert mapping used by cuda2metal.py output (plan 2.9). A violated `assert` in
// the CUDA source sets RV32IM_FAULT_ASSERT in the thread's ctx.tables.bad, which the kernel ORs
// into its fault buffer once; the host bails after sync(). Define C2M_ASSERT before this point to
// change that (e.g. to ((void)0) for CUDA release semantics).
#ifndef RV32IM_FAULT_ASSERT
#define RV32IM_FAULT_ASSERT 0x10u
#endif
#ifndef C2M_ASSERT
#define C2M_ASSERT(ctx, cond) ((void)((ctx).tables.bad |= (cond) ? 0u : RV32IM_FAULT_ASSERT))
#endif
#ifndef C2M_ASSERT_NOCTX
#define C2M_ASSERT_NOCTX(cond) ((void)0)
#endif
''' + LB_PRELUDE


class TranslateError(Exception):
    pass


# ------------------------------------------------------------------------------------------------
# Text helpers
# ------------------------------------------------------------------------------------------------

_MASK_RE = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'', re.S)


def mask(s):
    """Blank comments and string/char literals (same length, newlines kept)."""
    return _MASK_RE.sub(lambda m: re.sub(r'[^\n]', ' ', m.group(0)), s)


_OPEN = {'(': ')', '[': ']', '{': '}'}
_CLOSE = {')', ']', '}'}


def match_close(m, i):
    """Index of the bracket closing the one at m[i] (m is masked text)."""
    depth = 0
    for j in range(i, len(m)):
        c = m[j]
        if c in _OPEN:
            depth += 1
        elif c in _CLOSE:
            depth -= 1
            if depth == 0:
                return j
    raise TranslateError('unbalanced bracket at offset %d' % i)


def split_top(m, s, e):
    """Split m[s:e] at top-level commas; returns (start, end) spans."""
    out, depth, cur = [], 0, s
    for j in range(s, e):
        c = m[j]
        if c in _OPEN:
            depth += 1
        elif c in _CLOSE:
            depth -= 1
        elif c == ',' and depth == 0:
            out.append((cur, j))
            cur = j + 1
    out.append((cur, e))
    return out


def line_indent(s, i):
    ls = s.rfind('\n', 0, i) + 1
    j = ls
    while j < len(s) and s[j] in ' \t':
        j += 1
    return s[ls:j]


def apply_edits(s, edits):
    """edits: (start, end, text), non-overlapping."""
    edits.sort(key=lambda e: e[0])
    out, pos = [], 0
    for a, b, t in edits:
        if a < pos:
            raise TranslateError('overlapping edits at offset %d' % a)
        out.append(s[pos:a])
        out.append(t)
        pos = b
    out.append(s[pos:])
    return ''.join(out)


def where(s, i):
    return 'line %d' % (s.count('\n', 0, i) + 1)


class Counts(dict):
    def add(self, k, n=1):
        self[k] = self.get(k, 0) + n


# ------------------------------------------------------------------------------------------------
# Function discovery (top-level / namespace-level / struct-level function bodies)
# ------------------------------------------------------------------------------------------------

_FUNC_HDR_END = re.compile(
    r'\)\s*(?:const\s*|thread\s*|device\s*|constant\s*|override\s*|noexcept\s*)*'
    r'(?:->\s*[\w:<>, ]+)?\s*$')
_CTOR_INIT = re.compile(r'\)\s*:\s*[\w\s,(){}.\-*&]*$')


def _strip_pp(hdr):
    return '\n'.join(l for l in hdr.split('\n') if not l.lstrip().startswith('#'))


def classify_block(hdr):
    h = _strip_pp(hdr).strip()
    if re.search(r'\bnamespace\b', h) or re.match(r'extern\s+"C"\s*$', h) or h == 'extern':
        return 'ns'
    if re.search(r'(^|\s)(struct|class|union|enum)\b[^()=]*$', h):
        return 'struct'
    if '=' in h.replace('==', '').replace('<=', '').replace('>=', '').replace('!=', ''):
        if not re.search(r'\boperator\s*\S*=', h):
            return 'init'
    if _FUNC_HDR_END.search(h) or _CTOR_INIT.search(h):
        return 'func'
    return 'init'


_HIP_COND = re.compile(r'__HIPCC__|__HIP_DEVICE_COMPILE__|__HIP_PLATFORM')


def blank_hip_branches(m):
    """Blank the branches of #if/#ifdef/#ifndef blocks that only a HIP (ROCm) compile takes
    (Metal never defines __HIPCC__), so that alternative function headers such as
    `#ifdef __HIPCC__ f(a, b) { #else f(a) { #endif` do not unbalance the brace scan."""
    lines = m.split('\n')
    stack = []  # per open #if: 'hip' (the true branch is HIP-only), 'nothip', or None
    out = []
    for line in lines:
        t = line.strip()
        d = re.match(r'#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)', t)
        if d:
            kw, rest = d.group(1), d.group(2)
            if kw in ('ifdef', 'ifndef', 'if'):
                if not _HIP_COND.search(rest) or '||' in rest:
                    stack.append(None)  # not decided by HIP alone: keep both branches
                elif kw == 'ifndef' or re.match(r'\s*!', rest):
                    stack.append('nothip')
                else:
                    stack.append('hip')
            elif kw == 'else' and stack:
                stack[-1] = {'hip': 'nothip', 'nothip': 'hip'}.get(stack[-1])
            elif kw == 'elif' and stack:
                stack[-1] = None  # conservative: keep the remaining branches
            elif kw == 'endif' and stack:
                stack.pop()
            out.append(line)
            continue
        out.append(' ' * len(line) if 'hip' in stack else line)
    return '\n'.join(out)


_PP_LINE_RE = re.compile(r'(?m)^[ \t]*#(?:[^\n]*\\\n)*[^\n]*')


def find_functions(m):
    """Function definitions not nested in another function: (hdr_start, open, close).
    Preprocessor lines (with continuations) are ignored."""
    m = _PP_LINE_RE.sub(lambda mo: re.sub(r'[^\n]', ' ', mo.group(0)), blank_hip_branches(m))
    stack, spans, boundary = [], [], 0
    nfunc = 0
    for mo in re.finditer(r'[{};]', m):
        c, i = mo.group(), mo.start()
        if c == ';':
            if nfunc == 0 and (not stack or stack[-1][0] in ('ns', 'struct')):
                boundary = i + 1
            continue
        if c == '{':
            if nfunc or (stack and stack[-1][0] == 'init'):
                stack.append(('blk' if nfunc else 'init', i, 0))
                continue
            kind = classify_block(m[boundary:i])
            stack.append((kind, i, boundary))
            if kind == 'func':
                nfunc += 1
            boundary = i + 1
            continue
        # '}'
        if not stack:
            raise TranslateError('unbalanced } at offset %d' % i)
        kind, o, b = stack.pop()
        if kind == 'func':
            nfunc -= 1
            spans.append((b, o, i))
        if nfunc == 0 and (not stack or stack[-1][0] in ('ns', 'struct')):
            boundary = i + 1
    if stack:
        raise TranslateError('unbalanced { (%d open blocks)' % len(stack))
    return spans


def header_text(m, hdr_start, open_):
    """Function header with preprocessor lines and leading blank lines removed."""
    return _strip_pp(m[hdr_start:open_]).strip()


def header_name(hdr):
    hdr = re.sub(r'__attribute__\s*\(\(.*?\)\)', '', hdr)
    hdr = re.sub(r'\b__launch_bounds__\s*\([^()]*\)', '', hdr)
    mo = re.search(r'([~\w]+)\s*\(', hdr)
    return mo.group(1) if mo else None


# ------------------------------------------------------------------------------------------------
# Passes
# ------------------------------------------------------------------------------------------------


def pass_strip(s, counts):
    m = mask(s)
    edits = []
    for mo in re.finditer(r'\b__restrict__\b', m):
        edits.append((mo.start(), mo.end(), ''))
        counts.add('strip.restrict')
    for mo in re.finditer(r'\bnvtx3::scoped_range\s+\w+\s*\(|\bnvtxRange(?:Push|Pop)\w*\s*\(', m):
        e = match_close(m, mo.end() - 1) + 1
        mo2 = re.match(r'\s*;', m[e:])
        if not mo2:
            raise TranslateError('nvtx call not a statement at ' + where(s, mo.start()))
        edits.append((mo.start(), e + mo2.end(), ''))
        counts.add('strip.nvtx')
    for mo in re.finditer(r'\b(?:asm|__asm__)\s*(?:volatile\s*|__volatile__\s*)?\(', m):
        o = mo.end() - 1
        e = match_close(m, o) + 1
        tmpl = re.match(r'\s*((?:"(?:\\.|[^"\\])*"\s*)+)', s[o + 1:e - 1])
        text = ''.join(re.findall(r'"((?:\\.|[^"\\])*)"', tmpl.group(1))) if tmpl else None
        if text is None or not (text.strip() == '' or text.lstrip().startswith(';')):
            raise TranslateError('asm with instructions at %s cannot be stripped: %s'
                                 % (where(s, mo.start()), s[mo.start():e][:80]))
        mo2 = re.match(r'\s*;', m[e:])
        edits.append((mo.start(), e + (mo2.end() if mo2 else 0), ''))
        counts.add('strip.asm')
    return apply_edits(s, edits)


def pass_include(s, counts, include_map=None, metal_ns=METAL_NS):
    include_map = INCLUDE_MAP if include_map is None else include_map
    out = []
    for line in s.split('\n'):
        mo = re.match(r'\s*#\s*include\s+(\S+)', line)
        if mo:
            key = mo.group(1)
            if key not in include_map:
                raise TranslateError('no Metal mapping for #include %s' % key)
            rep = include_map[key]
            counts.add('include')
            if rep is None:
                continue
            line = '#include ' + rep
        out.append(line)
    s = '\n'.join(out)
    n = s.count(CUDA_NS)
    if n:
        s = s.replace(CUDA_NS, metal_ns)
    return s


def pass_accum_guard(s, counts):
    n_open = s.count(ACCUM_GUARD_OPEN)
    n_close = s.count(ACCUM_GUARD_CLOSE)
    if n_open != n_close or n_open > 1:
        raise TranslateError('steps.cu accum guard unbalanced (%d open, %d close)'
                             % (n_open, n_close))
    if n_open:
        s = s.replace(ACCUM_GUARD_OPEN, '#ifndef RV32IM_METAL_STEPS_NO_ACCUM  // CUDA: '
                      '#ifdef __HIPCC__ (stepAccumA in accum_tu.cu)')
        s = s.replace(ACCUM_GUARD_CLOSE, '#endif  // RV32IM_METAL_STEPS_NO_ACCUM')
        counts.add('accum_guard', 2)
    return s


def pass_array(s, counts):
    m = mask(s)
    edits = [(mo.start(), mo.end(), 'r0::array<')
             for mo in re.finditer(r'(?:::)?\bcuda::std::array\s*<', m)]
    counts.add('array', len(edits))
    return apply_edits(s, edits)


_TYPEDEF_RE = re.compile(
    r'\busing\s+(\w+)\s*=\s*(?:::)?(?:cuda::std::array|r0::array|metal::array)\s*<\s*'
    r'([^<>;]+?)\s*,\s*(\w+)\s*>\s*;')


def collect_array_typedefs(*texts):
    table = {}
    for t in texts:
        for mo in _TYPEDEF_RE.finditer(mask(t)):
            table[mo.group(1)] = (mo.group(2).strip(), mo.group(3))
    return table


def pass_value_type(s, counts, typedefs):
    m = mask(s)
    edits = []
    for mo in re.finditer(r'\b(\w+)\s*::\s*value_type\b', m):
        name = mo.group(1)
        if name not in typedefs:
            raise TranslateError('%s::value_type at %s: %s is not a known array typedef'
                                 % (name, where(s, mo.start()), name))
        edits.append((mo.start(), mo.end(), typedefs[name][0]))
    counts.add('value_type', len(edits))
    return apply_edits(s, edits)


class Uniq:
    def __init__(self):
        self.n = 0

    def __call__(self):
        self.n += 1
        return self.n


def pass_sbind(s, counts, uniq):
    m = mask(s)
    edits = []
    for mo in re.finditer(r'\bauto\s*(?:&\s*)?\[\s*([\w\s,]+?)\s*\]\s*=', m):
        if m[mo.start():mo.end()].replace(' ', '').startswith('auto&'):
            raise TranslateError('structured binding by reference at ' + where(s, mo.start()))
        names = [n.strip() for n in mo.group(1).split(',')]
        # statement end: next ';' at depth 0
        j, depth = mo.end(), 0
        while True:
            c = m[j]
            if c in _OPEN:
                depth += 1
            elif c in _CLOSE:
                depth -= 1
            elif c == ';' and depth == 0:
                break
            j += 1
        k = uniq()
        ind = line_indent(s, mo.start())
        tmp = 'c2m_sb%d' % k
        rep = 'auto %s =%s;' % (tmp, s[mo.end():j])
        for idx, n in enumerate(names):
            rep += '\n%sauto %s = %s[%d];' % (ind, n, tmp, idx)
        edits.append((mo.start(), j + 1, rep))
        counts.add('sbind')
    return apply_edits(s, edits)


def _parse_lambda(s, m, a, b):
    """s[a:b] is a lambda argument (optionally parenthesized). Returns (params, body_s, body_e)
    with params as [(type, name)] and the body span (inside the braces) in s coordinates."""
    i = a
    while m[i].isspace():
        i += 1
    while m[i] == '(':
        close = match_close(m, i)
        if m[close + 1:b].strip():
            break
        i += 1
        while m[i].isspace():
            i += 1
    if not m.startswith('[&]', i):
        raise TranslateError('map/reduce functor at %s is not a [&] lambda' % where(s, i))
    i += 3
    while m[i].isspace():
        i += 1
    if m[i] != '(':
        raise TranslateError('lambda without parameter list at ' + where(s, i))
    pe = match_close(m, i)
    params = []
    for ps, pe2 in split_top(m, i + 1, pe):
        p = ' '.join(s[ps:pe2].split())
        mo = re.match(r'(.+?)\s*\b(\w+)$', p)
        if not mo:
            raise TranslateError('cannot parse lambda parameter %r at %s' % (p, where(s, ps)))
        params.append((mo.group(1).strip(), mo.group(2)))
    j = pe + 1
    while m[j].isspace():
        j += 1
    if m[j] != '{':
        raise TranslateError('lambda at %s has a trailing return type or specifier' % where(s, j))
    be = match_close(m, j)
    return params, j + 1, be


def _lambda_return(s, m, bs, be):
    """The lambda body's single top-level `return E;` (last statement): (ret_s, expr_s, expr_e,
    ret_end)."""
    rets = []
    depth = 0
    i = bs
    while i < be:
        c = m[i]
        if c in _OPEN:
            depth += 1
        elif c in _CLOSE:
            depth -= 1
        elif c == 'r' and m.startswith('return', i) and (i == 0 or not (m[i - 1].isalnum() or
                                                                         m[i - 1] == '_')) \
                and not (m[i + 6].isalnum() or m[i + 6] == '_'):
            rets.append((i, depth))
        i += 1
    if len(rets) != 1 or rets[0][1] != 0:
        raise TranslateError('lambda body at %s needs exactly one top-level return (found %d)'
                             % (where(s, bs), len(rets)))
    r = rets[0][0]
    j, depth = r + 6, 0
    while True:
        c = m[j]
        if c in _OPEN:
            depth += 1
        elif c in _CLOSE:
            depth -= 1
        elif c == ';' and depth == 0:
            break
        j += 1
    if m[j + 1:be].strip():
        raise TranslateError('lambda return at %s is not the last statement' % where(s, r))
    return r, r + 6, j, j + 1


def _reindent(text, old, new):
    if not old and not new:
        return text
    out = []
    for line in text.split('\n'):
        if line.startswith(old):
            line = new + line[len(old):]
        elif line.strip():
            line = new + line.lstrip()
        out.append(line)
    return '\n'.join(out)


def expand_one_site(s, uniq, counts, map_lowering='loop', typedefs=None):
    """Expand the last map/reduce call site of s (it contains no other site). Returns new s or
    None when there is no site."""
    m = mask(s)
    sites = list(re.finditer(r'(?<![\w.>:])(map|reduce)\s*\(', m))
    if not sites:
        return None
    mo = sites[-1]
    kind = mo.group(1)
    o = mo.end() - 1
    c = match_close(m, o)
    tail = re.match(r'\s*;', m[c + 1:])
    if not tail:
        raise TranslateError('%s(...) at %s is not a whole statement'
                             % (kind, where(s, mo.start())))
    stmt_end = c + 1 + tail.end()
    st = max(m.rfind(';', 0, mo.start()), m.rfind('{', 0, mo.start()),
             m.rfind('}', 0, mo.start())) + 1
    pre = m[st:mo.start()]
    dm = re.match(r'(\s*)([A-Za-z_][\w:<>, ]*?)\s+(\w+)\s*=\s*$', pre)
    if not dm:
        raise TranslateError('%s(...) at %s: unsupported statement form %r'
                             % (kind, where(s, mo.start()), pre.strip()))
    decl_s = st + dm.end(1)
    rtype, rname = dm.group(2).strip(), dm.group(3)
    args = split_top(m, o + 1, c)
    if kind == 'map' and len(args) not in (2, 3) or kind == 'reduce' and len(args) not in (3, 4):
        raise TranslateError('%s with %d arguments at %s' % (kind, len(args), where(s, mo.start())))
    params, bs, be = _parse_lambda(s, m, *args[-1])
    nin = len(args) - 1 if kind == 'map' else len(args) - 1  # value args incl. start for reduce
    if len(params) != nin:
        raise TranslateError('%s lambda at %s takes %d parameters, expected %d'
                             % (kind, where(s, bs), len(params), nin))
    r, es, ee, rend = _lambda_return(s, m, bs, be)
    k = uniq()
    ind = line_indent(s, decl_s)
    ind2 = ind + '  '
    it = 'c2m_i%d' % k
    av = 'c2m_a%d' % k
    bv = 'c2m_b%d' % k
    arg_txt = [s[x:y].strip() for x, y in args[:-1]]
    lines = []
    if kind == 'map':
        lines.append('auto %s = %s;' % (av, arg_txt[0]))
        elem_params = [(params[0], '%s[%s]' % (av, it))]
        if len(args) == 3:
            lines.append('%sauto %s = %s;' % (ind, bv, arg_txt[1]))
            pt, pn = params[1]
            src = ('LAYOUT_SUBSCRIPT(%s, %s)' % (bv, it) if pt.startswith('BoundLayout<')
                   else '%s[%s]' % (bv, it))
            elem_params.append((params[1], src))
        lines.append('%s%s %s;' % (ind, rtype, rname))
        sink = '%s[%s] = ' % (rname, it)
    else:
        acc = 'c2m_acc%d' % k
        lines.append('auto %s = %s;' % (av, arg_txt[0]))
        lines.append('%sauto %s = %s;' % (ind, acc, arg_txt[1]))
        elem_params = [(params[0], acc), (params[1], '%s[%s]' % (av, it))]
        if len(args) == 4:
            lines.append('%sauto %s = %s;' % (ind, bv, arg_txt[2]))
            pt, pn = params[2]
            if not pt.startswith('BoundLayout<'):
                raise TranslateError('reduce layout parameter at %s is not a BoundLayout'
                                     % where(s, bs))
            elem_params.append((params[2], 'LAYOUT_SUBSCRIPT(%s, %s)' % (bv, it)))
        sink = '%s = ' % acc
    body = s[bs:r].rstrip()
    body_lines = body.split('\n')
    # body indentation: that of its first non-blank line
    first = next((l for l in body_lines if l.strip()), '')
    old_ind = first[:len(first) - len(first.lstrip())]
    body = _reindent(body.strip('\n'), old_ind, ind2) if body.strip() else ''
    expr = ' '.join(s[es:ee].split())
    if map_lowering == 'unroll':
        # One block per element, in element order (CUDA's loop order, so the calls' side effects
        # keep their order), each leaving its value in its own scalar; a map's array is then
        # brace-initialized from the scalars, a reduce's accumulator is threaded through the
        # blocks as c2m_accK_0 .. c2m_accK_N. No private array is default-constructed and then
        # written element by element in a loop that calls a function: the shape the Apple GPU
        # loses the last store of (docs/metal-port/notes/R17.md).
        if kind == 'map':
            arr = rtype
        else:
            decls = re.findall(r'\b(\w+)\s+%s\s*=' % re.escape(arg_txt[0]), m[:mo.start()]) \
                if re.match(r'^\w+$', arg_txt[0]) else []
            arr = decls[-1] if decls else None
        if arr not in (typedefs or {}):
            raise TranslateError('%s at %s: the array type %s is not a known array typedef '
                                 '(needed to unroll it)' % (kind, where(s, mo.start()), arr))
        etype, n = typedefs[arr]
        n = int(n)
        if kind == 'map':
            lines.pop()  # the default-constructed result declaration
            outs = ['c2m_r%d_%d' % (k, j) for j in range(n)]
        else:
            lines[1] = lines[1].replace('auto %s =' % acc, '%s %s_0 =' % (rtype, acc), 1)
            outs = ['%s_%d' % (acc, j + 1) for j in range(n)]
        # R17b: each element's scalar is declared with its value (a GNU statement expression,
        # `T out = ({ ...; expr; });`), not default-constructed and assigned after the block's
        # calls: a default-constructed private aggregate (an IR constructor loop for array
        # members) that is written after a call and then read is the lost-store shape again,
        # without a loop (docs/metal-port/notes/R17.md, "R17b hardening").
        for j in range(n):
            lines.append('%s%s %s = ({' % (ind, etype if kind == 'map' else rtype, outs[j]))
            for (pt, pn), src in elem_params:
                src = '%s_%d' % (acc, j) if kind == 'reduce' and src == acc \
                    else src.replace('[%s]' % it, '[%d]' % j).replace(', %s)' % it, ', %d)' % j)
                lines.append('%s%s %s = %s;' % (ind2, pt, pn, src))
            if body.strip():
                lines.append(body)
            lines.append('%s%s;' % (ind2, expr))
            lines.append('%s});' % ind)
        if kind == 'map':
            lines.append('%s%s %s = {{%s}};' % (ind, rtype, rname, ', '.join(outs)))
        else:
            lines.append('%s%s %s = %s;' % (ind, rtype, rname, outs[-1]))
        counts.add('lambda.' + kind)
        counts.add('lambda.%s.unroll' % kind)
        return s[:decl_s] + '\n'.join(lines) + s[stmt_end:]
    if map_lowering != 'loop':
        raise TranslateError('unknown map_lowering %r' % map_lowering)
    lines.append('%sfor (uint %s = 0; %s < %s.size(); %s++) {' % (ind, it, it, av, it))
    for (pt, pn), src in elem_params:
        lines.append('%s%s %s = %s;' % (ind2, pt, pn, src))
    if body.strip():
        lines.append(body)
    lines.append('%s%s%s;' % (ind2, sink, expr))
    lines.append('%s}' % ind)
    if kind == 'reduce':
        lines.append('%s%s %s = %s;' % (ind, rtype, rname, acc))
    counts.add('lambda.' + kind)
    return s[:decl_s] + '\n'.join(lines) + s[stmt_end:]


_STRUCT_RE = re.compile(r'(?ms)^struct (\w+) \{\n(.*?)^\};')
_FIELD_RE = re.compile(r'^\s*([A-Za-z_][\w:<>, ]*?)\s+(\w+);\s*$')


def zero_init_fn(text):
    """R17b: T -> an explicit brace initializer of T with every Val zero, from the generated
    types (the text of kernels/metal/types.metal.inc), e.g. `BigIntTopStateStruct{Val(0), Val(0),
    Val16Array{{Val(0), ...}}}`. Value-initializing T (T{}) default-constructs its Val arrays in
    an IR constructor loop; this is the same value written by straight-line stores."""
    arrays = {}
    for m in re.finditer(r'(?m)^using (\w+) = r0::array<([\w:]+),\s*(\d+)>;', text):
        arrays[m.group(1)] = (m.group(2), int(m.group(3)))
    structs = {}
    for m in _STRUCT_RE.finditer(text):
        fields = []
        for line in m.group(2).split('\n'):
            if not line.strip():
                continue
            fm = _FIELD_RE.match(line)
            if not fm:
                raise TranslateError('zero_init: unsupported member line %r in struct %s'
                                         % (line, m.group(1)))
            fields.append(fm.group(1).strip())
        structs[m.group(1)] = fields

    def zero(t):
        if t in ('Val', 'Fp'):
            return '%s(0)' % t
        if t in ('ExtVal', 'FpExt'):
            return '%s(0u)' % t
        if t in arrays:
            et, n = arrays[t]
            return '%s{{%s}}' % (t, ', '.join(zero(et) for _ in range(n)))
        if t in structs:
            return '%s{%s}' % (t, ', '.join(zero(f) for f in structs[t]))
        raise TranslateError('zero_init: unknown type %r (not Val, an r0::array alias or a '
                                 'struct of types.metal.inc)' % t)
    return zero


_MUX_DECL_RE = re.compile(r'(?m)^([ \t]*)([A-Za-z_][\w:<>, ]*?)[ \t]+(\w+);\n\1if \(')


def _mux_arm(s, m, ob, cb, name, where_):
    """Body of a mux arm s[ob+1:cb] -> (statements before the value, value text). The arm's
    last top-level statement must be `name = VALUE;` and name must not appear before it."""
    body = s[ob + 1:cb]
    mb = m[ob + 1:cb]
    ends, depth = [], 0
    for j, c in enumerate(mb):
        if c in _OPEN:
            depth += 1
        elif c in _CLOSE:
            depth -= 1
        elif c == ';' and depth == 0:
            ends.append(j)
    if not ends or mb[ends[-1] + 1:].strip():
        raise TranslateError('mux arm at %s does not end with a statement' % where_)
    st = (ends[-2] + 1) if len(ends) > 1 else 0
    last = mb[st:ends[-1]]
    am = re.match(r'\s*%s\s*=(?!=)' % re.escape(name), last)
    if not am:
        return body.rstrip(), None
    if re.search(r'\b%s\b' % re.escape(name), mb[:st]) or re.search(r'\breturn\b', mb):
        raise TranslateError('mux arm at %s reads or assigns %s before its value (or returns)'
                             % (where_, name))
    return body[:st].rstrip(), body[st + am.end():ends[-1]].strip()


def pass_mux_init(s, counts, zero=None):
    """R17b: `T x; if (c0) { ...; x = v0; } else if (c1) { ...; x = v1; } else { <no value> }`
    -> `T x = (c0) ? ({ ...; v0; }) : (c1) ? ({ ...; v1; }) : ({ <no value>; T{}; });`, so
    the mux result is declared with its value instead of default-constructed (an IR
    constructor loop for array members) and assigned after the arm's calls: the R17 lost-store
    shape without a loop. Same conditions, evaluated in the same order; the same arm runs. A
    final arm without a value (the unreachable-arm C2M_ASSERT, or no else) yields zero(T) (an
    explicit all-zero brace initializer, so no constructor loop either), or T{} without zero,
    the value the default-constructed declaration had. Innermost (last) first."""
    while True:
        m = mask(s)
        hits = list(_MUX_DECL_RE.finditer(m))
        if not hits:
            return s
        mo = hits[-1]
        ind, typ, name = mo.group(1), mo.group(2).strip(), mo.group(3)
        if typ in ('return', 'break', 'continue', 'goto'):
            raise TranslateError('mux_init: unexpected statement %r at %s'
                                 % (mo.group(0), where(s, mo.start())))
        pos = mo.end() - len('if (')
        arms = []  # (cond or None, stmts, value or None)
        while True:
            assert m.startswith('if (', pos), pos
            po = pos + 3
            pc = match_close(m, po)
            cond = s[po + 1:pc].strip()
            ob = pc + 1
            while m[ob] in ' \t\n':
                ob += 1
            if m[ob] != '{':
                raise TranslateError('mux at %s: if without a block' % where(s, pos))
            cb = match_close(m, ob)
            stmts, val = _mux_arm(s, m, ob, cb, name, where(s, ob))
            if val is None:
                raise TranslateError('mux at %s: arm %d does not end with `%s = ...;`'
                                     % (where(s, pos), len(arms), name))
            arms.append((cond, stmts, val))
            em = re.match(r'[ \t\n]*else[ \t\n]*', m[cb + 1:])
            end = cb + 1
            if not em:
                break
            nxt = cb + 1 + em.end()
            if m.startswith('if (', nxt):
                pos = nxt
                continue
            if m[nxt] != '{':
                raise TranslateError('mux at %s: else without a block' % where(s, nxt))
            ecb = match_close(m, nxt)
            stmts, val = _mux_arm(s, m, nxt, ecb, name, where(s, nxt))
            arms.append((None, stmts, val))
            end = ecb + 1
            break
        if arms[-1][0] is not None:
            arms.append((None, '', None))
        ind2 = ind + '  '
        out = ['%s%s %s =' % (ind, typ, name)]
        for k, (cond, stmts, val) in enumerate(arms):
            head = ('%s  (%s) ? ({' % (ind, cond)) if cond is not None else ('%s  ({' % ind)
            out.append(head)
            if stmts.strip():
                out.append(_reindent(stmts.strip('\n'), ind2, ind2 + '  '))
            out.append('%s    %s;' % (ind, val if val is not None else
                                        (zero(typ) if zero else '%s{}' % typ)))
            out.append('%s  })%s' % (ind, ' :' if cond is not None else ';'))
        s = s[:mo.start()] + '\n'.join(out) + s[end:]
        counts.add('mux.init')


def pass_lambda(s, counts, uniq, map_lowering='loop', typedefs=None):
    m = mask(s)
    if '[&]' not in m and '[=]' not in m:
        return s
    funcs = find_functions(m)
    edits = []
    for hs, o, c in funcs:
        if '[&]' not in m[o:c] and '[=]' not in m[o:c]:
            continue
        body = s[o:c + 1]
        while True:
            nb = expand_one_site(body, uniq, counts, map_lowering, typedefs)
            if nb is None:
                break
            body = nb
        mb = mask(body)
        if '[&]' in mb or '[=]' in mb:
            raise TranslateError('lambda outside a map/reduce call in function at '
                                 + where(s, o))
        edits.append((o, c + 1, body))
    return apply_edits(s, edits)


_FPP_RE = re.compile(r'\b((?:const\s+)?(?:Fp|FpExt|Val|ExtVal))\s*\*\s*\*\s*(\w+)')


def _fpp_param_pos(m, hs, o, fname, at):
    """Index of the parameter containing offset `at` in the header m[hs:o] of fname."""
    mo = re.search(r'\b%s\s*\(' % re.escape(fname), m[hs:o])
    po = hs + mo.end() - 1
    pc = match_close(m, po)
    for i, (a, b) in enumerate(split_top(m, po + 1, pc)):
        if a <= at < b:
            return i
    raise TranslateError('cannot locate the Fp** parameter of %s' % fname)


def pass_fpp(s, counts):
    """`Fp** args` parameters -> K explicit `device Fp*` parameters (definitions, prototypes),
    `args[k]` -> `args_k`, and call sites: an argument that is the caller's own Fp** parameter
    is forwarded as `args_0, ..., args_{K-1}`; an argument that is a local pointer array
    `Fp* a[N] = {e0, e1, ...};` is replaced by its first K initializer elements (`a[k]` ->
    `(ek)` elsewhere in the function and the array declaration is removed). Anything else is an
    error."""
    m = mask(s)
    if not _FPP_RE.search(m):
        return s
    funcs = find_functions(m)
    edits = []
    # 1. definitions: fname -> {pos, pname, k, type, span}
    fdefs = {}
    for hs, o, c in funcs:
        hdr = m[hs:o]
        for mo in _FPP_RE.finditer(hdr):
            fname = header_name(_strip_pp(hdr))
            if fname in fdefs:
                raise TranslateError('two definitions of %s with an Fp** parameter' % fname)
            name = mo.group(2)
            idx = [int(x) for x in re.findall(r'\b%s\s*\[\s*(\d+)\s*\]' % name, m[o:c])]
            fdefs[fname] = dict(pos=_fpp_param_pos(m, hs, o, fname, hs + mo.start()),
                                pname=name, k=max(idx) + 1 if idx else 0, type=mo.group(1),
                                decl=(hs + mo.start(), hs + mo.end()), body=(o, c))
    if not fdefs:
        # prototypes only: the size is unknown
        mo = _FPP_RE.search(m)
        raise TranslateError('%s** %s at %s: no definition in this input gives its size'
                             % (mo.group(1), mo.group(2), where(s, mo.start())))
    # 2. call sites of those functions inside function bodies
    call_re = re.compile(r'(?<![\w.])(?:\w+::)*(%s)\s*\(' % '|'.join(map(re.escape, fdefs)))
    calls = []  # (enclosing (hs, o, c), callee, arg span, arg text)
    for hs, o, c in funcs:
        for mo in call_re.finditer(m, o, c):
            po = mo.end() - 1
            pc = match_close(m, po)
            spans = split_top(m, po + 1, pc)
            callee = mo.group(1)
            pos = fdefs[callee]['pos']
            if pos >= len(spans):
                raise TranslateError('call of %s at %s: too few arguments'
                                     % (callee, where(s, mo.start())))
            a, b = spans[pos]
            calls.append(((hs, o, c), callee, (a, b), m[a:b].strip()))
    # K of a forwarding function must cover its callees' K (fixpoint)
    changed = True
    while changed:
        changed = False
        for (hs, o, c), callee, _, arg in calls:
            for f, d in fdefs.items():
                if d['body'] == (o, c) and arg == d['pname'] and d['k'] < fdefs[callee]['k']:
                    d['k'] = fdefs[callee]['k']
                    changed = True
    forwarded = set()
    arrays = {}  # (o, decl_start) -> info
    for (hs, o, c), callee, (a, b), arg in calls:
        k = fdefs[callee]['k']
        own = next((d for d in fdefs.values() if d['body'] == (o, c)), None)
        if own is not None and arg == own['pname']:
            rep = ', '.join('%s_%d' % (arg, i) for i in range(k))
            forwarded.add(a)
        else:
            dm = None
            if re.match(r'^\w+$', arg):
                dm = re.search(r'\b(?:const\s+)?(?:Fp|FpExt|Val|ExtVal)\s*\*\s*%s\s*\[\s*\d*\s*\]'
                               r'\s*=\s*\{([^{}]*)\}\s*;' % arg, m[o:c])
            if dm is None:
                raise TranslateError('call of %s at %s: the Fp** argument %r is neither the '
                                     "caller's Fp** parameter nor a local `Fp* %s[N] = {...};`"
                                     % (callee, where(s, a), arg, arg))
            elems = [s[x:y].strip() for x, y in split_top(m, o + dm.start(1), o + dm.end(1))]
            if any(not re.match(r'^[\w.]+$', e) for e in elems):
                raise TranslateError('pointer array %s at %s: only plain names can be '
                                     'substituted, got %r' % (arg, where(s, o + dm.start()), elems))
            if len(elems) < k:
                raise TranslateError('pointer array %s at %s has %d elements, %s takes %d'
                                     % (arg, where(s, o + dm.start()), len(elems), callee, k))
            rep = ', '.join(elems[:k])
            arrays[(o, o + dm.start())] = dict(name=arg, elems=elems, span=(o + dm.start(),
                                                                          o + dm.end()),
                                               body=(o, c))
        edits.append((a, b, (' ' if s[a:a + 1] == ' ' else '') + rep))
        counts.add('fpp.call')
    # 3. local pointer arrays: remaining a[k] -> (ek); the declaration goes
    taken = [(a, b) for a, b, _ in edits]
    for info in arrays.values():
        name, (o, c) = info['name'], info['body']
        ds, de = info['span']
        for mo in re.finditer(r'\b%s\b' % name, m[o:c]):
            p = o + mo.start()
            if ds <= p < de or any(a <= p < b for a, b in taken):
                continue
            im = re.match(r'%s\s*\[\s*(\d+)\s*\]' % name, m[p:])
            if not im or int(im.group(1)) >= len(info['elems']):
                raise TranslateError('pointer array %s at %s is used other than as %s[<literal>] '
                                     'or an Fp** call argument' % (name, where(s, p), name))
            edits.append((p, p + im.end(), '(%s)' % info['elems'][int(im.group(1))]))
        edits.append((ds, de, '/* cuda2metal: %s[] passed as explicit pointers */' % name))
        counts.add('fpp.array')
    # 4. definitions: a[k] -> a_k, the parameter -> K pointers
    for fname, d in fdefs.items():
        o, c = d['body']
        name, k = d['pname'], d['k']
        idx = list(re.finditer(r'\b%s\s*\[\s*(\d+)\s*\]' % name, m[o:c]))
        bare = [o + x.start() for x in re.finditer(r'\b%s\b' % name, m[o:c])]
        fwd = [p for p in bare if any(p == a or (a < p and not m[a:p].strip()) for a in forwarded)]
        if len(bare) != len(idx) + len(fwd) or k == 0:
            raise TranslateError('%s** %s in %s is used other than as %s[<literal>] or forwarded '
                                 'to an Fp** call' % (d['type'], name, fname, name))
        for mo2 in idx:
            edits.append((o + mo2.start(), o + mo2.end(), '%s_%s' % (name, mo2.group(1))))
        a, b = d['decl']
        edits.append((a, b, ', '.join('device %s* %s_%d' % (d['type'], name, i)
                                      for i in range(k))))
        counts.add('fpp')
    # prototypes: Fp** outside the definitions found above
    covered = [d['decl'] for d in fdefs.values()]
    for mo in _FPP_RE.finditer(m):
        if any(a <= mo.start() < b for a, b in covered):
            continue
        st = max(m.rfind(';', 0, mo.start()), m.rfind('}', 0, mo.start()),
                 m.rfind('{', 0, mo.start())) + 1
        fname = header_name(_strip_pp(m[st:mo.start()]))
        d = fdefs.get(fname)
        if d is None or d['pname'] != mo.group(2):
            raise TranslateError('%s** %s at %s: no definition of %s in this input gives its size'
                                 % (mo.group(1), mo.group(2), where(s, mo.start()), fname))
        edits.append((mo.start(), mo.end(),
                      ', '.join('device %s* %s_%d' % (mo.group(1), mo.group(2), i)
                                for i in range(d['k']))))
        counts.add('fpp')
    return apply_edits(s, edits)


# ------------------------------------------------------------------------------------------------
# Eval mode passes (R6): eval_check_0..3.cu + eval_check.cuh + the eval_check kernel
# ------------------------------------------------------------------------------------------------

POLY_MIX = 'poly_mix'


def pass_poly_mix(s, counts, name=POLY_MIX):
    """`__constant__ FpExt poly_mix[N];` (no initializer; CUDA fills it with
    cudaMemcpyToSymbol) -> a `const FpExt* poly_mix` parameter (made `const device` by the
    eval_aspace / kernel passes) threaded through every function that uses it, directly or
    through a call, with `, poly_mix` appended at their call sites. A kernel in that set gets it
    as its last buffer binding. A function that already has a `poly_mix` parameter is left
    alone."""
    m = mask(s)
    decls = list(re.finditer(r'\b__constant__\s+FpExt\s+%s\s*\[([^\]]*)\]\s*([=;{])' % name, m))
    if not decls:
        return s
    edits = []
    for d in decls:
        if d.group(2) != ';':
            raise TranslateError('__constant__ FpExt %s at %s has an initializer: it is a '
                                 'constant table, not a host-filled symbol' % (name, where(s, d.start())))
        n = d.group(1).strip()
        edits.append((d.start(), d.end(), '// cuda2metal: __constant__ FpExt %s[%s] -> a '
                      '`const device FpExt* %s` parameter (a kernel buffer of %s entries)\n'
                      'static_assert((%s) > 0, "%s: kernel buffer size");'
                      % (name, n, name, n, n, name)))
        counts.add('poly_mix.decl')
    funcs = find_functions(m)
    info = []
    for hs, o, c in funcs:
        hdr = header_text(m, hs, o)
        fname = header_name(hdr)
        own = re.search(r'\b%s\b' % name, hdr[hdr.find('('):]) is not None
        info.append((fname, o, c, own))
    need = set(f for f, o, c, own in info if not own and re.search(r'\b%s\b' % name, m[o:c]))
    changed = bool(need)
    while changed:
        changed = False
        alt = re.compile(r'(?<![\w.])(?:%s)\s*\(' % '|'.join(map(re.escape, sorted(need))))
        for f, o, c, own in info:
            if f in need or own:
                continue
            if alt.search(m, o, c):
                need.add(f)
                changed = True
    if not need:
        return apply_edits(s, edits)
    bodies = sorted((o, c) for _, o, c, _ in info)
    call_re = re.compile(r'(?<![\w.])(?:\w+::)*(%s)\s*\(' % '|'.join(map(re.escape, sorted(need))))
    for mo in call_re.finditer(m):
        po = mo.end() - 1
        pc = match_close(m, po)
        empty = not m[po + 1:pc].strip()
        inside = any(o < mo.start() < c for o, c in bodies)
        if inside:
            edits.append((pc, pc, ('' if empty else ', ') + name))
            counts.add('poly_mix.call')
        else:
            edits.append((pc, pc, ('' if empty else ', ') + 'const FpExt* ' + name))
            counts.add('poly_mix.param')
    return apply_edits(s, edits)


def pass_fpext_index(s, counts):
    """`e[i]` on an FpExt value (CUDA FpExt::operator[]) -> `e.elems[i]` (Metal fpext.h has no
    operator[]). Only names declared as by-value FpExt parameters or `FpExt x` locals of the
    same function are rewritten; FpExt arrays and pointers keep their subscripts. Subscripting
    an `auto` local is refused (its type is not known here)."""
    m = mask(s)
    edits = []
    for hs, o, c in find_functions(m):
        hdr = header_text(m, hs, o)
        body = m[o:c]
        names = set()
        if '(' in hdr:
            for mo in re.finditer(r'\bFpExt\s+(\w+)\s*(?=[,)=])', hdr[hdr.find('('):]):
                names.add(mo.group(1))
        for mo in re.finditer(r'\bFpExt\s+(\w+)\s*(?=[=;({,])', body):
            names.add(mo.group(1))
        arrays = set(re.findall(r'\bFpExt\s+(\w+)\s*\[', body))
        names -= arrays
        autos = set(re.findall(r'\bauto\s+(\w+)\s*=', body))
        for mo in re.finditer(r'(?<![\w.>])(\w+)\s*\[', body):
            n = mo.group(1)
            if n in names:
                edits.append((o + mo.end(1), o + mo.end(1), '.elems'))
                counts.add('fpext_index')
            elif n in autos:
                raise TranslateError('subscript of auto %s at %s: type unknown (FpExt needs '
                                     '.elems[i])' % (n, where(s, o + mo.start())))
    return apply_edits(s, edits)


def pass_eval_aspace(s, counts):
    """Address spaces of the eval chain: `const Fp*`/`const FpExt*` -> `const device` (trace
    buffers, poly_mix); mutable `Fp*`/`FpExt*` -> `thread` (poly_fp's x33/x34 temporaries,
    which the chain functions fill and read); `Fp&`/`FpExt&` is an error. Pointers that already
    carry an address space (kernel parameters) are left alone."""
    m = mask(s)
    edits = []
    rx = re.compile(r'(?<![\w:.>])(?P<const>const\s+)?(?P<type>Fp|FpExt|Val|ExtVal)\b'
                    r'\s*(?P<op>\*(?!\s*\*)|&(?!&))')
    for mo in rx.finditer(m):
        start = mo.start()
        if re.search(r'\b(%s)\s+(const\s+)?$' % '|'.join(ASPACES), m[max(0, start - 20):start]):
            continue
        after = m[mo.end():mo.end() + 40]
        if not re.match(r'\s*(const\s+)?(\w|>|\)|,)', after):
            continue
        if mo.group('op') == '*' and re.match(r'\s+\w', after) and \
                re.search(r'\s$', m[start:mo.end() - 1]) and \
                re.search(r'[\w)\]]\s*$', m[max(0, start - 8):start]):
            continue  # `a Fp * b` in an expression
        if mo.group('op') == '&':
            raise TranslateError('%s& at %s: spell its address space'
                                 % (mo.group('type'), where(s, start)))
        if mo.group('const'):
            edits.append((start, mo.end('const'), 'const device '))
            counts.add('aspace.device')
        else:
            edits.append((start, start, 'thread '))
            counts.add('aspace.thread')
    return apply_edits(s, edits)


def _policy_lookup(base, op):
    key = 'BoundLayout<>' if base.startswith('BoundLayout<') else base
    for types, ops, aspace, why in ASPACE_POLICY:
        if key in types and op in ops:
            return aspace, why
    return None, None


_SCALAR_TYPES = ('uint32_t', 'int32_t', 'uint', 'int', 'size_t', 'Fp', 'FpExt', 'Val', 'ExtVal',
                 'bool', 'uint64_t')


def enclosing_namespaces(m, pos):
    """Names of the namespace blocks of masked text m that contain offset pos (outermost
    first)."""
    out = []
    for mo in re.finditer(r'\bnamespace\s+([\w:]+)\s*\{', m[:pos]):
        if match_close(m, mo.end() - 1) > pos:
            out.append(mo.group(1))
    return out


def pass_kernel(s, counts):
    m = mask(s)
    edits = []
    lb_re = r'__launch_bounds__\s*\(([^()]*)\)\s*'
    kre = re.compile(r'(?:%s)?\b__global__\s+(?:%s)?void\s+(?:%s)?(\w+)\s*\('
                     % (lb_re, lb_re, lb_re))
    for mo in kre.finditer(m):
        lbs = [x for x in mo.groups()[:3] if x is not None]
        name = mo.group(4)
        po = mo.end() - 1
        pc = match_close(m, po)
        bo = pc + 1
        while m[bo].isspace():
            bo += 1
        if m[bo] != '{':
            raise TranslateError('kernel %s is declared without a body' % name)
        bc = match_close(m, bo)
        body = m[bo:bc]
        params, binds = [], []
        for ps, pe in split_top(m, po + 1, pc):
            p = ' '.join(s[ps:pe].split())
            if not p:
                continue
            p = re.sub(r'\s*=\s*[^=]+$', '', p)  # default argument
            p = p.replace('__restrict__', '').strip()
            p = ' '.join(p.split())
            pm = re.match(r'(const\s+)?([\w:]+)\s*(\*|&)?\s*(const\s+)?(\w+)$', p)
            if not pm:
                raise TranslateError('kernel %s: cannot parse parameter %r' % (name, p))
            const, base, op, _, pname = pm.groups()
            const = const or ''
            idx = len(params)
            if op:
                aspace, why = _policy_lookup(base, op)
                if aspace in (None, 'REJECT', 'thread'):
                    raise TranslateError('kernel %s parameter %r: %s' % (
                        name, p, why or 'type not in the address-space policy table'))
                if op == '&':
                    aspace = 'constant'
                params.append('%s%s %s%s %s [[buffer(%d)]]' % (const, aspace, base, op, pname, idx))
            else:
                if base not in _SCALAR_TYPES:
                    raise TranslateError('kernel %s parameter %r: by-value non-scalar' % (name, p))
                params.append('constant %s& %s [[buffer(%d)]]' % (base, pname, idx))
            binds.append('//   buffer(%d): %s' % (idx, p))
        for var, attr in (('threadIdx', 'thread_position_in_threadgroup'),
                          ('blockIdx', 'threadgroup_position_in_grid'),
                          ('blockDim', 'threads_per_threadgroup'),
                          ('gridDim', 'threadgroups_per_grid')):
            if re.search(r'\b%s\b' % var, body):
                params.append('uint3 %s [[%s]]' % (var, attr))
        attr = ''
        if lbs:
            if len(lbs) > 1:
                raise TranslateError('kernel %s has two __launch_bounds__' % name)
            lb = lbs[0].strip()
            first = split_top(lb, 0, len(lb))[0]
            n = lb[first[0]:first[1]].strip()
            if re.match(r'^\d+$', n) or len(split_top(lb, 0, len(lb))) > 1:
                arg = n
            else:
                arg = 'C2M_LB_FIRST(%s)' % n  # a macro that may expand to "N, M"
            attr = '[[max_total_threads_per_threadgroup(%s)]]\n' % arg
            counts.add('strip.launch_bounds')
        ind = ',\n' + ' ' * (len('kernel void %s(' % name))
        hdr = ('// Metal kernel %s (cuda2metal.py): explicit bindings\n%s\n'
               '%skernel void %s(%s) ' % (name, '\n'.join(binds), attr, name, ind.join(params)))
        # Kernels live at global scope: a namespaced kernel gets a mangled metallib entry name
        # (plan 2.4). Close the enclosing namespaces around it and look names up through them.
        nss = enclosing_namespaces(m, mo.start())
        if nss:
            close = ''.join('} // namespace %s (cuda2metal: kernel %s at global scope)\n'
                            % (n, name) for n in reversed(nss))
            hdr = close + 'using namespace %s;\n' % '::'.join(nss) + hdr
            edits.append((bc + 1, bc + 1, '\n' + ''.join('namespace %s {\n' % n for n in nss)))
            counts.add('kernel.hoisted')
        edits.append((mo.start(), bo, hdr))
        counts.add('kernel')
    # any other __launch_bounds__ is an error (it must sit on a kernel)
    taken = [(a, b) for a, b, _ in edits]
    for mo in re.finditer(r'__launch_bounds__', m):
        if not any(a <= mo.start() < b for a, b in taken):
            raise TranslateError('__launch_bounds__ not on a __global__ kernel at '
                                 + where(s, mo.start()))
    return apply_edits(s, edits)


def pass_qual(s, counts):
    m = mask(s)
    edits = []
    funcs = find_functions(m)
    fstarts = sorted((o, c) for _, o, c in funcs)

    def in_func(i):
        # binary search would be faster; the function count is small enough
        lo, hi = 0, len(fstarts)
        while lo < hi:
            mid = (lo + hi) // 2
            if fstarts[mid][0] <= i:
                lo = mid + 1
            else:
                hi = mid
        return lo > 0 and fstarts[lo - 1][0] <= i <= fstarts[lo - 1][1]

    data_re = re.compile(r'(extern\s+)?__device__\s+((?:static\s+)?(?:const(?:expr)?\s+)+)?'
                         r'([A-Za-z_][\w:<>, ]*?)\s+(\w+)\s*(=|;|\[|\{)')
    for mo in re.finditer(r'(?:\b(extern|static)\s+)?\b__device__\b\s*', m):
        if in_func(mo.start()):
            raise TranslateError('__device__ inside a function at ' + where(s, mo.start()))
        dm = data_re.match(m, mo.start())
        if dm and '(' not in m[mo.start():dm.end()]:
            # program-scope data
            quals = (dm.group(2) or '')
            if dm.group(1):
                raise TranslateError('extern __device__ data at %s: Metal has no extern '
                                     'program-scope data; include the definitions'
                                     % where(s, mo.start()))
            rep = 'constant ' + ('constexpr ' if 'constexpr' in quals else '')
            edits.append((mo.start(), dm.start(3), rep))
            counts.add('qual.device_data')
            continue
        if mo.group(1) == 'extern':
            edits.append((mo.start(), mo.end(), ''))
            counts.add('qual.extern_device')
        elif mo.group(1) == 'static':
            edits.append((mo.start(), mo.end(), 'static '))
            counts.add('qual.device_fn')
        else:
            edits.append((mo.start(), mo.end(), ''))
            counts.add('qual.device_fn')
    for mo in re.finditer(r'\b__host__\b\s*', m):
        edits.append((mo.start(), mo.end(), ''))
        counts.add('qual.host')
    for mo in re.finditer(r'\b__noinline__\b', m):
        edits.append((mo.start(), mo.end(), '__attribute__((noinline))'))
        counts.add('qual.noinline')
    for mo in re.finditer(r'\b__forceinline__\b', m):
        edits.append((mo.start(), mo.end(), 'inline __attribute__((always_inline))'))
        counts.add('qual.forceinline')
    for mo in re.finditer(r'\b__constant__\b', m):
        edits.append((mo.start(), mo.end(), 'constant'))
        counts.add('qual.constant')
    # namespace-scope constexpr data (MSL rejects non-`constant` program-scope data)
    for mo in re.finditer(r'(?m)^[ \t]*(constexpr)\s+[A-Za-z_][\w:<>, ]*?\s+\w+\s*(=|\{)', m):
        if in_func(mo.start()) or any(a <= mo.start(1) < b for a, b, _ in edits):
            continue
        before = m[max(0, mo.start(1) - 12):mo.start(1)]
        if re.search(r'\b(constant|static)\s+$', before):
            continue
        edits.append((mo.start(1), mo.start(1), 'constant '))
        counts.add('qual.ns_constexpr')
    return apply_edits(s, edits)


def _aspace_type_alts():
    names = set()
    for types, _, _, _ in ASPACE_POLICY:
        names.update(t for t in types if t != 'BoundLayout<>')
    return '|'.join(sorted(names, key=len, reverse=True))


_ASPACE_RE = re.compile(
    r'(?<![\w:.>])(?P<const>const\s+)?(?P<type>(?:%s)\b|BoundLayout\s*<[^;(){}&*]*>)'
    r'\s*(?P<op>\*(?!\s*\*)|&(?!&))' % _aspace_type_alts())


def pass_aspace(s, counts):
    m = mask(s)
    edits = []
    for mo in _ASPACE_RE.finditer(m):
        start = mo.start()
        before = m[max(0, start - 16):start]
        if re.search(r'\b(%s)\s+$' % '|'.join(ASPACES), before):
            continue
        # `T * x` with spaces on both sides, or a cast/expression: only declarations follow
        # with an identifier, `>` (template argument) or `)` (cast / unnamed parameter).
        after = m[mo.end():mo.end() + 40]
        if not re.match(r'\s*(const\s+)?(\w|>|\)|,)', after):
            continue
        if mo.group('op') == '*' and re.match(r'\s+\w', after) and \
                re.search(r'\s$', m[start:mo.end() - 1]) and \
                re.search(r'[\w)\]]\s*$', m[max(0, start - 8):start]):
            continue  # looks like `a Type * b` inside an expression
        base = re.sub(r'\s+', '', mo.group('type'))
        op = mo.group('op')
        aspace, why = _policy_lookup(base, op)
        if aspace is None:
            continue
        if aspace == 'REJECT':
            raise TranslateError('%s%s at %s: %s' % (base, op, where(s, start), why))
        edits.append((start, start, '%s ' % aspace if not mo.group('const') else ''))
        if mo.group('const'):
            edits[-1] = (start, mo.end('const'), 'const %s ' % aspace)
        counts.add('aspace.' + aspace)
    return apply_edits(s, edits)


def pass_fp_zero(s, counts):
    m = mask(s)
    edits = []
    fp_alt = '|'.join(FP_TYPES)
    for mo in re.finditer(r'\b(%s)(\s+\w+\s*=\s*)0(\s*;)' % fp_alt, m):
        edits.append((mo.start(3) - 1, mo.start(3), '%s(0)' % mo.group(1)))
        counts.add('fp_zero')
    for hs, o, c in find_functions(m):
        hdr = header_text(m, hs, o)
        body = m[o:c]
        names = {}
        name = header_name(hdr)
        for mo in re.finditer(r'\b(%s)\s+(\w+)\s*(?=[,)=;\[{(])' % fp_alt, hdr[hdr.find('('):]
                              if '(' in hdr else ''):
            names[mo.group(2)] = mo.group(1)
        for mo in re.finditer(r'\b(%s)\s+(\w+)\s*(?=[=;,\[{(])' % fp_alt, body):
            if mo.group(2) not in ('operator',):
                names.setdefault(mo.group(2), mo.group(1))
        for mo in re.finditer(r'(?<![\w.>])(\w+)\s*=\s*(0)\s*;', body):
            t = names.get(mo.group(1))
            a = o + mo.start(2)
            if t and not any(x <= a < y for x, y, _ in edits):
                edits.append((a, a + 1, '%s(0)' % t))
                counts.add('fp_zero')
        hdr_na = re.sub(r'__attribute__\s*\(\(.*?\)\)', '', hdr).strip()
        rt = re.match(r'(?:(?:static|inline|constexpr)\s+)*(%s)\s+%s\s*\('
                      % (fp_alt, re.escape(name or '@')), hdr_na) if name else None
        if rt:
            for mo in re.finditer(r'\breturn\s+(0)\s*;', body):
                a = o + mo.start(1)
                edits.append((a, a + 1, '%s(0)' % rt.group(1)))
                counts.add('fp_zero')
    return apply_edits(s, edits)


def pass_assert(s, counts):
    m = mask(s)
    edits = []
    funcs = find_functions(m)
    for mo in re.finditer(r'(?<![\w.])assert\s*\(', m):
        fn = next(((hs, o, c) for hs, o, c in funcs if o <= mo.start() <= c), None)
        has_ctx = fn is not None and re.search(r'\bctx\b', header_text(m, fn[0], fn[1]))
        if has_ctx:
            edits.append((mo.start(), mo.end(), 'C2M_ASSERT(ctx, '))
            counts.add('assert')
        else:
            edits.append((mo.start(), mo.end(), 'C2M_ASSERT_NOCTX('))
            counts.add('assert_noctx')
    return apply_edits(s, edits)


RESIDUAL = [
    (r'\b__device__\b', '__device__'),
    (r'\b__global__\b', '__global__'),
    (r'\b__launch_bounds__\b', '__launch_bounds__'),
    (r'\b__shared__\b', '__shared__'),
    (r'\b__constant__\b', '__constant__'),
    (r'\b__restrict__\b', '__restrict__'),
    (r'\b__noinline__\b|\b__forceinline__\b|\b__host__\b', 'CUDA function qualifier'),
    (r'\[&\]|\[=\]', 'lambda'),
    (r'::\s*value_type\b', 'value_type'),
    (r'\bauto\s*&?\s*\[', 'structured binding'),
    (r'\bcuda::std::', 'cuda::std'),
    (r'\b(?:asm|__asm__)\s*(?:volatile\s*)?\(', 'asm'),
    (r'\bnvtx', 'nvtx'),
    (r'(?<![\w.>:])(?:map|reduce)\s*\(', 'map/reduce call'),
    (r'(?<![\w.])assert\s*\(', 'assert'),
    (r'(?<!thread )(?<!thread const )\bExecContext\s*&', 'ExecContext& without address space'),
    (r'\b(?:Fp|FpExt)\s*\*\s*\*', 'Fp**'),
    (r'\b%s\b' % re.escape(CUDA_NS), 'CUDA namespace'),
]


def residual_scan(s):
    m = mask(s)
    found = {}
    for pat, what in RESIDUAL:
        hits = [mo.start() for mo in re.finditer(pat, m)]
        if hits:
            found[what] = (len(hits), where(s, hits[0]))
    return found


# ------------------------------------------------------------------------------------------------
# Drivers
# ------------------------------------------------------------------------------------------------

ALL_PASSES = ('kernel', 'strip', 'include', 'accum_guard', 'array', 'value_type', 'sbind',
              'lambda', 'fpp', 'qual', 'aspace', 'fp_zero', 'assert')


def translate(src, typedefs=None, passes=ALL_PASSES, counts=None, uniq=None, strict=True,
              include_map=None, metal_ns=METAL_NS, map_lowering='loop'):
    """Translate CUDA text. Returns (text, counts). Raises TranslateError. include_map and
    metal_ns override INCLUDE_MAP and the target namespace (eval mode)."""
    counts = Counts() if counts is None else counts
    uniq = uniq or Uniq()
    typedefs = typedefs if typedefs is not None else {}
    typedefs = dict(typedefs)
    typedefs.update(collect_array_typedefs(src))
    s = src
    for p in passes:
        if p == 'kernel':
            s = pass_kernel(s, counts)
        elif p == 'strip':
            s = pass_strip(s, counts)
        elif p == 'include':
            s = pass_include(s, counts, include_map, metal_ns)
        elif p == 'poly_mix':
            s = pass_poly_mix(s, counts)
        elif p == 'fpext_index':
            s = pass_fpext_index(s, counts)
        elif p == 'eval_aspace':
            s = pass_eval_aspace(s, counts)
        elif p == 'accum_guard':
            s = pass_accum_guard(s, counts)
        elif p == 'array':
            s = pass_array(s, counts)
        elif p == 'value_type':
            s = pass_value_type(s, counts, typedefs)
        elif p == 'sbind':
            s = pass_sbind(s, counts, uniq)
        elif p == 'lambda':
            s = pass_lambda(s, counts, uniq, map_lowering, typedefs)
        elif p == 'fpp':
            s = pass_fpp(s, counts)
        elif p == 'qual':
            s = pass_qual(s, counts)
        elif p == 'aspace':
            s = pass_aspace(s, counts)
        elif p == 'fp_zero':
            s = pass_fp_zero(s, counts)
        elif p == 'assert':
            s = pass_assert(s, counts)
        else:
            raise ValueError(p)
    if strict:
        res = residual_scan(s)
        if res:
            raise TranslateError('residual CUDA constructs: ' + '; '.join(
                '%s x%d (first at %s)' % (k, n, w) for k, (n, w) in sorted(res.items())))
    return s, counts


def read(path):
    with open(path, newline='') as f:
        return f.read()


def write_or_check(path, text, check):
    if check:
        try:
            cur = read(path)
        except FileNotFoundError:
            cur = None
        if cur != text:
            raise TranslateError('%s is out of date (re-run cuda2metal.py without --check)' % path)
        return
    with open(path, 'w', newline='') as f:
        f.write(text)


def banner(src_name):
    return ('// GENERATED by rv32im-sys/kernels/cuda/cuda2metal.py from %s -- do not edit.\n'
            % src_name)


def translate_steps(cuda_dir=HERE, src=None):
    """steps.cu, steps.cuh -> (steps.metal, steps.metal.h, {name: counts}). src overrides the
    steps.cu text (selftest)."""
    typedefs = collect_array_typedefs(read(os.path.join(cuda_dir, 'types.cuh.inc')))
    uniq = Uniq()
    src = read(os.path.join(cuda_dir, 'steps.cu')) if src is None else src
    hdr = read(os.path.join(cuda_dir, 'steps.cuh'))
    out_h, ch = translate(hdr, typedefs, uniq=uniq)
    out, cs = translate(src, typedefs, uniq=uniq)
    # steps.cu includes "steps.metal.h" (from steps.cuh) and "witgen.h"; the generated .inc files
    # are included by witgen.h under RV32IM_METAL_INCLUDE_GENERATED.
    lic_end = out.index('#include')
    out = (out[:lic_end] + banner('steps.cu') +
           '#ifndef RV32IM_METAL_INCLUDE_GENERATED\n#define RV32IM_METAL_INCLUDE_GENERATED 1\n'
           '#endif\n' + out[lic_end:])
    out = out.replace('#include "witgen.h"\n', '#include "witgen.h"\n\n' + PRELUDE, 1)
    lic_end = out_h.index('#pragma once')
    out_h = out_h[:lic_end] + banner('steps.cuh') + out_h[lic_end:]
    return out, out_h, {'steps.cu': cs, 'steps.cuh': ch}


def translate_incs(cuda_dir=HERE):
    res = {}
    for src_name, dst in (('defs.cu.inc', 'defs.metal.inc'), ('types.cuh.inc', 'types.metal.inc'),
                          ('layout.cu.inc', 'layout.metal.inc')):
        text, c = translate(read(os.path.join(cuda_dir, src_name)), passes=('array', 'qual'))
        res[dst] = (banner(src_name) + text, c, src_name)
    return res


# Eval mode (R6): the baseline eval_check (phase-1 oracle) from eval_check.cuh, eval_check_0..3.cu
# and the eval_check kernel of eval_check_kernel.cu, as one Metal translation unit. The chain
# functions live in EVAL_NS (not rv32im_metal) so that they cannot collide with other
# translation units of the same metallib (evalfast also defines a poly_fp).
EVAL_SOURCES = ('eval_check.cuh', 'eval_check_0.cu', 'eval_check_1.cu', 'eval_check_2.cu',
                'eval_check_3.cu', 'eval_check_kernel.cu')
EVAL_NS = 'rv32im_metal_eval'
EVAL_INCLUDE_MAP = {
    '"supra/fp.h"': None,
    '"eval_check.cuh"': None,  # inlined at the top of eval_check.metal
    '<cstdint>': None,
}
EVAL_PASSES = ('poly_mix', 'kernel', 'strip', 'include', 'fpext_index', 'fpp', 'qual',
               'eval_aspace', 'fp_zero')
EVAL_KERNEL_ANCHOR = '__launch_bounds__(EVAL_CHECK_THREADS)\n__global__ void eval_check('

# Rewrite counts of `eval` for the checked-in sources (see PINNED_STEPS).
# 21 functions (poly_fp + rv32im_v2_0..19), each with a prototype in eval_check.cuh and a
# definition: poly_mix.param = 2 * 21 + the kernel; poly_mix.call = 20 chain calls + the kernel's
# poly_fp call; fpext_index = the kernel's ret[0..3].
PINNED_EVAL = {
    'eval_check': {
        'aspace.device': 182,
        'aspace.thread': 74,
        'fpext_index': 4,
        'include': 14,
        'kernel': 1,
        'kernel.hoisted': 1,
        'poly_mix.call': 21,
        'poly_mix.decl': 1,
        'poly_mix.param': 43,
        'qual.device_fn': 21,
        'qual.extern_device': 21,
        'qual.noinline': 20,
        'qual.ns_constexpr': 2,
        'strip.launch_bounds': 1,
    },
}


def eval_kernel_source(text):
    """The eval_check kernel of eval_check_kernel.cu (not eval_check_diag, not the host code),
    in the CUDA namespace, with the EVAL_CHECK_THREADS default."""
    e0 = text.index(EVAL_KERNEL_ANCHOR)
    m = mask(text)
    bo = m.index('{', m.index(')', e0 + len(EVAL_KERNEL_ANCHOR)))
    bc = match_close(m, bo)
    d0 = text.index('#ifndef EVAL_CHECK_THREADS')
    d1 = text.index('#endif', d0) + len('#endif')
    return ('// ---- eval_check_kernel.cu: the eval_check kernel ----\nnamespace %s {\n\n%s\n\n%s\n\n'
            '} // namespace %s\n' % (CUDA_NS, text[d0:d1], text[e0:bc + 1], CUDA_NS))


def eval_input(cuda_dir=HERE, srcs=None):
    """The combined CUDA input of eval mode. srcs overrides file texts by name (selftest)."""
    srcs = srcs or {}
    parts = []
    for name in EVAL_SOURCES:
        text = srcs.get(name)
        if text is None:
            text = read(os.path.join(cuda_dir, name))
        if name == 'eval_check_kernel.cu':
            text = eval_kernel_source(text)
        elif name == 'eval_check.cuh':
            text = text.replace('#pragma once\n', '', 1)
        parts.append('// ---- %s ----\n%s' % (name, text) if name != 'eval_check_kernel.cu'
                     else text)
    return '\n'.join(parts)


def translate_eval(cuda_dir=HERE, srcs=None):
    """-> (eval_check.metal text, counts)."""
    src = eval_input(cuda_dir, srcs)
    out, c = translate(src, passes=EVAL_PASSES, include_map=EVAL_INCLUDE_MAP, metal_ns=EVAL_NS)
    head = (banner(', '.join(EVAL_SOURCES)) +
            '// Baseline rv32im eval_check (phase-1 oracle, Metal port item R6): poly_fp and the\n'
            '// 20 noinline rv32im_v2_* chain functions in namespace %s, kernel `eval_check`\n'
            '// at global scope. CUDA\'s `__constant__ FpExt poly_mix[%s]` is the kernel\'s last\n'
            '// buffer (`const device FpExt*`, kNumPolyMixPows entries) threaded through the chain.\n'
            '#include <metal_stdlib>\n#include "fp.h"\n#include "fpext.h"\n\n'
            % (EVAL_NS, 'kNumPolyMixPows') + LB_PRELUDE + '\n')
    return head + out, c


def write_if_changed(path, text):
    """Write unless the file already holds exactly text (build.rs reruns on a new mtime)."""
    try:
        if read(path) == text:
            return False
    except FileNotFoundError:
        pass
    with open(path, 'w', newline='') as f:
        f.write(text)
    return True


def check_pins(all_counts, pins=PINNED_STEPS):
    bad = []
    for name, want in pins.items():
        got = all_counts[name]
        for k in sorted(set(want) | set(got)):
            if want.get(k, 0) != got.get(k, 0):
                bad.append('%s %s: pinned %d, got %d' % (name, k, want.get(k, 0), got.get(k, 0)))
    return bad


def fmt_counts(c):
    return ', '.join('%s=%d' % (k, v) for k, v in sorted(c.items()))


def default_headers():
    if os.path.exists(os.path.join(R1_METAL, 'witgen.h')):
        return R1_METAL
    return STUBS


def metal_compile(path, out_air, incs, defines=()):
    cmd = (['xcrun', '-sdk', 'macosx', 'metal'] + METAL_FLAGS +
           ['-D%s' % d for d in defines] + sum((['-I', i] for i in incs), []) +
           ['-c', path, '-o', out_air])
    t0 = time.time()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return p.returncode, p.stderr, time.time() - t0, cmd


def error_classes(stderr):
    classes = {}
    for line in stderr.split('\n'):
        mo = re.search(r'\berror: (.*)$', line)
        if not mo:
            continue
        msg = re.sub(r"'[^']*'", "'X'", mo.group(1))
        msg = re.sub(r'\d+', 'N', msg)
        classes[msg] = classes.get(msg, 0) + 1
    return classes


# ------------------------------------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------------------------------------


def _extract_function(text, anchor_re, nth=0):
    """The whole function definition (from its header start to its closing brace) containing
    the nth match of anchor_re."""
    m = mask(text)
    hits = list(re.finditer(anchor_re, text))
    if len(hits) <= nth:
        raise TranslateError('selftest anchor %r not found' % anchor_re)
    pos = hits[nth].start()
    for hs, o, c in find_functions(m):
        if hs <= pos <= c:
            # drop leading preprocessor/comment lines of the header region
            h = hs
            while True:
                nl = text.find('\n', h)
                line = text[h:nl]
                if line.strip() == '' or line.lstrip().startswith(('#', '//')):
                    h = nl + 1
                    continue
                break
            return text[h:c + 1]
    raise TranslateError('selftest anchor %r is not inside a function' % anchor_re)


def _selftest_cases():
    steps = read(os.path.join(HERE, 'steps.cu'))
    witgen = read(os.path.join(HERE, 'witgen.h'))
    ffi = read(os.path.join(HERE, 'ffi.cu'))
    ek = read(os.path.join(HERE, 'eval_check_kernel.cu'))
    ef = read(os.path.join(HERE, 'eval_check_ef.cu'))
    efh = read(os.path.join(HERE, 'eval_check_ef_hip.cu'))
    types = read(os.path.join(HERE, 'types.cuh.inc'))
    cases = []

    # qual + aspace: the first generated step function.
    cases.append(dict(name='qual_aspace', pass_='qual/aspace', ctx='steps',
                      src=_extract_function(steps, r'NondetRegStruct back_NondetReg\('),
                      expect={'qual.device_fn': 1, 'aspace.thread': 1}))
    # value_type + lambda (map with a BoundLayout, 3 args).
    cases.append(dict(name='lambda_map3', pass_='value_type/lambda', ctx='steps',
                      src=_extract_function(steps, r'NondetRegStruct3Array x2 =\s*map\('),
                      expect={'lambda.map': 1, 'value_type': 2}))
    # lambda: map over two arrays-with-layout whose body returns a struct literal.
    cases.append(dict(name='lambda_map_struct', pass_='lambda', ctx='steps',
                      src=_extract_function(steps, r'DigestRegStruct exec_DigestReg\('),
                      expect={'lambda.map': 1, 'value_type': 2}))
    # lambda: the only reduce (4 args, BoundLayout).
    cases.append(dict(name='lambda_reduce', pass_='lambda', ctx='steps',
                      src=_extract_function(steps, r'=\s*reduce\('),
                      expect={'lambda.reduce': 1, 'value_type': 2}))
    # The R17 unrolled lowering (witgen_tu_gen.py) of the same map and reduce sites.
    cases.append(dict(name='lambda_map3_unroll', pass_='value_type/lambda', ctx='steps',
                      src=_extract_function(steps, r'NondetRegStruct3Array x2 =\s*map\('),
                      topts={'map_lowering': 'unroll'},
                      expect={'lambda.map': 1, 'lambda.map.unroll': 1, 'value_type': 2}))
    cases.append(dict(name='lambda_reduce_unroll', pass_='lambda', ctx='steps',
                      src=_extract_function(steps, r'=\s*reduce\('),
                      topts={'map_lowering': 'unroll'},
                      expect={'lambda.reduce': 1, 'lambda.reduce.unroll': 1, 'value_type': 2}))
    # structured bindings (getMemoryTxn: 5 names; bigIntExtern: 16 names).
    cases.append(dict(name='sbind5', pass_='sbind', ctx='steps',
                      src=_extract_function(steps, r'auto \[x3, x4, x5, x6, x7\] = INVOKE_EXTERN'),
                      expect={'sbind': 1}))
    cases.append(dict(name='sbind16', pass_='sbind', ctx='steps',
                      src=_extract_function(steps, r'auto \[x5, x6, x7, x8, x9, x10, x11'),
                      expect={'sbind': 1}))
    # assert: the shortest function with an unreachable mux arm.
    m = mask(steps)
    best = None
    for hs, o, c in find_functions(m):
        if 'Reached unreachable mux arm' in steps[o:c] and \
                (best is None or c - hs < best[1] - best[0]):
            best = (hs, c)
    anchor = steps[best[0]:best[1]].lstrip().split('(')[0]
    cases.append(dict(name='assert', pass_='assert', ctx='steps',
                      src=_extract_function(steps, re.escape(anchor) + r'\('),
                      expect={'assert': None}))
    # array: types.cuh.inc's first array typedef with the struct using it.
    a0 = types.index('struct NondetRegLayout {')
    a1 = types.index('struct NondetExtRegLayout {')
    cases.append(dict(name='array', pass_='array', ctx='bare',
                      src=types[a0:a1]
                      .replace('struct NondetRegLayout {', 'struct C2mNondetRegLayout {')
                      .replace('<NondetRegLayout,', '<C2mNondetRegLayout,')
                      .replace('NondetRegLayout7LayoutArray', 'C2mNondetRegLayout7LayoutArray')
                      .replace('OneHot_7_Layout', 'C2mOneHot_7_Layout'),
                      expect={'array': 1}))
    # Fp(0): witgen.h's non-virtual MutableBufObj::load (`return 0;` in a Val function).
    l0 = witgen.index('__device__ __forceinline__ Val load(ExecContext& ctx, size_t col, '
                      'size_t back) {')
    l1 = witgen.index('\n  }\n', l0) + 4
    meth = witgen[l0:l1]
    cases.append(dict(name='fp_zero', pass_='fp_zero', ctx='steps',
                      src='struct C2mProbeBuf {\n  Buffer buf;\n  Buffer preDataBuf;\n'
                          '  size_t zeroBack;\n' + meth + '\n};\n',
                      expect={'fp_zero': 1}))
    # kernel synthesis: injectCycleCols (ffi.cu) and eval_check (__launch_bounds__ first).
    ic = _extract_function(ffi, r'__global__ void injectCycleCols\(')
    cases.append(dict(name='kernel_inject', pass_='kernel', ctx='steps',
                      src='namespace %s {\n%s\n} // namespace %s\n' % (CUDA_NS, ic, CUDA_NS),
                      expect={'kernel': 1, 'kernel.hoisted': 1, 'aspace.device': 1},
                      global_scope=True, kernels=['injectCycleCols']))
    e0 = ek.index('__launch_bounds__(EVAL_CHECK_THREADS)')
    e1 = ek.index('    FpExt tot = poly_fp(', e0)
    ev = ('#define EVAL_CHECK_THREADS 256\n' + ek[e0:e1] +
          '    check[cycle] = Fp(po2) + rou;\n  }\n}\n')
    cases.append(dict(name='kernel_eval_check', pass_='kernel/launch_bounds', ctx='bare',
                      src='namespace %s {\n%s} // namespace %s\n' % (CUDA_NS, ev, CUDA_NS),
                      expect={'kernel': 1, 'kernel.hoisted': 1, 'strip.launch_bounds': 1},
                      global_scope=True, kernels=['eval_check']))
    # Fp** -> explicit pointers: poly_fp's signature and its args[] casts (eval_check_ef.cu).
    p0 = ef.index('__device__ FpExt poly_fp(size_t cycle, size_t steps, FpExt* poly_mix, '
                  'Fp** args) {')
    p1 = ef.index('  FpExt x203 = FpExt(0);', p0)
    pf = (ef[p0:p1] + '  return FpExt(x33[0]) + x34[0] + FpExt(poly_mix[0]);\n}\n'
          + '__device__ FpExt poly_fp(size_t cycle, size_t steps, FpExt* poly_mix, Fp** args);\n')
    cases.append(dict(name='fpp', pass_='fpp', ctx='bare', src=pf,
                      expect={'fpp': 2, 'qual.device_fn': 2}))
    # strip: an asm("") barrier (eval_check_ef_hip.cu ef_pmu) and an nvtx range (ffi.cu).
    a0 = efh.index('  asm("" : "+s"(p) : "v"(dep));\n  return ((const')
    asm_line = efh[a0:efh.index('\n', a0) + 1]
    n0 = ffi.index('nvtx3::scoped_range range("par_stepExec");')
    nv_line = ffi[n0:ffi.index('\n', n0) + 1]
    st = ('__device__ uint32_t c2m_strip_probe(const uint32_t* __restrict__ tab, uint32_t k, '
          'uint32_t dep) {\n  uint64_t p = k;\n' + asm_line + '  ' + nv_line +
          '  return tab[p];\n}\n')
    cases.append(dict(name='strip', pass_='strip', ctx='bare', src=st,
                      expect={'strip.asm': 1, 'strip.nvtx': 1, 'strip.restrict': 1}))
    cases += _selftest_eval_cases()
    return cases


EVAL_TOPTS = dict(passes=EVAL_PASSES, include_map=EVAL_INCLUDE_MAP, metal_ns=EVAL_NS)

# Fp** call sites (R6): a callee with an Fp** parameter, a function forwarding its own Fp**,
# and a kernel passing a local pointer array (the eval_check_ef.cu ef_eval_split shape:
# `Fp* args[5] = {accum, data, out, mix, slab}; poly_fp(..., args)` + `args[4]` elsewhere).
FPP_CALL_PROBE = '''\
namespace %s {
__device__ FpExt c2m_fpp_leaf(uint32_t i, Fp** args) {
  return FpExt(args[0][i] + args[2][i]) * args[1][i];
}
__device__ FpExt c2m_fpp_fwd(uint32_t i, Fp** args);
__device__ FpExt c2m_fpp_fwd(uint32_t i, Fp** args) {
  return c2m_fpp_leaf(i, args) + FpExt(args[3][i]);
}
__global__ void c2m_fpp_kernel(Fp* out, Fp* a, Fp* b, Fp* c, Fp* d, uint32_t n) {
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;
  Fp* args[4] = {a, b, c, d};
  FpExt t = c2m_fpp_fwd(i, args);
  out[i] = t[0] + args[3][i];
}
} // namespace %s
''' % (CUDA_NS, CUDA_NS)


def _selftest_eval_cases():
    """Eval mode (R6): the full eval_check.metal (pins checked separately), the smallest chain
    function on its own, and an Fp** call-site probe."""
    cases = []
    out, c = translate_eval()
    cases.append(dict(name='eval_full', pass_='eval (all)', ctx='eval', translated=(out, c),
                      expect=dict(PINNED_EVAL['eval_check']), global_scope=True,
                      kernels=['eval_check']))
    # rv32im_v2_0 (the last link of the chain: no further call) with its prototype and the
    # poly_mix declaration: poly_mix threading on a definition + prototype, chain aspaces.
    e3 = read(os.path.join(HERE, 'eval_check_3.cu'))
    cuh = read(os.path.join(HERE, 'eval_check.cuh'))
    f0 = _extract_function(e3, r'__device__ FpExt rv32im_v2_0\(')
    p0 = cuh.index('extern __device__ __noinline__ FpExt rv32im_v2_0(')
    p1 = cuh.index(';', p0) + 1
    src = ('namespace %s {\n%s\nconstexpr size_t INV_RATE = 4;\nconstexpr size_t kNumPolyMixPows = '
           '458;\n__constant__ FpExt poly_mix[kNumPolyMixPows];\n%s\n} // namespace %s\n'
           % (CUDA_NS, cuh[p0:p1], f0, CUDA_NS))
    cases.append(dict(name='eval_chain_v2_0', pass_='poly_mix/eval_aspace', ctx='eval', src=src,
                      topts=EVAL_TOPTS, global_scope=True,
                      expect={'poly_mix.decl': 1, 'poly_mix.param': 2, 'poly_mix.call': 0,
                              'qual.noinline': 1, 'aspace.thread': 2, 'aspace.device': 4}))
    cases.append(dict(name='fpp_call', pass_='fpp call sites', ctx='eval', src=FPP_CALL_PROBE,
                      topts=EVAL_TOPTS, global_scope=True, kernels=['c2m_fpp_kernel'],
                      expect={'fpp': 3, 'fpp.call': 2, 'fpp.array': 1, 'fpext_index': 1,
                              'kernel': 1}))
    return cases


def _selftest_negative():
    ffi = read(os.path.join(HERE, 'ffi.cu'))
    wtu = read(os.path.join(HERE, 'witgen_tu.cu'))
    ef = read(os.path.join(HERE, 'eval_check_ef.cu'))
    steps = read(os.path.join(HERE, 'steps.cu'))
    neg = []
    k0 = wtu.index('__global__ void __launch_bounds__(256, 1) par_stepExecW(')
    neg.append(('struct-of-pointers kernel arg (witgen_tu.cu par_stepExecW)',
                wtu[k0:wtu.index('\n}\n', k0) + 3], 'struct of device pointers'))
    m0 = ef.index('__device__ __forceinline__ static uint64_t ef_madw(')
    neg.append(('asm with instructions (eval_check_ef.cu ef_madw)',
                ef[m0:ef.index('\n}\n', m0) + 3], 'asm with instructions'))
    f = _extract_function(steps, r'NondetRegStruct3Array x2 =\s*map\(')
    f2 = f.replace('return x5;', 'if (x3 == Val(0)) return x5;\n            return x5;')
    neg.append(('lambda with two returns', f2, 'exactly one top-level return'))
    f3 = f.replace('Val3Array::value_type', 'C2mUnknownArray::value_type')
    neg.append(('unknown value_type', f3, 'not a known array typedef'))
    neg.append(('Fp& of unknown address space', '__device__ void f(Fp& x) { x = Fp(1); }\n',
                'spell its address space'))
    neg.append(('unmapped include', '#include "unknown.cuh"\n', 'no Metal mapping'))
    return [(w, src, needle, None) for w, src, needle in neg] + _selftest_eval_negative()


def _selftest_eval_negative():
    neg = []
    neg.append(('__constant__ poly_mix with an initializer',
                '__constant__ FpExt poly_mix[2] = {FpExt(1u), FpExt(2u)};\n'
                '__device__ FpExt f() { return poly_mix[1]; }\n', 'has an initializer'))
    neg.append(('Fp** argument that is not a pointer array',
                FPP_CALL_PROBE.replace('c2m_fpp_fwd(i, args);\n  out', 'c2m_fpp_fwd(i, &a);\n  out'),
                'neither the'))
    neg.append(('pointer array used other than args[k]',
                FPP_CALL_PROBE.replace('out[i] = t[0] + args[3][i];',
                                       'out[i] = t[0] + *args[3];\n  Fp** q = args;'),
                'used other than'))
    neg.append(('subscript of an auto FpExt',
                '__device__ FpExt g();\n__device__ Fp f() { auto t = g(); return t[0]; }\n',
                'subscript of auto'))
    neg.append(('Fp& in eval mode', '__device__ void f(Fp& x) { x = Fp(1); }\n',
                'spell its address space'))
    return [(w, src, needle, EVAL_TOPTS) for w, src, needle in neg]


def selftest(headers=None, keep=None):
    work = keep or tempfile.mkdtemp(prefix='cuda2metal_selftest_')
    try:
        return _selftest(headers or default_headers(), work)
    finally:
        # also when a check raises: the temp dir holds ~10 MB of translations and .air files
        if not keep:
            shutil.rmtree(work, ignore_errors=True)


def _selftest(headers, work):
    os.makedirs(work, exist_ok=True)
    print('cuda2metal selftest: headers %s%s' % (headers, ' (R1 witgen.h)' if headers == R1_METAL
                                                 else ' (local stubs)' if headers == STUBS else ''))
    fails = []
    typedefs = collect_array_typedefs(read(os.path.join(HERE, 'types.cuh.inc')))

    # 1. generated support files the snippets compile against
    for name, (text, c, src_name) in translate_incs().items():
        with open(os.path.join(work, name), 'w') as f:
            f.write(text)
    steps_out, steps_h, all_counts = translate_steps()
    with open(os.path.join(work, 'steps.metal.h'), 'w') as f:
        f.write(steps_h)

    # 2. pinned counts of the full steps translation
    bad = check_pins(all_counts)
    print('  pins   steps.cu: %s' % ('OK' if not bad else 'MISMATCH'))
    for b in bad:
        fails.append('pin: ' + b)
        print('    ' + b)

    # 2b. the pins must catch a change of shape: drop one unreachable-arm assert and one
    # structured binding from steps.cu
    steps_src = read(os.path.join(HERE, 'steps.cu'))
    mut = steps_src.replace('assert(0 && "Reached unreachable mux arm");', ';', 1)
    mut = mut.replace('auto [x6, x7] = INVOKE_EXTERN(ctx, nextPagingIdx);',
                      'auto c2mx = INVOKE_EXTERN(ctx, nextPagingIdx); auto x6 = c2mx[0], '
                      'x7 = c2mx[1];', 1)
    bad = check_pins(translate_steps(src=mut)[2])
    want = {'steps.cu assert: pinned 134, got 133', 'steps.cu sbind: pinned 6, got 5'}
    if set(bad) != want:
        fails.append('pin check on a mutated steps.cu reported %r, expected %r'
                     % (bad, sorted(want)))
        print('  FAIL   pins catch a changed input: %r' % bad)
    else:
        print('  OK     pins catch a changed input (%s)' % '; '.join(bad))

    # 2c. eval mode pins (R6), and their sensitivity: drop one __noinline__ from eval_check.cuh
    bad = check_pins({'eval_check': translate_eval()[1]}, PINNED_EVAL)
    print('  pins   eval_check: %s' % ('OK' if not bad else 'MISMATCH'))
    for b in bad:
        fails.append('pin: ' + b)
        print('    ' + b)
    cuh = read(os.path.join(HERE, 'eval_check.cuh'))
    mut = cuh.replace('extern __device__ __noinline__ FpExt rv32im_v2_0(',
                      'extern __device__ FpExt rv32im_v2_0(', 1)
    bad = check_pins({'eval_check': translate_eval(srcs={'eval_check.cuh': mut})[1]},
                     PINNED_EVAL)
    want = ['eval_check qual.noinline: pinned 20, got 19']
    if bad != want:
        fails.append('eval pin check on a mutated eval_check.cuh reported %r, expected %r'
                     % (bad, want))
        print('  FAIL   eval pins catch a changed input: %r' % bad)
    else:
        print('  OK     eval pins catch a changed input (%s)' % bad[0])

    # 3. one snippet per pass: translate, check counts and residuals, compile
    incs = [work, headers, BK_METAL]
    for case in _selftest_cases():
        try:
            if 'translated' in case:
                out, c = case['translated']
            else:
                out, c = translate(case['src'], typedefs, **case.get('topts', {}))
        except TranslateError as e:
            fails.append('%s: %s' % (case['name'], e))
            print('  FAIL   %-18s translate: %s' % (case['name'], e))
            continue
        for k, v in case['expect'].items():
            if v is None:
                if not c.get(k):
                    fails.append('%s: expected some %s rewrites' % (case['name'], k))
            elif c.get(k, 0) != v:
                fails.append('%s: %s = %d, expected %d' % (case['name'], k, c.get(k, 0), v))
        if case['ctx'] == 'eval':
            tu = [] if 'translated' in case else ['#include "fp.h"', '#include "fpext.h"',
                                                  LB_PRELUDE]
        else:
            tu = ['#define RV32IM_METAL_INCLUDE_GENERATED 1', '#include "witgen.h"']
            if case['ctx'] == 'steps':
                tu.append('#include "steps.metal.h"')
            tu.append(PRELUDE)
        if case.get('global_scope'):
            tu += [out, '']
        else:
            tu += ['namespace %s {' % METAL_NS, out, '} // namespace %s' % METAL_NS, '']
        path = os.path.join(work, 'case_%s.metal' % case['name'])
        with open(path, 'w') as f:
            f.write('\n'.join(tu))
        rc, err, dt, _ = metal_compile(path, path[:-6] + '.air', incs)
        if rc != 0:
            fails.append('%s: xcrun metal failed:\n%s' % (case['name'], err[:3000]))
        elif case.get('kernels'):
            # entry points must be unmangled global symbols (plan 2.4)
            p = subprocess.run(['xcrun', '-sdk', 'macosx', 'metal-nm', path[:-6] + '.air'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)
            syms = set(l.split()[-1] for l in p.stdout.split('\n') if ' T ' in l)
            for kname in case['kernels']:
                if kname not in syms:
                    fails.append('%s: kernel %s is not a global entry point (symbols: %s)'
                                 % (case['name'], kname, sorted(syms)))
        ok = rc == 0 and not any(f.startswith(case['name'] + ':') for f in fails)
        print('  %-6s %-18s %-22s %5.1fs  %s' % ('OK' if ok else 'FAIL', case['name'],
                                                 case['pass_'], dt, fmt_counts(c)))
    # 4. negative cases: the translator must refuse
    for what, src, needle, topts in _selftest_negative():
        try:
            translate(src, typedefs, **(topts or {}))
            fails.append('negative "%s": translated without error' % what)
            print('  FAIL   refuse %s' % what)
        except TranslateError as e:
            if needle not in str(e):
                fails.append('negative "%s": wrong error: %s' % (what, e))
                print('  FAIL   refuse %s: %s' % (what, e))
            else:
                print('  OK     refuse %s' % what)
    if fails:
        print('cuda2metal selftest: %d FAILURE(S)' % len(fails))
        for f in fails:
            print('  - ' + f)
        return 1
    print('cuda2metal selftest: all checks passed')
    return 0


def cmd_steps(args):
    out, out_h, counts = translate_steps()
    for name in ('steps.cu', 'steps.cuh'):
        print('cuda2metal %s: %s' % (name, fmt_counts(counts[name])))
    bad = check_pins(counts)
    if bad:
        print('cuda2metal: rewrite counts differ from PINNED_STEPS (the generated input changed '
              'shape):')
        for b in bad:
            print('  ' + b)
        if not args.no_pin:
            return 1
    else:
        print('cuda2metal: counts match PINNED_STEPS')
    if not args.check:
        os.makedirs(args.out, exist_ok=True)
    write_or_check(os.path.join(args.out, 'steps.metal'), out, args.check)
    write_or_check(os.path.join(args.out, 'steps.metal.h'), out_h, args.check)
    if args.compile or args.check:
        for name, (text, c, src_name) in translate_incs().items():
            write_or_check(os.path.join(args.out, name), text, args.check)
    print('cuda2metal: %s steps.metal (%d lines), steps.metal.h (%d lines) in %s'
          % ('checked' if args.check else 'wrote', out.count('\n'), out_h.count('\n'), args.out))
    if args.compile:
        headers = args.headers or default_headers()
        incs = [args.out, headers, BK_METAL]
        path = os.path.join(args.out, 'steps.metal')
        rc, err, dt, cmd = metal_compile(path, os.path.join(args.out, 'steps.air'), incs)
        print('cuda2metal: %s' % ' '.join(cmd))
        print('cuda2metal: compile %s in %.1f s (headers %s)'
              % ('OK' if rc == 0 else 'FAILED', dt, headers))
        classes = error_classes(err)
        nerr = sum(classes.values())
        nwarn = len(re.findall(r'\bwarning: ', err))
        print('cuda2metal: %d error(s), %d warning(s)' % (nerr, nwarn))
        for k, v in sorted(classes.items(), key=lambda kv: -kv[1]):
            print('  %5d  %s' % (v, k))
        log = os.path.join(args.out, 'steps.compile.log')
        if rc != 0:
            with open(log, 'w') as f:
                f.write(err)
            print('cuda2metal: compiler output in %s' % log)
            return 1
        if os.path.exists(log):
            os.remove(log)
    return 0


def cmd_eval(args):
    t0 = time.time()
    out, counts = translate_eval()
    dt = time.time() - t0
    print('cuda2metal eval_check: %s' % fmt_counts(counts))
    bad = check_pins({'eval_check': counts}, PINNED_EVAL)
    if bad:
        print('cuda2metal: eval rewrite counts differ from PINNED_EVAL (the generated input '
              'changed shape):')
        for b in bad:
            print('  ' + b)
        if not args.no_pin:
            return 1
    else:
        print('cuda2metal: counts match PINNED_EVAL')
    if not args.check:
        os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, 'eval_check.metal')
    if args.check:
        write_or_check(path, out, True)
        what = 'checked'
    else:
        what = 'wrote' if write_if_changed(path, out) else 'unchanged'
    print('cuda2metal: %s %s (%d lines; translation %.2f s)' % (what, path, out.count('\n'), dt))
    if args.compile:
        rc, err, dt, cmd = metal_compile(path, os.path.join(args.out, 'eval_check.air'), [BK_METAL])
        print('cuda2metal: %s' % ' '.join(cmd))
        nwarn = len(re.findall(r'\bwarning: ', err))
        classes = error_classes(err)
        print('cuda2metal: compile %s in %.1f s, %d error(s), %d warning(s)'
              % ('OK' if rc == 0 else 'FAILED', dt, sum(classes.values()), nwarn))
        for k, v in sorted(classes.items(), key=lambda kv: -kv[1]):
            print('  %5d  %s' % (v, k))
        if rc != 0 or nwarn:
            sys.stderr.write(err[:5000])
            return 1
    return 0


def cmd_check():
    """CI drift check (R16; no Metal toolchain needed): the checked-in .metal.inc files are
    up to date and the steps / eval translations succeed with their pinned rewrite counts."""
    fails = []
    for name, (text, c, src_name) in translate_incs().items():
        try:
            write_or_check(os.path.join(R1_METAL, name), text, True)
            print('cuda2metal check: kernels/metal/%s is up to date' % name)
        except TranslateError as e:
            fails.append(str(e))
    _, _, counts = translate_steps()
    bad = check_pins(counts)
    fails += ['steps pin: ' + b for b in bad]
    print('cuda2metal check: steps translation %s' % ('matches PINNED_STEPS' if not bad else
                                                       'differs from PINNED_STEPS'))
    _, counts = translate_eval()
    bad = check_pins({'eval_check': counts}, PINNED_EVAL)
    fails += ['eval pin: ' + b for b in bad]
    print('cuda2metal check: eval translation %s' % ('matches PINNED_EVAL' if not bad else
                                                      'differs from PINNED_EVAL'))
    if fails:
        for f in fails:
            print('cuda2metal check: ' + f, file=sys.stderr)
        print('cuda2metal check: regenerate with `python3 kernels/cuda/cuda2metal.py incs -o '
              'kernels/metal` (in rv32im-sys) or review the counts with `steps|eval -o DIR '
              '--compile --no-pin` and update the pins', file=sys.stderr)
        return 1
    print('cuda2metal check: all up to date')
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--headers', help='rv32im Metal header dir (default: R1 witgen.h if present, '
                    'else the selftest stubs)')
    ap.add_argument('--keep', help='selftest: keep generated files in this dir')
    ap.add_argument('mode', nargs='?', choices=('steps', 'incs', 'eval', 'file', 'check'))
    ap.add_argument('input', nargs='?')
    ap.add_argument('-o', '--out')
    ap.add_argument('--types', default=os.path.join(HERE, 'types.cuh.inc'))
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--no-pin', action='store_true')
    args = ap.parse_args(argv)
    try:
        if args.selftest:
            return selftest(args.headers, args.keep)
        if args.mode == 'check':
            return cmd_check()
        if args.mode == 'steps':
            if not args.out:
                ap.error('steps needs -o OUTDIR')
            return cmd_steps(args)
        if args.mode == 'eval':
            if not args.out:
                ap.error('eval needs -o OUTDIR')
            return cmd_eval(args)
        if args.mode == 'incs':
            if not args.out:
                ap.error('incs needs -o OUTDIR')
            if not args.check:
                os.makedirs(args.out, exist_ok=True)
            for name, (text, c, src_name) in translate_incs().items():
                write_or_check(os.path.join(args.out, name), text, args.check)
                print('cuda2metal %s -> %s: %s' % (src_name, name, fmt_counts(c)))
            return 0
        if args.mode == 'file':
            if not (args.input and args.out):
                ap.error('file needs IN and -o OUT')
            typedefs = collect_array_typedefs(read(args.types))
            out, c = translate(read(args.input), typedefs)
            write_or_check(args.out, banner(os.path.basename(args.input)) + out, args.check)
            print('cuda2metal %s: %s' % (args.input, fmt_counts(c)))
            return 0
        ap.print_help()
        return 2
    except TranslateError as e:
        print('cuda2metal: error: %s' % e, file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
