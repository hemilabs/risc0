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

# usage: accum_tu_gen.py [--check]
#        accum_tu_gen.py --metal -o OUTDIR [--inline always|noinline] [--check] [--no-pin]
#
# CUDA and ROCm builds run the rv32im accum step from accum_tu.cu instead of
# steps.cu. This script, run after regenerating steps.cu and accum_batchinv.py:
#
# 1. writes accum_tu.cu: the step_TopAccum call graph of steps.cu, renamed
#    (_A suffix) and marked `static __device__ __forceinline__`, plus the
#    stepAccumA kernel (its context struct copied from ffi.cu's DeviceAccumContext) and its host
#    launcher. build.rs compiles it as a
#    whole-program TU (not -dc), so the graph is inlined into the kernel with no
#    ABI calls or stack-resident buffer objects. Same statements in the same
#    order: bit-exact by construction. It defines RISC0_HIP_ROWS_PRIVATE: on
#    ROCm, witgen.h's non-virtual buffer objects and buffers.h's accesses
#    without compiler barriers (stepAccum reads no row another thread writes).
# 2. fences exec_TopAccum and step_TopAccum in steps.cu to ROCm builds
#    (#ifdef __HIPCC__; ROCm keeps them for RISC0_HIP_ACCUM_TU=0), so the -dc
#    compile of steps.cu no longer carries the unused relocatable copy (most of
#    its ptxas time).
#
# Both edits are idempotent; --check (run in CI) fails if either file is out of
# date. Re-run it after changing DeviceAccumContext in ffi.cu too.
#
# --metal (Metal port item R4) writes OUTDIR/accum_tu.metal instead and touches no checked-in
# file: the same call graph (same functions, same order, same _A renaming), translated by
# cuda2metal.py (R2's passes, imported, not modified), plus a hand-written stepAccumA kernel with
# explicit [[buffer(n)]] bindings and an AccumParams struct passed with set_bytes (plan 2.0/2.4:
# no struct of device pointers). build.rs runs it into $OUT_DIR/metal on every Metal build.
# --inline selects how the graph is compiled: `always` (default) marks every function
# [[clang::always_inline]] (the CUDA __forceinline__ whole-program TU), `noinline` marks them
# __attribute__((noinline)) (the fallback if always-inline compile or PSO time is too high).
# The per-pass rewrite counts are pinned (PINNED_METAL); --no-pin only reports them. The output
# is written only when it changed (a fresh mtime would rerun the cargo build script); --check
# fails instead of writing.
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = os.path.join(HERE, 'steps.cu')
OUT = os.path.join(HERE, 'accum_tu.cu')
GUARD_OPEN = '#ifdef __HIPCC__  // CUDA: stepAccumA in accum_tu.cu (accum_tu_gen.py)\n'
GUARD_CLOSE = '#endif  // __HIPCC__\n'


def call_graph():
    """steps.cu -> (steps text, fenced steps text, names, defs, order, renamer)."""
    steps = open(STEPS, newline='').read()
    # 2. the guard around exec_TopAccum .. step_TopAccum
    gs = steps.index('__device__ ComponentStruct exec_TopAccum(ExecContext& ctx')
    ge = steps.index('\n}\n', steps.index('\n__device__ void step_TopAccum(', gs)) + 3
    fenced = steps[:gs].endswith(GUARD_OPEN)
    if fenced:
        assert steps[ge:].startswith(GUARD_CLOSE), 'steps.cu: unbalanced exec_TopAccum guard'
        new_steps = steps
    else:
        assert GUARD_OPEN not in steps and GUARD_CLOSE not in steps
        new_steps = steps[:gs] + GUARD_OPEN + steps[gs:ge] + GUARD_CLOSE + steps[ge:]
    src = new_steps.replace(GUARD_OPEN, '').replace(GUARD_CLOSE, '')

    ns_open = src.index('namespace risc0::circuit::rv32im_v2::cuda {') + len('namespace risc0::circuit::rv32im_v2::cuda {')
    ns_close = src.rindex('} // namespace risc0::circuit::rv32im_v2::cuda')
    body = src[ns_open:ns_close]
    starts = [(m.start(), m.group(1)) for m in re.finditer(r'^__device__ [^(;{}]*?\b(\w+)\(', body, re.M)]
    defs = {}
    for i, (s, n) in enumerate(starts):
        e = starts[i + 1][0] if i + 1 < len(starts) else len(body)
        assert n not in defs, n
        defs[n] = body[s:e]
    call_re = re.compile(r'\b(exec\w*|back\w*|step_\w+|extern_\w+)\(')
    seen, st = [], ['step_TopAccum']
    while st:
        n = st.pop()
        if n in seen: continue
        seen.append(n)
        st += [c for c in call_re.findall(defs[n]) if c != n and c in defs]
    missing = {c for n in seen for c in call_re.findall(defs[n]) if c not in defs}
    assert not missing, missing
    names = set(seen)
    assert not any(l.startswith('#') for n in names for l in defs[n].split('\n')), \
        'preprocessor line inside the copied call graph'
    ren = re.compile(r'\b(' + '|'.join(sorted(names, key=len, reverse=True)) + r')\(')
    order = [n for _, n in starts if n in names]   # steps.cu order
    return steps, new_steps, names, defs, order, ren


def graph_text(defs, order, ren, qual):
    """The renamed call graph with each `__device__ ` replaced by `qual`: (protos, fns)."""
    protos, fns = [], []
    for n in order:
        d = ren.sub(lambda m: m.group(1) + '_A(', defs[n]).rstrip() + '\n'
        assert d.startswith('__device__ ')
        d = qual + d[len('__device__ '):]
        protos.append(d[:d.index('{')].rstrip() + ';')
        fns.append(d)
    return protos, fns


def cuda_tu(defs, order, ren):
    # The kernel's context: ffi.cu's DeviceAccumContext (which launches it), copied and renamed.
    ffi = open(os.path.join(HERE, 'ffi.cu'), newline='').read()
    cs = ffi.index('struct DeviceAccumContext {')
    ACCUM_CTX = ffi[cs:ffi.index('\n};\n', cs) + 3].replace('struct DeviceAccumContext {', 'struct DeviceAccumContextA {')

    out = ['// GENERATED by kernels/cuda/accum_tu_gen.py from steps.cu and ffi.cu -- do not edit; re-run it',
           '// instead.',
           '// The step_TopAccum call graph only, renamed (_A) and force-inlined, compiled as a',
           '// whole-program TU (not -dc).',
           '#define RISC0_ROWS_POW2 1',
           '// ROCm: stepAccum reads no row another thread writes (witgen.h, buffers.h).',
           '#define RISC0_HIP_ROWS_PRIVATE 1', '#include "witgen.h"',
           '#ifdef __HIPCC__', '#include <hip/hip_runtime.h>', '#else', '#include <cuda_runtime.h>', '#endif',
           '', 'namespace risc0::circuit::rv32im_v2::cuda {']
    protos, fns = graph_text(defs, order, ren, 'static __device__ __forceinline__ ')
    out += protos + [''] + fns
    out.append('''
constexpr size_t kUserAccumSplitA = kLayout_TopAccum.columns[0].col;
''' + ACCUM_CTX + '''
#ifndef ACCUM_LB
#define ACCUM_LB 256, 1
#endif
__global__ void __launch_bounds__(ACCUM_LB) stepAccumA(DeviceAccumContextA* ctx, uint32_t count, const uint32_t* perm) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle >= count) return;
  if (perm) cycle = perm[cycle];
  ExecContext execCtx(*ctx->preflight, *ctx->tables, cycle);
  Buffer dataBuf = *ctx->data;
  MutableBufObj data(dataBuf);
  Buffer accumBuf = *ctx->accum;
  MutableBufObj accum(accumBuf, /*zeroBack=*/kUserAccumSplitA);
  Buffer mixBuf = *ctx->mix;
  GlobalBufObj mix(mixBuf);
  Buffer globalBuf = *ctx->global;
  GlobalBufObj global(globalBuf);
  step_TopAccum_A(execCtx, &accum, &data, &global, &mix);
}
} // namespace risc0::circuit::rv32im_v2::cuda

void risc0_rv32im_launch_stepAccumTU(unsigned grid, unsigned block, cudaStream_t stream, void* ctx, uint32_t count, const uint32_t* perm) {
  using namespace risc0::circuit::rv32im_v2::cuda;
  stepAccumA<<<grid, block, 0, stream>>>((DeviceAccumContextA*)ctx, count, perm);
}
''')
    return '\n'.join(out)


def main_cuda(check):
    steps, new_steps, names, defs, order, ren = call_graph()
    new_tu = cuda_tu(defs, order, ren)
    try:
        old_tu = open(OUT, newline='').read()
    except FileNotFoundError:
        old_tu = ''
    stale = [n for n, old_, new_ in (('steps.cu', steps, new_steps), ('accum_tu.cu', old_tu, new_tu)) if old_ != new_]
    if check:
        if stale:
            sys.exit('accum_tu_gen: %s out of date; run kernels/cuda/accum_tu_gen.py and commit the result'
                     % ' and '.join(stale))
        print('accum_tu_gen: accum_tu.cu and steps.cu are up to date (%d functions)' % len(names))
    else:
        if new_steps != steps:
            open(STEPS, 'w', newline='').write(new_steps)
        open(OUT, 'w', newline='').write(new_tu)
        print('accum_tu_gen: wrote accum_tu.cu (%d functions, %d lines)%s'
              % (len(names), sum(defs[n].count('\n') for n in names),
                 '' if new_steps == steps else ' and fenced exec_TopAccum in steps.cu'))


# ------------------------------------------------------------------------------------------------
# --metal (R4)
# ------------------------------------------------------------------------------------------------

METAL_OUT = 'accum_tu.metal'

# How each graph function is qualified before translation, and what cuda2metal's `qual` pass
# turns that qualifier into (then rewritten to the final MSL spelling).
INLINE_MODES = {
    # CUDA's own __forceinline__ -> `static inline __attribute__((always_inline))` (qual pass)
    # -> `[[clang::always_inline]] static inline` (same attribute, C++11 spelling).
    'always': ('static __device__ __forceinline__ ',
               'static inline __attribute__((always_inline)) ',
               '[[clang::always_inline]] static inline '),
    # The fallback: real calls (Metal supports non-inlined device function calls).
    'noinline': ('static __device__ __noinline__ ',
                 'static __attribute__((noinline)) ',
                 '__attribute__((noinline)) static '),
}

# Rewrite counts of `--metal` for the checked-in steps.cu (29 functions). A difference means the
# generated input changed shape: re-run with --no-pin, review the counts and the compile, then
# update these numbers. The two modes differ only in the inline qualifier count.
PINNED_METAL_COMMON = {
    'functions': 29,
    'assert': 93,
    'array': 0,
    'aspace.thread': 58,
    'include': 1,
    'lambda.map': 6,
    'qual.device_fn': 58,
    'value_type': 12,
    'qual_rewrite': 58,
}
PINNED_METAL = {
    'always': dict(PINNED_METAL_COMMON, **{'qual.forceinline': 58}),
    'noinline': dict(PINNED_METAL_COMMON, **{'qual.noinline': 58}),
}

METAL_HEADER = '''\
// GENERATED by rv32im-sys/kernels/cuda/accum_tu_gen.py --metal (inline mode: %(mode)s) from
// steps.cu -- do not edit; build.rs regenerates it into $OUT_DIR/metal on every Metal build.
//
// The step_TopAccum call graph of steps.cu (the same %(nfun)d functions as accum_tu.cu, same order,
// renamed _A), translated by cuda2metal.py, plus the stepAccumA kernel (Metal port item R4).
//
// stepAccumA buffer bindings (explicit, plan 2.4; mirrored on the Rust side, hal/metal/accum.rs):
//   0 data   (device Fp*, dataRows x dataCols, column-major; read only by the accum graph)
//   1 accum  (device Fp*, accumRows x accumCols, column-major)
//   2 global (device Fp*, globalRows x globalCols; row 0 is used)
//   3 mix    (device Fp*, mixRows x mixCols; row 0 is used)
//   4 perm   (const device uint*, >= count entries; read only when AccumParams::hasPerm != 0,
//             otherwise any buffer may be bound, e.g. accum)
//   5 fault  (device atomic_uint*, 1 entry; RV32IM_FAULT_* bits ORed once per thread)
//   6 AccumParams (set_bytes, 48 bytes)
// Grid: 1-D, `count` threads (dispatch_threads); threadgroup size <= the PSO's
// maxTotalThreadsPerThreadgroup (the kernel declares <= %(tg)d, CUDA's __launch_bounds__(256, 1)).
// rows of data and accum must be powers of two (RISC0_ROWS_POW2, as on CUDA).
'''

METAL_KERNEL = r'''
// ---------------------------------------------------------------------------------------------
// stepAccumA kernel (hand-written by accum_tu_gen.py; CUDA: accum_tu.cu stepAccumA, whose
// DeviceAccumContextA of device pointers becomes explicit bindings + AccumParams).
// ---------------------------------------------------------------------------------------------

namespace rv32im_metal {
/// CUDA: kUserAccumSplitA = kLayout_TopAccum.columns[0].col (r0::array's operator[] is not
/// constexpr; R3 note).
constant constexpr uint32_t kUserAccumSplitA = kLayout_TopAccum.columns.elems[0].col;
} // namespace rv32im_metal

#ifndef RV32IM_FAULT_ACCUM_PERM
#define RV32IM_FAULT_ACCUM_PERM 0x20u // stepAccumA: perm[i] >= count (that row is skipped)
#endif

/// POD params (set_bytes); #[repr(C)] mirror on the Rust side, size asserted on both sides.
struct AccumParams {
  uint32_t count;      // cycles to run = threads (CUDA: lastCycle)
  uint32_t hasPerm;    // 1: cycle = perm[thread] (major-grouped order); 0: cycle = thread
  uint32_t dataRows;   // power of two
  uint32_t dataCols;
  uint32_t accumRows;  // power of two
  uint32_t accumCols;
  uint32_t globalRows;
  uint32_t globalCols;
  uint32_t mixRows;
  uint32_t mixCols;
  uint32_t _pad[2];
};
RV32IM_ASSERT_SIZE(AccumParams, 48);

#ifndef ACCUM_TG
#define ACCUM_TG %(tg)d
#endif

[[max_total_threads_per_threadgroup(ACCUM_TG)]]
kernel void stepAccumA(device Fp* data [[buffer(0)]],
                       device Fp* accum [[buffer(1)]],
                       device Fp* global [[buffer(2)]],
                       device Fp* mix [[buffer(3)]],
                       const device uint32_t* perm [[buffer(4)]],
                       device metal::atomic_uint* fault [[buffer(5)]],
                       constant AccumParams& p [[buffer(6)]],
                       uint gid [[thread_position_in_grid]]) {
  using namespace rv32im_metal;
  if (gid >= p.count) {
    return;
  }
  uint32_t cycle = gid;
  if (p.hasPerm) {
    cycle = perm[gid];
    // Plan 2.9: a host-built index is still clamped (CUDA trusts it).
    if (cycle >= p.count) {
      metal::atomic_fetch_or_explicit(fault, RV32IM_FAULT_ACCUM_PERM, metal::memory_order_relaxed);
      return;
    }
  }
  // The accum graph calls no extern but assert (a no-op): preflight and lookup tables are never
  // read (CUDA binds them anyway). C2M_ASSERT writes tables.bad.
  PreflightTrace preflight;
  preflight.cycles = nullptr;
  preflight.txns = nullptr;
  preflight.bigintBytes = nullptr;
  preflight.txnsLen = 0;
  preflight.bigintBytesLen = 0;
  preflight.tableSplitCycle = 0;
  LookupTables tables = makeLookupTables(nullptr, nullptr);
  ExecContext execCtx(preflight, tables, cycle);
  BufferObj dataObj = makeMutableBuf(makeBuffer(data, p.dataRows, p.dataCols));
  BufferObj accumObj = makeMutableBuf(
      makeBuffer(accum, p.accumRows, p.accumCols), nullBuffer(), /*zeroBack=*/kUserAccumSplitA);
  BufferObj mixObj = makeGlobalBuf(makeBuffer(mix, p.mixRows, p.mixCols));
  BufferObj globalObj = makeGlobalBuf(makeBuffer(global, p.globalRows, p.globalCols));
  step_TopAccum_A(execCtx, &accumObj, &dataObj, &globalObj, &mixObj);
  if (tables.bad) {
    metal::atomic_fetch_or_explicit(fault, tables.bad, metal::memory_order_relaxed);
  }
}
'''


def metal_tu(mode, tg=256):
    """-> (accum_tu.metal text, counts)."""
    sys.path.insert(0, HERE)
    import cuda2metal as c2m
    qual_in, qual_mid, qual_out = INLINE_MODES[mode]
    _, _, names, defs, order, ren = call_graph()
    protos, fns = graph_text(defs, order, ren, qual_in)
    src = '\n'.join(['#include "witgen.h"', '', 'namespace %s {' % c2m.CUDA_NS] + protos + [''] +
                    fns + ['} // namespace %s' % c2m.CUDA_NS, ''])
    typedefs = c2m.collect_array_typedefs(c2m.read(os.path.join(HERE, 'types.cuh.inc')))
    text, counts = c2m.translate(src, typedefs)
    n = text.count(qual_mid)
    text = text.replace(qual_mid, qual_out)
    counts.add('qual_rewrite', n)
    counts.add('functions', len(names))
    head = METAL_HEADER % {'mode': mode, 'nfun': len(names), 'tg': tg}
    prelude = ('#define RISC0_ROWS_POW2 1\n'
               '#define RV32IM_METAL_INCLUDE_GENERATED 1\n'
               '#include "witgen.h"\n\n' + c2m.PRELUDE)
    assert text.startswith('#include "witgen.h"\n')
    text = head + prelude + text[len('#include "witgen.h"\n'):] + METAL_KERNEL % {'tg': tg}
    return text, counts


def main_metal(argv):
    import argparse
    ap = argparse.ArgumentParser(prog='accum_tu_gen.py --metal')
    ap.add_argument('--metal', action='store_true')
    ap.add_argument('-o', '--out', required=True, help='output directory')
    ap.add_argument('--inline', choices=sorted(INLINE_MODES), default='always')
    ap.add_argument('--tg', type=int, default=256, help='max_total_threads_per_threadgroup')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--no-pin', action='store_true')
    args = ap.parse_args(argv)
    sys.path.insert(0, HERE)
    import cuda2metal as c2m
    try:
        text, counts = metal_tu(args.inline, args.tg)
    except c2m.TranslateError as e:
        print('accum_tu_gen --metal: error: %s' % e, file=sys.stderr)
        return 1
    print('accum_tu_gen --metal (%s): %s' % (args.inline, c2m.fmt_counts(counts)))
    bad = c2m.check_pins({'accum': counts}, {'accum': PINNED_METAL[args.inline]})
    if bad:
        print('accum_tu_gen --metal: rewrite counts differ from PINNED_METAL (steps.cu changed '
              'shape):', file=sys.stderr)
        for b in bad:
            print('  ' + b, file=sys.stderr)
        if not args.no_pin:
            return 1
    path = os.path.join(args.out, METAL_OUT)
    try:
        cur = open(path, newline='').read()
    except FileNotFoundError:
        cur = None
    if args.check:
        if cur != text:
            print('accum_tu_gen --metal: %s is out of date (re-run without --check)' % path,
                  file=sys.stderr)
            return 1
        print('accum_tu_gen --metal: %s is up to date (%d lines)' % (path, text.count('\n')))
        return 0
    os.makedirs(args.out, exist_ok=True)
    if cur != text:
        with open(path, 'w', newline='') as f:
            f.write(text)
        print('accum_tu_gen --metal: wrote %s (%d lines)' % (path, text.count('\n')))
    else:
        print('accum_tu_gen --metal: %s unchanged (%d lines)' % (path, text.count('\n')))
    return 0


if __name__ == '__main__':
    if '--metal' in sys.argv[1:]:
        sys.exit(main_metal(sys.argv[1:]))
    main_cuda('--check' in sys.argv[1:])
