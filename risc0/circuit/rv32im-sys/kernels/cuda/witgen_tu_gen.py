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

# usage: witgen_tu_gen.py [--check]
#        witgen_tu_gen.py --metal -o OUTDIR [--check] [--no-pin] [--map-lowering unroll|loop]
#
# Writes witgen_tu.cu (CUDA only; build.rs compiles it as a whole-program TU, not
# -dc): the step_Top call graph of steps.cu renamed (_W suffix) and marked
# `static __device__ __noinline__`, the extern_* device functions and
# ExecBuffers copied from ffi.cu (renamed static externW_* copies), power-of-two
# rows (RISC0_ROWS_POW2), and the par_stepExecW kernel with its host launcher.
# Kept noinline on purpose: the fully inlined build reorders the back-row loads
# of rows written by neighbouring threads (latent race) and is not bit-exact.
# ROCm (build.rs witgen_tu / witgen_tu2, WITGEN_TU_FNATTR empty / __forceinline__):
# every Buffer access keeps its compiler barriers there (buffers.h,
# RISC0_HIP_GLOBAL_BUFS), as in steps.cu's par_stepExec, so the inlined call graph
# cannot move those loads either.
#
# Re-run it after regenerating steps.cu or changing the copied parts of ffi.cu;
# --check (run in CI) fails if witgen_tu.cu is out of date, or if the Metal translation
# (gen_metal(), in memory, no toolchain needed) fails or its rewrite counts differ from
# PINNED_METAL.
#
# --metal (Metal port, item R5): writes OUTDIR/witgen_tu.metal instead (rv32im-sys/build.rs runs
# it into $OUT_DIR/metal on every metal build; the output is never checked in). The same call
# graph, translated by cuda2metal.py's passes (lambdas -> loops, structured bindings -> temps,
# address spaces, Fp(0), assert -> fault bit; rewrite counts pinned in PINNED_METAL), kept
# noinline (WITGEN_TU_FNATTR = __attribute__((noinline))), plus a hand port of the extern_*
# section of ffi.cu (its text is pinned by hash: a change fails until the port is re-checked)
# and three kernels with explicit bindings (plan 2.4): par_stepExecW (one thread per cycle) and
# the chunked single-thread fwd_stepExec / rev_stepExec (seq_exec.h). An unchanged output file
# is not rewritten (build.rs: a new mtime would rerun the build script). With --check it only
# compares.
#
# map()/reduce() are unrolled in the Metal TU (R17, MAP_LOWERING): one block per element, values
# in scalars, the result array brace-initialized. CUDA's lowering -- a default-constructed array
# written element by element in a loop that calls a noinline step function -- is the code shape
# in which the Apple GPU (M5 Max, macOS 27.2) loses the last iteration's store and the array
# reads back its constructor value (docs/metal-port/notes/R17.md: ToBits<5>'s bit 4 in DynPo2,
# ~1-200 wrong rows per po2 20 segment). The Metal call graph must not contain any loop
# (gen_metal() refuses one); `--map-lowering loop` emits the old shape, only to build a
# same-session baseline that still shows the defect (R17.md, "Re-verifying").
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'witgen_tu.cu')
ARGS = sys.argv[1:]
CHECK = '--check' in ARGS
METAL = '--metal' in ARGS
src = open(os.path.join(HERE, 'steps.cu'), newline='').read()
ffi = open(os.path.join(HERE, 'ffi.cu'), newline='').read()
body = src[src.index('namespace risc0::circuit::rv32im_v2::cuda {') + len('namespace risc0::circuit::rv32im_v2::cuda {'):src.rindex('} // namespace risc0::circuit::rv32im_v2::cuda')]
starts = [(m.start(), m.group(1)) for m in re.finditer(r'^__device__ [^(;{}]*?\b(\w+)\(', body, re.M)]
defs = {}
for i, (s, n) in enumerate(starts):
    defs[n] = body[s:(starts[i + 1][0] if i + 1 < len(starts) else len(body))]
call_re = re.compile(r'\b(exec\w*|back\w*|step_\w+)\(')
seen, st = [], ['step_Top']
while st:
    n = st.pop()
    if n in seen: continue
    seen.append(n); st += [c for c in call_re.findall(defs[n]) if c != n and c in defs]
names = set(seen)
assert not any(l.startswith('#') for n in names for l in defs[n].split('\n')), \
    'preprocessor line inside the copied call graph'
ren = re.compile(r'\b(' + '|'.join(sorted(names, key=len, reverse=True)) + r')\(')
order = [n for _, n in starts if n in names]
# extern defs from ffi.cu: from divide_rv32im up to nextStep
e0 = ffi.index('__device__ ::cuda::std::array<uint32_t, 2>\ndivide_rv32im'); e1 = ffi.index('__device__ void nextStep(')
ext = re.sub(r'^__device__', 'static __device__', ffi[e0:e1], flags=re.M)
# witgen.h declares the extern_* functions non-static: use renamed static copies (externW_*)
# and point INVOKE_EXTERN at them; log/assert stay the witgen.h no-ops.
ext = re.sub(r'\bextern_(getMemoryTxn|lookupDelta|lookupCurrent|memoryDelta|getDiffCount|isFirstCycle_0|divide|print|getMajorMinor|hostReadPrepare|hostWrite|nextPagingIdx|bigIntExtern)\b', r'externW_\1', ext)
dx0 = ffi.index('struct ExecBuffers {'); dx1 = ffi.index('struct AccumBuffers {')
out = ['// GENERATED by kernels/cuda/witgen_tu_gen.py from steps.cu and ffi.cu -- do not edit; re-run it',
       '// instead. The step_Top call graph only, renamed (_W), noinline, compiled as a whole-program TU',
       '// (not -dc).',
       '#define RISC0_ROWS_POW2 1', '#include "witgen.h"', '#ifdef __HIPCC__', '#include <hip/hip_runtime.h>', '#else', '#include <cuda_runtime.h>', '#endif', '#undef INVOKE_EXTERN', '#define INVOKE_EXTERN(ctx, name, ...) externW_##name(ctx, ##__VA_ARGS__)', '#define externW_log extern_log', '#define externW_assert extern_assert',
       '// The call graph\'s attribute: noinline (on ROCm build.rs may define it empty: inlined at the',
       '// compiler\'s discretion).', '#ifndef WITGEN_TU_FNATTR', '#define WITGEN_TU_FNATTR __noinline__', '#endif',
       'namespace risc0::circuit::rv32im_v2::cuda {', ffi[dx0:dx1], ext]
protos, fns = [], []
for n in order:
    d = ren.sub(lambda m: m.group(1) + '_W(', defs[n]).rstrip() + '\n'
    d = 'static __device__ WITGEN_TU_FNATTR ' + d[len('__device__ '):]
    protos.append(d[:d.index('{')].rstrip() + ';'); fns.append(d)
out += protos + fns
out.append('''
#ifdef __HIPCC__
// ROCm (build.rs): WITGEN_TU_LB sets the launch bounds (HIP: max threads, min waves per SIMD),
// WITGEN_TU_KERNEL / WITGEN_TU_LAUNCH the names of a variant; perm (nullable): the launch's
// cycles in the order to run them (absolute cycle numbers), as par_stepExec in ffi.cu.
#ifndef WITGEN_TU_LB
#define WITGEN_TU_LB 256, 8
#endif
#ifndef WITGEN_TU_KERNEL
#define WITGEN_TU_KERNEL par_stepExecW
#define WITGEN_TU_LAUNCH risc0_rv32im_launch_par_stepExecTU
#endif
__global__ void __launch_bounds__(WITGEN_TU_LB) WITGEN_TU_KERNEL(DeviceExecContext* ctx, uint32_t start, uint32_t count, const uint32_t* perm) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle >= count) return;
  cycle = perm ? perm[cycle] : start + cycle;
#else
__global__ void __launch_bounds__(256, 1) par_stepExecW(DeviceExecContext* ctx, uint32_t start, uint32_t count) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle >= count) return;
  cycle += start;
#endif
  ExecContext execCtx(*ctx->preflight, *ctx->tables, cycle);
  Buffer dataBuf = *ctx->data;
  Buffer preDataBuf = *ctx->pre_data;
  if (preDataBuf.buf == dataBuf.buf) preDataBuf.buf = nullptr;
  MutableBufObj data(dataBuf, preDataBuf);
  Buffer globalBuf = *ctx->global;
  GlobalBufObj global(globalBuf);
  step_Top_W(execCtx, &data, &global);
}
} // namespace risc0::circuit::rv32im_v2::cuda
#ifdef __HIPCC__
void WITGEN_TU_LAUNCH(unsigned grid, unsigned block, cudaStream_t stream, void* ctx, uint32_t start, uint32_t count, const uint32_t* perm) {
  using namespace risc0::circuit::rv32im_v2::cuda;
  WITGEN_TU_KERNEL<<<grid, block, 0, stream>>>((DeviceExecContext*)ctx, start, count, perm);
}
#else
void risc0_rv32im_launch_par_stepExecTU(unsigned grid, unsigned block, cudaStream_t stream, void* ctx, uint32_t start, uint32_t count) {
  using namespace risc0::circuit::rv32im_v2::cuda;
  par_stepExecW<<<grid, block, 0, stream>>>((DeviceExecContext*)ctx, start, count);
}
#endif
''')
new = '\n'.join(out)
# ------------------------------------------------------------------------------------------------
# --metal: witgen_tu.metal (Metal port, item R5; plan 4.2 row par_stepExecW)
# ------------------------------------------------------------------------------------------------

# Rewrite counts of cuda2metal.py's passes over the call graph (194 functions: 194 prototypes +
# 194 definitions). A change means steps.cu changed shape: re-run with --no-pin, review the
# counts and the compile, then update these numbers.
PINNED_METAL = {
    'functions': 194,
    'fnattr': 388,
    'counts': {
        'aspace.thread': 388,
        'assert': 41,
        'lambda.map': 41,
        'lambda.map.unroll': 41,
        'lambda.reduce': 1,
        'lambda.reduce.unroll': 1,
        'qual.device_fn': 388,
        'qual.noinline': 388,
        'sbind': 6,
        'value_type': 84,
    },
}
# sha256 of ffi.cu's extern section (divide_rv32im .. nextStep, the text witgen_tu.cu copies).
# METAL_EXTERNS below is a hand port of it; when ffi.cu changes there, re-port and update this.
FFI_EXTERN_SHA256 = '2f7c235c64764221bafb61042fd535c4cec37405772e13af28a7b5ff7a7cf1c9'

# How the call graph's map()/reduce() sites are lowered (cuda2metal.py `map_lowering`): 'unroll'
# (the default, R17) or 'loop' (CUDA's shape, which the Apple GPU miscompiles/misexecutes; only
# for a baseline build). See the header comment.
MAP_LOWERINGS = ('unroll', 'loop')
# A `for`/`while`/`do` statement in the translated call graph (not the externs or kernels).
_GRAPH_LOOP_RE = re.compile(r'\b(for|while|do)\s*[({]')



def gen_metal(no_pin=False, map_lowering='unroll'):
    import hashlib
    import cuda2metal as c2m
    if map_lowering not in MAP_LOWERINGS:
        raise c2m.TranslateError('unknown map lowering %r (%s)'
                                 % (map_lowering, ' | '.join(MAP_LOWERINGS)))
    ext_sha = hashlib.sha256(ffi[e0:e1].encode()).hexdigest()
    if ext_sha != FFI_EXTERN_SHA256:
        raise c2m.TranslateError(
            'ffi.cu extern section (divide_rv32im .. nextStep) changed (sha256 %s, ported %s): '
            're-port METAL_EXTERNS in witgen_tu_gen.py and update FFI_EXTERN_SHA256'
            % (ext_sha, FFI_EXTERN_SHA256))
    # The call graph as CUDA text in the CUDA namespace, every function
    # `static __device__ __noinline__` (cuda2metal.py's qual pass maps both).
    protos, fns = [], []
    for n in order:
        d = ren.sub(lambda m: m.group(1) + '_W(', defs[n]).rstrip() + '\n'
        d = 'static __device__ __noinline__ ' + d[len('__device__ '):]
        protos.append(d[:d.index('{')].rstrip() + ';')
        fns.append(d)
    cuda = '\n'.join(['namespace risc0::circuit::rv32im_v2::cuda {'] + protos + fns +
                     ['} // namespace risc0::circuit::rv32im_v2::cuda', ''])
    typedefs = c2m.collect_array_typedefs(c2m.read(os.path.join(HERE, 'types.cuh.inc')))
    passes = tuple(p for p in c2m.ALL_PASSES if p not in ('kernel', 'accum_guard'))
    text, counts = c2m.translate(cuda, typedefs, passes=passes, map_lowering=map_lowering)
    # The attribute becomes WITGEN_TU_FNATTR (default noinline; see the TU header).
    text, nattr = re.subn(r'(?m)^static __attribute__\(\(noinline\)\) ', 'static WITGEN_TU_FNATTR ',
                          text)
    got = dict(counts)
    bad = []
    if len(names) != PINNED_METAL['functions']:
        bad.append('functions: pinned %d, got %d' % (PINNED_METAL['functions'], len(names)))
    if nattr != PINNED_METAL['fnattr'] or text.count('__attribute__((noinline))'):
        bad.append('fnattr: pinned %d, got %d (+%d left)'
                   % (PINNED_METAL['fnattr'], nattr, text.count('__attribute__((noinline))')))
    want = dict(PINNED_METAL['counts'])
    if map_lowering == 'loop':
        want = {k: v for k, v in want.items() if not k.endswith('.unroll')}
    for k in sorted(set(want) | set(got)):
        if want.get(k, 0) != got.get(k, 0):
            bad.append('%s: pinned %d, got %d' % (k, want.get(k, 0), got.get(k, 0)))
    report = 'functions=%d, fnattr=%d, %s, map lowering %s' % (len(names), nattr,
                                                               c2m.fmt_counts(got), map_lowering)
    if bad and not no_pin:
        raise c2m.TranslateError('rewrite counts differ from PINNED_METAL (steps.cu changed '
                                 'shape; re-run with --no-pin and review):\n  ' + '\n  '.join(bad)
                                 + '\n  got: ' + report)
    ns0 = text.index('namespace %s {' % c2m.METAL_NS) + len('namespace %s {' % c2m.METAL_NS)
    ns1 = text.rindex('} // namespace %s' % c2m.METAL_NS)
    graph = text[ns0:ns1].strip('\n')
    # R17: no loop in the call graph (every loop there would be a lowered map/reduce calling a
    # noinline step function, the shape of the Apple GPU lost-store defect). Not waivable with
    # --no-pin: a new loop shape in steps.cu must be lowered without a loop first.
    loops = len(_GRAPH_LOOP_RE.findall(c2m.mask(graph)))
    if map_lowering == 'unroll' and loops:
        raise c2m.TranslateError(
            '%d loop(s) in the translated call graph: steps.cu has a loop shape that the '
            'unrolled lowering does not cover (docs/metal-port/notes/R17.md); lower it without '
            'a loop in cuda2metal.py first' % loops)
    out = METAL_HEADER + c2m.PRELUDE + METAL_PRE + METAL_EXTERNS + graph + '\n' + METAL_KERNELS
    return out, report, bad


def main_metal(args):
    import cuda2metal as c2m
    if '-o' not in args or args.index('-o') + 1 >= len(args):
        sys.exit('usage: witgen_tu_gen.py --metal -o OUTDIR [--check] [--no-pin] '
                 '[--map-lowering unroll|loop]')
    outdir = args[args.index('-o') + 1]
    lowering = 'unroll'
    if '--map-lowering' in args:
        if args.index('--map-lowering') + 1 >= len(args):
            sys.exit('witgen_tu_gen: --map-lowering needs a value (%s)' % ' | '.join(MAP_LOWERINGS))
        lowering = args[args.index('--map-lowering') + 1]
    try:
        new, report, bad = gen_metal(no_pin='--no-pin' in args, map_lowering=lowering)
    except c2m.TranslateError as e:
        sys.exit('witgen_tu_gen: error: %s' % e)
    path = os.path.join(outdir, 'witgen_tu.metal')
    try:
        cur = open(path, newline='').read()
    except FileNotFoundError:
        cur = None
    if bad:
        print('witgen_tu_gen: counts differ from PINNED_METAL:\n  ' + '\n  '.join(bad))
    if CHECK:
        if cur != new:
            sys.exit('witgen_tu_gen: %s is out of date; run kernels/cuda/witgen_tu_gen.py --metal -o %s'
                     % (path, outdir))
        print('witgen_tu_gen: %s is up to date (%s)' % (path, report))
        return 0
    os.makedirs(outdir, exist_ok=True)
    if cur != new:
        open(path, 'w', newline='').write(new)
        print('witgen_tu_gen: wrote %s (%d lines; %s)' % (path, new.count('\n'), report))
    else:
        print('witgen_tu_gen: %s unchanged (%s)' % (path, report))
    return 0


METAL_HEADER = r'''// GENERATED by rv32im-sys/kernels/cuda/witgen_tu_gen.py --metal from steps.cu and ffi.cu -- do
// not edit; re-run it instead (rv32im-sys/build.rs does, into $OUT_DIR/metal).
//
// Metal port of witgen_tu.cu (plan 4.2, item R5): step_Top's call graph renamed (_W), every
// function WITGEN_TU_FNATTR (default __attribute__((noinline)): the CUDA comment in
// witgen_tu_gen.py says the fully inlined build reorders the back-row loads of rows written by
// neighbouring threads and is not bit-exact), translated by cuda2metal.py's passes with every
// map()/reduce() unrolled into straight-line blocks (R17: CUDA's loop lowering is the code shape
// in which the Apple GPU loses a private-array store, docs/metal-port/notes/R17.md); the
// extern_* functions as static externW_* copies (hand port of ffi.cu, see below); and the
// kernels par_stepExecW (CUDA par_stepExecW, one thread per cycle; the HIP form's optional perm
// is kept behind WitgenParams::usePerm) and fwd_stepExec / rev_stepExec (CUDA ffi.cu
// fwd_stepExec / rev_stepExec, single thread, chunked by the host per seq_exec.h).
//
// Buffer bindings (explicit, plan 2.4), identical for the three kernels up to index 9:
//   0 data        device Fp*                 rows x dataCols, column-major
//   1 pre_data    device Fp*                 rows x dataCols: par_stepExecW's back > 0 loads
//                                            read it. Bind a separate copy of data (default,
//                                            plan 4.2); binding the data buffer itself selects
//                                            single-buffer mode (CUDA: pre_data == data ->
//                                            nullptr). fwd_stepExec / rev_stepExec ignore it
//                                            and read back rows from data (CUDA ffi.cu nulls
//                                            pre_data for the sequential modes); any valid
//                                            buffer may be bound there.
//   2 global      device Fp*                 globalRows x globalCols
//   3 cycles      device PreflightCycle*     cyclesLen entries; txnIdx is incremented in place
//   4 txns        const device MemoryTransaction*   txnsLen entries
//   5 bigintBytes const device uint8_t*      bigintBytesLen bytes
//   6 tableU8     device atomic_uint*        256 counters (zeroed by the host)
//   7 tableU16    device atomic_uint*        65536 counters (zeroed by the host)
//   8 fault       device atomic_uint*        1 word; RV32IM_FAULT_* bits ORed once per thread
//   9 WitgenParams (set_bytes, 48 bytes)
//  10 par_stepExecW: perm   const device uint*   count entries, read only if usePerm != 0
//     fwd_stepExec / rev_stepExec: SeqParams (set_bytes, 16 bytes; one 1x1 dispatch per
//     RV32IM_SEQ_CHUNK positions, in order, in one encoder)
// Every pointer binding must be a valid buffer (bind a small dummy for empty txns/bigint/perm).
// The host asserts rows * dataCols < 2^32 (32-bit indexing, cuda_compat.h).
#define RISC0_ROWS_POW2 1
#define RV32IM_METAL_INCLUDE_GENERATED 1
#include "witgen.h"
#include "seq_exec.h"

'''

METAL_PRE = r'''
#undef INVOKE_EXTERN
#define INVOKE_EXTERN(ctx, name, ...) externW_##name(ctx, ##__VA_ARGS__)
#define externW_log extern_log
#define externW_assert extern_assert
// The call graph's attribute (an experiment may define it empty or always_inline; only the
// noinline default is the CUDA-equivalent build).
#ifndef WITGEN_TU_FNATTR
#define WITGEN_TU_FNATTR __attribute__((noinline))
#endif
// par_stepExecW's [[max_total_threads_per_threadgroup]] (CUDA __launch_bounds__(256, 1)).
#ifndef WITGEN_TU_MAX_THREADS
#define WITGEN_TU_MAX_THREADS 256
#endif
// RV32IM_FAULT_CYCLE_INDEX / RV32IM_FAULT_DIFF_COUNT_INDEX: cuda_compat.h.

namespace rv32im_metal {

'''

METAL_EXTERNS = r'''// ---- extern_*: hand port of ffi.cu divide_rv32im .. extern_bigIntExtern (pinned by
// FFI_EXTERN_SHA256 in witgen_tu_gen.py). static externW_* copies as in witgen_tu.cu (so other
// TUs of the metallib may define their own). Differences, on invalid input only (plan 2.9):
// a txnIdx / bigintIdx outside the uploaded preflight sets a RV32IM_FAULT_* bit in
// ctx.tables.bad (ORed into the fault buffer by the kernel) and yields zeros instead of an
// out-of-bounds read; so does a getDiffCount cycle past the preflight cycles. The 7th
// structured binding of the TU (extern_divide) is a temp here.

__attribute__((unused)) static inline r0::array<uint32_t, 2>
divide_rv32im(uint32_t numer, uint32_t denom, uint32_t signType) {
  uint32_t onesComp = (signType == 2);
  bool negNumer = signType && int32_t(numer) < 0;
  bool negDenom = signType == 1 && int32_t(denom) < 0;
  if (negNumer) {
    numer = -numer - onesComp;
  }
  if (negDenom) {
    denom = -denom - onesComp;
  }
  uint32_t quot;
  uint32_t rem;
  if (denom == 0) {
    quot = 0xffffffff;
    rem = numer;
  } else {
    quot = numer / denom;
    rem = numer % denom;
  }
  uint32_t quotNegOut = (negNumer ^ negDenom) - ((denom == 0) * negNumer);
  uint32_t remNegOut = negNumer;
  if (quotNegOut) {
    quot = -quot - onesComp;
  }
  if (remNegOut) {
    rem = -rem - onesComp;
  }
  r0::array<uint32_t, 2> ret;
  ret[0] = quot;
  ret[1] = rem;
  return ret;
}

/// The cycle's memory transaction index, post-incremented in the preflight (ffi.cu: plain
/// post-increment, each thread handles exactly one cycle); false if it is out of range.
static inline bool externW_txnIndex(thread ExecContext& ctx, bool advance, thread uint32_t& txnIdx) {
  device PreflightCycle& cyc = ctx.preflight.cycles[ctx.cycle];
  txnIdx = advance ? cyc.txnIdx++ : cyc.txnIdx;
  if (txnIdx >= ctx.preflight.txnsLen) {
    ctx.tables.bad |= RV32IM_FAULT_TXN_INDEX;
    return false;
  }
  return true;
}

static inline r0::array<Val, 5> externW_getMemoryTxn(thread ExecContext& ctx, Val addrElem) {
  uint32_t addr = addrElem.asUInt32();
  r0::array<Val, 5> ret;
  uint32_t txnIdx;
  if (!externW_txnIndex(ctx, true, txnIdx)) {
    for (uint32_t i = 0; i < 5; i++) {
      ret[i] = Val(0);
    }
    return ret;
  }
  const device MemoryTransaction& txn = ctx.preflight.txns[txnIdx];
  // Assertions removed (ffi.cu): in parallel witgen, union layout overwrites can cause transient
  // address mismatches; eval_check verifies all constraints on the final witness.
  ret[0] = Val(txn.prevCycle);
  ret[1] = Val(txn.prevWord & 0xffff);
  ret[2] = Val(txn.prevWord >> 16);
  ret[3] = Val(txn.word & 0xffff);
  ret[4] = Val(txn.word >> 16);
  return ret;
}

static inline void externW_lookupDelta(thread ExecContext& ctx, Val table, Val index, Val count) {
  ctx.tables.lookupDelta(table, index, count);
}

static inline Val externW_lookupCurrent(thread ExecContext& ctx, Val table, Val index) {
  return ctx.tables.lookupCurrent(table, index);
}

static inline void externW_memoryDelta(
    thread ExecContext& ctx, Val addr, Val cycle, Val dataLow, Val dataHigh, Val count) {}

/// The argument is the witness value of the current or a neighbouring cycle, 2 * cycle (+ 1);
/// an index past the preflight cycles (a corrupt witness value) sets
/// RV32IM_FAULT_DIFF_COUNT_INDEX and yields 0 instead of an out-of-bounds read (plan 2.9).
static inline uint32_t externW_getDiffCount(thread ExecContext& ctx, Val cycle) {
  uint32_t cycleU32 = cycle.asUInt32();
  uint32_t idx = cycleU32 / 2;
  if (idx >= ctx.preflight.cyclesLen) {
    ctx.tables.bad |= RV32IM_FAULT_DIFF_COUNT_INDEX;
    return 0;
  }
  return ctx.preflight.cycles[idx].diffCount[cycleU32 % 2];
}

static inline Val externW_isFirstCycle_0(thread ExecContext& ctx) {
  return Val(ctx.cycle == 0 ? 1u : 0u);
}

__attribute__((unused)) static inline r0::array<Val, 4> externW_divide(
    thread ExecContext& ctx, Val numerLow, Val numerHigh, Val denomLow, Val denomHigh, Val signType) {
  uint32_t numer = numerLow.asUInt32() | (numerHigh.asUInt32() << 16);
  uint32_t denom = denomLow.asUInt32() | (denomHigh.asUInt32() << 16);
  r0::array<uint32_t, 2> qr = divide_rv32im(numer, denom, signType.asUInt32());
  uint32_t quot = qr[0];
  uint32_t rem = qr[1];
  r0::array<Val, 4> ret;
  ret[0] = Val(quot & 0xffff);
  ret[1] = Val(quot >> 16);
  ret[2] = Val(rem & 0xffff);
  ret[3] = Val(rem >> 16);
  return ret;
}

__attribute__((unused)) static inline void externW_print(thread ExecContext& ctx, Val v) {}

static inline r0::array<Val, 2> externW_getMajorMinor(thread ExecContext& ctx) {
  r0::array<Val, 2> ret;
  ret[0] = Val(uint32_t(ctx.preflight.cycles[ctx.cycle].major));
  ret[1] = Val(uint32_t(ctx.preflight.cycles[ctx.cycle].minor));
  return ret;
}

static inline Val externW_hostReadPrepare(thread ExecContext& ctx, Val fp, Val len) {
  uint32_t txnIdx;
  if (!externW_txnIndex(ctx, false, txnIdx)) {
    return Val(0);
  }
  return Val(ctx.preflight.txns[txnIdx].word);
}

static inline Val
externW_hostWrite(thread ExecContext& ctx, Val fdVal, Val addrLow, Val addrHigh, Val lenVal) {
  uint32_t txnIdx;
  if (!externW_txnIndex(ctx, false, txnIdx)) {
    return Val(0);
  }
  return Val(ctx.preflight.txns[txnIdx].word);
}

static inline r0::array<Val, 2> externW_nextPagingIdx(thread ExecContext& ctx) {
  r0::array<Val, 2> ret;
  ret[0] = Val(ctx.preflight.cycles[ctx.cycle].pagingIdx);
  ret[1] = Val(uint32_t(ctx.preflight.cycles[ctx.cycle].machineMode));
  return ret;
}

static inline r0::array<Val, 16> externW_bigIntExtern(thread ExecContext& ctx) {
  r0::array<Val, 16> ret;
  uint32_t bigintIdx = ctx.preflight.cycles[ctx.cycle].bigintIdx;
  uint32_t len = ctx.preflight.bigintBytesLen;
  bool ok = bigintIdx <= len && len - bigintIdx >= 16;
  if (!ok) {
    ctx.tables.bad |= RV32IM_FAULT_BIGINT_INDEX;
  }
  for (uint32_t i = 0; i < 16; i++) {
    ret[i] = ok ? Val(uint32_t(ctx.preflight.bigintBytes[bigintIdx + i])) : Val(0);
  }
  return ret;
}

// ---- step_Top's call graph (cuda2metal.py passes over steps.cu) ----

'''

METAL_KERNELS = r'''
} // namespace rv32im_metal

using namespace rv32im_metal;

/// par_stepExecW / fwd_stepExec / rev_stepExec params (binding 9; mirror #[repr(C)] on the host).
struct WitgenParams {
  uint32_t rows;            // rows of data / pre_data (a power of two: RISC0_ROWS_POW2)
  uint32_t dataCols;        // columns of data / pre_data
  uint32_t globalRows;      // rows of global (RawBuffer::rows of the CPU HAL)
  uint32_t globalCols;      // columns of global
  uint32_t cyclesLen;       // entries in cycles
  uint32_t txnsLen;         // entries in txns
  uint32_t bigintBytesLen;  // bytes in bigintBytes
  uint32_t tableSplitCycle; // PreflightTrace::tableSplitCycle
  uint32_t start;           // par_stepExecW: first cycle of the launch (usePerm == 0)
  uint32_t count;           // par_stepExecW: threads that run a cycle
  uint32_t usePerm;         // par_stepExecW: 1 -> cycle = perm[i], 0 -> start + i
  uint32_t _pad;
};
RV32IM_ASSERT_SIZE(WitgenParams, 48);
RV32IM_ASSERT_OFFSET(WitgenParams, rows, 0);
RV32IM_ASSERT_OFFSET(WitgenParams, dataCols, 4);
RV32IM_ASSERT_OFFSET(WitgenParams, globalRows, 8);
RV32IM_ASSERT_OFFSET(WitgenParams, globalCols, 12);
RV32IM_ASSERT_OFFSET(WitgenParams, cyclesLen, 16);
RV32IM_ASSERT_OFFSET(WitgenParams, txnsLen, 20);
RV32IM_ASSERT_OFFSET(WitgenParams, bigintBytesLen, 24);
RV32IM_ASSERT_OFFSET(WitgenParams, tableSplitCycle, 28);
RV32IM_ASSERT_OFFSET(WitgenParams, start, 32);
RV32IM_ASSERT_OFFSET(WitgenParams, count, 36);
RV32IM_ASSERT_OFFSET(WitgenParams, usePerm, 40);

/// The kernels' bindings, as a thread value (CUDA: DeviceExecContext in device memory).
struct WitgenBindings {
  device Fp* data;
  device Fp* preData;
  device Fp* global;
  device PreflightCycle* cycles;
  const device MemoryTransaction* txns;
  const device uint8_t* bigintBytes;
  device metal::atomic_uint* tableU8;
  device metal::atomic_uint* tableU16;
  constant WitgenParams* p;
};

/// CUDA ffi.cu nextStep / witgen_tu.cu par_stepExecW body for one cycle (calling step_Top_W).
/// Returns the RV32IM_FAULT_* bits the cycle raised.
static inline uint32_t witgenStep(thread const WitgenBindings& b, uint32_t cycle) {
  constant WitgenParams& p = *b.p;
  if (cycle >= p.cyclesLen || cycle >= p.rows) {
    return RV32IM_FAULT_CYCLE_INDEX;
  }
  PreflightTrace preflight;
  preflight.cycles = b.cycles;
  preflight.cyclesLen = p.cyclesLen;
  preflight.txns = b.txns;
  preflight.bigintBytes = b.bigintBytes;
  preflight.txnsLen = p.txnsLen;
  preflight.bigintBytesLen = p.bigintBytesLen;
  preflight.tableSplitCycle = p.tableSplitCycle;
  LookupTables tables = makeLookupTables(b.tableU8, b.tableU16);
  ExecContext execCtx(preflight, tables, cycle);
  Buffer dataBuf = makeBuffer(b.data, p.rows, p.dataCols);
  Buffer preDataBuf = makeBuffer(b.preData, p.rows, p.dataCols);
  // When pre_data is the data buffer itself, null it out (CUDA: one check per cycle instead of
  // one per load).
  if (preDataBuf.buf == dataBuf.buf) {
    preDataBuf.buf = nullptr;
  }
  BufferObj data = makeMutableBuf(dataBuf, preDataBuf);
  Buffer globalBuf = makeBuffer(b.global, p.globalRows, p.globalCols);
  BufferObj global = makeGlobalBuf(globalBuf);
  step_Top_W(execCtx, &data, &global);
  return tables.bad;
}

#define WITGEN_TU_BINDINGS                                                                     \
  device Fp *data [[buffer(0)]], device Fp *preData [[buffer(1)]],                             \
      device Fp *global [[buffer(2)]], device PreflightCycle *cycles [[buffer(3)]],            \
      const device MemoryTransaction *txns [[buffer(4)]],                                      \
      const device uint8_t *bigintBytes [[buffer(5)]],                                         \
      device metal::atomic_uint *tableU8 [[buffer(6)]],                                        \
      device metal::atomic_uint *tableU16 [[buffer(7)]],                                       \
      device metal::atomic_uint *fault [[buffer(8)]], constant WitgenParams &p [[buffer(9)]]

#define WITGEN_TU_MAKE_BINDINGS                                                                \
  WitgenBindings b = {data, preData, global, cycles, txns, bigintBytes, tableU8, tableU16, &p}

/// CUDA witgen_tu.cu par_stepExecW: thread i runs cycle start + i (or perm[i]), i < count.
[[max_total_threads_per_threadgroup(WITGEN_TU_MAX_THREADS)]]
kernel void par_stepExecW(WITGEN_TU_BINDINGS,
                          const device uint32_t* perm [[buffer(10)]],
                          uint gid [[thread_position_in_grid]]) {
  uint32_t cycle = gid;
  if (cycle >= p.count) {
    return;
  }
  cycle = p.usePerm ? perm[cycle] : p.start + cycle;
  WITGEN_TU_MAKE_BINDINGS;
  uint32_t bad = witgenStep(b, cycle);
  if (bad) {
    metal::atomic_fetch_or_explicit(fault, bad, metal::memory_order_relaxed);
  }
}

/// seq_exec.h step functor: one cycle of step_Top_W, fault bits accumulated.
struct WitgenSeqStep {
  WitgenBindings b;
  uint32_t bad;

  void operator()(uint32_t cycle) thread { bad |= witgenStep(b, cycle); }
};

/// CUDA ffi.cu fwd_stepExec (cycles 0 .. lastCycle - 1 in order), positions [begin, end).
kernel void fwd_stepExec(WITGEN_TU_BINDINGS,
                         constant SeqParams& sp [[buffer(10)]],
                         uint gid [[thread_position_in_grid]]) {
  if (gid != 0) {
    return;
  }
  WITGEN_TU_MAKE_BINDINGS;
  // CUDA ffi.cu:1596-1602 nulls pre_data for the sequential modes (no cross-thread race), so
  // back > 0 loads read the rows this pass already wrote. witgenStep nulls preData == data.
  b.preData = data;
  WitgenSeqStep step{b, 0};
  rv32im_seq_fwd(step, sp);
  if (step.bad) {
    metal::atomic_fetch_or_explicit(fault, step.bad, metal::memory_order_relaxed);
  }
}

/// CUDA ffi.cu rev_stepExec (split - 1 .. 0, then lastCycle - 1 .. split), positions
/// [begin, end) of that order.
kernel void rev_stepExec(WITGEN_TU_BINDINGS,
                         constant SeqParams& sp [[buffer(10)]],
                         uint gid [[thread_position_in_grid]]) {
  if (gid != 0) {
    return;
  }
  WITGEN_TU_MAKE_BINDINGS;
  // CUDA ffi.cu:1596-1602 nulls pre_data for the sequential modes (no cross-thread race), so
  // back > 0 loads read the rows this pass already wrote. witgenStep nulls preData == data.
  b.preData = data;
  WitgenSeqStep step{b, 0};
  rv32im_seq_rev(step, sp);
  if (step.bad) {
    metal::atomic_fetch_or_explicit(fault, step.bad, metal::memory_order_relaxed);
  }
}
'''


def main_cuda(check):
    """Writes witgen_tu.cu, or with check compares it and translates --metal in memory."""
    try:
        cur = open(OUT, newline='').read()
    except FileNotFoundError:
        cur = ''
    if not check:
        open(OUT, 'w', newline='').write(new)
        print('witgen_tu_gen: wrote witgen_tu.cu (%d functions)' % len(names))
        return 0
    if cur != new:
        sys.exit('witgen_tu_gen: witgen_tu.cu is out of date; run kernels/cuda/witgen_tu_gen.py '
                 'and commit the result')
    # The Metal translation too (its output lives in the build's OUT_DIR, never checked in): in
    # memory, with the pins and the no-loop check (R16 open issue, as accum_tu_gen.py).
    sys.dont_write_bytecode = True
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import cuda2metal as c2m
    try:
        _, report, bad = gen_metal()
    except c2m.TranslateError as e:
        sys.exit('witgen_tu_gen: --metal translation: %s' % e)
    print('witgen_tu_gen: witgen_tu.cu is up to date (%d functions); --metal translates (%s)'
          % (len(names), report))
    return 0


if __name__ == '__main__':
    sys.dont_write_bytecode = True  # no __pycache__ in the source tree (build.rs runs this)
    sys.path.insert(0, HERE)
    sys.exit(main_metal(ARGS) if METAL else main_cuda(CHECK))
