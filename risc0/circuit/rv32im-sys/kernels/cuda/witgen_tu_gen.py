#!/usr/bin/env python3
"""gen_witgen_tu.py <steps.cu> <ffi.cu> <out.cu> [--inline]
Standalone TU with the step_Top call graph of steps.cu renamed (_W), static (noinline by default),
RISC0_ROWS_POW2, the extern_* device functions copied from ffi.cu, and par_stepExecW + launcher."""
import re, sys
src = open(sys.argv[1]).read(); ffi = open(sys.argv[2]).read()
FI = '__forceinline__' if '--inline' in sys.argv else '__noinline__'
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
ren = re.compile(r'\b(' + '|'.join(sorted(names, key=len, reverse=True)) + r')\(')
order = [n for _, n in starts if n in names]
# extern defs from ffi.cu: from divide_rv32im up to nextStep
e0 = ffi.index('__device__ ::cuda::std::array<uint32_t, 2>\ndivide_rv32im'); e1 = ffi.index('__device__ void nextStep(')
ext = re.sub(r'^__device__', 'static __device__', ffi[e0:e1], flags=re.M)
# witgen.h declares the extern_* functions non-static: use renamed static copies (externW_*)
# and point INVOKE_EXTERN at them; log/assert stay the witgen.h no-ops.
ext = re.sub(r'\bextern_(getMemoryTxn|lookupDelta|lookupCurrent|memoryDelta|getDiffCount|isFirstCycle_0|divide|print|getMajorMinor|hostReadPrepare|hostWrite|nextPagingIdx|bigIntExtern)\b', r'externW_\1', ext)
dx0 = ffi.index('struct ExecBuffers {'); dx1 = ffi.index('struct AccumBuffers {')
out = ['#define RISC0_ROWS_POW2 1', '#include "witgen.h"', '#include <cuda_runtime.h>', '#undef INVOKE_EXTERN', '#define INVOKE_EXTERN(ctx, name, ...) externW_##name(ctx, ##__VA_ARGS__)', '#define externW_log extern_log', '#define externW_assert extern_assert', 'namespace risc0::circuit::rv32im_v2::cuda {', ffi[dx0:dx1], ext]
protos, fns = [], []
for n in order:
    d = ren.sub(lambda m: m.group(1) + '_W(', defs[n]).rstrip() + '\n'
    d = f'static __device__ {FI} ' + d[len('__device__ '):]
    protos.append(d[:d.index('{')].rstrip() + ';'); fns.append(d)
out += protos + fns
out.append('''
__global__ void __launch_bounds__(256, 1) par_stepExecW(DeviceExecContext* ctx, uint32_t start, uint32_t count) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle >= count) return;
  cycle += start;
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
void risc0_rv32im_launch_par_stepExecTU(unsigned grid, unsigned block, cudaStream_t stream, void* ctx, uint32_t start, uint32_t count) {
  using namespace risc0::circuit::rv32im_v2::cuda;
  par_stepExecW<<<grid, block, 0, stream>>>((DeviceExecContext*)ctx, start, count);
}
''')
open(sys.argv[3], 'w').write('\n'.join(out))
print(len(names), 'functions')
