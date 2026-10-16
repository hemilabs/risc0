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

# usage: eval_check_split.py [--check]
#
# Writes eval_check_split.cu (CUDA only, next to this script) from the generated
# eval_check.cu: poly_fp is split into kNumParts kernels (ec_part<P>) along its
# gated-sum tree. Each part re-applies the gate/mix factors of the tree nodes it
# touches and adds the previous parts' sum left in `check`; the last part scales
# by the quotient. Only the regrouping of field sums changes, so the result is
# bit-identical to eval_check.cu. Subtrees costing more than TARGET_OPS are
# split into their terms; leaves are packed into parts of at most GROUP_OPS ops
# (the measured optimum on sm_120: 6 parts).
#
# build.rs compiles the result standalone (not relocatable, which is ~25%
# slower), so it keeps its own __constant__ poly_mix, filled by
# eval_check_split() itself. Re-run this after regenerating eval_check.cu;
# --check (run in CI) fails if eval_check_split.cu is out of date.
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'eval_check_split.cu')
CHECK = '--check' in sys.argv[1:]
TARGET_OPS, GROUP_OPS = 2200, 3000
EXPECT = 'total ops 12075 leaves 65 groups 6 max leaf 2049'
target, gtarget = TARGET_OPS, GROUP_OPS
src = open(os.path.join(HERE, 'eval_check.cu'), newline='').read().split('\n')
defs = {}; order = []
rx = re.compile(r'^\s*(Fp|FpExt|auto) (x\d+) = (.*);\s*$')
rc = re.compile(r'^\s*Fp (x\d+)\((\d+)\);\s*$')
ret = None
for l in src:
    m = rx.match(l)
    if m: defs[m.group(2)] = (m.group(1), m.group(3)); order.append(m.group(2)); continue
    m = rc.match(l)
    if m: defs[m.group(1)] = ('Fp', None, m.group(2)); order.append(m.group(1)); continue
    m = re.match(r'^\s*return (x\d+);', l)
    if m: ret = m.group(1)
tok = re.compile(r'\bx\d+\b')
def deps(v):
    d = defs[v]; return tok.findall(d[1]) if d[1] is not None else []
def slice_of(vs):
    seen = set(); st = list(vs)
    while st:
        v = st.pop()
        if v in seen: continue
        seen.add(v); st.extend(deps(v))
    return seen
def cost(vs): return sum(1 for v in slice_of(vs) if defs[v][1] is not None)
ra = re.compile(r'^(x\d+) \+ (x\d+) \* (x\d+) \* (poly_mix\[\d+\])$')
rb = re.compile(r'^(x\d+) \+ (poly_mix\[\d+\]) \* (x\d+)$')
# node: dict(f=[factors], kids=[nodes]) or leaf dict(f=[factors], b=var)
def terms_of(v):
    d = defs[v]
    if d[1] == 'FpExt(0)': return []
    m = ra.match(d[1])
    if m and defs[m.group(1)][0] == 'FpExt' and defs[m.group(3)][0] == 'FpExt':
        return terms_of(m.group(1)) + [([m.group(2), m.group(4)], m.group(3))]
    m = rb.match(d[1])
    if m and defs[m.group(1)][0] == 'FpExt':
        return terms_of(m.group(1)) + [([m.group(2)], m.group(3))]
    return [([], v)]
def leafvars(n):
    return [n['b']] + [x for x in n['f'] if x.startswith('x')]
def build(f, b, depth):
    n = {'f': f, 'b': b}
    if defs[b][0] == 'FpExt':
        sub = terms_of(b)
        if len(sub) > 1 and cost([b]) > target:
            return {'f': f, 'kids': [build(sf, sb, depth + 1) for sf, sb in sub]}
    return n
root = {'f': [], 'kids': [build(sf, sb, 0) for sf, sb in terms_of(ret)]}
# leaves with paths
leaves = []
def walk(n, path):
    if 'kids' in n:
        for i, k in enumerate(n['kids']): walk(k, path + [(id(n), i)])
    else:
        leaves.append((path, n))
walk(root, [])
lc = [cost(leafvars(n)) for p, n in leaves]
groups = []; cur = []; curv = []
for (p, n), c in zip(leaves, lc):
    if cur and cost(curv + leafvars(n)) > gtarget:
        groups.append(cur); cur = []; curv = []
    cur.append((p, n)); curv += leafvars(n)
if cur: groups.append(cur)
STATS = 'total ops %d leaves %d groups %d max leaf %d' % (cost([ret]), len(leaves), len(groups), max(lc))
def emit_node(n, members, lines, ctr):
    """value expression of node restricted to member leaves (by id)"""
    if 'kids' not in n:
        if id(n) not in members: return None
        b = n['b']; base = b if defs[b][0] == 'FpExt' else None
        if base is None:   # Fp base: pm * x
            return ' * '.join(n['f'] + [b])
        return ' * '.join([b] + n['f']) if n['f'] else b
    parts = [emit_node(k, members, lines, ctr) for k in n['kids']]
    parts = [p for p in parts if p is not None]
    if not parts: return None
    ctr[0] += 1; t = 's%d' % ctr[0]
    lines.append('  FpExt %s = %s;' % (t, parts[0] if parts[0].startswith('s') or ' ' not in parts[0] and defs.get(parts[0], ('',))[0] == 'FpExt' else parts[0]))
    for p in parts[1:]:
        lines.append('  %s = %s + %s;' % (t, t, p))
    if n['f']:
        return ' * '.join([t] + n['f'])
    return t
if STATS != EXPECT:
    sys.exit('eval_check_split: the split changed shape (%s, expected %s): re-verify the kernel '
             '(bit-exactness and timing) before updating EXPECT' % (STATS, EXPECT))
out = []
for gi, g in enumerate(groups):
    members = {id(n) for p, n in g}
    vs = set()
    for p, n in g: vs |= set(leafvars(n))
    # factors of ancestor nodes are needed too
    def anc(n):
        r = set()
        if 'kids' in n:
            for k in n['kids']:
                a = anc(k)
                if a is not None:
                    r |= a
                    r |= {x for x in n['f'] if x.startswith('x')}
            return r if r or any(('kids' not in k and id(k) in members) for k in n['kids']) else None
        return set() if id(n) in members else None
    a = anc(root) or set()
    vs |= a
    sl = slice_of(vs)
    out.append('__device__ __forceinline__ FpExt poly_fp_part%d(uint32_t idx, uint32_t size, const Fp* ctrl, const Fp* out, const Fp* data, const Fp* mix, const Fp* accum) {' % gi)
    out.append('  uint32_t mask = size - 1;')
    for v in order:
        if v not in sl: continue
        d = defs[v]
        out.append('  Fp %s(%s);' % (v, d[2]) if d[1] is None else '  %s %s = %s;' % (d[0], v, d[1]))
    lines = []; ctr = [0]
    e = emit_node(root, members, lines, ctr)
    out += lines
    out.append('  return %s;\n}' % e)
out.append('constexpr int kNumParts = %d;' % len(groups))
out.append('__device__ __forceinline__ FpExt poly_fp_part(int p, uint32_t idx, uint32_t size, const Fp* ctrl, const Fp* out, const Fp* data, const Fp* mix, const Fp* accum) {')
out.append('  switch (p) {')
for gi in range(len(groups)):
    out.append('  case %d: return poly_fp_part%d(idx, size, ctrl, out, data, mix, accum);' % (gi, gi))
out.append('  }\n  return FpExt(0);\n}')
new = ('''// GENERATED by kernels/cuda/eval_check_split.py from eval_check.cu -- do not edit; re-run it instead.
// Recursion eval_check split into kernels by the gated-sum tree of poly_fp (sum regrouping only,
// so the result is bit-identical). CUDA only; HIP keeps eval_check.cu.
#ifndef __HIPCC__
#include "cuda.h"
#include "supra/fp.h"
#include <cstdint>

constexpr size_t INV_RATE = 4;
// Compiled standalone (not relocatable), so this module needs its own copy of the mix powers:
// an `extern __constant__` would not alias ffi_supra.cu's poly_mix without -rdc.
__constant__ FpExt poly_mix[158];

namespace risc0::circuit::recursion::cuda {
''' + '\n' + '\n'.join(out) + '\n' + '''
template <int P>
__launch_bounds__(256, 1) __global__ void ec_part(Fp* check, const Fp* ctrl, const Fp* data, const Fp* accum,
                                                  const Fp* mix, const Fp* out, const Fp rou, uint32_t po2,
                                                  uint32_t domain) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle >= domain) return;
  FpExt tot = poly_fp_part(P, cycle, domain, ctrl, out, data, mix, accum);
  if (P > 0) {
    // The previous parts' sum, left in `check` by the kernel before on the same stream.
    tot = FpExt(check[cycle], check[domain + cycle], check[2 * domain + cycle], check[3 * domain + cycle]) + tot;
  }
  if (P == kNumParts - 1) {
    Fp x = pow(rou, cycle);
    Fp y = pow(Fp(3) * x, 1 << po2);
    tot = tot * inv(y - Fp(1));
  }
  check[domain * 0 + cycle] = tot[0];
  check[domain * 1 + cycle] = tot[1];
  check[domain * 2 + cycle] = tot[2];
  check[domain * 3 + cycle] = tot[3];
}

template <int P>
static void launch_parts(cudaStream_t stream, uint32_t grid, uint32_t block, Fp* check, const Fp* ctrl,
                         const Fp* data, const Fp* accum, const Fp* mix, const Fp* out, Fp rou, uint32_t po2,
                         uint32_t domain) {
  if constexpr (P < kNumParts) {
    ec_part<P><<<grid, block, 0, stream>>>(check, ctrl, data, accum, mix, out, rou, po2, domain);
    launch_parts<P + 1>(stream, grid, block, check, ctrl, data, accum, mix, out, rou, po2, domain);
  }
}

void eval_check_split(cudaStream_t stream, Fp* check, const Fp* ctrl, const Fp* data, const Fp* accum,
                      const Fp* mix, const Fp* out, Fp rou, uint32_t po2, uint32_t domain,
                      const void* poly_mix_host) {
  // poly_mix_host is pageable: the copy is staged before this call returns.
  CUDA_OK(cudaMemcpyToSymbolAsync(poly_mix, poly_mix_host, sizeof(poly_mix), 0, cudaMemcpyHostToDevice, stream));
  uint32_t block = 256, grid = (domain + block - 1) / block;
  launch_parts<0>(stream, grid, block, check, ctrl, data, accum, mix, out, rou, po2, domain);
}

} // namespace risc0::circuit::recursion::cuda
#endif // __HIPCC__
''')

try:
    cur = open(OUT, newline='').read()
except FileNotFoundError:
    cur = ''
if CHECK:
    if cur != new:
        sys.exit('eval_check_split: eval_check_split.cu is out of date; '
                 'run kernels/cuda/eval_check_split.py and commit the result')
    print('eval_check_split: eval_check_split.cu is up to date (%s)' % STATS)
else:
    open(OUT, 'w', newline='').write(new)
    print('eval_check_split: wrote eval_check_split.cu (%s)' % STATS)
