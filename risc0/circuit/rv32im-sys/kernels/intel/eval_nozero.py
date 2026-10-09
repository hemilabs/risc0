#!/usr/bin/env python3
# Intel mono eval_check post-pass (RISC0_INTEL_EVAL_FAST): stop zero-filling
# the per-row private arrays of poly_fp (e.g. `Fp x33[1007];`,
# `FpExt x34[144];`). Value-initialising them costs a store per slot per row
# in per-lane private memory (measured -123 ms per po2=20 segment when
# removed). That is only sound if every slot is written before it is read, so
# this script first PROVES that statically, following the arrays through the
# rv32im_v2_N call chain in program order, and fails the build otherwise
# (non-literal index, aliasing, or read-before-write).
#
# usage: eval_nozero.py <in.cpp> <out.cpp>
import re
import sys

src = open(sys.argv[1]).read()
cut = src.find('// Intel SYCL eval_check kernel')
assert cut > 0, 'kernel wrapper marker not found'
body, wrapper = src[:cut], src[cut:]

funcs = {}
for m in re.finditer(
        r'^(?:__attribute__\(\(noinline\)\) )?FpExt (rv32im_v2_\d+|poly_fp)\(([^)]*)\) \{\n(.*?)^\}\n',
        body, re.S | re.M):
    params = [p.strip().split()[-1] for p in m.group(2).split(',')]
    funcs[m.group(1)] = (params, m.group(3))
assert 'poly_fp' in funcs, 'poly_fp not found'

decl_re = re.compile(r'^(\s*)(Fp|FpExt) (x\d+)\[(\d+)\];$', re.M)
arrays = {m.group(3): (m.group(2), int(m.group(4))) for m in decl_re.finditer(funcs['poly_fp'][1])}
if not arrays:
    sys.exit('eval_nozero: no private arrays found in poly_fp')

written, bad, reads = set(), [], 0
callre = re.compile(r'(rv32im_v2_\d+)\(([^;]*)\)\s*$')
idxre = re.compile(r'\b(\w+)\[(\d+)\]')


def run(fn, amap):
    global reads
    params, fbody = funcs[fn]
    for st in fbody.split(';\n'):
        st = re.sub(r'//[^\n]*', '', st).strip()
        if not st or decl_re.match(st + ';'):
            continue
        lhs, rhs = None, st
        mw = re.match(r'(\w+)\[(\d+)\]\s*=\s*(.*)$', st, re.S)
        if mw and mw.group(1) in amap:
            lhs, rhs = (amap[mw.group(1)], int(mw.group(2))), mw.group(3)
        for mr in idxre.finditer(rhs):
            if mr.group(1) in amap:
                reads += 1
                if (amap[mr.group(1)], int(mr.group(2))) not in written:
                    bad.append(('READ-BEFORE-WRITE', fn, st[:120]))
        for nm in amap:
            for mm in re.finditer(r'\b' + nm + r'\[([^\]]*)\]', st):
                if not mm.group(1).isdigit():
                    bad.append(('NONLITERAL', fn, st[:120]))
            if re.search(r'\b' + nm + r'\b(?!\[)', st) and not callre.search(st):
                bad.append(('ALIAS', fn, st[:120]))
        mc = callre.search(rhs)
        if mc:
            callee = mc.group(1)
            args = [a.strip() for a in re.sub(r'/\*\w+=\*/', '', mc.group(2)).split(',')]
            run(callee, {p: amap[a] for a, p in zip(args, funcs[callee][0]) if a in amap})
        if lhs:
            written.add(lhs)


run('poly_fp', {n: n for n in arrays})
if bad:
    for b in bad[:20]:
        print('eval_nozero:', b, file=sys.stderr)
    sys.exit('eval_nozero: arrays may be read before written; refusing to drop the zero-fill')


def raw_decl(m):
    ind, ty, name, n = m.groups()
    words = int(n) * (4 if ty == 'FpExt' else 1)
    return (f'{ind}alignas(16) uint32_t {name}_raw[{words}]; '
            f'{ty}* {name} = reinterpret_cast<{ty}*>({name}_raw);')


pbody = funcs['poly_fp'][1]
new_pbody = decl_re.sub(raw_decl, pbody)
assert body.count(pbody) == 1
body = body.replace(pbody, new_pbody)
open(sys.argv[2], 'w').write(body + wrapper)
print(f'eval_nozero: {sorted(arrays)} uninitialised; {reads} reads checked, '
      f'{len(written)} slots, all written before read', file=sys.stderr)
