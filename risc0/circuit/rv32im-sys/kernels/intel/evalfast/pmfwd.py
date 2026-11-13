#!/usr/bin/env python3
# pmfwd.py in.cpp out.cpp   (pre-slab; run right before eval_slab.py)
# Store-to-load forwarding of row-invariant SLM values: when a per-row array slot (x33/x34,
# through any alias in the rv32im_v2_* chain) is written with a value defined as
# `FpExt xV = poly_mix[K];` (an SLM, work-group-uniform value, e.g. from ec-algebra --hoist /
# --absorb tables), drop the store and read poly_mix[K] at every read of that slot instead.
# The slab then neither stores nor reloads it (and IGC sees a uniform value).
import os, re, sys
# eval_slab.py lives in the parent directory (kernels/intel).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval_slab as ES

def main():
    src = open(sys.argv[1]).read().split('\n')
    funcs = ES.parse(src)
    pf = funcs['poly_fp']
    arrays = {'names': {}, 'decl_lines': set()}
    for i in pf['lines']:
        m = ES.DECL.match(src[i])
        if m:
            ind, name, words, ty = m.groups()
            width = 4 if ty == 'FpExt' else 1
            arrays['names'][name] = (ty, width, int(words) // width)
            arrays['decl_lines'].add(i)
    tr, calls = ES.trace(src, funcs, arrays)
    line_fn = {}
    for f, d in funcs.items():
        for i in d['lines']: line_fn[i] = f
    pmdef = {}
    for f, d in funcs.items():
        for i in d['lines']:
            m = re.match(r'^\s*FpExt (x\d+) = (poly_mix\[\d+\]);\s*$', src[i])
            if m: pmdef[(f, m.group(1))] = m.group(2)
    fwd = {}; dels = set()
    for a in tr:
        if a['rw'] != 'W': continue
        m = re.match(r'^\s*%s\[%d\] = (x\d+);\s*$' % (re.escape(a['name']), a['idx']), src[a['line']])
        if not m: continue
        key = (line_fn[a['line']], m.group(1))
        if key in pmdef:
            fwd[(a['arr'], a['idx'])] = pmdef[key]; dels.add(a['line'])
    out = list(src); nrep = 0
    for a in sorted(tr, key=lambda x: (x['line'], -x['s'])):
        k = (a['arr'], a['idx'])
        if k in fwd and a['rw'] == 'R':
            l = out[a['line']]
            out[a['line']] = l[:a['s']] + fwd[k] + l[a['e']:]; nrep += 1
    for i in dels:
        out[i] = '  // pmfwd: store of a poly_mix value forwarded to its reads'
    open(sys.argv[2], 'w').write('\n'.join(out))
    print('pmfwd: %d slots forwarded, %d reads rewritten' % (len(fwd), nrep), file=sys.stderr)

if __name__ == '__main__':
    main()
