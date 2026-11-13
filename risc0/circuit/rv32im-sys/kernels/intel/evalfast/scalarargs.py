# Replace by-value FpExt parameters of the rv32im_v2_* functions (lowered to
# byval private-memory pointers by clang's SPIR ABI) with 4 scalar Fp params.
# usage: scalarargs.py in.cpp out.cpp
import re, sys
src = open(sys.argv[1]).read()
cut = src.find('// Intel SYCL eval_check kernel')
head, tail = src[:cut], src[cut:]
lines = head.split('\n')
SIG = re.compile(r'^(__attribute__\(\(noinline\)\) )?FpExt (rv32im_v2_\d+(?:_p\d+)?)\((.*)\)( \{|;)$')
# pass 1: collect which param positions are FpExt by value per function
fpos = {}
for l in lines:
    m = SIG.match(l)
    if not m: continue
    ps = [p.strip() for p in m.group(3).split(',')]
    fpos[m.group(2)] = [i for i, p in enumerate(ps) if re.match(r'^FpExt \w+$', p)]
def newsig(m):
    ps = [p.strip() for p in m.group(3).split(',')]
    out = []
    for i, p in enumerate(ps):
        mm = re.match(r'^FpExt (\w+)$', p)
        if mm:
            n = mm.group(1); out += ['Fp %s_e%d' % (n, j) for j in range(4)]
        else:
            out.append(p)
    return out
res = []
CALL = re.compile(r'^(\s*auto \w+ = )(rv32im_v2_\d+(?:_p\d+)?)\((.*)\);$')
CALL2 = re.compile(r'^(\s*return )(rv32im_v2_\d+(?:_p\d+)?)\((.*)\);$')
for l in lines:
    m = SIG.match(l)
    if m:
        ps = newsig(m)
        res.append('%sFpExt %s(%s)%s' % (m.group(1) or '', m.group(2), ', '.join(ps), m.group(4)))
        if m.group(4) == ' {':
            for p in [p.strip() for p in m.group(3).split(',')]:
                mm = re.match(r'^FpExt (\w+)$', p)
                if mm:
                    n = mm.group(1)
                    res.append('  FpExt %s(%s);' % (n, ', '.join('%s_e%d' % (n, j) for j in range(4))))
        continue
    mc = CALL.match(l) or CALL2.match(l)
    if mc and mc.group(2) in fpos:
        args = [a.strip() for a in mc.group(3).split(',')]
        out = []
        for i, a in enumerate(args):
            if i in fpos[mc.group(2)]:
                out += ['%s.elems[%d]' % (a, j) for j in range(4)]
            else:
                out.append(a)
        res.append('%s%s(%s);' % (mc.group(1), mc.group(2), ', '.join(out)))
        continue
    res.append(l)
open(sys.argv[2], 'w').write('\n'.join(res) + tail)
print('functions:', len(fpos), 'fpext params:', sum(len(v) for v in fpos.values()) // 2)
