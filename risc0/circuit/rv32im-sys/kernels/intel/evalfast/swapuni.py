#!/usr/bin/env python3
# swapuni.py in.cpp out.cpp  (run on the FINAL post-slab amalgamation)
# FpExt*FpExt folds NBETA into the RIGHT operand (3 Montgomery muls). When the left operand
# is work-group-uniform (poly_mix / hoisted SLM values, constants, mix/global scalars) and the
# right one is per-row, swap them so the fold is uniform (scalar) work. Multiplication is
# commutative and the result is the canonical field element either way (bit-identical).
import re, sys
src = open(sys.argv[1]).read()
k = '// Intel SYCL eval_check kernel'
head, tail = src[:src.find(k)], src[src.find(k):]
L = head.split('\n')
FD = re.compile(r'^(__attribute__\(\([a-z_]+\)\) )?(inline )?FpExt (rv32im_v2_\d+|poly_fp)\(.*\{$')
DEF = re.compile(r'^\s*(?:auto|FpExt|Fp|constexpr Fp|constexpr FpExt|const FpExt|const Fp) (x\d+)\s*(=|\()\s*(.*?)\)?;\s*$')
out = []; U = set(); infn = False; nsw = 0
for l in L:
    if FD.match(l): infn = True; U = set(); out.append(l); continue
    if infn and l.startswith('}'): infn = False; out.append(l); continue
    if infn:
        m = DEF.match(l)
        if m:
            v, kind, rhs = m.group(1), m.group(2), m.group(3)
            uni = False
            if l.strip().startswith('constexpr'): uni = True
            elif re.match(r'^poly_mix\[\d+\](\.elems\[\d\])?$', rhs): uni = True
            elif re.match(r'^(/\*\w+=\*/)?arg\d+\[\d+\]$', rhs) or re.match(r'^(/\*\w+=\*/)?args\[\d\]\[\d+\]$', rhs): uni = True
            else:
                toks = re.findall(r'\bx\d+\b', rhs)
                if toks and all(t in U for t in toks) and re.match(r'^[x\d\s+\-*()]+$', rhs): uni = True
            if uni: U.add(v)
            mm = re.match(r'^(\s*auto (x\d+) = )(x\d+) \* (x\d+);\s*$', l)
            if mm and mm.group(3) in U and mm.group(4) not in U:
                l = '%s%s * %s;' % (mm.group(1), mm.group(4), mm.group(3)); nsw += 1
    out.append(l)
open(sys.argv[2], 'w').write('\n'.join(out) + tail)
print('swapuni: swapped', nsw, file=sys.stderr)
