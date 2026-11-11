# hfix.py in.cpp out.cpp : inside rv32im_v2_* bodies, give every expression use of a hoisted
# _h_data_C_B parameter its own `auto xN = _h_data_C_B;` statement (the form ec_sched_pass's unhoist knows).
import re, sys
src = open(sys.argv[1]).read()
k = '// Intel SYCL eval_check kernel'
head, tail = src[:src.find(k)], src[src.find(k):]
L = head.split('\n'); out = []; infn = False; n = 3000000; cnt = 0
FD = re.compile(r'^(__attribute__\(\(noinline\)\) )?FpExt rv32im_v2_\d+\(.*\{$')
for l in L:
    if FD.match(l): infn = True; out.append(l); continue
    if infn and l.startswith('}'): infn = False; out.append(l); continue
    if infn and '_h_data' in l and not l.strip().startswith('//') and not re.match(r'^\s*auto x\d+ = _h_data_\d+_\d+;$', l) and ' = rv32im_v2_' not in l and 'return rv32im_v2_' not in l:
        ind = re.match(r'^(\s*)', l).group(1)
        def rep(m):
            global n, cnt
            n += 1; cnt += 1
            out.append('%sauto x%d = %s;' % (ind, n, m.group(0)))
            return 'x%d' % n
        l = re.sub(r'\b_h_data_\d+_\d+\b', rep, l)
    out.append(l)
open(sys.argv[2], 'w').write('\n'.join(out) + tail)
print('hfix: rewrote', cnt, 'uses', file=sys.stderr)
