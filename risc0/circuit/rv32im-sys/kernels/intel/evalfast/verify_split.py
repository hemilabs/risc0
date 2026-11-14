#!/usr/bin/env python3
# verify_split.py full.cpp part_0.cpp ... part_k.cpp [seeds]: checks poly_fp(full) == the sum of
# the parts' poly_fp numerically on random inputs (the interpreter of verify.py), as degsplit.py's
# split must hold for any inputs.
import sys, re
from dag import build_all
from interp import Interp, add


def qt(path):
    src = open(path).read()
    m = re.search(r'constexpr uint16_t rk_qa\[\] = \{([^}]*)\};\s*constexpr uint16_t rk_qb\[\] = \{([^}]*)\};', src)
    return list(zip([int(v) for v in m.group(1).split(',')], [int(v) for v in m.group(2).split(',')])) if m else []


args = sys.argv[1:]
seeds = int(args.pop()) if args[-1].isdigit() else 3
paths = args
assert len(paths) >= 3, 'usage: verify_split.py full.cpp part_0.cpp part_1.cpp ... [seeds]'
fs = [build_all(p)[4] for p in paths]
ok = True
for seed in range(seeds):
    r = []
    for p, F in zip(paths, fs):
        I = Interp(F, seed=seed, qt=qt(p))
        I.run_hoist(open(p).read())
        r.append(I.go())
    s = r[1]
    for x in r[2:]:
        s = add(s, x)
    good = s == r[0]
    print(seed, r[0], s, 'OK' if good else 'MISMATCH')
    ok &= good
print('ALL OK' if ok else 'FAIL')
