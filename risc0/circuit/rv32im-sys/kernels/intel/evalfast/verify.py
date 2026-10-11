import sys
from dag import build_all
from interp import Interp
import re
def qt(path):
    src=open(path).read()
    m=re.search(r'constexpr uint16_t rk_qa\[\] = \{([^}]*)\};\s*constexpr uint16_t rk_qb\[\] = \{([^}]*)\};',src)
    return list(zip([int(v) for v in m.group(1).split(',')],[int(v) for v in m.group(2).split(',')])) if m else []
A=build_all(sys.argv[1])[4]; B=build_all(sys.argv[2])[4]; QA=qt(sys.argv[1]); QB=qt(sys.argv[2])
ok=True
for seed in range(int(sys.argv[3]) if len(sys.argv)>3 else 5):
    IA=Interp(A,seed=seed,qt=QA); IA.run_hoist(open(sys.argv[1]).read()); ra=IA.go()
    IB=Interp(B,seed=seed,qt=QB); IB.run_hoist(open(sys.argv[2]).read()); rb=IB.go()
    print(seed, ra, rb, 'OK' if ra==rb else 'MISMATCH'); ok&=(ra==rb)
print('ALL OK' if ok else 'FAIL')
