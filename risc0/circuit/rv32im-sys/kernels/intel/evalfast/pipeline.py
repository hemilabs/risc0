#!/usr/bin/env python3
# Intel mono eval_check source pipeline (RISC0_INTEL_EVAL_FAST). Turns the
# build.rs amalgamation (generated poly_fp + eval_check.cpp wrapper) into the
# optimized kernel source. Every step is exact: the algebraic rewrites are
# checked numerically (verify.py), and each pass asserts the code shapes it
# understands and fails the build on anything else.
#
#  1. eval_algebra.py  zero-link, absorb, chain differencing, hoist (verified)
#  2. rewrite.py       --factor --xshift, then --factorF          (verified)
#  3. graft.py         re-attach the kernel wrapper + absorb/hoist SLM edits
#  4. ../eval_nozero.py  no zero-fill of per-row arrays (write-before-read proof)
#  5. lazy2.py         lazy split-word poly_mix accumulation chains
#  6. hfix.py, ec_sched_pass.py (un-hoist, sink, scalar FpExt args)
#  7. pmfwd.py, ../eval_slab.py   per-row arrays -> slot-major global slab
#  8. swapuni.py, idx32.py, lincomb.py (3), lincombF.py (2)   final peepholes
#  9. xmxlink.py       poly_mix linking chains on XMX (DPAS); RISC0_INTEL_EVAL_XMX=0 skips
#
# usage: pipeline.py <in amalgamation> <out amalgamation> <work dir>
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
src, out, work = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(work, exist_ok=True)
env = dict(os.environ, PYTHONHASHSEED='0')  # the passes iterate dicts/sets


LOG = []


def run(script, *args, capture=False):
    cmd = [sys.executable, script, *args]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    sys.stderr.write(r.stderr)
    LOG.append(r.stderr)
    if r.returncode != 0:
        sys.exit('evalfast pipeline: %s failed (%d):\n%s' % (os.path.basename(script), r.returncode, r.stdout[-2000:]))
    return r.stdout


# The peephole passes skip shapes they do not recognise (always exact, but a
# skipped fusion silently costs time). Pin what they do on the current
# generated circuit so a generator change fails here instead of regressing.
EXPECT = [
    'idx32: 21 functions, 3474 loads',
    'lincomb: 248 trees fused',
    'lincombF: 346 Fp trees fused',
]


def w(name):
    return os.path.join(work, name)


run(os.path.join(HERE, 'eval_algebra.py'), src, w('a1.cpp'))
run(os.path.join(HERE, 'rewrite.py'), w('a1.cpp'), w('a2.cpp'), '--factor', '--xshift')
if 'ALL OK' not in run(os.path.join(HERE, 'verify.py'), w('a1.cpp'), w('a2.cpp'), '2'):
    sys.exit('evalfast pipeline: factor/xshift rewrite failed numeric verification')
run(os.path.join(HERE, 'rewrite.py'), w('a2.cpp'), w('a3.cpp'), '--factorF')
if 'ALL OK' not in run(os.path.join(HERE, 'verify.py'), w('a2.cpp'), w('a3.cpp'), '2'):
    sys.exit('evalfast pipeline: factorF rewrite failed numeric verification')
run(os.path.join(HERE, 'graft.py'), w('a3.cpp'), src, w('orig.cpp'), '--absorb')
run(os.path.join(PARENT, 'eval_nozero.py'), w('orig.cpp'), w('nz.cpp'))
run(os.path.join(HERE, 'lazy2.py'), w('nz.cpp'), w('lazy.cpp'))
run(os.path.join(HERE, 'hfix.py'), w('lazy.cpp'), w('lazyh.cpp'))
run(os.path.join(HERE, 'ec_sched_pass.py'), w('lazyh.cpp'), w('pre.cpp'))
run(os.path.join(HERE, 'pmfwd.py'), w('pre.cpp'), w('pre_pmf.cpp'))
run(os.path.join(PARENT, 'eval_slab.py'), w('pre_pmf.cpp'), w('slab.cpp'))
run(os.path.join(HERE, 'swapuni.py'), w('slab.cpp'), w('sw.cpp'))
run(os.path.join(HERE, 'idx32.py'), w('sw.cpp'), w('i32.cpp'))
run(os.path.join(HERE, 'lincomb.py'), w('i32.cpp'), w('lc.cpp'), '3')
XMX = os.environ.get('RISC0_INTEL_EVAL_XMX', '1') not in ('0', 'false', 'off', 'OFF')
if XMX:
    run(os.path.join(HERE, 'lincombF.py'), w('lc.cpp'), w('lcf.cpp'), '2')
    run(os.path.join(HERE, 'xmxlink.py'), w('lcf.cpp'), out)
    EXPECT.append('xmxlink: 166 chains (>= 4 terms) of 282, 773 blocks')
else:
    run(os.path.join(HERE, 'lincombF.py'), w('lc.cpp'), out, '2')
log = ''.join(LOG)
for e in EXPECT:
    if e not in log:
        sys.exit('evalfast pipeline: expected "%s" in the pass output; the generated '
                 'circuit or a pass changed shape (re-measure and update EXPECT)' % e)
print('evalfast pipeline: wrote %s' % out, file=sys.stderr)
