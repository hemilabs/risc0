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

"""Regenerate kernels/cuda/eval_check_ef.cu from the generated circuit.

usage: regen.py [--check]

The CUDA eval_check kernel is the Intel evalfast pipeline output converted by
to_cuda.py. The chain, the same one build.rs runs for the Intel mono kernel
(with RISC0_INTEL_EVAL_XMX=0, as XMX is Intel-only):

  1. kernels/intel/poly_fp_cse.py kernels/cxx --top 16 --min-share 10
  2. the amalgamation build.rs writes as intel_eval_check_amalg.cpp
  3. kernels/intel/evalfast/pipeline.py
  4. kernels/cuda/evalfast/to_cuda.py

followed by the CUDA post-passes in this directory (PASSES below):

  5. remat.py: slab values that are small Fp trees over input loads are
     recomputed at their reads; the slab is re-coloured
  6. lazyfp.py: lazy [0, 2P) Fp add/sub/mul where every consumer allows it
  7. close.py: closures B + u * PM_k accumulated lazily in 64 bits with an
     NBETA-folded constant table (+ inline PTX mad.wide, CUDA only)
  8. linkext.py: rk_link with an FpExt term accumulated the same way
  9. accfirst_cuda.py: the LogUp/accum block sliced out of the main chain
 10. mksplit2.py: main / LogUp block-type phase split (ef_eval_split, even
     blocks main and odd blocks LogUp, plus ef_combine)
 11. remat_lu.py: LogUp slab FpExt values recomputed at their reads
 12. mksm2.py: per-phase 64 hottest slab words in shared memory
 13. mksub.py: every chain function split in two at its narrowest point
     between 45% and 55% of its body (the tail is a __noinline__ template)

Each pass prints what it did; EXPECT pins those lines, so a change in the
circuit or a pass that changes the shape of the kernel fails here first: re-run
the bit-exactness battery on the new kernel before updating the pins.

The result is committed, so the CUDA build needs no Python. Re-run this after
changing the generated circuit (kernels/cxx/rust_poly_fp_*.cpp) or any of the
passes; --check (run in CI) fails if the committed file is out of date.
"""

import difflib
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SYS = os.path.normpath(os.path.join(HERE, '..', '..', '..'))  # rv32im-sys
OUT = os.path.join(SYS, 'kernels', 'cuda', 'eval_check_ef.cu')
NS = 'namespace risc0::circuit::rv32im_v2 {'

# mksub.py cut points: function, body lines, cut line, live values passed to the tail (+ input
# loads re-issued there instead).
MKSUB = [
    'rv32im_v2_0: 1329 lines, cut at 640 (0.48), live 28 (+90 reloaded)',
    'rv32im_v2_1: 1238 lines, cut at 592 (0.48), live 32 (+69 reloaded)',
    'rv32im_v2_2: 1131 lines, cut at 528 (0.47), live 35 (+57 reloaded)',
    'rv32im_v2_3: 999 lines, cut at 452 (0.45), live 8 (+23 reloaded)',
    'rv32im_v2_4: 1322 lines, cut at 594 (0.45), live 76 (+2 reloaded)',
    'rv32im_v2_5: 1283 lines, cut at 581 (0.45), live 15 (+0 reloaded)',
    'rv32im_v2_6: 1132 lines, cut at 509 (0.45), live 85 (+40 reloaded)',
    'rv32im_v2_7: 1335 lines, cut at 687 (0.51), live 29 (+10 reloaded)',
    'rv32im_v2_8: 1023 lines, cut at 481 (0.47), live 35 (+1 reloaded)',
    'rv32im_v2_9: 1077 lines, cut at 486 (0.45), live 35 (+7 reloaded)',
    'rv32im_v2_10: 1378 lines, cut at 620 (0.45), live 12 (+36 reloaded)',
    'rv32im_v2_11: 1088 lines, cut at 489 (0.45), live 102 (+43 reloaded)',
    'rv32im_v2_12: 1222 lines, cut at 642 (0.53), live 65 (+25 reloaded)',
    'rv32im_v2_13: 1250 lines, cut at 591 (0.47), live 13 (+51 reloaded)',
    'rv32im_v2_14: 1240 lines, cut at 672 (0.54), live 43 (+83 reloaded)',
    'rv32im_v2_15: 1280 lines, cut at 610 (0.48), live 67 (+57 reloaded)',
    'rv32im_v2_16: 998 lines, cut at 541 (0.54), live 15 (+24 reloaded)',
    'rv32im_v2_17: 1186 lines, cut at 533 (0.45), live 46 (+18 reloaded)',
    'rv32im_v2_18: 1038 lines, cut at 504 (0.49), live 19 (+13 reloaded)',
    'rv32im_v2_19: 1015 lines, cut at 465 (0.46), live 112 (+25 reloaded)',
]

# (pass, extra arguments, lines its output must contain)
PASSES = [
    ('remat.py', ['--maxops', '4', '--maxloads', '4'],
     ['remat: 675 values, 1421 loads rematerialized, 675 stores dropped, slab words 324']),
    ('lazyfp.py', [], ["lazyfp: {'cand': 6032, 'add': 94, 'sub': 503, 'mul': 265}"]),
    ('close.py', ['--chain'], ['close: 231 closures, 157 chained']),
    ('linkext.py', [], ['linkext: 5118 call sites']),
    ('accfirst_cuda.py', [], ['accfirst_cuda: 209 tail lines, 41 pulled-in defs']),
    ('mksplit2.py', [], []),
    ('remat_lu.py', ['--maxg', '4', '--maxops', '8'],
     ["loads 135 {'keep': 41, 'remat': 94} stores 115 dropped 89"]),
    ('mksm2.py', ['64', '64'], ['main words 85 acc 751 hot 683 | lu words 104 acc 268 hot 188']),
    ('mksub.py', ['--fns', ','.join('rv32im_v2_%d' % i for i in range(20)), '--lo', '0.45', '--hi', '0.55'],
     MKSUB),
]

POST_HEADER = '''//
// CUDA post-passes (kernels/cuda/evalfast, chained by regen.py): slab values
// rematerialised from the inputs (remat.py), lazy [0, 2P) Fp results
// (lazyfp.py), lazy 64-bit closure and FpExt link accumulation with inline PTX
// mad.wide (close.py, linkext.py), the LogUp block sliced out of the main chain
// (accfirst_cuda.py), a main / LogUp block-type phase split (ef_eval_split +
// ef_combine, mksplit2.py), LogUp slab values rematerialised at their reads
// (remat_lu.py), the per-phase 64 hottest slab words in shared memory
// (mksm2.py) and every chain function split in two at its narrowest point
// (mksub.py). The rematerialisation relies on data, accum and mix being
// read-only for the whole kernel.
'''
HEADER_END = '// kernel happens to tolerate them.\n'


def run(env, *args):
    r = subprocess.run([sys.executable, *args], env=env, cwd=SYS, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit('regen: %s failed (%d):\n%s%s' % (os.path.basename(args[0]), r.returncode,
                                                   r.stdout[-2000:], r.stderr[-2000:]))
    return r.stdout + r.stderr


def rust_lines(s):
    # str::lines(): split on '\n', strip one trailing '\r', no final empty line.
    lines = s.split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    return [l[:-1] if l.endswith('\r') else l for l in lines]


def amalgamate(poly_fp_dir):
    # Must match build.rs (build_intel_kernels, "intel_eval_check_amalg.cpp")
    # without RISC0_LSC_HINTS.
    out = ['// Auto-generated amalgamation for SYCL device code\n',
           '#include "fp.h"\n', '#include "fpext.h"\n', '#include <cstdint>\n',
           NS + '\n', 'constexpr size_t kInvRate = 4;\n']
    for i in range(4):
        with open(os.path.join(poly_fp_dir, 'rust_poly_fp_%d.cpp' % i), newline='') as f:
            src = f.read()
        start = src.index(NS) + len(NS)
        for line in rust_lines(src[start:src.rindex('}')]):
            if line.startswith('FpExt rv32im_v2_') and '{' in line:
                out.append('__attribute__((noinline)) ')
            out.append(line + '\n')
    out.append('} // namespace risc0::circuit::rv32im_v2\n')
    with open(os.path.join(SYS, 'kernels', 'intel', 'eval_check.cpp'), newline='') as f:
        out.append(f.read())
    return ''.join(out)


def main():
    check = '--check' in sys.argv[1:]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('RISC0_', 'SINK_LEAD', 'PYTHON'))}
    env.update(PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1', RISC0_INTEL_EVAL_XMX='0')
    with tempfile.TemporaryDirectory() as tmp:
        cse = os.path.join(tmp, 'cse')
        os.makedirs(cse)
        run(env, 'kernels/intel/poly_fp_cse.py', 'kernels/cxx', cse,
            '--top', '16', '--min-share', '10')
        amalg = os.path.join(tmp, 'amalg.cpp')
        with open(amalg, 'w', newline='') as f:
            f.write(amalgamate(cse))
        fast = os.path.join(tmp, 'fast.cpp')
        run(env, 'kernels/intel/evalfast/pipeline.py', amalg, fast, os.path.join(tmp, 'work'))
        gen = os.path.join(tmp, 'base.cu')
        run(env, os.path.join(HERE, 'to_cuda.py'), fast, gen)
        for n, (tool, extra, expect) in enumerate(PASSES):
            nxt = os.path.join(tmp, 'p%d.cu' % n)
            log = run(env, os.path.join(HERE, tool), gen, nxt, *extra)
            for line in expect:
                if line not in log:
                    sys.exit('regen: EXPECT failed: %s did not print\n  %s\nit printed:\n%s'
                             % (tool, line, log))
            gen = nxt
        with open(gen, newline='') as f:
            new = f.read()
    assert new.count(HEADER_END) == 1
    new = new.replace(HEADER_END, HEADER_END + POST_HEADER)
    bad = [i + 1 for i, l in enumerate(new.split('\n')) if l != l.rstrip()]
    if bad:
        sys.exit('regen: trailing whitespace on %d generated lines (first: %d)' % (len(bad), bad[0]))
    try:
        with open(OUT, newline='') as f:
            old = f.read()
    except FileNotFoundError:
        old = ''
    if check:
        if old != new:
            diff = difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                        'committed/eval_check_ef.cu', 'regenerated/eval_check_ef.cu')
            sys.stdout.writelines(list(diff)[:200])
            sys.exit('regen: kernels/cuda/eval_check_ef.cu is out of date; '
                     'run kernels/cuda/evalfast/regen.py and commit the result')
        print('regen: kernels/cuda/eval_check_ef.cu is up to date')
    else:
        with open(OUT, 'w', newline='') as f:
            f.write(new)
        print('regen: wrote %s' % OUT)


if __name__ == '__main__':
    main()
