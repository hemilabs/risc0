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


def run(env, *args):
    r = subprocess.run([sys.executable, *args], env=env, cwd=SYS, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit('regen: %s failed (%d):\n%s%s' % (os.path.basename(args[0]), r.returncode,
                                                   r.stdout[-2000:], r.stderr[-2000:]))


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
        gen = os.path.join(tmp, 'eval_check_ef.cu')
        run(env, os.path.join(HERE, 'to_cuda.py'), fast, gen)
        with open(gen, newline='') as f:
            new = f.read()
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
