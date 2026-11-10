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

"""Shared pieces of the recursion CUDA -> Metal translators (Metal port items C2 and C3).

`cuda2metal_steps.py` (C2) and `cuda2metal_eval.py` (C3) import this module. It holds:

  - paths: the recursion-sys kernel directories and the shared MSL headers (`METAL_INCS`,
    risc0/build_kernel/kernels/metal). `kernels/cuda` is never put on a Metal include path
    (its extern.h would shadow kernels/metal/extern.h; plan 2.0).
  - `Rule`: one reversible, regex-based text rewrite with a pinned count. The translators are
    sequences of rules; `apply_rules` runs them forward and `unapply_rules` backward, which is
    what the `--check` round trip uses (translate, strip the generated blocks, undo every rule,
    and require the original CUDA source byte for byte). Before a rule runs, the input must not
    already contain its output spelling (`collision` check), so the inverse is exact.
  - generated blocks: everything the translator adds that has no CUDA counterpart (prelude,
    kernel wrappers) sits between `GEN_BEGIN` / `GEN_END` marker lines and is dropped by the
    round trip.
  - pin checking, residual scans, `write_or_check`, and the `xcrun metal` compile helper.

Every failure raises `TranslateError`; there are no silent fallbacks.
"""

import os
import re
import shutil
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RECURSION_SYS = os.path.dirname(os.path.dirname(HERE))
RISC0_ROOT = os.path.dirname(os.path.dirname(RECURSION_SYS))
BK_METAL = os.path.join(RISC0_ROOT, 'build_kernel', 'kernels', 'metal')
REC_METAL = os.path.join(RECURSION_SYS, 'kernels', 'metal')
CUDA_DIR = HERE

METAL_FLAGS = ['-std=metal3.2', '-O2']
INCLUDE_DIRS = [BK_METAL, REC_METAL]

GEN_BEGIN = '// ---- cuda2metal: generated (begin) ----'
GEN_END = '// ---- cuda2metal: generated (end) ----'

LICENSE = """\
// Copyright 2026 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
"""


class TranslateError(Exception):
    pass


class Rule:
    """A reversible rewrite: `fwd` -> `fwd_repl` (forward), `rev` -> `rev_repl` (inverse).

    `collision` is a regex that must not match the input before the rule runs (defaults to
    `rev`): it guarantees that every `rev` match in the output came from this rule, so the
    inverse restores exactly the original text. Patterns are compiled with re.MULTILINE.
    """

    def __init__(self, key, fwd, fwd_repl, rev, rev_repl, collision=None):
        self.key = key
        self.fwd = re.compile(fwd, re.M)
        self.fwd_repl = fwd_repl
        self.rev = re.compile(rev, re.M)
        self.rev_repl = rev_repl
        self.collision = re.compile(collision, re.M) if collision else self.rev

    def forward(self, text):
        hit = self.collision.search(text)
        if hit:
            raise TranslateError('rule %s: input already contains its output spelling at %s: %r'
                                 % (self.key, where(text, hit.start()), hit.group(0)))
        return self.fwd.subn(self.fwd_repl, text)

    def backward(self, text):
        return self.rev.subn(self.rev_repl, text)


class Counter:
    """Rule hit counts (`counts[key]`) plus free-form tallies (`counts.add`)."""

    def __init__(self):
        self.d = {}

    def add(self, key, n=1):
        self.d[key] = self.d.get(key, 0) + n

    def __getitem__(self, key):
        return self.d.get(key, 0)

    def items(self):
        return sorted(self.d.items())


def where(text, pos):
    line = text.count('\n', 0, pos) + 1
    return 'line %d' % line


def apply_rules(text, rules, counts):
    for r in rules:
        text, n = r.forward(text)
        counts.add(r.key, n)
    return text


def unapply_rules(text, rules):
    for r in reversed(rules):
        text, _ = r.backward(text)
    return text


def gen_block(body):
    """Wrap generated text (no CUDA counterpart) in marker lines."""
    if GEN_BEGIN in body or GEN_END in body:
        raise TranslateError('generated block contains a marker line')
    return '%s\n%s%s\n' % (GEN_BEGIN, body if body.endswith('\n') else body + '\n', GEN_END)


def strip_gen_blocks(text):
    """Drop every generated block (marker lines included)."""
    out = []
    inside = False
    for line in text.splitlines(keepends=True):
        s = line.rstrip('\n')
        if s == GEN_BEGIN:
            if inside:
                raise TranslateError('nested generated block')
            inside = True
            continue
        if s == GEN_END:
            if not inside:
                raise TranslateError('unbalanced generated block end')
            inside = False
            continue
        if not inside:
            out.append(line)
    if inside:
        raise TranslateError('unterminated generated block')
    return ''.join(out)


def check_no_markers(text, name):
    if GEN_BEGIN in text or GEN_END in text:
        raise TranslateError('%s already contains a cuda2metal marker line' % name)


def check_pins(name, counts, pinned, strict=True):
    """Compare `counts` against `pinned` (a dict key -> int). Returns a list of mismatch lines;
    raises when `strict`."""
    bad = []
    keys = sorted(set(pinned) | set(k for k, _ in counts.items()))
    for k in keys:
        want = pinned.get(k, 0)  # an absent pin means 0
        got = counts[k]
        if want != got:
            bad.append('%s: %s = %d, pinned %d' % (name, k, got, want))
    if bad and strict:
        raise TranslateError('rewrite counts differ from the pins (the generated input changed '
                             'shape; re-run with --no-pin, review, update the pins):\n  '
                             + '\n  '.join(bad))
    return bad


def residual_scan(name, text, patterns):
    """Refuse output that still contains a CUDA-only construct. `patterns` is a list of
    (regex, description)."""
    for pat, desc in patterns:
        m = re.search(pat, text, re.M)
        if m:
            raise TranslateError('%s: residual %s at %s: %r'
                                 % (name, desc, where(text, m.start()), m.group(0)))


def read(path):
    with open(path, 'r', newline='') as f:
        return f.read()


def write_or_check(path, text, check):
    """Write `text` to `path` (only if it changed, so mtimes stay stable), or with `check`
    compare it with the existing file. Returns 'written', 'unchanged', 'match' or 'absent'."""
    if check:
        if not os.path.exists(path):
            return 'absent'
        if read(path) != text:
            raise TranslateError('%s is out of date (re-run without --check)' % path)
        return 'match'
    if os.path.exists(path) and read(path) == text:
        return 'unchanged'
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='') as f:
        f.write(text)
    os.replace(tmp, path)
    return 'written'


def metal_tool(tool):
    """`xcrun <tool>` argv prefix; requires the Metal toolchain (source env.sh)."""
    if shutil.which('xcrun') is None:
        raise TranslateError('xcrun not found (source /Users/user/zkminer/env.sh)')
    return ['xcrun', '-sdk', 'macosx', tool]


def compile_metal(src, air, extra_flags=(), defines=()):
    """Compile one .metal to .air. Returns (seconds, stderr text); raises on failure."""
    cmd = metal_tool('metal') + METAL_FLAGS + list(extra_flags)
    for d in defines:
        cmd.append('-D' + d)
    for inc in INCLUDE_DIRS:
        cmd += ['-I', inc]
    cmd += ['-c', src, '-o', air]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if p.returncode != 0:
        errs = [l for l in p.stderr.splitlines() if ' error: ' in l]
        raise TranslateError('%s: xcrun metal failed (%d error lines, %.1f s):\n%s'
                             % (src, len(errs), dt, '\n'.join(p.stderr.splitlines()[:60])))
    return dt, p.stderr


def link_metallib(airs, out):
    cmd = metal_tool('metallib') + list(airs) + ['-o', out]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise TranslateError('xcrun metallib failed:\n' + p.stderr)
    return time.time() - t0
