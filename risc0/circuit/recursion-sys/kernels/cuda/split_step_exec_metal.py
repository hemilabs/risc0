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

"""Split zirgen's monolithic recursion `step_exec.cu` into 19 `__noinline__` arm functions
(Metal port item P31-07; the input form C2's `cuda2metal_steps.py` translates).

usage:
  split_step_exec_metal.py IN -o OUT [-q]   # write the split (only when its text changes)
  split_step_exec_metal.py IN --stats       # detect + split in memory, print the structure
  split_step_exec_metal.py --selftest [-q]  # the byte-exact regression (see below)

zirgen emits `step_exec` as one function: a constants block (`Fp x0(..)` .. `Fp xK(0)`), then one
`if (sel != 0) { .. }` arm per OneHot selector of the top mux, then a tail of small blocks guarded
by the same selectors. The split it produces (identical to the hemilabs fork's F30 `step_exec.cu`,
which was made by its `split_step_exec.py`, `split_macro_ops.py` and `split_micro_ops.py`):

  step_exec                 dispatcher: one `if (arg0[col * steps + ..] != 0) arm(..)` per arm
  step_exec_<arm> x 7       micro_ops, macro_ops, poseidon2_{load,full,partial,store},
                            checked_bytes (columns 1..7); each repeats the constants block, then
                            the arm body and the arm's tail blocks
  step_exec_macro_<op> x 9  macro_ops is split again by its own OneHot (columns 9..17): wom_init,
                            wom_fini, bit_and_elem, bit_op_shorts, sha_{init,fini,load,mix},
                            set_global; step_exec_macro_ops keeps the write_addr prologue, the
                            dispatch and the macro tail
  step_exec_micro_inst0..2  micro_ops runs 3 micro instructions in sequence (opcode columns 8, 12,
                            16, write_addr = arg0[0] + 0/1/2); step_exec_micro_ops calls the three
                            and keeps the micro tail

The fork's scripts named the selector variables (`x8811`, `x1726`, ...), which zirgen renumbers on
every circuit change (3.1's BIT_AND_ELEM fix shifted them by 12). This generator finds them
instead:
  - arms: function-body-level `auto xN = arg0[C * steps + ((cycle - 0) & mask)];` followed within 5
    lines by `if (xN != 0) {`; the arm's name comes from the zirgen `loc(..)` comment above the read
    (`select(OneHot)/<name>(Reg)` at the top, `opcode(OneHot)/<name>(Reg)` inside macro_ops);
  - micro instructions: the `inst[k](recursion::MicroInst)/opcode(Reg)` reads, the write_addr
    `arg0[0 * steps ..]` read next to inst 0, and the `write_addr + one` / `write_addr + two`
    transitions (`one`/`two` are the constants `Fp xN(1)` / `Fp xN(2)`);
  - the constants block: the contiguous `Fp xN(..)` run starting at `Fp x0(`, numbered 0..K.
Any difference from the expected structure (the arm names, their columns and order, 3 micro
instructions, 19 functions) raises SplitError: the build fails loudly instead of translating a
wrongly cut kernel. So does any statement the split would drop, or would emit more than once per
cycle unless it is side-effect free (`Coverage`, `PURE_RES`). The cut and the emitted text are the fork scripts' own, line for line.

`--selftest` re-splits upstream 3.0's monolith (`testdata/step_exec_up30.cu.xz`, the blob
`git show 9fc79b9ef:risc0/circuit/recursion-sys/kernels/cuda/step_exec.cu`) and requires the
result to equal F30's committed split (`git show 68b3eacbb:...` blob f3117ffa) byte for byte (by
sha256, UP30_SPLIT_SHA256), then splits the current `step_exec.cu` (structure only).
"""

import argparse
import hashlib
import os
import re
import sys

sys.dont_write_bytecode = True  # never leave __pycache__ in the source tree (build.rs runs this)

HERE = os.path.dirname(os.path.abspath(__file__))

# The expected structure: (name, selector column) in source order.
TOP_ARMS = [
    ('micro_ops', 1),
    ('macro_ops', 2),
    ('poseidon2_load', 3),
    ('poseidon2_full', 4),
    ('poseidon2_partial', 5),
    ('poseidon2_store', 6),
    ('checked_bytes', 7),
]
MACRO_ARMS = [
    ('wom_init', 9),
    ('wom_fini', 10),
    ('bit_and_elem', 11),
    ('bit_op_shorts', 12),
    ('sha_init', 13),
    ('sha_fini', 14),
    ('sha_load', 15),
    ('sha_mix', 16),
    ('set_global', 17),
]
MICRO_OPCODE_COLS = [8, 12, 16]
NUM_FUNCTIONS = len(TOP_ARMS) + len(MACRO_ARMS) + len(MICRO_OPCODE_COLS)  # 19 noinline functions

# --selftest fixture and expectations (see the module docstring and notes/P31-07.md).
UP30_FIXTURE = os.path.join(HERE, 'testdata', 'step_exec_up30.cu.xz')
UP30_SHA256 = '0377b13c58ddbd58e56df437a9d3747faa9909d2ba95d03972d5fe8e0d503cb6'
UP30_SPLIT_SHA256 = '964665e6a6cf5c6ceced905d3138fed6730c8a1c4baf7f7a3ca6729011c7099e'
UP30_SPLIT_LINES = 53092
# The current 3.1 `step_exec.cu` split (zirgen output of release-3.1 incl. the BIT_AND_ELEM fix
# 84a449919). A change here means zirgen re-emitted the circuit: review the new split and the
# cuda2metal_steps.py pins together, then update this value.
CUR_SPLIT_SHA256 = 'ea7ae4820cc1cdfe776325c907cee718f059890a149b0058557c443287fbd581'
CUR_SPLIT_LINES = 53141

SEL_RE = re.compile(r'^\s*auto (x\d+) = arg0\[(\d+) \* steps \+ \(\(cycle - 0\) & mask\)\];$')
CONST_RE = re.compile(r'^  Fp x(\d+)\((-?\d+)\);$')
ARG_SIG = ('    void* ctx, uint32_t steps, uint32_t cycle, uint32_t mask,\n',
           '    Fp* arg0, Fp* arg1, Fp* arg2')


class SplitError(Exception):
    pass


def check(cond, msg):
    if not cond:
        raise SplitError(msg)


def find_matching_brace(lines, start_idx):
    """Line index of the brace closing the one opened on `start_idx` (naive count, as the fork's
    scripts; zirgen emits no braces in strings or comments)."""
    depth = 0
    for i in range(start_idx, len(lines)):
        depth += lines[i].count('{') - lines[i].count('}')
        if depth == 0:
            return i
    raise SplitError('no matching brace for line %d' % (start_idx + 1))


def body_level_lines(lines, open_line, close_line):
    """Indices of the lines that start at brace depth 1 inside the block open_line..close_line."""
    out = []
    depth = 0
    for i in range(open_line, close_line + 1):
        if depth == 1:
            out.append(i)
        depth += lines[i].count('{') - lines[i].count('}')
    return out


def find_constants(lines, lo, hi):
    """The constants block in lines[lo:hi]: (start incl. the comment before x0, end = last Fp
    line, {value: name})."""
    x0 = next((i for i in range(lo, hi) if lines[i].startswith('  Fp x0(')), None)
    check(x0 is not None, 'no constants block (`Fp x0(`) in lines %d..%d' % (lo + 1, hi))
    start = x0 - 1 if lines[x0 - 1].strip().startswith('//') else x0
    end = x0
    values = {}
    expect = 0
    i = x0
    while i < hi:
        m = CONST_RE.match(lines[i])
        if m:
            check(int(m.group(1)) == expect,
                  'constants not numbered 0..K: x%s at line %d' % (m.group(1), i + 1))
            v = int(m.group(2))
            values.setdefault(v, []).append('x' + m.group(1))
            expect += 1
            end = i
            i += 1
        elif lines[i].strip().startswith('//') and i + 1 < hi and CONST_RE.match(lines[i + 1]):
            i += 1
        else:
            break
    return start, end, values


def detect_arms(lines, open_line, close_line, after, loc_re, expected, what):
    """Body-level OneHot arms of the block: [{name, col, var, sel_line, if_line, end_line}].
    A guarded read without the arm's loc comment is allowed only after the last arm (macro_ops'
    tail section has one: `arg0[8]` under component.cpp)."""
    arms = []
    others = []
    for i in body_level_lines(lines, open_line, close_line):
        if i <= after:
            continue
        m = SEL_RE.match(lines[i])
        if not m:
            continue
        var, col = m.group(1), int(m.group(2))
        if_line = next((j for j in range(i, min(i + 5, close_line))
                        if lines[j].strip() == 'if (%s != 0) {' % var), None)
        if if_line is None:
            continue
        lm = loc_re.search(lines[i - 1])
        if lm is None:
            others.append((i, var, col))
            continue
        arms.append(dict(name=lm.group(1), col=col, var=var, sel_line=i, if_line=if_line,
                         end_line=find_matching_brace(lines, if_line)))
    got = [(a['name'], a['col']) for a in arms]
    check(got == expected, '%s: detected arms %s, expected %s' % (what, got, expected))
    for i, var, col in others:
        check(i > arms[-1]['end_line'], '%s: guarded read %s (column %d) at line %d without a '
              '`%s` loc comment before the last arm' % (what, var, col, i + 1, loc_re.pattern))
    for a, b in zip(arms, arms[1:]):
        check(a['end_line'] < b['sel_line'], '%s: arms %s and %s overlap'
              % (what, a['name'], b['name']))
    return arms


# Lines the split may emit more than once per cycle (the constants block in every arm function,
# macro_ops' write_addr prologue in the dispatcher and in each of the 9 sub-arm functions, a
# comment shared by two micro instructions): running them twice must be indistinguishable from
# running them once, so only side-effect-free shapes are allowed. A read is limited to the code
# columns (arg0, never written by step_exec) at the current cycle; an assert re-checks a value.
# Anything else there (an extern call, `auto& reg =` witness write, a brace, an `if`) raises.
PURE_RES = [
    re.compile(r'^\s*//.*$'),
    re.compile(r'^\s*Fp x\d+\(-?\d+\);$'),
    re.compile(r'^\s*auto x\d+ = arg0\[\d+ \* steps \+ \(\(cycle - 0\) & mask\)\];$'),
    re.compile(r'^\s*assert\(x\d+ != Fp::invalid\(\)\);$'),
    re.compile(r'^\s*auto x\d+ = x\d+ [-+*] x\d+;$'),
]


def is_pure(line):
    line = line.rstrip('\n')
    return line.strip() == '' or any(r.match(line) for r in PURE_RES)


class Coverage:
    """Accounts for every line of a function body the split consumes: each one must be emitted
    (`take`) or be one of the shapes the split regenerates or deliberately drops (`drop`, which
    checks the exact text). `finish` raises SplitError on any other non-blank line, so a statement
    zirgen emits outside the places the split knows about (between arms, in an unguarded tail, in
    a block guarded by a non-selector, between macro sub-arms, before the first micro instruction,
    ...) fails the build instead of vanishing from the kernel. It also counts how many times each
    line is emitted (`take(.., copies=n)`, overlapping takes add up) and raises on any line
    emitted more than once that is not side-effect free (PURE_RES): a statement must neither
    vanish from nor be duplicated into the kernel (P31-07 review rounds 1 and 2)."""

    def __init__(self, lines, lo, hi, what):
        self.lines, self.lo, self.hi, self.what = lines, lo, hi, what
        self.covered = set()
        self.copies = {}

    def take(self, lo, hi, copies=1):
        self.covered.update(range(lo, hi))
        for i in range(lo, hi):
            self.copies[i] = self.copies.get(i, 0) + copies
        return self.lines[lo:hi]

    def drop(self, i, expected, why):
        got = self.lines[i].rstrip('\n')
        ok = expected.match(got) if hasattr(expected, 'match') else got == expected
        check(ok, '%s: line %d is %r, expected %s (%r)'
              % (self.what, i + 1, got, why, getattr(expected, 'pattern', expected)))
        self.covered.add(i)

    def finish(self):
        lost = [i for i in range(self.lo, self.hi)
                if i not in self.covered and self.lines[i].strip() != '']
        check(not lost, '%s: %d line(s) would be dropped by the split, first: %s'
              % (self.what, len(lost), '; '.join('line %d %r' % (i + 1, self.lines[i].strip()[:120])
                                                for i in lost[:5])))
        dup = sorted(i for i, n in self.copies.items() if n > 1 and not is_pure(self.lines[i]))
        check(not dup, '%s: %d line(s) would run more than once per cycle in the split, first: %s'
              % (self.what, len(dup), '; '.join('line %d %r (x%d)' % (i + 1,
                                                self.lines[i].strip()[:120], self.copies[i])
                                                for i in dup[:5])))


# Lines of a function's preamble the split regenerates in every function it emits.
PREAMBLE = {'  uint32_t mask = steps - 1;', '  Fp extern_args[96];', '  Fp extern_outs[32];'}


def drop_preamble(cov, open_brace, const_start, allowed):
    for i in range(open_brace + 1, const_start):
        if cov.lines[i].strip() == '':
            continue
        check(cov.lines[i].rstrip('\n') in allowed, '%s: line %d %r before the constants block is '
              'not one of %s' % (cov.what, i + 1, cov.lines[i].strip(), sorted(allowed)))
        cov.covered.add(i)


def drop_arm_guard(cov, a):
    """The arm's selector read, its invalid assert(s), the `if` and the closing brace (the split's
    dispatcher re-reads the column). The loc comment above the read is handled by the caller."""
    v = a['var']
    line = cov.lines[a['sel_line']]
    indent = line[:len(line) - len(line.lstrip())]
    cov.drop(a['sel_line'], SEL_RE, 'the selector read')
    for j in range(a['sel_line'] + 1, a['if_line']):
        cov.drop(j, '%sassert(%s != Fp::invalid());' % (indent, v), 'the selector assert')
    cov.drop(a['if_line'], '%sif (%s != 0) {' % (indent, v), 'the arm guard')
    cov.drop(a['end_line'], indent + '}', 'the arm closing brace')


def find_function(lines, name, what):
    """(definition line, opening-brace line, closing-brace line) of a split function, and the
    (start, end) of its forward declaration."""
    head = '__device__ __noinline__ void %s(' % name
    hits = [i for i, l in enumerate(lines) if head in l]
    check(len(hits) == 2, '%s: expected a declaration and a definition of %s, found %d'
          % (what, name, len(hits)))
    decl, defn = hits
    decl_end = next((j for j in range(decl, decl + 5) if ';' in lines[j]), None)
    check(decl_end is not None, '%s: %s declaration has no `;`' % (what, name))
    open_brace = defn
    while '{' not in lines[open_brace]:
        open_brace += 1
    return defn, open_brace, find_matching_brace(lines, open_brace), decl, decl_end


def fn_head(name, term):
    return ['__device__ __noinline__ void %s(\n' % name, ARG_SIG[0], ARG_SIG[1] + term]


def split_top(lines, log):
    """step_exec -> dispatcher + 7 arm functions (the fork's split_step_exec.py)."""
    starts = [i for i, l in enumerate(lines) if '__device__ void step_exec(' in l]
    check(len(starts) == 1, 'expected one `__device__ void step_exec(`, found %d' % len(starts))
    func_start = starts[0]
    open_brace = func_start
    while '{' not in lines[open_brace]:
        open_brace += 1
    func_close = find_matching_brace(lines, open_brace)
    header = lines[:func_start]
    const_start, const_end, consts = find_constants(lines, func_start, func_close)
    const_lines = lines[const_start:const_end + 1]
    arms = detect_arms(lines, open_brace, func_close, const_end,
                       re.compile(r'/select\(OneHot\)/(\w+)\(Reg\)'), TOP_ARMS, 'step_exec')
    log('step_exec: constants x0..x%d, arms %s' % (len(sum(consts.values(), [])) - 1,
                                                   ' '.join('%s=%s' % (a['name'], a['var'])
                                                            for a in arms)))
    cov = Coverage(lines, open_brace + 1, func_close, 'step_exec')
    drop_preamble(cov, open_brace, const_start, PREAMBLE)
    cov.take(const_start, const_end + 1, copies=len(arms))
    for a in arms:
        cov.drop(a['sel_line'] - 1, re.compile(r'^  // loc\(.*/select\(OneHot\)/%s\(Reg\)'
                                               % a['name']), 'the arm\'s loc comment')
        drop_arm_guard(cov, a)
        cov.take(a['if_line'] + 1, a['end_line'])

    # Tail: after the last arm, body-level `if (<arm selector> != 0) { .. }` blocks only.
    tail_start = arms[-1]['end_line'] + 1
    func_end = len(lines) - 1
    while func_end > tail_start and lines[func_end].strip() == '':
        func_end -= 1
    check(func_end == func_close, 'step_exec is not the last thing in the file')
    guards = {'  if (%s != 0) {\n' % a['var']: a['var'] for a in arms}
    tail_blocks = {}
    i = tail_start
    while i < func_end:
        var = guards.get(lines[i])
        if var is not None:
            end = find_matching_brace(lines, i)
            cov.drop(end, '  }', 'a tail block closing brace')
            cov.covered.add(i)
            cov.take(i + 1, end)
            tail_blocks.setdefault(var, []).append((i, end))
            i = end
        i += 1
    cov.finish()
    log('  tail blocks: %s' % ', '.join('%s x%d' % (a['name'], len(tail_blocks.get(a['var'], [])))
                                        for a in arms))

    out = list(header)
    for a in arms:
        out += fn_head('step_exec_' + a['name'], ');\n') + ['\n']
    out.append('__device__ void step_exec(\n')
    out.append('    void* ctx, uint32_t steps, uint32_t cycle, Fp* arg0, Fp* arg1, Fp* arg2, '
               'Fp* arg3, Fp* arg4) {\n')
    out.append('  uint32_t mask = steps - 1;\n')
    out.append('\n')
    for a in arms:
        out.append('  if (arg0[%d * steps + ((cycle - 0) & mask)] != 0)\n' % a['col'])
        out.append('    step_exec_%s(ctx, steps, cycle, mask, arg0, arg1, arg2);\n' % a['name'])
    out.append('}\n')
    out.append('\n')
    for a in arms:
        out += fn_head('step_exec_' + a['name'], ') {\n')
        out.append('  Fp extern_args[96];\n')
        out.append('  Fp extern_outs[32];\n')
        out += const_lines
        out.append('\n')
        out += lines[a['if_line'] + 1:a['end_line']]
        if a['var'] in tail_blocks:
            out.append('\n')
            out.append('  // tail section\n')
            for tb_start, tb_end in tail_blocks[a['var']]:
                out += lines[tb_start + 1:tb_end]
        out.append('}\n')
        out.append('\n')
    return out


def split_macro(lines, log):
    """step_exec_macro_ops -> dispatcher + 9 sub-arm functions (the fork's split_macro_ops.py)."""
    what = 'step_exec_macro_ops'
    func_start, open_brace, func_end, _, fwd_decl_end = find_function(lines, what, what)
    const_start, const_end, _ = find_constants(lines, func_start, func_end)
    const_lines = lines[const_start:const_end + 1]
    subs = detect_arms(lines, open_brace, func_end, const_end,
                       re.compile(r'/opcode\(OneHot\)/(\w+)\(Reg\)'), MACRO_ARMS, what)
    log('macro_ops: %s' % ' '.join('%s=%s' % (s['name'], s['var']) for s in subs))
    has_tail = any(l.strip() for l in lines[subs[-1]['end_line'] + 1:func_end])
    cov = Coverage(lines, open_brace + 1, func_end, what)
    drop_preamble(cov, open_brace, const_start, PREAMBLE)
    cov.take(const_start, const_end + 1, copies=len(subs) + int(has_tail))
    # The prologue runs in the dispatcher and again in every sub-arm function (Coverage requires
    # it to be side-effect free).
    prologue_lines = cov.take(const_end + 1, subs[0]['sel_line'], copies=1 + len(subs))
    for k, s in enumerate(subs):
        if k > 0:
            cov.drop(s['sel_line'] - 1, re.compile(r'^\s*// loc\(.*/opcode\(OneHot\)/%s\(Reg\)'
                                                   % s['name']), 'the sub-arm\'s loc comment')
        drop_arm_guard(cov, s)
        cov.take(s['if_line'] + 1, s['end_line'])
    tail_lines = cov.take(subs[-1]['end_line'] + 1, func_end)
    cov.finish()

    out = lines[:fwd_decl_end + 1]
    out.append('\n')
    for s in subs:
        out += fn_head('step_exec_macro_' + s['name'], ');\n') + ['\n']
    out += lines[fwd_decl_end + 1:func_start]
    out += fn_head(what, ') {\n')
    if tail_lines:
        out.append('  Fp extern_args[96];\n')
        out.append('  Fp extern_outs[32];\n')
        out += const_lines
        out.append('\n')
    out += prologue_lines
    out.append('\n')
    for s in subs:
        out.append('  if (arg0[%d * steps + ((cycle - 0) & mask)] != 0)\n' % s['col'])
        out.append('    step_exec_macro_%s(ctx, steps, cycle, mask, arg0, arg1, arg2);\n'
                   % s['name'])
    if tail_lines:
        out.append('\n')
        out.append('  // tail section\n')
        out += tail_lines
    out.append('}\n')
    out.append('\n')
    for s in subs:
        out += fn_head('step_exec_macro_' + s['name'], ') {\n')
        out.append('  Fp extern_args[96];\n')
        out.append('  Fp extern_outs[32];\n')
        out += const_lines
        out.append('\n')
        out += prologue_lines
        out.append('\n')
        out += lines[s['if_line'] + 1:s['end_line']]
        out.append('}\n')
        out.append('\n')
    out += lines[func_end + 1:]
    return out


def split_micro(lines, log):
    """step_exec_micro_ops -> dispatcher + 3 instance functions (the fork's split_micro_ops.py)."""
    what = 'step_exec_micro_ops'
    func_start, open_brace, func_end, _, fwd_decl_end = find_function(lines, what, what)
    const_start, const_end, consts = find_constants(lines, func_start, func_end)
    const_lines = lines[const_start:const_end + 1]
    one, two = consts.get(1, []), consts.get(2, [])
    check(len(one) == 1 and len(two) == 1, '%s: constants 1/2 are %s/%s' % (what, one, two))
    one, two = one[0], two[0]

    # The micro instructions' opcode reads, by their loc comment.
    inst_re = re.compile(r'/inst\[(\d+)\]\(recursion::MicroInst\)/opcode\(Reg\)')
    opcode_lines = []
    for i in range(const_end + 1, func_end):
        m = inst_re.search(lines[i])
        if m and lines[i].lstrip().startswith('// loc('):
            s = SEL_RE.match(lines[i + 1])
            check(s is not None, '%s: inst[%s] opcode loc at line %d is not followed by an '
                  'arg0 read' % (what, m.group(1), i + 1))
            opcode_lines.append((int(m.group(1)), int(s.group(2)), i + 1))
    got = [(k, col) for k, col, _ in opcode_lines]
    want = list(enumerate(MICRO_OPCODE_COLS))
    check(got == want, '%s: micro instruction opcode reads %s, expected %s' % (what, got, want))
    opcode_lines = [i for _, _, i in opcode_lines]

    wa = [i for i in range(opcode_lines[0] - 5, opcode_lines[0] + 10)
          if 'arg0[0 * steps' in lines[i]]
    check(len(wa) >= 1, '%s: no write_addr (arg0[0]) read next to inst 0' % what)
    m = SEL_RE.match(lines[wa[0]])
    check(m is not None, '%s: the write_addr line %d %r is not a plain `auto xN = arg0[0 * steps '
          '+ ((cycle - 0) & mask)];` read' % (what, wa[0] + 1, lines[wa[0]].strip()))
    write_addr_var = m.group(1)

    def transition(lo, hi, const):
        hits = [i for i in range(lo, hi) if '%s + %s' % (write_addr_var, const) in lines[i]]
        check(hits, '%s: no `%s + %s` transition in lines %d..%d'
              % (what, write_addr_var, const, lo + 1, hi))
        return hits[-1]

    trans_01 = transition(opcode_lines[0], opcode_lines[1], one)
    trans_12 = transition(opcode_lines[1], opcode_lines[2], two)
    tail_start = next((i for i in range(opcode_lines[2], func_end)
                       if '// tail section' in lines[i]), func_end)
    tail_lines = lines[tail_start:func_end]
    log('micro_ops: opcode reads %s, write_addr %s, transitions +%s/+%s'
        % ([SEL_RE.match(lines[i]).group(1) for i in opcode_lines], write_addr_var, one, two))

    def span(start, end, back_comment):
        if back_comment and lines[start - 1].strip().startswith('//'):
            start -= 1
        while end > start and lines[end].strip() == '':
            end -= 1
        return start, end

    i0 = opcode_lines[0]
    if i0 > const_end + 1 and lines[i0 - 1].strip().startswith('//'):
        i0 -= 1
    instances = [
        ('inst0',) + span(i0, trans_01 - 1, False) + (True,),
        ('inst1',) + span(trans_01, trans_12 - 1, True) + (False,),
        ('inst2',) + span(trans_12, tail_start - 1, True) + (False,),
    ]
    check(wa[0] >= i0 and wa[0] <= instances[0][2], '%s: the write_addr read at line %d is not '
          'inside inst0 (lines %d..%d)' % (what, wa[0] + 1, i0 + 1, instances[0][2] + 1))
    cov = Coverage(lines, open_brace + 1, func_end, what)
    drop_preamble(cov, open_brace, const_start, PREAMBLE)
    cov.take(const_start, const_end + 1, copies=len(instances) + int(bool(tail_lines)))
    for _, start, end, _ in instances:  # adjacent instances may share a comment line
        cov.take(start, end + 1)
    cov.take(tail_start, func_end)
    cov.finish()

    out = lines[:fwd_decl_end + 1]
    out.append('\n')
    for name, _, _, _ in instances:
        out += fn_head('step_exec_micro_' + name, ');\n') + ['\n']
    out += lines[fwd_decl_end + 1:func_start]
    out += fn_head(what, ') {\n')
    if tail_lines:
        out.append('  Fp extern_args[96];\n')
        out.append('  Fp extern_outs[32];\n')
        out += const_lines
        out.append('\n')
    for name, _, _, _ in instances:
        out.append('  step_exec_micro_%s(ctx, steps, cycle, mask, arg0, arg1, arg2);\n' % name)
    if tail_lines:
        out.append('\n')
        out += tail_lines
    out.append('}\n')
    out.append('\n')
    for name, start, end, reads_wa in instances:
        out += fn_head('step_exec_micro_' + name, ') {\n')
        out.append('  Fp extern_args[96];\n')
        out.append('  Fp extern_outs[32];\n')
        out += const_lines
        out.append('\n')
        if not reads_wa:
            # inst1/inst2 use write_addr through their transition variable.
            out.append('    auto %s = arg0[0 * steps + ((cycle - 0) & mask)];\n' % write_addr_var)
            out.append('    assert(%s != Fp::invalid());\n' % write_addr_var)
        out += lines[start:end + 1]
        out.append('}\n')
        out.append('\n')
    out += lines[func_end + 1:]
    return out


def split(text, log=lambda *a: None):
    """The split form of a monolithic step_exec.cu (str -> str). Raises SplitError."""
    check('__noinline__' not in text, 'input is already split (it contains __noinline__)')
    check(text.endswith('\n'), 'input does not end with a newline')
    lines = text.splitlines(keepends=True)
    lines = split_top(lines, log)
    lines = split_macro(lines, log)
    lines = split_micro(lines, log)
    out = ''.join(lines)
    n = out.count('__device__ __noinline__ void ')
    check(n == 2 * NUM_FUNCTIONS, 'split has %d noinline declarations + definitions, expected %d'
          % (n, 2 * NUM_FUNCTIONS))
    return out


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read(path):
    with open(path, 'r', newline='') as f:
        return f.read()


def write_if_changed(path, text):
    if os.path.exists(path) and read(path) == text:
        return 'unchanged'
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='') as f:
        f.write(text)
    os.replace(tmp, path)
    return 'written'


# Statements the split must never drop silently (P31-07 review round 1: each of these used to
# vanish from the output with no error, and the cuda2metal_steps.py pins still passed).
PROBE_CALL = '  extern_markerFn(ctx, cycle, "PROBE", extern_args, extern_outs);\n'
PROBE_IOP = ('  extern_args[0] = x0;\n'
             '  extern_readIOPBody(ctx, cycle, "", extern_args, extern_outs);\n')
PROBE_WRITE = ('  {\n    auto& reg = arg2[5 * steps + cycle];\n'
               '    assert(reg == Fp::invalid() || reg == x0);\n    reg = x0;\n  }\n')


def indent(text, n):
    return ''.join(' ' * n + l if l.strip() else l for l in text.splitlines(keepends=True))


def drop_mutations(text):
    """(what, mutated text) pairs that insert a statement where the split has no place for it."""
    lines = text.splitlines(keepends=True)

    def first(pred, what):
        hits = [i for i, l in enumerate(lines) if pred(l)]
        check(hits, 'selftest: no anchor for %r' % what)
        return hits[0]

    def at(i, ins):
        return ''.join(lines[:i]) + ins + ''.join(lines[i:])

    guard = re.compile(r'^  if \(x\d+ != 0\) \{$')
    guards = [i for i, l in enumerate(lines) if guard.match(l.rstrip('\n'))]
    check(len(guards) > len(TOP_ARMS), 'selftest: no step_exec tail blocks')
    tail0 = guards[len(TOP_ARMS)]  # the first tail block
    macro_loc = first(lambda l: '/select(OneHot)/macro_ops(Reg)' in l, 'macro_ops loc')
    micro_loc = first(lambda l: '/select(OneHot)/micro_ops(Reg)' in l, 'micro_ops loc')
    fini_loc = first(lambda l: '/opcode(OneHot)/wom_fini(Reg)' in l and '// loc(' in l,
                     'wom_fini loc')
    inst0_loc = first(lambda l: '/inst[0](recursion::MicroInst)/opcode(Reg)' in l
                      and '// loc(' in l, 'inst[0] loc')
    init_loc = first(lambda l: '/opcode(OneHot)/wom_init(Reg)' in l and '// loc(' in l,
                     'wom_init loc')  # the end of macro_ops' write_addr prologue
    last = len(lines) - 1
    check(lines[last] == '}\n', 'selftest: step_exec does not end the file')
    return [
        ('a call between the micro_ops and macro_ops arms', at(macro_loc, PROBE_CALL)),
        ('a call between the constants and the first arm', at(micro_loc, PROBE_CALL)),
        ('an unguarded call in the tail', at(tail0, PROBE_CALL)),
        ('a tail block guarded by a non-selector',
         at(tail0, '  if (x5 != 0) {\n' + indent(PROBE_CALL, 2) + '  }\n')),
        ('a call before step_exec\'s closing brace', at(last, PROBE_CALL)),
        ('a witness write between the macro sub-arms', at(fini_loc, indent(PROBE_WRITE, 2))),
        ('a witness write before micro inst[0]', at(inst0_loc, indent(PROBE_WRITE, 2))),
        ('an unguarded witness write in the tail', at(tail0, PROBE_WRITE)),
        # Review round 2: the macro_ops prologue is emitted 10 times (dispatcher + 9 sub-arms).
        ('an extern call in the macro_ops prologue', at(init_loc, indent(PROBE_IOP, 2))),
        ('a witness write in the macro_ops prologue', at(init_loc, indent(PROBE_WRITE, 2))),
        ('a guarded witness write in the macro_ops prologue',
         at(init_loc, '    if (x0 != 0) {\n' + indent(PROBE_WRITE, 4) + '    }\n')),
        ('a read of the witness in the macro_ops prologue',
         at(init_loc, '    auto x99999 = arg2[5 * steps + ((cycle - 0) & mask)];\n')),
    ]


def keep_mutations(text):
    """(what, mutated text) pairs that insert PROBE_CALL where the split must copy it once."""
    lines = text.splitlines(keepends=True)
    out = []
    for what, anchor in (('a call in the poseidon2_full arm', '/select(OneHot)/poseidon2_full(Reg)'),
                         ('a call in the sha_mix macro sub-arm', '/opcode(OneHot)/sha_mix(Reg)')):
        i = next(i for i, l in enumerate(lines) if anchor in l and '// loc(' in l)
        j = next(j for j in range(i, i + 6) if lines[j].rstrip().endswith('!= 0) {')) + 1
        pad = len(lines[j]) - len(lines[j].lstrip())
        out.append((what, ''.join(lines[:j]) + ' ' * (pad - 2) + PROBE_CALL + ''.join(lines[j:])))
    guard = re.compile(r'^  if \(x\d+ != 0\) \{$')
    tail0 = [i for i, l in enumerate(lines) if guard.match(l.rstrip('\n'))][len(TOP_ARMS)]
    out.append(('a call in a tail block', ''.join(lines[:tail0 + 1]) + indent(PROBE_CALL, 2)
                + ''.join(lines[tail0 + 1:])))
    return out


def pure_prologue_mutation(text):
    """A side-effect-free line in the macro_ops prologue: allowed, emitted in the dispatcher and
    in each of the 9 sub-arm functions."""
    lines = text.splitlines(keepends=True)
    i = next(i for i, l in enumerate(lines)
             if '/opcode(OneHot)/wom_init(Reg)' in l and '// loc(' in l)
    return ''.join(lines[:i]) + '    auto x99999 = x0 + x0;\n' + ''.join(lines[i:])


def expect_split_error(what, text, log):
    try:
        split(text)
    except SplitError as e:
        log('selftest: %s -> SplitError (%s)' % (what, e))
        return
    raise SplitError('selftest: %s was not detected' % what)


def selftest(log):
    import lzma  # only the selftest needs it (a Python without xz support can still split)
    with lzma.open(UP30_FIXTURE, 'rb') as f:
        up30 = f.read().decode()
    check(sha256(up30) == UP30_SHA256, 'fixture %s: sha256 %s, expected %s (UP30 step_exec.cu)'
          % (UP30_FIXTURE, sha256(up30), UP30_SHA256))
    out = split(up30, log)
    got = sha256(out)
    if got != UP30_SPLIT_SHA256:
        raise SplitError('UP30 re-split differs from F30\'s committed split: sha256 %s (%d lines), '
                         'expected %s (%d lines); compare with `git show 68b3eacbb:risc0/circuit/'
                         'recursion-sys/kernels/cuda/step_exec.cu`'
                         % (got, out.count('\n'), UP30_SPLIT_SHA256, UP30_SPLIT_LINES))
    log('selftest: UP30 re-split == F30 split (sha256 %s, %d lines)' % (got, out.count('\n')))
    # Structure changes must fail loudly, not produce a differently cut kernel.
    mutations = [
        ('a top arm renamed', '/select(OneHot)/poseidon2_full(Reg)',
         '/select(OneHot)/poseidon2_fast(Reg)'),
        ('a macro arm moved to another column', 'auto x2049 = arg0[13 * steps',
         'auto x2049 = arg0[19 * steps'),
        ('a micro instruction missing', '/inst[2](recursion::MicroInst)/opcode(Reg)',
         '/inst[3](recursion::MicroInst)/opcode(Reg)'),
        ('the constant 1 missing', '  Fp x310(1);\n', '  Fp x310(77);\n'),
        ('an already split input', '__device__ void step_exec(',
         '__device__ __noinline__ void step_exec('),
        ('micro write_addr not a current-cycle read', 'auto x314 = arg0[0 * steps + ((cycle - 0)',
         'auto x314 = arg0[0 * steps + ((cycle - 1)'),
    ]
    for what, old, new in mutations:
        check(up30.count(old) == 1, 'selftest mutation %r: %r occurs %d times'
              % (what, old, up30.count(old)))
        expect_split_error(what, up30.replace(old, new), log)
    cur_path = os.path.join(HERE, 'step_exec.cu')
    cur = read(cur_path)
    # Statements outside every place the split copies must fail the split, on both inputs.
    for name, text in (('UP30', up30), ('3.1', cur)):
        for what, mutated in drop_mutations(text):
            expect_split_error('%s: %s' % (name, what), mutated, log)
        # Positive control: the same statements inside an arm body or a tail block are kept.
        for what, mutated in keep_mutations(text):
            kept = split(mutated).count('"PROBE"')
            check(kept == 1, 'selftest: %s: %s kept %d times, expected once' % (name, what, kept))
            log('selftest: %s: %s -> kept' % (name, what))
        kept = split(pure_prologue_mutation(text)).count('auto x99999 = x0 + x0;')
        check(kept == 1 + len(MACRO_ARMS), 'selftest: %s: a pure macro_ops prologue line kept %d '
              'times, expected %d' % (name, kept, 1 + len(MACRO_ARMS)))
        log('selftest: %s: a pure line in the macro_ops prologue -> kept %d times' % (name, kept))
    out = split(cur, log)
    got = sha256(out)
    if got != CUR_SPLIT_SHA256:
        raise SplitError('%s split: sha256 %s (%d lines), expected %s (%d lines); zirgen changed '
                         'the circuit: review the split and the cuda2metal_steps.py pins, then '
                         'update CUR_SPLIT_SHA256' % (cur_path, got, out.count('\n'),
                                                      CUR_SPLIT_SHA256, CUR_SPLIT_LINES))
    log('selftest: %s splits: %d lines, %d noinline functions, sha256 %s'
        % (cur_path, out.count('\n'), NUM_FUNCTIONS, got))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('input', nargs='?')
    ap.add_argument('-o', '--output')
    ap.add_argument('--stats', action='store_true', help='split in memory and print the structure')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('-q', '--quiet', action='store_true')
    a = ap.parse_args(argv)
    log = (lambda *x: None) if a.quiet else print
    if a.selftest:
        selftest(log)
        return 0
    if not a.input or not (a.output or a.stats):
        ap.error('IN with -o OUT or --stats is required unless --selftest')
    out = split(read(a.input), log)
    log('%s: %d lines -> %d lines, %d noinline functions, sha256 %s'
        % (a.input, read(a.input).count('\n'), out.count('\n'), NUM_FUNCTIONS, sha256(out)))
    if a.output:
        log('-> %s (%s)' % (a.output, write_if_changed(a.output, out)))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except SplitError as e:
        print('split_step_exec_metal.py: error: %s' % e, file=sys.stderr)
        sys.exit(1)
