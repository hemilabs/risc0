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

"""Metal-only regrouping of the recursion eval_check split (Metal port item C15, LEADS #17).

usage: ec_regroup_metal.py [--check] [--leaves]

eval_check_split.py splits poly_fp into 6 parts along its gated-sum tree, tuned for sm_120. On
the Apple GPU two of those parts (0 and 5, 28 of 45 ms per po2 18 call) compile to large
per-thread stacks (2.6 and 3.1 KB, tools/metal_audit/agx_size.py) and run 2-4x slower per op
than the others; R21 found that eval time on this GPU follows the stack size. This module runs
eval_check_split.py's own generator (the same tree, the same emitted part text, the same
ec_lazyredc.py pass) with two Metal-only changes:

- FORCE_SPLIT: tree nodes split into their terms even below TARGET_OPS (x1101 in part 0, x12357
  in part 5: each one leaf whose subtree alone spilled to a ~2 KB stack);
- GROUPS: an explicit packing of the leaves into parts (14 instead of 6).

Every part still re-applies the gate/mix factors of the tree nodes it touches and adds the
previous parts' sum, so only the grouping of field sums changes and the result is bit-identical
(the argument eval_check_split.py relies on; the Metal tests compare every variant with the CPU).
The CUDA/HIP sources are untouched: the output is never written to the source tree, it is the
input of `cuda2metal_eval.py --split-rg` (eval_check_split_rg.metal, kernels ec_rg_part_0..13).

The generator is eval_check_split.py itself, executed from its source with a few pinned text
hooks (PATCHES, each must apply exactly once), so a change to the split algorithm reaches this
module too. `--check` (and cuda2metal_eval.py --check --split-rg) also runs the hooked generator
with no override and requires eval_check_split.cu byte for byte, and pins the regrouped shape
(EXPECT, PART_OPS). If eval_check.cu changes shape, re-derive GROUPS: `--leaves` lists the
leaves (index, tree depth, base, factors, ops); per-part stack sizes come from agx_size.py on
the built metallib and the times from an interleaved per-part harness (notes/C15.md).

Known limit of ec_lazyredc.py (not a problem for GROUPS): its materialized-variable set is
global to the file, so a grouping in which two parts both read the materialized (`xN_m`) form of
the same lazy chain emits the materialization only in the first part and the second fails to
compile (an undeclared `xN_m`). That is a compile error, never wrong code; pick another
grouping.
"""

import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT_PY = os.path.join(HERE, 'eval_check_split.py')
SPLIT_CU = os.path.join(HERE, 'eval_check_split.cu')

# Leaves split into their terms although their cost is below TARGET_OPS.
FORCE_SPLIT = ('x1101', 'x12357')


def _r(a, b):
    return list(range(a, b + 1))


# The leaves (indices in tree order with FORCE_SPLIT applied; `--leaves`) of each Metal part.
# Original part 0 = leaves 0-44 (0-38 are x1101's terms), 1 = 45-64, 2 = 65-66, 3 = 67-68,
# 4 = 69-91, 5 = 92-114 (102-114 are x12357's terms). Measured per-part GPU ms at po2 18
# (C15 harness, quiet window) in the comments; the 6-part split was 10.85, 5.80, 4.41, 3.24,
# 6.03, 14.69 = 45.0 ms.
GROUPS = [
    _r(0, 19),             # x1101 terms, first half                      2.07
    _r(20, 38),            # x1101 terms, second half                     2.22
    _r(39, 44),            # x284 x1110 x1377 x1621 x1844 x2299             2.06
    _r(45, 63),            # x3863 and the Fp mix leaves of part 1          4.05
    [64],                  # x4809                                          0.54
    [65],                  # x5224                                          0.79
    [66],                  # x6152                                          1.68
    _r(67, 68),            # x7908 (a serial chain), x8069                  3.24
    [69],                  # x9404                                          1.98
    _r(70, 91),            # the small gated leaves of part 4               2.84
    [92, 93, 94, 97],      # x11026 (x3), x11736                            2.36
    [95, 96] + _r(98, 101),  # x11746 x11850 x11863 x12000                  0.86
    _r(102, 109),          # x12357 terms: the short chains, x12211 x12214  0.93
    _r(110, 114),          # x12357 terms: x12335 (x4), x12356              2.46
]
NUM_PARTS = len(GROUPS)

# Pins: the hooked generator's statistics with FORCE_SPLIT + GROUPS (generator; lazy REDC) and
# each part's op count (eval_check_split.py's cost()).
EXPECT = ('total ops 12075 leaves 115 groups 14 max leaf 2049; '
          'lazy REDC: chain terms 1074, folds 406, from 16, materialized 115')
PART_OPS = [452, 389, 1282, 2851, 146, 468, 1030, 2017, 1499, 416, 2097, 527, 301, 411]

# Text hooks into eval_check_split.py (old, new); each `old` must occur exactly once.
PATCHES = [
    ('if len(sub) > 1 and cost([b]) > target:',
     'if len(sub) > 1 and (cost([b]) > target or b in C15_FORCE_SPLIT):'),
    ('if cur: groups.append(cur)\n',
     'if cur: groups.append(cur)\ngroups = C15_REGROUP(leaves, groups)\n'),
    ('if STATS != EXPECT:', 'if C15_PINS and STATS != EXPECT:'),
    ('if counts != ec_lazyredc.EXPECT:', 'if C15_PINS and counts != ec_lazyredc.EXPECT:'),
    ('if vcounts != EC_VARIANTS_EXPECT:', 'if C15_PINS and vcounts != EC_VARIANTS_EXPECT:'),
]
# The generator's source is executed up to (not including) its file write / --check compare.
STOP = "\ntry:\n    cur = open(OUT, newline='').read()\n"


class RegroupError(Exception):
    pass


def _run(force, groups, pins):
    """Execute eval_check_split.py's generator with the hooks; returns its namespace."""
    src = open(SPLIT_PY, newline='').read()
    for old, new in PATCHES:
        n = src.count(old)
        if n != 1:
            raise RegroupError('eval_check_split.py hook %r found %d times (expected 1): '
                               're-derive ec_regroup_metal.PATCHES' % (old, n))
        src = src.replace(old, new)
    if src.count(STOP) != 1:
        raise RegroupError('eval_check_split.py: file-write block not found')
    src = src[:src.index(STOP) + 1]

    def regroup(leaves, default):
        if groups is None:
            return default
        flat = sorted(i for g in groups for i in g)
        if flat != list(range(len(leaves))):
            raise RegroupError('GROUPS must cover leaves 0..%d exactly once' % (len(leaves) - 1))
        return [[leaves[i] for i in g] for g in groups]

    ns = {'__name__': 'ec_regroup_metal_split', '__file__': SPLIT_PY,
          'C15_FORCE_SPLIT': frozenset(force), 'C15_REGROUP': regroup, 'C15_PINS': pins}
    argv, path = sys.argv, list(sys.path)
    sys.argv = [SPLIT_PY]  # never --check: the hooked source stops before the compare/write
    try:
        exec(compile(src, SPLIT_PY, 'exec'), ns)
    except SystemExit as e:
        raise RegroupError('eval_check_split.py (hooked): %s' % e)
    finally:
        sys.argv = argv
        sys.path[:] = path
    return ns


def _part_ops(ns):
    cost, leafvars = ns['cost'], ns['leafvars']
    return [cost([v for _, n in g for v in leafvars(n)]) for g in ns['groups']]


def regrouped(check=True):
    """(CUDA-form split source with the Metal grouping, statistics string). With `check`, the
    hooked generator must reproduce eval_check_split.cu without the override, and the regrouped
    shape must match EXPECT / PART_OPS."""
    if check:
        base = _run((), None, True)['new']
        cur = open(SPLIT_CU, newline='').read()
        if base != cur:
            raise RegroupError('the hooked eval_check_split.py does not reproduce '
                               'eval_check_split.cu (run eval_check_split.py --check first)')
    ns = _run(FORCE_SPLIT, GROUPS, False)
    stats = ns['STATS'].split('; HIP variants')[0]
    ops = _part_ops(ns)
    if check and (stats != EXPECT or ops != PART_OPS):
        raise RegroupError('the regrouped split changed shape (%s, part ops %s; expected %s, %s): '
                           're-derive GROUPS (see the module doc), re-verify bit-exactness and '
                           'timing, then update the pins' % (stats, ops, EXPECT, PART_OPS))
    return ns['new'], stats


def leaves_table():
    ns = _run(FORCE_SPLIT, None, False)
    cost, leafvars = ns['cost'], ns['leafvars']
    return ['%d\t%d\t%s\t%s\t%d' % (i, len(p), n['b'], ' '.join(n['f']), cost(leafvars(n)))
            for i, (p, n) in enumerate(ns['leaves'])]


def main():
    if '--leaves' in sys.argv[1:]:
        print('\n'.join(leaves_table()))
        return 0
    _, stats = regrouped(check=True)
    print('ec_regroup_metal: %d parts, %s, part ops %s' % (NUM_PARTS, stats, PART_OPS))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except RegroupError as e:
        print('ec_regroup_metal.py: error: %s' % e, file=sys.stderr)
        sys.exit(1)
