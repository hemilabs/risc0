#!/usr/bin/env python3
"""Split recursion/poly_fp.cpp into N noinline sub-functions, with buffer
loads hoisted to the entry function to avoid SSA-closure duplication.

## Status (2026-06-08): v2 hoist design ALSO produces broken kernel.

Setting RISC0_RECURSION_SPLIT_POLY_FP=1 with this v2 hoist design produces
a kernel that compiles cleanly (.so = 55.5 MB, vs the v1 PoC's 59.7 MB)
but still triggers UR_RESULT_ERROR_DEVICE_LOST at runtime during
composite_to_succinct. Diagnostic showed failure at any WG (tested 64,
128, 1024) — not a private-memory-pressure issue from `Fp bufs[721]`.

The likely root cause is more fundamental:
- Splitting a 24K-line monolithic function into 10 noinline sub-fns
  forces the compiler to commit `bufs[]` to private memory (it can't
  CSE/register-allocate across noinline boundaries).
- Each sub-fn then reads bufs[K] via slow private-memory loads,
  multiplied by the deep call chain.
- The total per-cycle cost exceeds the TDR timeout, or some other
  GPU resource cap is hit at runtime.

In other words: the monolithic poly_fp WORKS BECAUSE IGC has full
visibility over all 12K SSA defs and can optimize globally. Any split
that introduces noinline boundaries degrades this enough to break the
kernel at runtime, regardless of how clever the hoist design is.

Possible further directions (not yet tried):
- Aggressive `__attribute__((always_inline))` on EVERYTHING and a
  marker function call to break the IGC compile into stages without
  introducing real call boundaries — but the original goal was
  noinline-for-IGC; that conflicts.
- Split into TWO files via a different mechanism: each translation
  unit's poly_fp is `static inline`, build object files separately,
  link via LTO. Compile-time wins but no runtime split.
- Hand-curated split of just the most-arithmetic-heavy sections; keep
  buffer-read-heavy front matter in one piece.

DEFAULT OFF in build.rs. This script is kept in-tree as a starting
point for any future iteration that finds a working split topology.

## Design (v2, currently produces broken kernel)

Recursion's codegen produces ~721 buffer reads of the form
`auto xN = /*code=*/args[K][col * steps + ((cycle - kInvRate * D) & mask)];`.
All buffer reads have constant column indices and depend only on the
function parameters (cycle, steps, mask, kInvRate). They do not depend on
any other SSA xid.

The original PoC (May 2026) split the body into N noinline sub-functions
and re-emitted each sub-function's transitive SSA closure at the top.
Because the 721 buffer reads were referenced THROUGHOUT the body, every
sub-function pulled most of them into its closure, inflating the .so from
13MB to 59.7MB (4.6x). The resulting kernel triggered
UR_RESULT_ERROR_DEVICE_LOST at runtime.

The hoist design:

1. The entry `poly_fp` allocates `Fp bufs[NUM]` on the stack.
2. The entry emits all NUM buffer reads as `bufs[i] = args[K][...];` in
   source order.
3. Each sub-fn receives `Fp* bufs` instead of `Fp** args` (no per-buffer
   pointers; sub-fns never touch args[] directly).
4. References to buffer-read xids inside sub-fns rewrite to `bufs[slot]`.
5. Buffer-read xids are excluded from the SSA closure walk's
   external_needed set (they live in `bufs[]`, not as locals).

Effect: each buffer read exists exactly once (in the entry), and each
sub-fn references it via a single LSC load through `bufs`. This preserves
the "split-for-IGC-compile" benefit while keeping total code size
bounded near the original 13MB.

## Bit-exactness

The transformation is structural: buffer reads happen in source order in
the entry (cycle/steps/mask are parameters, atomic evaluation). Sub-fns
reference the cached values rather than reloading. With no side effects
in args[K][...] indexing, the rewrite preserves semantics.

## Usage (called from build.rs)

  python3 split_poly_fp.py <in_path> <out_dir> [--n 10] [--num-files 2]

Produces `rust_poly_fp_0.cpp`, ..., `rust_poly_fp_<num_files-1>.cpp`.
The last file contains the entry `poly_fp`.
"""

import argparse
import os
import re
import sys

SSA_DEF_RE = re.compile(r'^  (?:constexpr )?(?:auto|FpExt|Fp)\s+x(\d+)\s*(?:=|\()')
POLY_FP_DEF_RE = re.compile(r'^FpExt\s+poly_fp\(.*\)\s*\{\s*$')
NS_CLOSE_RE = re.compile(r'^\}\s*//\s*namespace')
XREF_RE = re.compile(r'\bx(\d+)\b')

# Buffer reads: `  auto xN = /*name=*/args[K][index_expr];`
BUFFER_READ_RE = re.compile(r'^  auto x(\d+)\s*=\s*/\*\w+=\*/args\[(\d+)\]\[(.+?)\];\s*$')

HEADER_TEMPLATE = """\
// Copyright 2025 RISC Zero, Inc.
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

// This code is automatically generated by split_poly_fp.py - do not edit.

#include "fp.h"
#include "fpext.h"

#include <cstdint>

constexpr size_t kInvRate = 4;

// clang-format off
namespace risc0::circuit::recursion {

"""

NS_CLOSE = """\
} // namespace risc0::circuit::recursion
// clang-format on
"""


def parse_file(path):
    with open(path) as f:
        lines = f.readlines()
    n = len(lines)
    def_open = next(i for i, ln in enumerate(lines) if POLY_FP_DEF_RE.match(ln))
    mask_idx = next(i for i in range(def_open, n) if 'size_t mask' in lines[i])
    header_end = mask_idx + 1
    p = header_end
    while p < n:
        s = lines[p].strip()
        if s.startswith('//') or s.startswith('constexpr'):
            p += 1
            continue
        break
    prologue_end = p
    t_idx = next(i for i in range(n - 1, -1, -1) if NS_CLOSE_RE.match(lines[i]))
    bend = t_idx - 1
    while bend > 0 and lines[bend].strip() == '':
        bend -= 1
    body_end = bend
    return {
        'lines': lines,
        'header_end': header_end,
        'prologue_end': prologue_end,
        'body_start': prologue_end,
        'body_end': body_end,
    }


def find_buffer_reads(lines, body_start, body_end):
    """Return ordered list of (xid, def_line, buffer_idx, index_expr)."""
    out = []
    for li in range(body_start, body_end):
        m = BUFFER_READ_RE.match(lines[li])
        if m:
            out.append((int(m.group(1)), li, int(m.group(2)), m.group(3)))
    return out


def build_ssa_table(lines, body_start, body_end):
    table = {}
    last_use = {}
    for li in range(body_start, body_end):
        ln = lines[li]
        m = SSA_DEF_RE.match(ln)
        if not m:
            continue
        xid = int(m.group(1))
        if ln.startswith('  FpExt'):
            t = 'FpExt'
        elif ln.startswith('  auto'):
            t = 'auto'
        else:
            t = 'Fp'
        eq = ln.find('=')
        rhs = ln[eq + 1:] if eq >= 0 else ln
        deps = {int(m2.group(1)) for m2 in XREF_RE.finditer(rhs) if int(m2.group(1)) != xid}
        table[xid] = {'def_line': li, 'type': t, 'deps': deps}
    for li in range(body_start, body_end):
        ln = lines[li]
        eq = ln.find('=')
        rhs = ln[eq + 1:] if eq >= 0 else ln
        for m in XREF_RE.finditer(rhs):
            uid = int(m.group(1))
            if uid in table:
                last_use[uid] = li
    for xid, info in table.items():
        info['last_use_line'] = last_use.get(xid, info['def_line'])
    return table


def pick_cuts(table, body_start, body_end, n_funcs, window=200, hoisted_ids=None):
    """Pick N-1 cuts that minimise live FpExt count near each geometric target.
    Skip hoisted (buffer-read) def lines as candidates."""
    if hoisted_ids is None:
        hoisted_ids = set()
    span = body_end - body_start
    chunk = span // n_funcs
    def_lines = sorted(info['def_line']
                       for xid, info in table.items() if xid not in hoisted_ids)
    def_set = set(def_lines)
    cuts = []
    prev = body_start
    for k in range(1, n_funcs):
        target = body_start + k * chunk
        lo, hi = max(prev + 1, target - window), min(body_end - 1, target + window)
        best = None
        for L in range(lo, hi + 1):
            if L not in def_set:
                continue
            n_live = sum(1 for xid, info in table.items()
                         if info['type'] == 'FpExt' and xid not in hoisted_ids
                         and info['def_line'] < L and info['last_use_line'] >= L)
            score = (n_live, abs(L - target))
            if best is None or score < best[0]:
                best = (score, L)
        cuts.append(best[1] if best else target)
        prev = cuts[-1]
    return cuts


def rewrite_xrefs(line, name_map):
    """Rewrite xN -> name_map[xN] for any xN in name_map."""
    if not name_map:
        return line
    def sub(m):
        xid = int(m.group(1))
        return name_map.get(xid, m.group(0))
    return XREF_RE.sub(sub, line)


def compute_closure(table, region_xids):
    needed = set()
    work = list(region_xids)
    while work:
        x = work.pop()
        if x in needed:
            continue
        needed.add(x)
        info = table.get(x)
        if not info:
            continue
        for d in info['deps']:
            if d not in needed:
                work.append(d)
    return needed


def emit_subfn(buf, region_idx, region_lo, region_hi, lines, table,
               prologue_lines, num_funcs, fpext_imports, callee_imports,
               hoisted_map):
    """Emit recursion_v2_<region_idx>.

    fpext_imports: list of FpExt xids that flow IN (rewritten to farg0..fargN).
    callee_imports: list of FpExt xids the callee K-1 expects as imports.
    hoisted_map: dict {xid -> 'bufs[slot]'} for buffer-read xids.
    """
    name_map = dict(hoisted_map)
    for i, xid in enumerate(fpext_imports):
        name_map[xid] = f'farg{i}'
    # Signature: drops Fp** args, adds Fp* bufs.
    sig_parts = ['size_t cycle', 'size_t steps', 'FpExt* poly_mix', 'Fp* bufs']
    sig_parts.extend(f'FpExt farg{i}' for i in range(len(fpext_imports)))
    sig = (f'__attribute__((noinline)) FpExt recursion_v2_{region_idx}('
           + ', '.join(sig_parts) + ')')
    buf.append(sig + ' {\n')
    buf.append('  size_t mask = steps - 1;\n')
    buf.extend(prologue_lines)
    referenced = set()
    for li in range(region_lo, region_hi):
        ln = lines[li]
        eq = ln.find('=')
        rhs = ln[eq + 1:] if eq >= 0 else ln
        for m in XREF_RE.finditer(rhs):
            referenced.add(int(m.group(1)))
    closure = compute_closure(table, referenced)
    region_defs = {xid for xid, info in table.items()
                   if region_lo <= info['def_line'] < region_hi}
    fpext_import_set = set(fpext_imports)
    hoisted_set = set(hoisted_map.keys())
    # External deps to re-emit: closure minus region-local, FpExt imports
    # (passed as fargK), and hoisted xids (read via bufs[]).
    to_emit_external = closure - region_defs - fpext_import_set - hoisted_set
    to_emit_external_sorted = sorted(
        (xid for xid in to_emit_external if xid in table),
        key=lambda x: table[x]['def_line']
    )
    for xid in to_emit_external_sorted:
        dl = table[xid]['def_line']
        buf.append(rewrite_xrefs(lines[dl], name_map))
    for li in range(region_lo, region_hi):
        ln = lines[li]
        # Skip buffer-read defs in the region - they're hoisted to entry.
        if BUFFER_READ_RE.match(ln):
            continue
        buf.append(rewrite_xrefs(ln, name_map))
    if region_idx > 0:
        names = []
        for xid in callee_imports:
            if xid in region_defs:
                names.append(f'x{xid}')
            elif xid in name_map:
                names.append(name_map[xid])
            else:
                names.append(f'/* MISSING x{xid} */')
        fwd = ', '.join(names)
        call = (f'  return recursion_v2_{region_idx - 1}(cycle, steps, '
                f'poly_mix, bufs')
        if fwd:
            call += f', {fwd}'
        call += ');\n'
        buf.append(call)
    else:
        last_fpext = max(
            (xid for xid, info in table.items()
             if info['type'] == 'FpExt' and info['def_line'] < region_hi),
            key=lambda x: table[x]['def_line']
        )
        buf.append(f'  return x{last_fpext};\n')
    buf.append('}\n\n')


def emit_poly_fp(buf, prologue_lines, head_idx, buffer_reads):
    """Entry: prologue + bufs[] population + call to head sub-fn."""
    buf.append('FpExt poly_fp(size_t cycle, size_t steps, FpExt* poly_mix, '
               'Fp** args) {\n')
    buf.append('  size_t mask = steps - 1;\n')
    buf.extend(prologue_lines)
    n_bufs = len(buffer_reads)
    buf.append(f'  Fp bufs[{n_bufs}];\n')
    for slot, (xid, _, buf_idx, idx_expr) in enumerate(buffer_reads):
        buf.append(f'  bufs[{slot}] = args[{buf_idx}][{idx_expr}];\n')
    buf.append(f'  return recursion_v2_{head_idx}(cycle, steps, poly_mix, '
               f'bufs);\n')
    buf.append('}\n\n')


def emit_forward_decls(buf, num_funcs, fpext_imports_per_region):
    for region_idx in range(num_funcs - 1, -1, -1):
        n_in = len(fpext_imports_per_region[region_idx])
        parts = ['size_t cycle', 'size_t steps', 'FpExt* poly_mix', 'Fp* bufs']
        parts.extend(f'FpExt farg{i}' for i in range(n_in))
        buf.append(f'FpExt recursion_v2_{region_idx}('
                   + ', '.join(parts) + ');\n')
    buf.append('FpExt poly_fp(size_t cycle, size_t steps, FpExt* poly_mix, '
               'Fp** args);\n\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('in_path')
    ap.add_argument('out_dir')
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--num-files', type=int, default=2)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    parsed = parse_file(args.in_path)
    lines = parsed['lines']
    body_start = parsed['body_start']
    body_end = parsed['body_end']
    prologue_lines = lines[parsed['header_end']:parsed['prologue_end']]
    print(f'[split_poly_fp] body lines {body_start}..{body_end} '
          f'({body_end-body_start} lines)', file=sys.stderr)

    buffer_reads = find_buffer_reads(lines, body_start, body_end)
    hoisted_map = {xid: f'bufs[{slot}]'
                   for slot, (xid, _, _, _) in enumerate(buffer_reads)}
    hoisted_set = set(hoisted_map.keys())
    print(f'[split_poly_fp] hoisting {len(buffer_reads)} buffer reads',
          file=sys.stderr)

    table = build_ssa_table(lines, body_start, body_end)
    print(f'[split_poly_fp] SSA defs: {len(table)} '
          f'(FpExt={sum(1 for i in table.values() if i["type"]=="FpExt")}, '
          f'hoisted={len(hoisted_set)})',
          file=sys.stderr)

    cuts = pick_cuts(table, body_start, body_end, args.n, hoisted_ids=hoisted_set)
    region_los = [body_start] + cuts + [body_end]

    print(f'[split_poly_fp] cuts:', file=sys.stderr)
    for i, c in enumerate(cuts):
        print(f'  cut[{i}] @ line {c}', file=sys.stderr)

    fpext_imports = [None] * args.n
    for s in range(args.n):
        region_idx = args.n - 1 - s
        lo = region_los[s]
        hi = region_los[s + 1]
        imports = sorted(
            (xid for xid, info in table.items()
             if info['type'] == 'FpExt' and info['def_line'] < lo
             and info['last_use_line'] >= lo),
            key=lambda x: table[x]['def_line']
        )
        fpext_imports[region_idx] = imports
        n_lines = hi - lo
        print(f'  region S={s} (rec_v2_{region_idx}) lines [{lo}..{hi}) '
              f'({n_lines} lines) FpExt imports={len(imports)}',
              file=sys.stderr)

    if args.dry_run:
        return

    os.makedirs(args.out_dir, exist_ok=True)

    region_idxs = list(range(args.n - 1, -1, -1))
    per_file = [args.n // args.num_files] * args.num_files
    for i in range(args.n % args.num_files):
        per_file[i] += 1

    cursor = 0
    for f_idx in range(args.num_files):
        buf = [HEADER_TEMPLATE]
        emit_forward_decls(buf, args.n, fpext_imports)
        my_regions = region_idxs[cursor:cursor + per_file[f_idx]]
        cursor += per_file[f_idx]
        for region_idx in my_regions:
            s = args.n - 1 - region_idx
            lo = region_los[s]
            hi = region_los[s + 1]
            callee_imports = fpext_imports[region_idx - 1] if region_idx > 0 else []
            emit_subfn(buf, region_idx, lo, hi, lines, table, prologue_lines,
                       args.n, fpext_imports[region_idx], callee_imports,
                       hoisted_map)
        if f_idx == args.num_files - 1:
            emit_poly_fp(buf, prologue_lines, args.n - 1, buffer_reads)
        buf.append(NS_CLOSE)
        out_path = os.path.join(args.out_dir, f'rust_poly_fp_{f_idx}.cpp')
        text = ''.join(buf)
        with open(out_path, 'w') as f:
            f.write(text)
        print(f'[split_poly_fp] wrote {out_path} '
              f'({text.count(chr(10))} lines)', file=sys.stderr)


if __name__ == '__main__':
    main()
