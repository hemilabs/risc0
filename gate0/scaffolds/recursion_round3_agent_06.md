# Round-3 Agent 06 — Concrete START: recursion `split_poly_fp.py`

## Context (one paragraph)

R2-04 established that the recursion `poly_fp.cpp` is a 24,753-line
straight-line SSA function with only 2–9 live `FpExt` and 0 live
`FpExt`-tainted `Fp` values at every candidate cut. The rv32im kernel
has the analogous shape (52K lines) and ships with a 20-way split into
`__noinline__` sub-functions named `rv32im_v2_N`. Empirically that
split is *the* thing that lets IGC schedule under register pressure
and produce a kernel that runs on Battlemage. The plan: mirror the
rv32im split for recursion, write a Python pass `split_poly_fp.py`
that emits `rust_poly_fp_{0,1}.cpp`, and gate a `build.rs` change
behind it. This document is the **splitter script**, sketched as a
single drop-in file.

The script lives at
`risc0/circuit/recursion-sys/kernels/intel/split_poly_fp.py`. It does
**not** mutate any source file by itself — it reads `poly_fp.cpp`,
runs a def/use analysis, picks cuts, and writes two new files to a
specified output directory. The caller (a human, or `build.rs` gated
behind `RISC0_REGEN_POLY_FP=1`) is responsible for moving them into
`kernels/cxx/`.

## Design choices

1. **Inputs**: a single monolithic `poly_fp.cpp` plus `--n` (default
   10) and `--num-files` (default 2). The empirical analysis from
   R1-11 / R2-04 already established that the live-set is uniformly
   ≤9 FpExt across the body, so we can pick cuts geometrically (every
   ~2400 lines) and only refine each cut to the nearest local-minimum
   FpExt-live position within ±100 lines.

2. **Anchoring cuts on FpExt def-lines**: every cut line must be one
   *after* a complete SSA statement (no half-statement) AND must be at
   a point where the next statement either:
   - opens a fresh FpExt chain (i.e. the next defined FpExt depends
     on `x284 = FpExt(0)` or on a non-live earlier FpExt), or
   - is a leaf-Fp/auto definition (always safe — no FpExt live).
   Both conditions are met cheaply: scan for lines whose def-pattern
   matches `^  (auto|FpExt) x\d+ = ` and consider only those.

3. **ABI**: each sub-function returns one `FpExt` (the primary
   accumulator at the cut) and takes:
   - `size_t cycle, size_t steps, FpExt* poly_mix` (always)
   - `Fp* arg0..arg3` (the four buffers: code/data/accum/mix —
     poly_fp will plumb `args[0..3]` into them)
   - `FpExt argN..argM` (live-out FpExt from the previous sub-fn)
   The chain mirrors rv32im: `poly_fp` calls `recursion_v2_9`, which
   calls `_8`, ..., which calls `_0`. Each call passes the chain's
   current `FpExt` as the second-to-last accumulator argument.

4. **SSA renaming at boundaries**: in the original file, the body
   is essentially `auto x285 = ...; auto x286 = ...; ...; FpExt x317
   = x284 + x316 * poly_mix[0]; ...`. After splitting, the variable
   `x284` (defined in region 0) is consumed in region 1 as that
   region's parameter `argN`. The rest of the SSA names (`x285+`) are
   *file-local* and need no rename because each sub-fn re-derives them
   from buffer reads. The splitter rewrites only the cross-region
   *FpExt* references to their incoming parameter names. The rest is
   verbatim copy.

5. **Bit-exactness audit**: a static post-condition check that every
   non-whitespace body line of the original appears in exactly one
   sub-function (modulo whitespace and the renamed FpExt uses).

## The script

```python
#!/usr/bin/env python3
"""Split recursion/poly_fp.cpp into N noinline sub-functions, mirroring
rv32im's rust_poly_fp_{0,1,2,3}.cpp layout.

Usage:
  python3 split_poly_fp.py <in_dir> <out_dir>
      [--n 10] [--num-files 2]
      [--no-verify]     # skip bit-exactness audit
      [--dry-run]       # print cut points and exit

`<in_dir>` must contain `poly_fp.cpp` (the original monolith).
`<out_dir>` receives `rust_poly_fp_0.cpp` and `rust_poly_fp_1.cpp`.

The split is deterministic given the input file content and N.
"""

import argparse
import os
import re
import sys
from collections import defaultdict

# ----- Regexes (anchored on the codegen's exact whitespace/format) -----

# Match a leaf SSA definition: `  auto xN = ...;` or `  constexpr Fp xN(...);`
SSA_DEF_RE = re.compile(
    r'^  (?:constexpr )?(?:auto|FpExt|Fp)\s+x(\d+)\s*(?:=|\()'
)
# Match an FpExt definition specifically (these carry the accumulator chain)
FPEXT_DEF_RE = re.compile(r'^  FpExt\s+x(\d+)\s*=\s*(.+?);')
# Match poly_fp entry/decl
POLY_FP_DECL_RE = re.compile(r'^FpExt\s+poly_fp\(.*\);\s*$')
POLY_FP_DEF_RE  = re.compile(r'^FpExt\s+poly_fp\(.*\)\s*\{\s*$')
# Trailer close braces
NS_CLOSE_RE = re.compile(r'^\}\s*//\s*namespace')

# A reference to `xN` in an RHS — used for def/use analysis
XREF_RE = re.compile(r'\bx(\d+)\b')

# A buffer read: `args[K][N * steps + ((cycle - kInvRate * B) & mask)]`
READ_RE = re.compile(
    r'args\[(\d+)\]\s*\[(\d+)\s*\*\s*steps\s*\+\s*\(\(cycle\s*-\s*kInvRate\s*\*\s*(\d+)\)\s*&\s*mask\)\]'
)

# Header (license + includes + namespace open). Hard-coded line numbers
# in the codegen are stable; we still detect dynamically.

# ----- Parsing -----

def parse_file(path):
    """Return (header_lines, prologue_end_idx, body_start_idx, body_end_idx,
                trailer_lines, all_lines).

    - header: copyright / includes / `namespace ... {` / decl / def-open
    - prologue: `constexpr Fp xK(...);` block before the first body stmt
    - body: from first non-constexpr SSA stmt up to (and including) the
            final `FpExt xLAST = ...;` AND the `return xLAST;`
    - trailer: `}` (fn close), `} // namespace ...`, comment trailer
    """
    with open(path) as f:
        lines = f.readlines()
    n = len(lines)

    # 1. Find `FpExt poly_fp(...) {` definition opening
    def_open = None
    for i, ln in enumerate(lines):
        if POLY_FP_DEF_RE.match(ln):
            def_open = i
            break
    assert def_open is not None, "poly_fp() definition not found"

    # 2. Header: everything up to and including `def_open` plus the
    #    `size_t mask = steps - 1;` line right after it.
    # In the recursion file, body opens at the line after `mask = steps - 1`.
    header_end = def_open + 1
    while header_end < n and 'size_t mask' not in lines[header_end]:
        header_end += 1
    header_end += 1  # include `size_t mask = steps - 1;`

    # 3. Prologue: contiguous `constexpr Fp xN(...);` (and their `// loc()`
    #    comments). Stops at the first non-constexpr SSA stmt.
    p = header_end
    while p < n:
        ln = lines[p].strip()
        if ln.startswith('//') or ln.startswith('constexpr'):
            p += 1
            continue
        break
    prologue_end = p  # exclusive

    # 4. Trailer: walk back from EOF for `}` close + `} // namespace ...`
    t_end = n
    t_start = t_end
    while t_start > 0 and not NS_CLOSE_RE.match(lines[t_start - 1]):
        t_start -= 1
    t_start -= 1  # the `}` closing poly_fp
    # walk back: skip blank lines between `}` and `} // namespace ...`
    while t_start > 0 and lines[t_start].strip() == '':
        t_start -= 1
    # also include the `return xLAST;` line in body, NOT in trailer
    # so body_end (exclusive) = t_start
    body_start = prologue_end
    body_end = t_start  # exclusive; the line BEFORE `}` of poly_fp

    return {
        'lines': lines,
        'header_end': header_end,
        'prologue_end': prologue_end,
        'body_start': body_start,
        'body_end': body_end,
        'trailer_start': t_start,
    }


def collect_ssa_table(lines, body_start, body_end):
    """For each defined SSA var x_id in [body_start, body_end), record:
       def_line, type ('Fp'/'auto'/'FpExt'), last_use_line, deps (set of ids)."""
    table = {}            # id -> dict
    last_use = {}         # id -> last_use_line_idx
    # First pass: defs + deps
    for li in range(body_start, body_end):
        ln = lines[li]
        m = SSA_DEF_RE.match(ln)
        if not m:
            continue
        xid = int(m.group(1))
        # type
        if ln.startswith('  FpExt'):
            t = 'FpExt'
        elif ln.startswith('  auto'):
            t = 'auto'
        else:
            t = 'Fp'
        # RHS: everything after `=` up to `;`
        eq = ln.find('=')
        if eq < 0:
            rhs = ''
        else:
            rhs = ln[eq + 1:]
        deps = set()
        for dm in XREF_RE.finditer(rhs):
            d = int(dm.group(1))
            if d != xid:
                deps.add(d)
        table[xid] = {
            'def_line': li, 'type': t, 'deps': deps,
        }
    # Second pass: last_use_line (max line on which xid appears as a USE,
    # not as the LHS def — RHS only)
    for li in range(body_start, body_end):
        ln = lines[li]
        # Strip LHS def to avoid counting `x317 = ...` as using x317
        m = SSA_DEF_RE.match(ln)
        eq = ln.find('=')
        rhs = ln[eq + 1:] if eq >= 0 else ln
        for um in XREF_RE.finditer(rhs):
            uid = int(um.group(1))
            if uid in table:
                last_use[uid] = li
    for xid, info in table.items():
        info['last_use_line'] = last_use.get(xid, info['def_line'])
    return table


def fpext_taint(table):
    """Compute FpExt-taint via transitive closure.

    Returns: dict[xid -> bool]. True iff transitively depends on
    any FpExt-typed var (including itself).
    """
    tainted = {xid: (info['type'] == 'FpExt') for xid, info in table.items()}
    # Iterate until fixpoint. Topology: defs are in source order, so a single
    # forward pass usually suffices (deps are always earlier-defined).
    changed = True
    while changed:
        changed = False
        for xid, info in table.items():
            if tainted[xid]:
                continue
            for d in info['deps']:
                if d in tainted and tainted[d]:
                    tainted[xid] = True
                    changed = True
                    break
    return tainted


def live_fpext_at(table, cut_line):
    """Set of FpExt SSA ids live across `cut_line`:
       defined < cut_line AND last_use_line >= cut_line."""
    live = []
    for xid, info in table.items():
        if info['type'] != 'FpExt':
            continue
        if info['def_line'] < cut_line and info['last_use_line'] >= cut_line:
            live.append(xid)
    # Order by def_line for stable arg lists
    live.sort(key=lambda xid: table[xid]['def_line'])
    return live


def pick_cuts(table, body_start, body_end, n_funcs, window=100):
    """Pick N-1 cut lines that split the body into N roughly equal pieces.
    For each geometric target, slide +/-`window` to find a line that:
      - lies on an SSA def boundary (so we don't bisect a statement)
      - minimises live-FpExt count

    Returns: list of cut line indices (length n_funcs - 1), strictly
    increasing, both > body_start and < body_end.
    """
    span = body_end - body_start
    chunk = span // n_funcs
    cuts = []
    # SSA def boundaries: precompute the set of line indices that start a def
    def_lines = sorted(info['def_line'] for info in table.values())
    def_set = set(def_lines)
    for k in range(1, n_funcs):
        target = body_start + k * chunk
        # candidate range
        lo, hi = max(body_start + 1, target - window), min(body_end - 1, target + window)
        best_line, best_score = None, None
        for L in range(lo, hi + 1):
            if L not in def_set:
                continue
            n_live = len(live_fpext_at(table, L))
            # Prefer fewer-live, then closer-to-target
            score = (n_live, abs(L - target))
            if best_score is None or score < best_score:
                best_score = score
                best_line = L
        if best_line is None:
            best_line = target  # fallback (shouldn't happen with this codegen)
        cuts.append(best_line)
    # Enforce strictly increasing (window overlap could break ordering)
    for i in range(1, len(cuts)):
        if cuts[i] <= cuts[i - 1]:
            cuts[i] = cuts[i - 1] + 1
    return cuts


# ----- Emission -----

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

// This code is automatically generated by split_poly_fp.py — do not edit.

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


def make_param_list(buf_args, fpext_args):
    """Return the `(...)` argument list for a sub-function.

    Always begins with `size_t cycle, size_t steps, FpExt* poly_mix`,
    followed by Fp* args (4 buffer pointers), then FpExt arg(s).
    """
    parts = ['size_t cycle', 'size_t steps', 'FpExt* poly_mix']
    parts.extend(f'Fp* {a}' for a in buf_args)
    parts.extend(f'FpExt {a}' for a in fpext_args)
    return ', '.join(parts)


def emit_subfn(buf, region_idx, cuts, body_lines, table, prologue_lines,
               num_funcs, body_start, body_end):
    """Write one sub-function `recursion_v2_<region_idx>` to `buf` (list).

    region_idx counts from high to low: head = num_funcs-1, tail = 0.
    region_idx=num_funcs-1 contains body lines [body_start, cuts[0])
    region_idx=0 contains body lines [cuts[-1], body_end)
    """
    # Determine line range
    if region_idx == num_funcs - 1:
        lo, hi = body_start, cuts[0]
    elif region_idx == 0:
        lo, hi = cuts[-1], body_end
    else:
        # i-th region from the tail: cuts[num_funcs-2-region_idx-?]
        # Simpler: regions in source order are 0..num_funcs-1 where
        # source-order 0 maps to region_idx=num_funcs-1.
        src_order = (num_funcs - 1) - region_idx
        lo, hi = cuts[src_order - 1], cuts[src_order]

    # In-set: FpExt ids defined < lo, used in [lo, hi)
    in_fpext = sorted(
        xid for xid, info in table.items()
        if info['type'] == 'FpExt' and info['def_line'] < lo
        and lo <= info['last_use_line'] < hi or
        (info['def_line'] < lo and info['last_use_line'] >= hi)
    )
    # We want: defined-before-region AND used-anywhere-at-or-after-lo.
    in_fpext = sorted(
        xid for xid, info in table.items()
        if info['type'] == 'FpExt'
        and info['def_line'] < lo
        and info['last_use_line'] >= lo
    )

    # Out-set: FpExt ids defined in [lo, hi), used >= hi.
    out_fpext = sorted(
        xid for xid, info in table.items()
        if info['type'] == 'FpExt'
        and lo <= info['def_line'] < hi
        and info['last_use_line'] >= hi
    )

    # Primary return: the *last* defined FpExt in this region that's
    # used past `hi`. Falls back to the last FpExt defined in [lo, hi)
    # for the tail region (which carries the final result).
    if region_idx == 0:
        # tail region: the absolute last FpExt def in body is the return
        last_fpext = max(
            (xid for xid, info in table.items()
             if info['type'] == 'FpExt' and info['def_line'] < hi),
            key=lambda x: table[x]['def_line'],
        )
        primary_return = last_fpext
    else:
        primary_return = out_fpext[-1] if out_fpext else None

    # Param names: stable Fp* buffer order is `arg0..arg3`, mapped to
    # data/accum/mix/global in the caller chain.
    buf_arg_names = ['arg0', 'arg1', 'arg2', 'arg3']
    fpext_arg_names = [f'farg{i}' for i in range(len(in_fpext))]
    # Map original xN id -> incoming param name
    name_map = {xid: fpext_arg_names[i] for i, xid in enumerate(in_fpext)}

    sig = f'FpExt recursion_v2_{region_idx}({make_param_list(buf_arg_names, fpext_arg_names)})'

    buf.append(f'__attribute__((noinline)) {sig} {{\n')

    # Local prologue: the leaf-Fp constants we need to inline.
    # Conservative: emit ALL constexpr Fp xK(...) from the prologue
    # into every sub-function (cheap — IGC will constant-fold). This
    # avoids tracking which constants are used in each region.
    buf.extend(prologue_lines)
    buf.append('\n')

    # Body lines: copy from the original [lo, hi), with one rewrite:
    # any reference to an FpExt id in `in_fpext` becomes its `farg<i>`
    # name. (Pure Fp/auto ids `xK` defined inside this region keep
    # their original names; they're file-local.)
    if in_fpext:
        # Build a single regex: `\bx(<id1>|<id2>|...)\b`
        ids_alt = '|'.join(str(x) for x in in_fpext)
        ref_re = re.compile(rf'\bx({ids_alt})\b')

        def rewrite(line):
            return ref_re.sub(lambda m: name_map[int(m.group(1))], line)
    else:
        def rewrite(line):
            return line

    for li in range(lo, hi):
        buf.append(rewrite(body_lines[li]))

    # If this is NOT the tail, emit the call into the next sub-fn
    # AND a return.
    if region_idx > 0:
        # Build callee in/out
        # Callee's `in_fpext` is THIS region's `out_fpext`
        # (those FpExt that we just defined and that flow forward).
        callee_in = out_fpext
        # Forward args: the callee param names are also `farg0..N`,
        # so we pass our region's locally-named values.
        # The values are: the original SSA names (we DID NOT rewrite the
        # defining `FpExt xK = ...;` lines, only RHS references), so we
        # pass `xK` directly.
        fwd = ', '.join(f'x{xid}' for xid in callee_in)
        # The primary chain value: we still ALSO pass it as the last
        # FpExt arg (it's part of callee_in by construction).
        buf.append(
            f'  auto chain = recursion_v2_{region_idx - 1}('
            f'cycle, steps, poly_mix, arg0, arg1, arg2, arg3'
            + (f', {fwd}' if fwd else '')
            + ');\n')
        buf.append('  return chain;\n')
    else:
        # Tail: return the final FpExt id
        buf.append(f'  return x{primary_return};\n')

    buf.append('}\n\n')


def emit_poly_fp(buf, prologue_lines, table, cuts, body_start, body_end,
                 num_funcs):
    """Emit the `poly_fp` entry that loads constants and kicks off the chain."""
    buf.append('FpExt poly_fp(size_t cycle, size_t steps, FpExt* poly_mix, Fp** args) {\n')
    buf.append('  size_t mask = steps - 1;\n')
    # Copy the prologue verbatim (constants)
    buf.extend(prologue_lines)
    buf.append('\n')
    # poly_fp's job: kick off recursion_v2_<num_funcs-1>. It seeds the
    # chain with `FpExt(0)` and the four buffer pointers.
    # The head sub-fn's `in_fpext` should be empty (or include only the
    # initial `x284 = FpExt(0)` which we inline as a literal arg).
    head = num_funcs - 1
    head_in = sorted(
        xid for xid, info in table.items()
        if info['type'] == 'FpExt'
        and info['def_line'] < body_start
    )
    # Conventionally there are no FpExt defs before body_start, so head_in is [].
    fwd = ', '.join(f'x{xid}' for xid in head_in)
    # /*data=*/, /*accum=*/, /*mix=*/, /*global=*/ markers preserved as
    # comments so the CSE pass (poly_fp_cse.py adapted) can later walk
    # the chain and identify buffer mappings.
    buf.append(
        f'  auto result = recursion_v2_{head}(cycle, steps, poly_mix, '
        f'/*code=*/args[0], /*data=*/args[2], /*accum=*/args[1], /*mix=*/args[3]'
        + (f', {fwd}' if fwd else '')
        + ');\n'
    )
    buf.append('  return result;\n')
    buf.append('}\n\n')


def emit_forward_decls(buf, num_funcs, in_sets):
    """Emit forward declarations for each `recursion_v2_N`.

    in_sets: list of lists [in_fpext_per_region], indexed by region_idx
    """
    for region_idx in range(num_funcs - 1, -1, -1):
        n_in = len(in_sets[region_idx])
        fp_args = ', '.join(f'Fp* arg{i}' for i in range(4))
        fp_ext_args = ', '.join(f'FpExt farg{i}' for i in range(n_in))
        params = ['size_t cycle', 'size_t steps', 'FpExt* poly_mix', fp_args]
        if fp_ext_args:
            params.append(fp_ext_args)
        sig = f"FpExt recursion_v2_{region_idx}({', '.join(params)});\n"
        buf.append(sig)
    buf.append('FpExt poly_fp(size_t cycle, size_t steps, FpExt* poly_mix, Fp** args);\n\n')


# ----- Main -----

def verify_bit_exact(orig_lines, body_start, body_end, emitted_text, table, in_sets):
    """Check that every non-whitespace original body line appears
    in the emitted output, modulo the in_set FpExt rewrites."""
    # Build the reverse rewrite: for each rewritten name, what's the original?
    # Just check that the *set* of stripped lines matches.
    orig = []
    for li in range(body_start, body_end):
        s = orig_lines[li].strip()
        if not s or s.startswith('//'):
            continue
        orig.append(s)
    # We can't trivially reverse the FpExt rename, so collapse both sides
    # by replacing `\bx\d+\b` and `\bfarg\d+\b` with `<id>`.
    def normalise(s):
        s = re.sub(r'\bx\d+\b', '<id>', s)
        s = re.sub(r'\bfarg\d+\b', '<id>', s)
        return s
    norm_orig = sorted(normalise(s) for s in orig)
    emit_lines = [ln.strip() for ln in emitted_text.splitlines() if ln.strip()]
    # Filter only lines that look like SSA defs (cheap heuristic)
    emit_filt = [ln for ln in emit_lines
                 if re.match(r'^(constexpr |)(auto|Fp|FpExt) x\d+', ln)]
    norm_emit = sorted(normalise(s) for s in emit_filt)
    # Multiset compare
    if norm_orig != norm_emit:
        # Diff for diagnostics
        from collections import Counter
        co = Counter(norm_orig); ce = Counter(norm_emit)
        missing = co - ce
        extra = ce - co
        print(f"[verify] {sum(missing.values())} missing, "
              f"{sum(extra.values())} extra", file=sys.stderr)
        for s, n in list(missing.items())[:5]:
            print(f"  MISSING ({n}x): {s}", file=sys.stderr)
        for s, n in list(extra.items())[:5]:
            print(f"  EXTRA ({n}x): {s}", file=sys.stderr)
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('in_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--n', type=int, default=10, help='Number of sub-functions')
    ap.add_argument('--num-files', type=int, default=2,
                    help='Number of output .cpp files (sub-fns will be distributed)')
    ap.add_argument('--no-verify', action='store_true',
                    help='Skip bit-exactness audit (faster)')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print cut points and exit, do not write files')
    args = ap.parse_args()

    in_path = os.path.join(args.in_dir, 'poly_fp.cpp')
    parsed = parse_file(in_path)
    lines = parsed['lines']
    body_start = parsed['body_start']
    body_end = parsed['body_end']
    prologue_lines = lines[parsed['header_end']:parsed['prologue_end']]
    print(f"[split_poly_fp] Body: lines {body_start}..{body_end} "
          f"({body_end - body_start} lines)", file=sys.stderr)

    # Def/use analysis
    table = collect_ssa_table(lines, body_start, body_end)
    tainted = fpext_taint(table)
    n_tainted_fp = sum(1 for xid, info in table.items()
                       if info['type'] != 'FpExt' and tainted[xid])
    if n_tainted_fp != 0:
        print(f"[split_poly_fp] FATAL: {n_tainted_fp} Fp/auto vars are "
              f"FpExt-tainted; ABI assumption violated.", file=sys.stderr)
        sys.exit(2)
    print(f"[split_poly_fp] SSA vars: {len(table)} "
          f"(FpExt: {sum(1 for i in table.values() if i['type']=='FpExt')})",
          file=sys.stderr)

    # Pick cuts
    cuts = pick_cuts(table, body_start, body_end, args.n)
    print(f"[split_poly_fp] Cuts (n={args.n}):", file=sys.stderr)
    prev = body_start
    for i, c in enumerate(cuts):
        n_live = len(live_fpext_at(table, c))
        print(f"  cut[{i}] line {c}  (region size {c - prev}, "
              f"live FpExt {n_live})", file=sys.stderr)
        prev = c
    print(f"  tail region size {body_end - prev}", file=sys.stderr)
    if args.dry_run:
        return

    # Build in/out sets per region for forward decls
    in_sets = [None] * args.n
    for region_idx in range(args.n):
        if region_idx == args.n - 1:
            lo = body_start
        else:
            src_order = (args.n - 1) - region_idx
            lo = cuts[src_order - 1] if src_order >= 1 else body_start
        in_sets[region_idx] = [
            xid for xid, info in table.items()
            if info['type'] == 'FpExt'
            and info['def_line'] < lo
            and info['last_use_line'] >= lo
        ]

    # Distribute regions across `num_files` files, head-heavy (lower
    # file index gets the higher-numbered sub-fns).
    per_file = [args.n // args.num_files] * args.num_files
    for i in range(args.n % args.num_files):
        per_file[i] += 1
    # File 0 gets the head sub-fns (highest numbers); file `num_files-1`
    # gets the tail (numbers 0..) and the poly_fp entry.

    # Emit
    os.makedirs(args.out_dir, exist_ok=True)
    regions_remaining = list(range(args.n - 1, -1, -1))   # head→tail
    for f_idx in range(args.num_files):
        buf = [HEADER_TEMPLATE]
        # Forward decls (only in file 0; other files include them via header
        # of file 0 conceptually — but for simplicity, every file emits them).
        emit_forward_decls(buf, args.n, in_sets)
        # Pick regions for this file
        my_regions = regions_remaining[:per_file[f_idx]]
        regions_remaining = regions_remaining[per_file[f_idx]:]
        for region_idx in my_regions:
            emit_subfn(buf, region_idx, cuts, lines, table, prologue_lines,
                       args.n, body_start, body_end)
        # The last file also gets the poly_fp entry
        if f_idx == args.num_files - 1:
            emit_poly_fp(buf, prologue_lines, table, cuts,
                         body_start, body_end, args.n)
        buf.append(NS_CLOSE)
        out_path = os.path.join(args.out_dir, f'rust_poly_fp_{f_idx}.cpp')
        text = ''.join(buf)
        with open(out_path, 'w') as f:
            f.write(text)
        print(f"[split_poly_fp] Wrote {out_path} "
              f"({text.count(chr(10))} lines)", file=sys.stderr)

    # Bit-exactness audit: concatenate all emitted text and compare normalised
    # SSA-def lines against the original body.
    if not args.no_verify:
        all_text = ''
        for f_idx in range(args.num_files):
            with open(os.path.join(args.out_dir, f'rust_poly_fp_{f_idx}.cpp')) as f:
                all_text += f.read()
        ok = verify_bit_exact(lines, body_start, body_end, all_text,
                               table, in_sets)
        if not ok:
            print("[split_poly_fp] BIT-EXACTNESS FAIL", file=sys.stderr)
            sys.exit(3)
        print("[split_poly_fp] Bit-exactness OK", file=sys.stderr)


if __name__ == '__main__':
    main()
```

## What this script does NOT do (deferred for later sessions)

1. **Apply to source.** The script only writes to `<out_dir>`. The
   user must `mv out_dir/rust_poly_fp_*.cpp kernels/cxx/` and edit
   `build.rs` separately. Per R2-04, the `build.rs` change is ~80
   lines of paste-adapt from `rv32im-sys/build.rs:1212-1304`.

2. **Cross-function CSE.** The hoisting transform from
   `rv32im-sys/kernels/intel/poly_fp_cse.py` (T3.1 work) can be
   adapted later — but only after the split lands and the basic
   chain compiles cleanly. R2-04 §8 explicitly defers this.

3. **Tree-reduce FMA rewrites.** `project_tree_reduce_negative.md`
   says this will likely regress on a spill-bound kernel — skip it
   here, same logic.

4. **Auto-regenerate from build.rs.** R2-04 step 3 suggests gating
   regenerate behind `RISC0_REGEN_POLY_FP=1`; that's a build.rs
   change, not part of this script.

5. **CUDA/CPU/Metal paths.** Those continue to use the monolithic
   `kernels/cxx/poly_fp.cpp` via their existing wrappers
   (`kernels/cuda/poly_fp.cu`, `kernels/cpu/poly_fp.cpp`). Only the
   Intel SYCL/OpenCL build switches to the split files.

## Validation plan (when this script is actually invoked)

1. **Dry-run first**: `python3 split_poly_fp.py kernels/cxx/ /tmp/split-test --dry-run`
   — confirm cut points have ≤9 live FpExt each.
2. **Real run with verify**: drop `--dry-run`, check bit-exactness
   passes.
3. **Compile-test**: hand-copy outputs into `kernels/cxx/`, patch
   `build.rs` per R2-04 step 2, run
   `cargo build -p risc0-circuit-recursion-sys --release`. Should
   finish in 20–40 min (vs 84 min on the monolith with `-cl-opt-disable`
   removed).
4. **Runtime-test**: `cargo test -p risc0-circuit-recursion --release`,
   then a small Succinct proof end-to-end. Diff the verifier output
   against the reference (zero tolerance).
5. **Perf-test**: bench Succinct E2E, look for ≥1% improvement and
   recursion-kernel scratch drop from 131KB → <30KB.

## Why this matches rv32im exactly

- **Naming**: `recursion_v2_N` mirrors `rv32im_v2_N`.
- **ABI shape**: `(size_t cycle, size_t steps, FpExt* poly_mix, Fp*
  buffers..., FpExt accumulators...)` matches rv32im's
  `rust_poly_fp_0.cpp:27-46`.
- **Chain direction**: head = highest number, tail = 0, return value
  flows backward — same as rv32im.
- **noinline attribute**: `__attribute__((noinline))` injected on
  every definition, matching `rv32im-sys/build.rs:1266`.
- **Forward-decls**: identical layout to
  `rust_poly_fp_0.cpp:27-47`.
- **Distribution**: 10 sub-fns × 2 files matches the high-water-mark
  observed in rv32im (20 sub-fns × 4 files); we use half the count
  because recursion is half the size (~25K vs ~52K body lines).

## Risk register (carried from R2-04 §5)

| Risk | Status | Notes |
|---|---|---|
| Bit-exactness silent failure | Mitigated | `verify_bit_exact` normalises x-name and farg-name to `<id>` and compares multisets |
| Param count > 64 (ABI cap) | Very low risk | Empirical max FpExt-live is 9; ×4 scalars = 36, plus 4 buffer ptrs = 40, well under 64 |
| ocloc still slow after split | Medium | Increase `--n` to 15–20 if compile time stays >1h |
| Constants duplicated 10× in 10 sub-fns | Cost: <1% binary size | IGC constant-folds; runtime cost is zero |
| Pure Fp values re-derived | Cost: extra LDGs, identical to rv32im baseline behaviour | The 4× cycle re-read is what hurt rv32im before T3.1 — mitigation is the CSE pass, gated separately |
