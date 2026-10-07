#!/usr/bin/env python3
"""Cross-function CSE on shared args[buf][col*steps+back] reads in eval_check.

Background:
  - rust_poly_fp_*.cpp is auto-generated; the entry `poly_fp` calls a LINEAR chain
    of 20 sub-functions rv32im_v2_19 → 18 → 17 → ... → 0, all marked __noinline__.
  - Each sub-function independently re-issues global memory loads for shared
    columns of the four buffers (data, accum, mix, global). Empirically, columns
    48..102 are read by 16-19 different sub-functions independently.
  - T3.1 (Agent A #2): hoist the most-shared reads into poly_fp once, then pass
    them down the call chain as additional value parameters.

Approach (this script):
  1. Build a per-sub-function parameter map: which Fp* parameter is bound to
     which logical buffer (data/accum/mix/global), by tracing the call chain
     starting from poly_fp.
  2. Scan each sub-function body for reads `argK[N*steps+((cycle-kInvRate*B)&mask)]`
     and tally (buffer, N, B) tuples globally.
  3. Pick the top-N most-shared tuples (configurable).
  4. Emit a hoisted block in poly_fp that loads each into a local Fp scalar.
  5. Append the loaded scalars to every sub-function's call site (and signature),
     selectively along the chain — only sub-functions that USE the value, plus
     the ones in between (forwarders).
  6. Rewrite each sub-function body to use the new param instead of the read.

Constraints:
  - The transform must be bit-exact. Math is identity (same load, just hoisted).
  - The call chain is LINEAR so threading parameters down is straightforward.
  - All four files share the same forward declarations (in fp_0.cpp) — those
    declarations must be updated to match the new signatures.

Safety / risk:
  - Each hoisted value adds 1 scalar parameter to every sub-function on the path
    from poly_fp to its latest consumer. This grows the function signatures
    (~60 added params worst case) and may pressure the GPU stackcall ABI.
  - Adding more SSA values risks spill, which T3.2 taught us is catastrophic on
    this kernel. So we keep N small (configurable, default 32) and validate
    zebin private_size after each compile.
"""

import os
import re
import sys
import argparse
from collections import defaultdict, OrderedDict

# Read pattern: arg<digits>[<int>*steps+((cycle-kInvRate*<int>)&mask)]
# Capture: arg_idx, col, back-offset
# Also handle: arg<digits>[(cycle-...)&mask + N*steps]  — but in this codebase
# the codegen always writes col*steps first.
READ_RE = re.compile(
    r'arg(\d+)\[(\d+) \* steps \+ \(\(cycle - kInvRate \* (\d+)\) & mask\)\]'
)

# Same shape but for the entry poly_fp's hoisted reads: args[K][...].
READ_RE_ARGS = re.compile(
    r'args\[(\d+)\]\[(\d+) \* steps \+ \(\(cycle - kInvRate \* (\d+)\) & mask\)\]'
)

# Lookbehind guards used by apply_lsc_hints() so a second pass over an
# already-wrapped file is a no-op. Python's re module doesn't support
# variable-width lookbehind, so these are fixed-width "&" negative lookbehinds.
LSC_WRAPPED_READ_RE = re.compile(
    r'(?<!&)arg(\d+)\[(\d+) \* steps \+ \(\(cycle - kInvRate \* (\d+)\) & mask\)\]'
)
LSC_WRAPPED_READ_RE_ARGS = re.compile(
    r'(?<!&)args\[(\d+)\]\[(\d+) \* steps \+ \(\(cycle - kInvRate \* (\d+)\) & mask\)\]'
)

# Signature header for a sub-function definition (not declaration)
DEF_RE = re.compile(
    r'^FpExt rv32im_v2_(\d+)\((.*)\)\s*\{?\s*$'
)
# Forward declaration: same prefix but ends with ");"
DECL_RE = re.compile(
    r'^FpExt rv32im_v2_(\d+)\((.*)\);\s*$'
)
# poly_fp entry definition
POLY_FP_RE = re.compile(r'^FpExt poly_fp\(.*\)\s*\{?\s*$')
# poly_fp's call to rv32im_v2_19 with explicit /*data=*/, /*accum=*/, ... markers
POLY_FP_CALL_RE = re.compile(
    r'rv32im_v2_19\(cycle, steps, poly_mix,(.*)\);'
)
# Inter-sub-fn call: rv32im_v2_N(cycle, steps, poly_mix, ARGS);
CALL_RE = re.compile(
    r'(rv32im_v2_(\d+)\(cycle, steps, poly_mix, )(.*?)(\);)'
)


# Recognise parameter list in a signature: returns list of (typestr, name, is_fp_ptr).
PARAM_SPLIT_RE = re.compile(r',\s*')
PARAM_RE = re.compile(r'^(Fp\*|FpExt\*|Fp|FpExt)\s+(arg\d+)$')


def parse_signature_params(sig_body):
    """sig_body: the parenthesized parameter list (without parens).
    Returns list of (type, name)."""
    # The leading cycle/steps/poly_mix are common to every sub-fn; we skip them.
    raw_params = [p.strip() for p in sig_body.split(',')]
    out = []
    for p in raw_params:
        if p.startswith('size_t cycle') or p.startswith('size_t steps') \
                or p.startswith('FpExt* poly_mix'):
            continue
        m = PARAM_RE.match(p)
        if not m:
            # Could be a pointer with funny spacing
            mm = re.match(r'^(Fp\s*\*|FpExt\s*\*|Fp|FpExt)\s+(arg\d+)$', p)
            if mm:
                t = re.sub(r'\s+', '', mm.group(1))
                out.append((t, mm.group(2)))
                continue
            raise SystemExit(f"Cannot parse param: {p!r}")
        out.append((m.group(1), m.group(2)))
    return out


def split_call_args(call_args_str):
    """Split a call argument list by top-level commas (no nested parens here,
    but be defensive)."""
    out = []
    depth = 0
    cur = ''
    for c in call_args_str:
        if c == '(' or c == '[':
            depth += 1
            cur += c
        elif c == ')' or c == ']':
            depth -= 1
            cur += c
        elif c == ',' and depth == 0:
            out.append(cur.strip())
            cur = ''
        else:
            cur += c
    if cur.strip():
        out.append(cur.strip())
    return out


def find_fn_bodies(lines):
    """Return list of (start_idx, end_idx_inclusive, header_match) for every
    sub-function definition (not declaration) and for poly_fp."""
    out = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m_def = DEF_RE.match(line)
        m_poly = POLY_FP_RE.match(line)
        if m_def or m_poly:
            # Verify it's a definition, not declaration (ends with '{' or
            # has '{' on next line). Look ahead for the matching brace count.
            # Use brace counting.
            # Find the opening '{'.
            depth = 0
            j = i
            found_open = False
            while j < n:
                for c in lines[j]:
                    if c == '{':
                        depth += 1
                        found_open = True
                    elif c == '}':
                        depth -= 1
                        if found_open and depth == 0:
                            break
                if found_open and depth == 0:
                    break
                j += 1
            if not found_open:
                # forward declaration line — skip
                i += 1
                continue
            out.append((i, j, m_def, m_poly))
            i = j + 1
        else:
            i += 1
    return out


def trace_buffer_map_global(file_state):
    """Trace buffer map across all 4 files at once.

    file_state: dict[fname -> {'lines': [...], 'fn_starts': [...]}]
    Returns:
      bufmap: dict[fn_num -> {paramName -> bufName}]
      sub_fn_params: dict[fn_num -> list[(type, name)]] (signatures from defs)
      poly_fp_loc: (fname, start, end)
      sub_fn_loc: dict[fn_num -> (fname, start, end)]
    """
    sub_fn_params = {}
    sub_fn_loc = {}
    poly_fp_loc = None
    for fname, st in file_state.items():
        for s, e, mdef, mpoly in st['fn_starts']:
            if mpoly:
                poly_fp_loc = (fname, s, e)
            elif mdef:
                fn_num = int(mdef.group(1))
                sub_fn_loc[fn_num] = (fname, s, e)
                sub_fn_params[fn_num] = parse_signature_params(mdef.group(2))
    assert poly_fp_loc is not None, "poly_fp not found"

    # Initialise bufmap[19] from poly_fp's call
    pf_fname, pf_s, pf_e = poly_fp_loc
    pf_lines = file_state[pf_fname]['lines']
    fp_call_args_str = None
    for ln in pf_lines[pf_s:pf_e + 1]:
        m = POLY_FP_CALL_RE.search(ln)
        if m:
            fp_call_args_str = m.group(1)
            break
    assert fp_call_args_str is not None, "rv32im_v2_19 call from poly_fp not found"
    call_args = split_call_args(fp_call_args_str)
    v19_params = sub_fn_params[19]
    bufmap = {19: {}}
    for i, arg in enumerate(call_args):
        if i >= len(v19_params):
            break
        pname = v19_params[i][1]
        for buf, marker in (('data', '/*data=*/'), ('accum', '/*accum=*/'),
                            ('mix', '/*mix=*/'), ('global', '/*global=*/')):
            if marker in arg:
                bufmap[19][pname] = buf
                break

    # Walk the chain: caller's bufmap propagates to callee through arg names
    for caller in range(19, 0, -1):
        callee = caller - 1
        if caller not in sub_fn_loc or callee not in sub_fn_loc:
            continue
        c_fname, c_s, c_e = sub_fn_loc[caller]
        c_lines = file_state[c_fname]['lines']
        # Find call site inside caller
        call_args_str = None
        call_re = re.compile(
            rf'rv32im_v2_{callee}\(cycle, steps, poly_mix, (.*?)\)'
        )
        for li in range(c_s, c_e + 1):
            m = call_re.search(c_lines[li])
            if m:
                call_args_str = m.group(1)
                break
        if call_args_str is None:
            continue
        call_args = split_call_args(call_args_str)
        callee_params = sub_fn_params[callee]
        bufmap[callee] = {}
        for i, arg_expr in enumerate(call_args):
            if i >= len(callee_params):
                break
            ptype, pname = callee_params[i]
            if ptype != 'Fp*':
                continue
            arg_expr = arg_expr.strip()
            if arg_expr in bufmap.get(caller, {}):
                bufmap[callee][pname] = bufmap[caller][arg_expr]

    return bufmap, sub_fn_params, poly_fp_loc, sub_fn_loc


def count_reads(lines, sub_fn_ranges, sub_fn_params, bufmap):
    """Count (buffer, col, back) tuples per sub-function, returning:
       counts[(buffer, col, back)] = set(fn_nums that use it)
       and the raw list of (fn_num, line_idx, col, back, buf, argname) usages."""
    counts = defaultdict(set)
    usages = []  # (fn_num, line_idx, col, back, buf, argname)
    # Build a fast paramName -> buf lookup per sub-fn
    for fn_num, (st, en) in sub_fn_ranges.items():
        pmap = bufmap.get(fn_num, {})
        for li in range(st, en + 1):
            line = lines[li]
            for m in READ_RE.finditer(line):
                arg_idx = int(m.group(1))
                col = int(m.group(2))
                back = int(m.group(3))
                argname = f'arg{arg_idx}'
                buf = pmap.get(argname)
                if buf is None:
                    # Sub-fn uses a Fp* not derived from the four root buffers
                    # (shouldn't happen in eval_check; skip).
                    continue
                key = (buf, col, back)
                counts[key].add(fn_num)
                usages.append((fn_num, li, col, back, buf, argname))
    return counts, usages


def analyze_files(in_dir):
    """Load all 4 files and trace buffer maps globally."""
    files = sorted(os.listdir(in_dir))
    files = [f for f in files if re.match(r'rust_poly_fp_\d+\.cpp$', f)]
    per_file = {}
    for fname in files:
        path = os.path.join(in_dir, fname)
        with open(path) as f:
            lines = f.readlines()
        fn_starts = find_fn_bodies(lines)
        per_file[fname] = {'lines': lines, 'fn_starts': fn_starts}

    bufmap, sub_fn_params, poly_fp_loc, sub_fn_loc = \
        trace_buffer_map_global(per_file)

    # Count reads per file
    agg_counts = defaultdict(set)
    for fname, st in per_file.items():
        lines = st['lines']
        sub_fn_ranges = {}
        for s, e, mdef, mpoly in st['fn_starts']:
            if mdef:
                fn_num = int(mdef.group(1))
                sub_fn_ranges[fn_num] = (s, e)
        st['sub_fn_ranges'] = sub_fn_ranges
        counts, usages = count_reads(lines, sub_fn_ranges, sub_fn_params, bufmap)
        st['usages'] = usages
        for k, fns in counts.items():
            agg_counts[k].update(fns)

    return agg_counts, per_file, bufmap, sub_fn_params, poly_fp_loc, sub_fn_loc


def transform(in_dir, out_dir, top_n=16, min_share=10):
    """Apply the CSE hoist transform.

    in_dir: directory with the original rust_poly_fp_*.cpp files.
    out_dir: where to write the transformed files (also copies non-transformed files).
    top_n: hoist at most this many tuples.
    min_share: only hoist tuples shared by at least this many sub-fns.
    """
    agg, per_file, bufmap, sub_fn_params, poly_fp_loc, sub_fn_loc = \
        analyze_files(in_dir)
    # Rank by share count
    items = sorted(agg.items(), key=lambda kv: (-len(kv[1]),
                                                 kv[0][0], kv[0][1], kv[0][2]))
    hoist_keys = []
    for key, fns in items:
        if len(fns) < min_share:
            break
        if len(hoist_keys) >= top_n:
            break
        hoist_keys.append((key, fns))
    if not hoist_keys:
        print("No tuples meet the share threshold; nothing to do.", file=sys.stderr)
        os.makedirs(out_dir, exist_ok=True)
        for fname in os.listdir(in_dir):
            src = os.path.join(in_dir, fname)
            dst = os.path.join(out_dir, fname)
            if os.path.isfile(src):
                with open(src, 'rb') as fin, open(dst, 'wb') as fout:
                    fout.write(fin.read())
        return 0

    # Build a hoisted-name table: a fixed canonical name per (buf, col, back).
    # Using `_h_<buf>_<col>_<back>` for legibility.
    def hname(buf, col, back):
        return f'_h_{buf}_{col}_{back}'

    # Build per-sub-fn lists:
    #   needed[fn_num] = list of (key, hname) the sub-fn needs to RECEIVE
    #   used[fn_num]   = list of (key, hname) the sub-fn USES internally
    # A sub-fn USES h if h is in `fns` for that key. It needs to RECEIVE h iff
    # some fn ≤ fn_num in the call chain USES h (it's a forwarder).
    used = defaultdict(list)
    needed = defaultdict(list)
    # Lowest user (smallest fn_num) along the chain for each hoisted key
    for key, fns in hoist_keys:
        # Every sub-fn that uses key
        for fn_num in fns:
            used[fn_num].append(key)
        deepest = min(fns)
        # Every sub-fn from 19 down to `deepest` (inclusive) must RECEIVE the value
        for fn_num in range(19, deepest - 1, -1):
            needed[fn_num].append(key)

    # Deterministic order for params: by (buf, col, back)
    hoist_order = [k for k, _ in hoist_keys]
    # Sort needed/used into hoist_order
    def sort_by_order(keys):
        idx = {k: i for i, k in enumerate(hoist_order)}
        return sorted(keys, key=lambda k: idx[k])
    for fn_num in list(needed.keys()):
        needed[fn_num] = sort_by_order(needed[fn_num])
    for fn_num in list(used.keys()):
        used[fn_num] = sort_by_order(used[fn_num])

    # Apply transforms per file.
    os.makedirs(out_dir, exist_ok=True)
    files_changed = []
    files_unchanged = []
    for fname, st in per_file.items():
        lines = list(st['lines'])  # mutable copy
        n_changes = 0

        # 1. Update forward declarations in this file.
        # Forward decls match DECL_RE: `FpExt rv32im_v2_N(...);` on a single line.
        for i, line in enumerate(lines):
            m = DECL_RE.match(line)
            if not m:
                continue
            fn_num = int(m.group(1))
            extra = needed.get(fn_num, [])
            if not extra:
                continue
            # Append extra Fp params before the closing `);`
            extra_params = ', '.join(f'Fp {hname(*k)}' for k in extra)
            # Replace ");" with ", extra_params);"
            new_line = re.sub(
                r'\);\s*$',
                f', {extra_params});\n',
                line,
            )
            lines[i] = new_line
            n_changes += 1

        # 2. Update sub-function DEFINITIONS in this file.
        # Definition: `FpExt rv32im_v2_N(...)` followed by `{`. Could be on
        # the same line or the next. Search for DEF_RE-matched headers.
        # Build a list of (line_idx, fn_num, header_text) for definitions.
        # Then update the header.
        for i, line in enumerate(lines):
            m = DEF_RE.match(line)
            if not m:
                continue
            fn_num = int(m.group(1))
            extra = needed.get(fn_num, [])
            if not extra:
                continue
            extra_params = ', '.join(f'Fp {hname(*k)}' for k in extra)
            # Replace `)` (followed by optional space + `{`) with `, EXTRA) {`
            new_line = re.sub(
                r'\)(\s*\{?\s*)$',
                f', {extra_params})\\1',
                line,
            )
            lines[i] = new_line
            n_changes += 1

        # 3. Update inter-sub-fn call sites in this file.
        # Pattern: `rv32im_v2_N(cycle, steps, poly_mix, ARGS);` — usually at the
        # end of a line. Could be inside `auto xN = rv32im_v2_N(...);`.
        # Find all such call sites.
        call_call_re = re.compile(
            r'(rv32im_v2_(\d+)\(cycle, steps, poly_mix,)(.+?)(\);)'
        )
        for i, line in enumerate(lines):
            for m in call_call_re.finditer(line):
                callee = int(m.group(2))
                extra = needed.get(callee, [])
                if not extra:
                    continue
                # Don't double-apply if line already has hoisted names.
                # Check by searching for the first hoist name.
                first_hn = hname(*extra[0])
                if first_hn in line:
                    continue
                # Forward args from caller's frame. The CALLER is the function
                # containing this line — figure out which fn.
                # For poly_fp: caller is poly_fp, which has the hoisted Fp
                # variables in scope at this point.
                # For sub-fn: caller has the hoisted values as PARAMS (named
                # hname(*k)) — same name, so forwarding is just `hname(*k)`.
                # So the new call args list adds: ", hname(*k)" for each k.
                forward_args = ', '.join(hname(*k) for k in extra)
                # Insert before the `);`
                replacement = m.group(1) + m.group(3) + ', ' + forward_args + m.group(4)
                line = line.replace(m.group(0), replacement, 1)
            lines[i] = line

        # 4. Replace global LDGs inside sub-function bodies with the hoist name.
        # For each sub-fn, find lines in its body matching READ_RE, and check
        # if the read matches one of the USED hoisted keys.
        for fn_num, (fname_loc, s_loc, e_loc) in sub_fn_loc.items():
            if fname_loc != fname:
                continue
            if fn_num not in used:
                continue
            pmap = bufmap.get(fn_num, {})
            # Build a (argname, col, back) → key lookup for used keys
            argname_for_buf = {v: k for k, v in pmap.items()}
            key_set = set(used[fn_num])
            for li in range(s_loc, e_loc + 1):
                def repl(m):
                    arg_idx = int(m.group(1))
                    col = int(m.group(2))
                    back = int(m.group(3))
                    argname = f'arg{arg_idx}'
                    buf = pmap.get(argname)
                    if buf is None:
                        return m.group(0)
                    key = (buf, col, back)
                    if key not in key_set:
                        return m.group(0)
                    return hname(*key)
                new_line, count = READ_RE.subn(repl, lines[li])
                if count > 0:
                    lines[li] = new_line
                    n_changes += count

        # 5. For poly_fp's file, insert hoist block BEFORE the call to v_19.
        if fname_loc := next((f for f, _, _ in (poly_fp_loc,) if f == fname), None):
            # Find the call line
            pf_fname, pf_s, pf_e = poly_fp_loc
            # Look up the call to rv32im_v2_19 within poly_fp's range.
            for li in range(pf_s, pf_e + 1):
                if POLY_FP_CALL_RE.search(lines[li]):
                    # Inject hoist reads BEFORE this line.
                    indent = re.match(r'^(\s*)', lines[li]).group(1)
                    inject = []
                    inject.append(f'{indent}// [poly-fp-cse] hoisted shared reads (top {len(hoist_order)})\n')
                    # The /*data=*/ etc. comments tell us which args[N] corresponds.
                    bufN = {'accum': 0, 'data': 1, 'global': 2, 'mix': 3}
                    for (buf, col, back) in hoist_order:
                        idx = bufN[buf]
                        inject.append(
                            f'{indent}auto {hname(buf, col, back)} = '
                            f'/*{buf}=*/args[{idx}]'
                            f'[{col} * steps + ((cycle - kInvRate * {back}) & mask)];\n'
                        )
                    lines[li:li] = inject
                    n_changes += len(inject)
                    break

        # Write out
        out_path = os.path.join(out_dir, fname)
        with open(out_path, 'w') as f:
            f.writelines(lines)
        if n_changes > 0:
            files_changed.append((fname, n_changes))
        else:
            files_unchanged.append(fname)

    # Copy non-transformed files through unchanged (anything not rust_poly_fp_*)
    for fname in os.listdir(in_dir):
        if fname in per_file:
            continue
        src = os.path.join(in_dir, fname)
        dst = os.path.join(out_dir, fname)
        if os.path.isfile(src):
            with open(src, 'rb') as fin, open(dst, 'wb') as fout:
                fout.write(fin.read())

    print(f"[poly_fp_cse] Hoisted {len(hoist_order)} reads (min_share={min_share}, top_n={top_n})",
          file=sys.stderr)
    for fname, n in files_changed:
        print(f"  {fname}: {n} edits", file=sys.stderr)
    return len(hoist_order)


def apply_lsc_hints(out_dir):
    """Tier B4: wrap all remaining argK[...] / args[K][...] reads in the
    rust_poly_fp_*.cpp output files with ::risc0::lsc::cached_load(&...).

    Idempotent: uses regexes with a `(?<!&)` negative lookbehind so a read
    that's already preceded by `&` (the wrap we emit) is skipped. Running
    this a second time over the same files is a no-op."""
    def wrap(m):
        return f'::risc0::lsc::cached_load(&{m.group(0)})'
    total = 0
    for fname in sorted(os.listdir(out_dir)):
        if not fname.startswith('rust_poly_fp_') or not fname.endswith('.cpp'):
            continue
        path = os.path.join(out_dir, fname)
        with open(path) as f:
            content = f.read()
        new_content, n1 = LSC_WRAPPED_READ_RE.subn(wrap, content)
        new_content, n2 = LSC_WRAPPED_READ_RE_ARGS.subn(wrap, new_content)
        if n1 + n2:
            with open(path, 'w') as f:
                f.write(new_content)
            print(f"[lsc-hints] {fname}: wrapped {n1 + n2} reads", file=sys.stderr)
        total += n1 + n2
    print(f"[lsc-hints] total reads wrapped: {total}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('in_dir')
    ap.add_argument('out_dir', nargs='?', default=None,
                    help='If given, apply transform; if not, just analyze')
    ap.add_argument('--top', type=int, default=16,
                    help='Hoist at most this many tuples')
    ap.add_argument('--min-share', type=int, default=10,
                    help='Only hoist tuples shared by at least this many sub-fns')
    ap.add_argument('--lsc-hints', action='store_true',
                    help='Tier B4: wrap remaining buffer reads in '
                         '::risc0::lsc::cached_load() for IGC LSC cache hints')
    args = ap.parse_args()

    if args.out_dir:
        transform(args.in_dir, args.out_dir,
                  top_n=args.top, min_share=args.min_share)
        if args.lsc_hints:
            apply_lsc_hints(args.out_dir)
        return

    agg, _, bufmap, _, _, _ = analyze_files(args.in_dir)
    items = sorted(agg.items(), key=lambda kv: (-len(kv[1]),
                                                 kv[0][0], kv[0][1], kv[0][2]))
    print(f"Total unique (buf,col,back) tuples: {len(items)}")
    print(f"Top {args.top}:")
    print(f"{'#':>4} {'count':>5}  {'buf':<8} {'col':>4} {'back':>4}")
    for i, ((buf, col, back), fns) in enumerate(items[:args.top]):
        print(f"{i:>4} {len(fns):>5}  {buf:<8} {col:>4} {back:>4}")
    from collections import Counter
    hist = Counter(len(fns) for _, fns in items)
    print("\nShare-count histogram (count -> #tuples):")
    for k in sorted(hist.keys(), reverse=True):
        print(f"  {k:>3} → {hist[k]}")
    print("\nSub-function buffer maps:")
    for fn_num in sorted(bufmap.keys(), reverse=True):
        m = bufmap[fn_num]
        s = ', '.join(f'{k}={v}' for k, v in sorted(m.items()))
        print(f"  rv32im_v2_{fn_num}: {s}")


if __name__ == '__main__':
    main()
