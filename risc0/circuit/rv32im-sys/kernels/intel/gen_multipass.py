#!/usr/bin/env python3
"""Generate a 2-way multi-pass eval_check amalgamation for Intel GPU.

Splits the monolithic poly_fp kernel into two passes at the rv32im_v2_10 → rv32im_v2_9
boundary. Each pass becomes a separate kernel function:
  - poly_fp_pass1: preamble + functions 19..10 → writes intermediate
  - poly_fp_pass2: preamble + reads intermediate + functions 9..0 → writes check

The intermediate buffer stores:
  - 225 x33 (Fp) indices that cross the boundary
  - 3 FpExt by-value arguments (x1368, x708, x699)
  - x34[5] FpExt values (indices 0-4, written by fn 13/12/9, read by fn 7)

Usage: python3 gen_multipass.py <cxx_root> <output_amalg.cpp>
"""

import re
import sys
import os
from collections import defaultdict

def parse_functions(files):
    """Parse all function definitions and extract their bodies."""
    functions = {}  # name -> { 'sig': str, 'body': str, 'file': str }

    for fname in files:
        with open(fname) as f:
            content = f.read()

        # Find namespace body
        ns_marker = "namespace risc0::circuit::rv32im_v2 {"
        ns_start = content.find(ns_marker)
        if ns_start < 0:
            continue
        body_start = ns_start + len(ns_marker)
        body_end = content.rfind('}')
        ns_body = content[body_start:body_end]

        # Extract function definitions
        # Match: FpExt rv32im_v2_N(... or FpExt poly_fp(...
        pattern = re.compile(r'^(FpExt\s+(?:rv32im_v2_\d+|poly_fp)\([^)]*\))\s*\{', re.MULTILINE)
        for m in pattern.finditer(ns_body):
            sig = m.group(1)
            func_name = re.search(r'(rv32im_v2_\d+|poly_fp)', sig).group(1)

            # Find matching closing brace
            start = m.start()
            brace_start = ns_body.index('{', start)
            depth = 1
            pos = brace_start + 1
            while depth > 0 and pos < len(ns_body):
                if ns_body[pos] == '{': depth += 1
                elif ns_body[pos] == '}': depth -= 1
                pos += 1

            body = ns_body[brace_start+1:pos-1]

            if func_name not in functions:
                functions[func_name] = {
                    'sig': sig,
                    'body': body,
                    'file': fname,
                }

    return functions

def extract_x33_cross_boundary(functions, pass1_fns, pass2_fns):
    """Find x33 indices that are written in pass1 and read in pass2."""
    # Identify x33 arg for each function
    x33_arg = {}
    for fn_name, fn_data in functions.items():
        if fn_name == 'poly_fp':
            continue
        body = fn_data['body']
        arg_max = {}
        for m in re.finditer(r'(arg\d+)\[(\d+)\]', body):
            arg = m.group(1)
            idx = int(m.group(2))
            if arg not in arg_max or idx > arg_max[arg]:
                arg_max[arg] = idx
        # x33 is the arg with highest index (up to 1006)
        best = max(((a, v) for a, v in arg_max.items() if v > 10), key=lambda x: x[1], default=(None, -1))
        x33_arg[fn_name] = best[0]

    # Collect writes and reads
    writes = set()
    for fn in pass1_fns:
        if fn not in functions or fn not in x33_arg or x33_arg[fn] is None:
            continue
        arg = x33_arg[fn]
        body = functions[fn]['body']
        for m in re.finditer(rf'{arg}\[(\d+)\]\s*=\s', body):
            writes.add(int(m.group(1)))

    reads = set()
    for fn in pass2_fns:
        if fn not in functions or fn not in x33_arg or x33_arg[fn] is None:
            continue
        arg = x33_arg[fn]
        body = functions[fn]['body']
        for m in re.finditer(rf'{arg}\[(\d+)\]', body):
            idx = int(m.group(1))
            rest = body[m.end():m.end()+5].lstrip()
            if not rest.startswith('=') or rest.startswith('=='):
                reads.add(idx)

    cross = sorted(writes & reads)
    return cross, x33_arg

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <cxx_root> <output_amalg.cpp>")
        sys.exit(1)

    cxx_root = sys.argv[1]
    output_path = sys.argv[2]

    files = [f"kernels/cxx/rust_poly_fp_{i}.cpp" for i in range(4)]

    # Parse all functions
    functions = parse_functions(files)
    print(f"Parsed {len(functions)} functions", file=sys.stderr)

    # Define passes
    pass1_fns = [f'rv32im_v2_{i}' for i in range(19, 9, -1)]  # 19..10
    pass2_fns = [f'rv32im_v2_{i}' for i in range(9, -1, -1)]   # 9..0

    # Compute cross-boundary x33 indices
    cross_indices, x33_arg = extract_x33_cross_boundary(functions, pass1_fns, pass2_fns)
    print(f"Cross-boundary x33 indices: {len(cross_indices)}", file=sys.stderr)

    # The call at boundary: rv32im_v2_10 calls rv32im_v2_9
    # rv32im_v2_9(cycle, steps, poly_mix, arg0, x1368, arg2, x708, arg23, arg24, x699, arg25, arg26, arg27, arg28, arg29)
    # By-value FpExt: x1368, x708, x699 (3 values)
    # Pointers: arg0=x33, arg2=x34, arg23-29=buffer passthroughs

    # For now, just output the cross-boundary analysis
    print(f"\nCross-boundary summary:", file=sys.stderr)
    print(f"  x33 Fp indices: {len(cross_indices)} ({len(cross_indices)*4} bytes/WI)", file=sys.stderr)
    print(f"  FpExt by-value: 3 (48 bytes/WI)", file=sys.stderr)
    print(f"  x34 FpExt: ~5 (80 bytes/WI)", file=sys.stderr)
    total = len(cross_indices)*4 + 48 + 80
    print(f"  Total: {total} bytes/WI", file=sys.stderr)

    # Generate the amalgamation with both pass functions
    # For now, generate the standard monolithic amalgamation (we'll add multi-pass later)
    print(f"\nGenerated analysis. Multi-pass code generation not yet implemented.", file=sys.stderr)

if __name__ == '__main__':
    main()
