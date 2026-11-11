#!/usr/bin/env python3
"""Further split step_exec_macro_ops into 9 sub-arm __noinline__ functions.

The macro_ops arm (16.7 MB ISA) contains 9 sub-opcodes gated by OneHot selectors
on arg0 columns 9-17. The largest (sha_mix at ~7MB) still causes I-cache thrashing
on AMD MI300X. Splitting into separate __noinline__ functions allows each to compile
independently with -fgpu-rdc.
"""

import sys
import re


def read_file(path):
    with open(path, 'r') as f:
        return f.readlines()


def find_matching_brace(lines, start_idx):
    """Find line index of matching closing brace for the opening brace on start_idx."""
    depth = 0
    for i in range(start_idx, len(lines)):
        depth += lines[i].count('{') - lines[i].count('}')
        if depth == 0:
            return i
    raise RuntimeError(f"No matching brace found starting from line {start_idx + 1}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'step_exec.cu'
    lines = read_file(path)
    print(f"Read {len(lines)} lines from {path}")

    # Sub-arm definitions: (name, selector_column, selector_variable)
    # These are the 9 sub-opcodes within macro_ops, each gated by a OneHot selector
    sub_arms = [
        ('wom_init',        9,  'x1362'),
        ('wom_fini',       10,  'x1363'),
        ('bit_and_elem',   11,  'x1364'),
        ('bit_op_shorts',  12,  'x1726'),
        ('sha_init',       13,  'x2049'),
        ('sha_fini',       14,  'x2515'),
        ('sha_load',       15,  'x3415'),
        ('sha_mix',        16,  'x5752'),
        ('set_global',     17,  'x8654'),
    ]

    # Find step_exec_macro_ops function definition (skip forward declaration)
    func_start = None
    for i, line in enumerate(lines):
        if '__device__ __noinline__ void step_exec_macro_ops(' in line and i > 50:
            func_start = i
            break
    assert func_start is not None, "Could not find step_exec_macro_ops function definition"

    # Find opening brace of function body
    open_brace = func_start
    while '{' not in lines[open_brace]:
        open_brace += 1
    func_end = find_matching_brace(lines, open_brace)
    print(f"macro_ops function: lines {func_start+1}-{func_end+1} ({func_end - func_start + 1} lines)")

    # Find constants block within macro_ops (x0 through x311)
    const_start = const_end = None
    for i in range(func_start, func_end):
        if const_start is None and '  Fp x0(' in lines[i]:
            const_start = i - 1 if lines[i-1].strip().startswith('//') else i
        if '  Fp x311(' in lines[i]:
            const_end = i
            break
    assert const_start and const_end, "Could not find constants block in macro_ops"
    const_lines = lines[const_start:const_end + 1]
    print(f"Constants: lines {const_start+1}-{const_end+1} ({len(const_lines)} lines)")

    # Find prologue: code between constants end and first sub-arm selector read
    # This includes: auto x1361 = arg0[0 * steps + ...]; assert(x1361 != Fp::invalid());
    # and the comment before x1362's selector read

    # Find each sub-arm's if-block
    sub_info = []
    for name, col, var in sub_arms:
        # Find: if (var != 0) {
        if_line = None
        for i in range(const_end, func_end):
            if f'if ({var} != 0)' in lines[i]:
                if_line = i
                break
        assert if_line is not None, f"Could not find if block for {name} ({var})"

        # Find the selector read line (a few lines before the if)
        sel_line = if_line
        for i in range(if_line - 1, max(if_line - 10, const_end), -1):
            if f'auto {var} =' in lines[i]:
                sel_line = i
                break

        end_line = find_matching_brace(lines, if_line)
        body_lines = end_line - if_line - 1

        sub_info.append({
            'name': name,
            'col': col,
            'var': var,
            'sel_line': sel_line,
            'if_line': if_line,
            'end_line': end_line,
        })
        print(f"  {name} (col {col}, {var}): if@{if_line+1}, {body_lines} body lines")

    # Prologue: from constants end+1 to first sub-arm's selector read line
    prologue_start = const_end + 1
    prologue_end = sub_info[0]['sel_line']
    prologue_lines = lines[prologue_start:prologue_end]
    print(f"Prologue: {len(prologue_lines)} lines (lines {prologue_start+1}-{prologue_end})")

    # Tail: from last sub-arm's end_line+1 to func_end (exclusive of closing brace)
    tail_start = sub_info[-1]['end_line'] + 1
    tail_lines = lines[tail_start:func_end]
    print(f"Tail: {len(tail_lines)} lines (lines {tail_start+1}-{func_end})")

    # Find macro_ops forward declaration at the top of the file
    fwd_decl_start = None
    fwd_decl_end = None
    for i in range(min(50, len(lines))):
        if '__device__ __noinline__ void step_exec_macro_ops(' in lines[i]:
            fwd_decl_start = i
            # Find the semicolon ending the declaration
            for j in range(i, i + 5):
                if ';' in lines[j]:
                    fwd_decl_end = j
                    break
            break

    assert fwd_decl_start is not None, "Could not find macro_ops forward declaration"

    # Generate output
    out = []

    # Everything before the macro_ops forward declaration
    out.extend(lines[:fwd_decl_end + 1])
    out.append('\n')

    # Add forward declarations for 9 sub-arm functions
    for si in sub_info:
        out.append(f'__device__ __noinline__ void step_exec_macro_{si["name"]}(\n')
        out.append(f'    void* ctx, uint32_t steps, uint32_t cycle, uint32_t mask,\n')
        out.append(f'    Fp* arg0, Fp* arg1, Fp* arg2);\n')
        out.append('\n')

    # Everything between the forward declaration and the function definition
    # (includes other forward decls, step_exec dispatcher, and earlier arm functions)
    out.extend(lines[fwd_decl_end + 1:func_start])

    # Generate new macro_ops dispatcher function
    out.append('__device__ __noinline__ void step_exec_macro_ops(\n')
    out.append('    void* ctx, uint32_t steps, uint32_t cycle, uint32_t mask,\n')
    out.append('    Fp* arg0, Fp* arg1, Fp* arg2) {\n')

    # Tail blocks reference constants (x311 etc.) and use extern_args/outs
    if tail_lines:
        out.append('  Fp extern_args[96];\n')
        out.append('  Fp extern_outs[32];\n')
        out.extend(const_lines)
        out.append('\n')

    # Prologue (x1361 = write_addr)
    out.extend(prologue_lines)
    out.append('\n')

    # Dispatch to sub-arm functions
    for si in sub_info:
        out.append(f'  if (arg0[{si["col"]} * steps + ((cycle - 0) & mask)] != 0)\n')
        out.append(f'    step_exec_macro_{si["name"]}(ctx, steps, cycle, mask, arg0, arg1, arg2);\n')

    # Tail section (small, kept inline in dispatcher)
    if tail_lines:
        out.append('\n')
        out.append('  // tail section\n')
        out.extend(tail_lines)

    out.append('}\n')
    out.append('\n')

    # Generate 9 sub-arm functions
    for si in sub_info:
        name = si['name']
        if_line = si['if_line']
        end_line = si['end_line']

        out.append(f'__device__ __noinline__ void step_exec_macro_{name}(\n')
        out.append(f'    void* ctx, uint32_t steps, uint32_t cycle, uint32_t mask,\n')
        out.append(f'    Fp* arg0, Fp* arg1, Fp* arg2) {{\n')
        out.append(f'  Fp extern_args[96];\n')
        out.append(f'  Fp extern_outs[32];\n')

        # Constants
        out.extend(const_lines)
        out.append('\n')

        # Prologue re-read (x1361 = write_addr from arg0[0])
        out.extend(prologue_lines)
        out.append('\n')

        # Body: lines inside the if block (between if_line and end_line, exclusive)
        body = lines[if_line + 1:end_line]
        out.extend(body)

        out.append('}\n')
        out.append('\n')

    # Everything after macro_ops function (remaining arm functions)
    out.extend(lines[func_end + 1:])

    # Write output
    with open(path, 'w') as f:
        f.writelines(out)

    total_lines = len(out)
    print(f"\nWrote {total_lines} lines to {path}")
    print(f"macro_ops split into 1 dispatcher + {len(sub_info)} sub-arm functions")

    # Print size summary
    for si in sub_info:
        body_size = si['end_line'] - si['if_line'] - 1
        print(f"  step_exec_macro_{si['name']}: {body_size} body lines")


if __name__ == '__main__':
    main()
