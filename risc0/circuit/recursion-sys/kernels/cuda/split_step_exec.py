#!/usr/bin/env python3
"""Split step_exec.cu into separate __noinline__ functions per mux arm.

The monolithic step_exec function (41K lines, 17MB ISA) causes I-cache thrashing
on AMD MI300X (64KB L1 I-cache per CU). By splitting into 7 separate noinline
functions (one per mux arm), each function is ~1-3MB ISA, fitting much better
in cache since only one arm executes per cycle (OneHot dispatch).
"""

import sys

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

    # Find the function start
    func_start = None
    for i, line in enumerate(lines):
        if '__device__ void step_exec(' in line:
            func_start = i
            break
    assert func_start is not None, "Could not find step_exec function"

    # Header = everything before the function (copyright + includes)
    header = lines[:func_start]

    # Find constants block (x0 through x311)
    const_start = None
    const_end = None
    for i in range(func_start, len(lines)):
        if const_start is None and '  Fp x0(' in lines[i]:
            # Include the comment line before x0
            const_start = i - 1 if lines[i-1].strip().startswith('//') else i
        if '  Fp x311(' in lines[i]:
            const_end = i
            break
    assert const_start and const_end, "Could not find constants block"
    const_lines = lines[const_start:const_end + 1]
    print(f"Constants: lines {const_start+1}-{const_end+1} ({len(const_lines)} lines)")

    # Define arms with their selector column and variable name
    arms = [
        ('micro_ops',        1, 'x312'),
        ('macro_ops',        2, 'x1360'),
        ('poseidon2_load',   3, 'x8811'),
        ('poseidon2_full',   4, 'x9216'),
        ('poseidon2_partial',5, 'x10033'),
        ('poseidon2_store',  6, 'x11720'),
        ('checked_bytes',    7, 'x11850'),
    ]

    # Find arm boundaries
    arm_info = []
    for name, col, var in arms:
        # Find the selector read: auto xN = arg0[col * steps + ...]
        sel_line = None
        for i in range(func_start, len(lines)):
            if f'auto {var} = arg0[{col} * steps' in lines[i]:
                sel_line = i
                break
        assert sel_line is not None, f"Could not find selector for {name} ({var})"

        # Find the if block: if (xN != 0) {
        if_line = None
        for i in range(sel_line, sel_line + 5):
            if f'if ({var} != 0)' in lines[i]:
                if_line = i
                break
        assert if_line is not None, f"Could not find if block for {name}"

        end_line = find_matching_brace(lines, if_line)

        arm_info.append({
            'name': name,
            'col': col,
            'var': var,
            'if_line': if_line,
            'end_line': end_line,
        })
        print(f"  {name}: if at line {if_line+1}, body lines {if_line+2}-{end_line}, "
              f"~{end_line - if_line - 1} lines")

    # Tail section starts after last arm
    tail_start = arm_info[-1]['end_line'] + 1
    # Function end is the final closing brace
    func_end = len(lines) - 1
    while func_end > tail_start and lines[func_end].strip() == '':
        func_end -= 1
    # func_end should be the closing } of step_exec
    print(f"Tail section: lines {tail_start+1}-{func_end+1}")

    # Parse tail blocks - each is if (xN != 0) { ... }
    # Map: var -> list of (if_line, end_line) tuples
    tail_blocks = {}
    i = tail_start
    while i <= func_end:
        for name, col, var in arms:
            if f'if ({var} != 0)' in lines[i]:
                end = find_matching_brace(lines, i)
                if var not in tail_blocks:
                    tail_blocks[var] = []
                tail_blocks[var].append((i, end))
                print(f"  tail block for {var}: lines {i+1}-{end+1}")
                i = end
                break
        i += 1

    # Generate output
    out = []

    # Header (copyright + includes)
    out.extend(header)

    # Forward declarations
    for info in arm_info:
        out.append(f'__device__ __noinline__ void step_exec_{info["name"]}(\n')
        out.append(f'    void* ctx, uint32_t steps, uint32_t cycle, uint32_t mask,\n')
        out.append(f'    Fp* arg0, Fp* arg1, Fp* arg2);\n')
        out.append(f'\n')

    # Main dispatcher function
    out.append('__device__ void step_exec(\n')
    out.append('    void* ctx, uint32_t steps, uint32_t cycle, Fp* arg0, Fp* arg1, Fp* arg2, Fp* arg3, Fp* arg4) {\n')
    out.append('  uint32_t mask = steps - 1;\n')
    out.append('\n')
    for info in arm_info:
        out.append(f'  if (arg0[{info["col"]} * steps + ((cycle - 0) & mask)] != 0)\n')
        out.append(f'    step_exec_{info["name"]}(ctx, steps, cycle, mask, arg0, arg1, arg2);\n')
    out.append('}\n')
    out.append('\n')

    # Each arm function
    for info in arm_info:
        name = info['name']
        var = info['var']
        if_line = info['if_line']
        end_line = info['end_line']

        out.append(f'__device__ __noinline__ void step_exec_{name}(\n')
        out.append(f'    void* ctx, uint32_t steps, uint32_t cycle, uint32_t mask,\n')
        out.append(f'    Fp* arg0, Fp* arg1, Fp* arg2) {{\n')
        out.append(f'  Fp extern_args[96];\n')
        out.append(f'  Fp extern_outs[32];\n')

        # Constants
        out.extend(const_lines)
        out.append('\n')

        # Arm body: lines inside the if block (between if_line and end_line, exclusive)
        body_lines = lines[if_line + 1:end_line]
        out.extend(body_lines)

        # Tail blocks for this arm
        if var in tail_blocks:
            out.append('\n')
            out.append('  // tail section\n')
            for (tb_start, tb_end) in tail_blocks[var]:
                # Strip the if guard - include only the body inside the if block
                # tb_start is the `if (xN != 0) {` line
                # tb_end is the matching `}`
                # Inner body is lines tb_start+1 through tb_end-1
                inner = lines[tb_start + 1:tb_end]
                out.extend(inner)

        out.append('}\n')
        out.append('\n')

    # Write output
    out_path = path
    with open(out_path, 'w') as f:
        f.writelines(out)

    total_lines = len(out)
    print(f"\nWrote {total_lines} lines to {out_path}")
    print(f"Original: 1 function, {len(lines)} lines")
    print(f"Split: 1 dispatcher + {len(arm_info)} arm functions")

if __name__ == '__main__':
    main()
