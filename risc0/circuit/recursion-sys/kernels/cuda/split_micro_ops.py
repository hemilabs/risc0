#!/usr/bin/env python3
"""Split step_exec_micro_ops into 3 instance functions.

micro_ops contains 3 identical micro-instruction processing instances. At 347KB ISA
total, splitting into 3 functions of ~116KB each improves I-cache utilization on
AMD MI300X (64KB L1 I-cache per CU).

Each instance processes one micro-instruction slot:
- Instance 0: opcode=arg0[8], write_addr=arg0[0]
- Instance 1: opcode=arg0[12], write_addr=arg0[0]+1
- Instance 2: opcode=arg0[16], write_addr=arg0[0]+2
"""

import sys
import re


def read_file(path):
    with open(path, 'r') as f:
        return f.readlines()


def find_matching_brace(lines, start_idx):
    depth = 0
    for i in range(start_idx, len(lines)):
        depth += lines[i].count('{') - lines[i].count('}')
        if depth == 0:
            return i
    raise RuntimeError(f"No matching brace found from line {start_idx + 1}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'step_exec.cu'
    lines = read_file(path)
    print(f"Read {len(lines)} lines from {path}")

    # Find step_exec_micro_ops function definition (skip forward declaration)
    func_start = None
    for i, line in enumerate(lines):
        if '__device__ __noinline__ void step_exec_micro_ops(' in line and i > 80:
            func_start = i
            break
    assert func_start is not None, "Could not find step_exec_micro_ops function definition"

    # Find opening brace
    open_brace = func_start
    while '{' not in lines[open_brace]:
        open_brace += 1
    func_end = find_matching_brace(lines, open_brace)
    print(f"micro_ops function: lines {func_start+1}-{func_end+1} ({func_end - func_start + 1} lines)")

    # Find constants block
    const_start = const_end = None
    for i in range(func_start, func_end):
        if const_start is None and '  Fp x0(' in lines[i]:
            const_start = i - 1 if lines[i-1].strip().startswith('//') else i
        if '  Fp x311(' in lines[i]:
            const_end = i
            break
    assert const_start and const_end
    const_lines = lines[const_start:const_end + 1]
    print(f"Constants: lines {const_start+1}-{const_end+1}")

    # Find the opcode reads for each instance (arg0[8], arg0[12], arg0[16])
    # and the write_addr read (arg0[0])
    search_from = const_end
    opcode_lines = []
    for col in [8, 12, 16]:
        for i in range(search_from, func_end):
            if f'arg0[{col} * steps' in lines[i]:
                opcode_lines.append(i)
                search_from = i + 100
                break

    assert len(opcode_lines) == 3, f"Found {len(opcode_lines)} opcode reads, expected 3"
    print(f"Opcode reads at lines: {[l+1 for l in opcode_lines]}")

    # Find write_addr read (arg0[0]) near the first opcode read
    write_addr_line = None
    for i in range(opcode_lines[0] - 5, opcode_lines[0] + 10):
        if 'arg0[0 * steps' in lines[i]:
            write_addr_line = i
            break
    assert write_addr_line is not None
    m = re.search(r'auto (x\d+) = arg0\[0 \* steps', lines[write_addr_line])
    write_addr_var = m.group(1)  # x314
    print(f"write_addr variable: {write_addr_var} at line {write_addr_line+1}")

    # Find transition lines between instances
    # Between inst 0 and 1: auto xNNN = x314 + x310 (x310 = 1)
    # Between inst 1 and 2: auto xNNN = x314 + x309 (x309 = 2)
    trans_01 = trans_12 = None
    for i in range(opcode_lines[0], opcode_lines[1]):
        if f'{write_addr_var} + x310' in lines[i]:
            trans_01 = i
            m = re.search(r'auto (x\d+)', lines[i])
            inst1_wa_var = m.group(1) if m else None
    for i in range(opcode_lines[1], opcode_lines[2]):
        if f'{write_addr_var} + x309' in lines[i]:
            trans_12 = i
            m = re.search(r'auto (x\d+)', lines[i])
            inst2_wa_var = m.group(1) if m else None

    assert trans_01, "Could not find transition inst0->inst1"
    assert trans_12, "Could not find transition inst1->inst2"
    print(f"Transitions: 0->1 at line {trans_01+1} ({inst1_wa_var}), 1->2 at line {trans_12+1} ({inst2_wa_var})")

    # Find tail section
    tail_start = func_end
    for i in range(opcode_lines[2], func_end):
        if '// tail section' in lines[i]:
            tail_start = i
            break
    tail_lines = lines[tail_start:func_end]
    print(f"Tail: {len(tail_lines)} lines")

    # Instance boundaries:
    # inst0: from the comment before opcode_lines[0] through the line before trans_01
    # Look for the comment line before the first opcode read
    inst0_start = opcode_lines[0]
    if inst0_start > const_end + 1 and lines[inst0_start - 1].strip().startswith('//'):
        inst0_start -= 1
    inst0_end = trans_01 - 1
    # Skip trailing blank lines
    while inst0_end > inst0_start and lines[inst0_end].strip() == '':
        inst0_end -= 1

    # inst1: from trans_01 through line before trans_12
    inst1_start = trans_01
    # Include the comment before trans_01 if present
    if lines[inst1_start - 1].strip().startswith('//'):
        inst1_start -= 1
    inst1_end = trans_12 - 1
    while inst1_end > inst1_start and lines[inst1_end].strip() == '':
        inst1_end -= 1

    # inst2: from trans_12 through line before tail
    inst2_start = trans_12
    if lines[inst2_start - 1].strip().startswith('//'):
        inst2_start -= 1
    inst2_end = tail_start - 1
    while inst2_end > inst2_start and lines[inst2_end].strip() == '':
        inst2_end -= 1

    instances = [
        {'name': 'inst0', 'start': inst0_start, 'end': inst0_end, 'needs_wa_read': True},
        {'name': 'inst1', 'start': inst1_start, 'end': inst1_end, 'needs_wa_read': False},
        {'name': 'inst2', 'start': inst2_start, 'end': inst2_end, 'needs_wa_read': False},
    ]

    for inst in instances:
        size = inst['end'] - inst['start'] + 1
        print(f"  {inst['name']}: lines {inst['start']+1}-{inst['end']+1} ({size} lines)")

    # Find the forward declaration for micro_ops
    fwd_decl_start = fwd_decl_end = None
    for i in range(80):
        if '__device__ __noinline__ void step_exec_micro_ops(' in lines[i]:
            fwd_decl_start = i
            for j in range(i, i + 5):
                if ';' in lines[j]:
                    fwd_decl_end = j
                    break
            break
    assert fwd_decl_start is not None

    # Generate output
    out = []

    # Everything before the micro_ops forward declaration
    out.extend(lines[:fwd_decl_end + 1])
    out.append('\n')

    # Forward declarations for 3 instance functions
    for inst in instances:
        out.append(f'__device__ __noinline__ void step_exec_micro_{inst["name"]}(\n')
        out.append(f'    void* ctx, uint32_t steps, uint32_t cycle, uint32_t mask,\n')
        out.append(f'    Fp* arg0, Fp* arg1, Fp* arg2);\n')
        out.append('\n')

    # Everything between forward declaration and function definition
    out.extend(lines[fwd_decl_end + 1:func_start])

    # New micro_ops dispatcher
    out.append('__device__ __noinline__ void step_exec_micro_ops(\n')
    out.append('    void* ctx, uint32_t steps, uint32_t cycle, uint32_t mask,\n')
    out.append('    Fp* arg0, Fp* arg1, Fp* arg2) {\n')

    if tail_lines:
        out.append('  Fp extern_args[96];\n')
        out.append('  Fp extern_outs[32];\n')
        out.extend(const_lines)
        out.append('\n')

    # Call instance functions
    out.append('  step_exec_micro_inst0(ctx, steps, cycle, mask, arg0, arg1, arg2);\n')
    out.append('  step_exec_micro_inst1(ctx, steps, cycle, mask, arg0, arg1, arg2);\n')
    out.append('  step_exec_micro_inst2(ctx, steps, cycle, mask, arg0, arg1, arg2);\n')

    if tail_lines:
        out.append('\n')
        out.extend(tail_lines)

    out.append('}\n')
    out.append('\n')

    # Generate 3 instance functions
    for inst in instances:
        name = inst['name']
        body_start = inst['start']
        body_end = inst['end']

        out.append(f'__device__ __noinline__ void step_exec_micro_{name}(\n')
        out.append(f'    void* ctx, uint32_t steps, uint32_t cycle, uint32_t mask,\n')
        out.append(f'    Fp* arg0, Fp* arg1, Fp* arg2) {{\n')
        out.append(f'  Fp extern_args[96];\n')
        out.append(f'  Fp extern_outs[32];\n')

        # Constants
        out.extend(const_lines)
        out.append('\n')

        # For inst1 and inst2: need to read x314 (write_addr = arg0[0])
        # since their body references it via the transition variable
        if not inst['needs_wa_read']:
            out.append(f'    auto {write_addr_var} = arg0[0 * steps + ((cycle - 0) & mask)];\n')
            out.append(f'    assert({write_addr_var} != Fp::invalid());\n')

        # Body
        body = lines[body_start:body_end + 1]
        out.extend(body)

        out.append('}\n')
        out.append('\n')

    # Everything after micro_ops function
    out.extend(lines[func_end + 1:])

    # Write output
    with open(path, 'w') as f:
        f.writelines(out)

    print(f"\nWrote {len(out)} lines to {path}")
    print(f"micro_ops split into 1 dispatcher + 3 instance functions")


if __name__ == '__main__':
    main()
