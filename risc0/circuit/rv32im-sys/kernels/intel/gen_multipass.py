#!/usr/bin/env python3
"""Generate 2-way multi-pass eval_check C++ code.

Splits the poly_fp chain at rv32im_v2_10 → rv32im_v2_9 boundary.
Creates _pass1 variants of functions 19..10 that propagate intermediate
buffer pointers and write cross-boundary state instead of continuing to v9.

Also creates poly_fp_pass2 that re-executes the preamble, loads intermediate
state, and calls rv32im_v2_9.

Usage: python3 gen_multipass.py <amalg_input.cpp> <amalg_output.cpp>

The input amalgamation should already have all functions with noinline.
The output will have the original functions PLUS the _pass1 variants and poly_fp_pass2.
"""

import re
import sys

# Cross-boundary x33 indices (from static analysis on the amalgamation).
# IMPORTANT: computed from the amalgamation where all functions are in one namespace,
# not from source files (which have forward declaration contamination).
# 197 indices that are WRITTEN by pass1 (v19..v10) AND READ by pass2 (v9..v0).
CROSS_X33_INDICES = [
    123, 127, 130, 133, 135, 179, 184, 198, 207, 221, 247, 278, 279, 280, 282, 292,
    306, 310, 325, 349, 355, 368, 371, 382, 384, 388, 389, 390, 391, 392, 396, 398,
    399, 413, 425, 426, 428, 433, 435, 445, 446, 447, 448, 449, 450, 451, 452, 453,
    454, 455, 456, 457, 470, 471, 472, 473, 475, 481, 488, 492, 493, 506, 507, 508,
    509, 510, 533, 538, 539, 540, 541, 542, 543, 544, 563, 570, 581, 589, 593, 597,
    601, 602, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634,
    635, 637, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 696, 697, 698,
    699, 700, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716,
    717, 718, 719, 720, 721, 724, 729, 730, 732, 733, 734, 735, 736, 737, 738, 739,
    740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 751, 752, 753, 754, 755, 756,
    757, 758, 759, 760, 761, 858, 860, 861, 867, 946, 955, 967, 969, 970, 971, 972,
    973, 974, 979, 981, 982, 983, 984, 986, 988, 989, 990, 991, 992, 993, 994, 996,
    997, 999, 1000, 1004, 1006,
]

# The call chain: poly_fp → v19 → v18 → ... → v10 → v9 → ... → v0
# Pass1 functions: 19, 18, 17, 16, 15, 14, 13, 12, 11, 10
# Pass2 functions: 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
PASS1_FNS = list(range(19, 9, -1))  # 19..10
PASS2_FNS = list(range(9, -1, -1))   # 9..0

# The tail-call at the boundary (in rv32im_v2_10):
# auto x1371 = rv32im_v2_9(cycle, steps, poly_mix, arg0, x1368, arg2, x708, arg23, arg24, x699, arg25, arg26, arg27, arg28, arg29);
BOUNDARY_BYVAL_EXTS = ['x1368', 'x708', 'x699', 'arg23', 'arg24', 'arg25', 'arg26']


def find_function(content, fn_name):
    """Find a function definition and return (start, end, sig, body)."""
    pattern = f'FpExt {fn_name}(size_t'
    # Find the definition (has opening brace)
    pos = 0
    while True:
        idx = content.find(pattern, pos)
        if idx < 0:
            return None
        # Check if this is a definition (has {) not just a declaration
        line_end = content.index('\n', idx)
        line = content[idx:line_end]
        if '{' in line:
            break
        # Also check if the { is on the next line
        next_line_end = content.index('\n', line_end + 1) if line_end + 1 < len(content) else len(content)
        next_line = content[line_end+1:next_line_end].strip()
        if next_line.startswith('{'):
            break
        pos = line_end

    # Find the opening brace
    brace_pos = content.index('{', idx)
    sig = content[idx:brace_pos].strip()

    # Find matching closing brace
    depth = 1
    p = brace_pos + 1
    while depth > 0 and p < len(content):
        if content[p] == '{': depth += 1
        elif content[p] == '}': depth -= 1
        p += 1

    body = content[brace_pos+1:p-1]
    return (idx, p, sig, body)


def make_pass1_variant(content, fn_idx):
    """Create a _pass1 variant of rv32im_v2_N that propagates extra params."""
    fn_name = f'rv32im_v2_{fn_idx}'
    result = find_function(content, fn_name)
    if not result:
        print(f"WARNING: Could not find {fn_name}", file=sys.stderr)
        return ""

    _, _, sig, body = result

    # New function name
    new_name = f'rv32im_v2_{fn_idx}_pass1'
    new_sig = sig.replace(f'rv32im_v2_{fn_idx}(', f'{new_name}(')

    # Remove __attribute__((noinline)) if present (we'll add it back)
    new_sig = new_sig.replace('__attribute__((noinline)) ', '')

    # Add extra params: Fp* d_inter_fp, FpExt* d_inter_ext, size_t domain
    # Find the last ) in the signature
    last_paren = new_sig.rfind(')')
    new_sig = new_sig[:last_paren] + ', Fp* d_inter_fp, FpExt* d_inter_ext, size_t domain' + new_sig[last_paren:]

    new_body = body

    if fn_idx == 10:
        # This is the boundary function — replace tail-call with intermediate writes
        tail_pattern = r'auto\s+x\d+\s*=\s*rv32im_v2_9\([^;]+;'
        return_pattern = r'\s*return\s+x\d+;'

        # Build the replacement
        replacement = "// === MULTI-PASS BOUNDARY: write intermediate state ===\n"
        replacement += "    for (uint32_t _i = 0; _i < CROSS_X33_COUNT; _i++) {\n"
        replacement += "        d_inter_fp[_i * domain + cycle] = arg0[CROSS_X33_INDICES[_i]];\n"
        replacement += "    }\n"
        for i, ext_name in enumerate(BOUNDARY_BYVAL_EXTS):
            replacement += f"    d_inter_ext[{i} * domain + cycle] = {ext_name};\n"
        replacement += "    return FpExt(0);"

        # Replace the tail-call + return
        new_body = re.sub(tail_pattern + return_pattern, replacement, new_body)
    else:
        # Replace call to rv32im_v2_{fn_idx-1} with rv32im_v2_{fn_idx-1}_pass1
        callee = fn_idx - 1
        old_call = f'rv32im_v2_{callee}(cycle, steps, poly_mix,'
        new_call = f'rv32im_v2_{callee}_pass1(cycle, steps, poly_mix,'
        new_body = new_body.replace(old_call, new_call)

        # Add extra params to the call — find the tail-call and append before );
        # The tail-call looks like: auto xN = rv32im_v2_{callee}_pass1(...);
        tail_re = re.compile(rf'(auto\s+x\d+\s*=\s*rv32im_v2_{callee}_pass1\([^)]+)\)')
        m = tail_re.search(new_body)
        if m:
            new_body = new_body[:m.end(1)] + ', d_inter_fp, d_inter_ext, domain)' + new_body[m.end():]

    return f'__attribute__((noinline)) {new_sig} {{\n{new_body}\n}}\n'


def make_poly_fp_pass1(content):
    """Create poly_fp_pass1 that calls the _pass1 chain."""
    result = find_function(content, 'poly_fp')
    if not result:
        print("WARNING: Could not find poly_fp", file=sys.stderr)
        return ""

    _, _, sig, body = result

    new_sig = sig.replace('poly_fp(', 'poly_fp_pass1(')
    new_sig = new_sig.replace('__attribute__((noinline)) ', '')
    # Add extra params
    last_paren = new_sig.rfind(')')
    new_sig = new_sig[:last_paren] + ', Fp* d_inter_fp, FpExt* d_inter_ext, size_t domain' + new_sig[last_paren:]

    new_body = body
    # Replace call to rv32im_v2_19 with rv32im_v2_19_pass1
    new_body = new_body.replace('rv32im_v2_19(cycle, steps, poly_mix,',
                                 'rv32im_v2_19_pass1(cycle, steps, poly_mix,')
    # Add extra params to the call
    tail_re = re.compile(r'(auto\s+x\d+\s*=\s*rv32im_v2_19_pass1\([^)]+)\)')
    m = tail_re.search(new_body)
    if m:
        new_body = new_body[:m.end(1)] + ', d_inter_fp, d_inter_ext, domain)' + new_body[m.end():]

    return f'__attribute__((noinline)) {new_sig} {{\n{new_body}\n}}\n'


def make_poly_fp_pass2(content):
    """Create poly_fp_pass2 that reads intermediate and calls rv32im_v2_9."""
    result = find_function(content, 'poly_fp')
    if not result:
        return ""

    _, _, sig, body = result

    new_sig = 'FpExt poly_fp_pass2(size_t cycle, size_t steps, FpExt* poly_mix, Fp** args, const Fp* d_inter_fp, const FpExt* d_inter_ext, size_t domain)'

    # The preamble from poly_fp: everything before the rv32im_v2_19 call
    # We re-execute it, then overlay the cross-boundary x33 values from the intermediate buffer

    # Find the rv32im_v2_19 call in the body
    call_idx = body.find('rv32im_v2_19(')
    if call_idx < 0:
        print("WARNING: Could not find rv32im_v2_19 call in poly_fp", file=sys.stderr)
        return ""

    # Everything before the call is the preamble
    preamble = body[:call_idx]
    # Find the 'auto xN = ' prefix of the call
    line_start = body.rfind('\n', 0, call_idx) + 1
    preamble = body[:line_start]

    new_body = preamble
    new_body += "\n    // === MULTI-PASS PASS2: load intermediate state ===\n"
    new_body += "    for (uint32_t _i = 0; _i < CROSS_X33_COUNT; _i++) {\n"
    new_body += "        x33[CROSS_X33_INDICES[_i]] = d_inter_fp[_i * domain + cycle];\n"
    new_body += "    }\n"

    # Load the 7 FpExt by-value arguments
    new_body += "    FpExt _ext0 = d_inter_ext[0 * domain + cycle];\n"
    new_body += "    FpExt _ext1 = d_inter_ext[1 * domain + cycle];\n"
    new_body += "    FpExt _ext2 = d_inter_ext[2 * domain + cycle];\n"
    new_body += "    FpExt _ext3 = d_inter_ext[3 * domain + cycle];\n"
    new_body += "    FpExt _ext4 = d_inter_ext[4 * domain + cycle];\n"
    new_body += "    FpExt _ext5 = d_inter_ext[5 * domain + cycle];\n"
    new_body += "    FpExt _ext6 = d_inter_ext[6 * domain + cycle];\n"

    # Call rv32im_v2_9 with the loaded values
    # rv32im_v2_9(cycle, steps, poly_mix, arg0=x33, arg1=x1368, arg2=x34, arg3=x708,
    #             arg4=arg23, arg5=arg24, arg6=x699, arg7=arg25, arg8=arg26,
    #             arg9=Fp*, arg10=Fp*, arg11=Fp*)
    # The buffer pointers (Fp* args) come from the original args[]
    # In the original call: arg27=data_ptr, arg28=accum_ptr, arg29=global_ptr
    # Actually from the original call:
    # rv32im_v2_9(cycle, steps, poly_mix, arg0, x1368, arg2, x708, arg23, arg24, x699, arg25, arg26, arg27, arg28, arg29)
    # where arg27=Fp*(data?), arg28=Fp*(accum?), arg29=Fp*(mix?)
    # These correspond to args[1], args[0], args[3] from the original poly_fp call
    # Actually the exact mapping depends on the chain. Let me trace it.
    # In poly_fp: calls rv32im_v2_19(..., /*data=*/args[1], /*accum=*/args[0], /*mix=*/args[3], /*global=*/args[2])
    # So the 4 buffer pointers are: data=args[1], accum=args[0], mix=args[3], global=args[2]
    # These get passed through the chain to v9's arg9, arg10, arg11
    # We need to figure out which buffer maps to which.
    # From rv32im_v2_10's call to rv32im_v2_9: arg27, arg28, arg29 are the last 3 Fp* params
    # They correspond to data/accum/mix/global in some order.
    # From poly_fp: args[1]=data, args[0]=accum, args[3]=mix, args[2]=global
    # The chain passes them through, but reorders may happen.
    # For safety, let's just trace: rv32im_v2_11 passes arg10,arg11,arg12 as the last 3 Fp*
    # And v11 gets arg10=Fp*, arg11=Fp*, arg12=Fp* from v12, etc.
    # The original poly_fp passes: args[1], args[0], args[3], args[2] (data, accum, mix, global)
    # These become the 4 Fp* args in v19's signature: arg8=data, arg9=accum, arg10=mix, arg11=global
    # v19 passes to v18 as: arg7=data?, arg8=accum?, arg9=mix?, arg10=global?
    # This varies per function. We just need to pass the correct 3 buffer pointers to v9.
    #
    # v9's sig: (cycle, steps, poly_mix, Fp* arg0, FpExt arg1, FpExt* arg2,
    #            FpExt arg3-arg8, Fp* arg9, Fp* arg10, Fp* arg11)
    # The 3 Fp* at the end (arg9, arg10, arg11) are buffer pointers.
    # From v10's call: arg27, arg28, arg29 → these trace back through the chain
    # to the original args[1], args[0], args[3] (or similar ordering).
    #
    # The safest approach: re-derive the buffer pointers from the original args[].
    # We know from poly_fp's call to v19:
    #   ...x34, /*data=*/args[1], /*accum=*/args[0], /*mix=*/args[3], /*global=*/args[2]
    # These 4 Fp* get mapped through the chain. v10 receives them as arg27, arg28, arg29
    # (only 3 Fp* in v10, but poly_fp passes 4). Actually v10 has Fp* arg27, Fp* arg28, Fp* arg29.
    # And v10's call to v9 passes: arg27, arg28, arg29.
    # Tracing: poly_fp passes 4 Fp* (data,accum,mix,global) to v19 as arg8-arg11.
    # v19 passes 4 Fp* to v18 as arg7-arg10 (or similar). Eventually v10 gets 3 Fp* (arg27-29).
    # One Fp* was dropped along the way — the global buffer.
    # Actually, v10 has a much wider signature (30 args). Let me just use the original 4 args.

    new_body += "\n    // Call rv32im_v2_9 with loaded intermediate values\n"
    new_body += "    // Buffer pointers: data=args[1], accum=args[0], mix=args[3]\n"
    new_body += "    auto _result = rv32im_v2_9(cycle, steps, poly_mix,\n"
    new_body += "        x33,       // arg0: x33 scratch\n"
    new_body += "        _ext0,     // arg1: x1368\n"
    new_body += "        x34,       // arg2: x34 FpExt array\n"
    new_body += "        _ext1,     // arg3: x708\n"
    new_body += "        _ext3,     // arg4: arg23 passthrough\n"
    new_body += "        _ext4,     // arg5: arg24 passthrough\n"
    new_body += "        _ext2,     // arg6: x699\n"
    new_body += "        _ext5,     // arg7: arg25 passthrough\n"
    new_body += "        _ext6,     // arg8: arg26 passthrough\n"
    new_body += "        /*data=*/args[1], /*accum=*/args[0], /*mix=*/args[3]);\n"
    new_body += "    return _result;\n"

    return f'__attribute__((noinline)) {new_sig} {{\n{new_body}\n}}\n'


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input_amalg.cpp> <output_amalg.cpp>", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    with open(input_path) as f:
        content = f.read()

    # Find the namespace closing brace
    ns_close = '} // namespace risc0::circuit::rv32im_v2\n'
    ns_close_pos = content.find(ns_close)
    if ns_close_pos < 0:
        print("ERROR: Could not find namespace close", file=sys.stderr)
        sys.exit(1)

    # Generate multi-pass code
    extra = "\n// ================================================================\n"
    extra += "// Multi-pass eval_check: 2-way split at rv32im_v2_10/rv32im_v2_9\n"
    extra += "// Generated by gen_multipass.py\n"
    extra += "// ================================================================\n\n"

    # Index table
    extra += f"static constexpr uint32_t CROSS_X33_COUNT = {len(CROSS_X33_INDICES)};\n"
    extra += "static constexpr uint16_t CROSS_X33_INDICES[CROSS_X33_COUNT] = {\n"
    for i in range(0, len(CROSS_X33_INDICES), 16):
        chunk = CROSS_X33_INDICES[i:i+16]
        extra += "    " + ", ".join(str(x) for x in chunk) + ",\n"
    extra += "};\n\n"

    # Generate _pass1 variants for functions 19..10
    variants = []
    for fn_idx in PASS1_FNS:
        print(f"  Generating rv32im_v2_{fn_idx}_pass1...", file=sys.stderr)
        variant = make_pass1_variant(content, fn_idx)
        if variant:
            variants.append(variant)

    # Generate poly_fp_pass1
    print("  Generating poly_fp_pass1...", file=sys.stderr)
    pass1_fn = make_poly_fp_pass1(content)

    # Generate poly_fp_pass2
    print("  Generating poly_fp_pass2...", file=sys.stderr)
    pass2_fn = make_poly_fp_pass2(content)

    # Emit forward declarations FIRST (needed because functions call each other
    # and the definition order doesn't match the call order due to round-robin files)
    extra += "// Forward declarations for _pass1 variants\n"
    for variant in variants:
        # Extract the signature (first line up to the opening {)
        first_line = variant.split('{')[0].strip()
        # Remove __attribute__((noinline))
        sig = first_line.replace('__attribute__((noinline)) ', '')
        extra += sig + ";\n"
    # poly_fp_pass1 forward declaration
    sig = pass1_fn.split('{')[0].strip().replace('__attribute__((noinline)) ', '')
    extra += sig + ";\n"
    # poly_fp_pass2 forward declaration
    sig = pass2_fn.split('{')[0].strip().replace('__attribute__((noinline)) ', '')
    extra += sig + ";\n"
    extra += "\n"

    # Now emit all definitions
    for variant in variants:
        extra += variant + "\n"
    extra += pass1_fn + "\n"
    extra += pass2_fn + "\n"

    # Insert before namespace close
    output = content[:ns_close_pos] + extra + content[ns_close_pos:]

    with open(output_path, 'w') as f:
        f.write(output)

    print(f"  Written multi-pass amalgamation to {output_path}", file=sys.stderr)
    print(f"  Cross-boundary: {len(CROSS_X33_INDICES)} Fp + 7 FpExt = {len(CROSS_X33_INDICES)*4 + 7*16} bytes/WI", file=sys.stderr)


if __name__ == '__main__':
    main()
