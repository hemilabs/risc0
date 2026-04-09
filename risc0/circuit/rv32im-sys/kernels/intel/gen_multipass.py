#!/usr/bin/env python3
"""Generate 2-way multi-pass eval_check as TWO separate amalgamation files.

Splits poly_fp at the rv32im_v2_10 → rv32im_v2_9 boundary into:
  pass1_amalg.cpp: poly_fp_pass1 + rv32im_v2_{19..10}_pass1 → writes intermediate
  pass2_amalg.cpp: poly_fp_pass2 + rv32im_v2_{9..0} → reads intermediate, writes check

Each file is ~27K lines — half the monolithic 52K — and compiles to a separate .so
so the GPU binary per kernel is half the size and fits in L2 cache.

Usage: python3 gen_multipass.py <mono_amalg.cpp> <pass1_output.cpp> <pass2_output.cpp>
"""

import re
import sys

# Cross-boundary x33 indices (197 indices, computed from amalgamation analysis)
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

PASS1_FNS = list(range(19, 9, -1))  # 19..10
PASS2_FNS = list(range(9, -1, -1))   # 9..0

# The by-value FpExt args at the v10→v9 boundary
BOUNDARY_BYVAL_EXTS = ['x1368', 'x708', 'x699', 'arg23', 'arg24', 'arg25', 'arg26']


def find_function(content, fn_name):
    """Find a function definition (with __attribute__ prefix) and return (start, end, sig, body)."""
    # Try with noinline attribute first
    patterns = [
        f'__attribute__((noinline)) FpExt {fn_name}(size_t',
        f'FpExt {fn_name}(size_t',
    ]
    for pattern in patterns:
        pos = 0
        while True:
            idx = content.find(pattern, pos)
            if idx < 0:
                break
            # Check for opening brace (definition, not declaration)
            brace_search = content[idx:idx+5000]
            brace_pos_rel = None
            for ci, ch in enumerate(brace_search):
                if ch == '{':
                    brace_pos_rel = ci
                    break
                if ch == ';':
                    break  # Declaration, skip
            if brace_pos_rel is None:
                pos = idx + 1
                continue

            brace_pos = idx + brace_pos_rel
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

            pos = idx + 1
    return None


def extract_declarations(content, fn_indices):
    """Extract forward declarations for the given function indices."""
    decls = []
    for idx in fn_indices:
        fn_name = f'rv32im_v2_{idx}'
        # Find declaration (ends with ;)
        pattern = f'FpExt {fn_name}(size_t'
        pos = content.find(pattern)
        if pos >= 0:
            end = content.index(';', pos) + 1
            decl = content[pos:end].strip()
            if '__attribute__' not in decl:
                decls.append(decl)
    return decls


def make_pass1_variant(content, fn_idx):
    """Create a _pass1 variant that propagates intermediate buffer params."""
    fn_name = f'rv32im_v2_{fn_idx}'
    result = find_function(content, fn_name)
    if not result:
        print(f"WARNING: Could not find {fn_name}", file=sys.stderr)
        return ""

    _, _, sig, body = result
    new_name = f'rv32im_v2_{fn_idx}_pass1'
    new_sig = sig.replace(f'{fn_name}(', f'{new_name}(')
    new_sig = new_sig.replace('__attribute__((noinline)) ', '')

    # Add extra params before the closing paren
    last_paren = new_sig.rfind(')')
    new_sig = new_sig[:last_paren] + ', Fp* d_inter_fp, FpExt* d_inter_ext, size_t domain' + new_sig[last_paren:]

    new_body = body

    if fn_idx == 10:
        # Boundary function: replace tail-call with intermediate writes
        tail_pattern = r'auto\s+x\d+\s*=\s*rv32im_v2_9\([^;]+;'
        return_pattern = r'\s*return\s+x\d+;'

        replacement = "// === MULTI-PASS BOUNDARY: write intermediate state ===\n"
        replacement += "    for (uint32_t _i = 0; _i < CROSS_X33_COUNT; _i++) {\n"
        replacement += "        d_inter_fp[_i * domain + cycle] = arg0[CROSS_X33_INDICES[_i]];\n"
        replacement += "    }\n"
        for i, ext_name in enumerate(BOUNDARY_BYVAL_EXTS):
            replacement += f"    d_inter_ext[{i} * domain + cycle] = {ext_name};\n"
        replacement += "    return FpExt(0);"

        new_body = re.sub(tail_pattern + return_pattern, replacement, new_body)
    else:
        callee = fn_idx - 1
        old_call = f'rv32im_v2_{callee}(cycle, steps, poly_mix,'
        new_call = f'rv32im_v2_{callee}_pass1(cycle, steps, poly_mix,'
        new_body = new_body.replace(old_call, new_call)

        # Add extra params to the tail-call
        tail_re = re.compile(rf'(auto\s+x\d+\s*=\s*rv32im_v2_{callee}_pass1\([^)]+)\)')
        m = tail_re.search(new_body)
        if m:
            new_body = new_body[:m.end(1)] + ', d_inter_fp, d_inter_ext, domain)' + new_body[m.end():]

    return f'__attribute__((noinline)) {new_sig} {{\n{new_body}\n}}\n'


def make_poly_fp_pass1(content):
    """Create poly_fp_pass1 that calls the _pass1 chain."""
    result = find_function(content, 'poly_fp')
    if not result:
        return ""

    _, _, sig, body = result
    new_sig = sig.replace('poly_fp(', 'poly_fp_pass1(')
    new_sig = new_sig.replace('__attribute__((noinline)) ', '')
    last_paren = new_sig.rfind(')')
    new_sig = new_sig[:last_paren] + ', Fp* d_inter_fp, FpExt* d_inter_ext, size_t domain' + new_sig[last_paren:]

    new_body = body
    new_body = new_body.replace('rv32im_v2_19(cycle, steps, poly_mix,',
                                 'rv32im_v2_19_pass1(cycle, steps, poly_mix,')
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

    # Preamble: everything before the rv32im_v2_19 call
    call_idx = body.find('rv32im_v2_19(')
    if call_idx < 0:
        return ""
    line_start = body.rfind('\n', 0, call_idx) + 1
    preamble = body[:line_start]

    new_body = preamble
    new_body += "\n    // === MULTI-PASS PASS2: load intermediate state ===\n"
    new_body += "    for (uint32_t _i = 0; _i < CROSS_X33_COUNT; _i++) {\n"
    new_body += "        x33[CROSS_X33_INDICES[_i]] = d_inter_fp[_i * domain + cycle];\n"
    new_body += "    }\n"

    for i in range(7):
        new_body += f"    FpExt _ext{i} = d_inter_ext[{i} * domain + cycle];\n"

    # v9 sig: (cycle, steps, poly_mix, Fp* arg0, FpExt arg1, FpExt* arg2,
    #          FpExt arg3..arg8, Fp* arg9, Fp* arg10, Fp* arg11)
    # Mapping: arg0=x33, arg1=x1368, arg2=x34, arg3=x708,
    #          arg4=arg23, arg5=arg24, arg6=x699, arg7=arg25, arg8=arg26,
    #          arg9=data(args[1]), arg10=accum(args[0]), arg11=mix(args[3])
    new_body += "\n    auto _result = rv32im_v2_9(cycle, steps, poly_mix,\n"
    new_body += "        x33, _ext0, x34, _ext1, _ext3, _ext4, _ext2, _ext5, _ext6,\n"
    new_body += "        /*data=*/args[1], /*accum=*/args[0], /*mix=*/args[3]);\n"
    new_body += "    return _result;\n"

    return f'__attribute__((noinline)) {new_sig} {{\n{new_body}\n}}\n'


def gen_index_table():
    """Generate C++ cross-boundary index table."""
    code = f"static constexpr uint32_t CROSS_X33_COUNT = {len(CROSS_X33_INDICES)};\n"
    code += "static constexpr uint16_t CROSS_X33_INDICES[CROSS_X33_COUNT] = {\n"
    for i in range(0, len(CROSS_X33_INDICES), 16):
        chunk = CROSS_X33_INDICES[i:i+16]
        code += "    " + ", ".join(str(x) for x in chunk) + ",\n"
    code += "};\n"
    return code


def gen_kernel_wrapper(pass_num):
    """Generate SYCL kernel wrapper for pass1 or pass2."""
    if pass_num == 1:
        return '''
#include <sycl/sycl.hpp>
#include <cstring>

using namespace risc0;

static const char* make_error(const char* msg) { return strdup(msg); }

extern "C" const char* risc0_circuit_rv32im_intel_eval_check_pass1(
    void* queue_ptr, void* d_inter_fp, void* d_inter_ext,
    const void* d_data, const void* d_accum, const void* d_out, const void* d_mix,
    const void* d_poly_mix, uint32_t domain)
{
    auto* q = static_cast<sycl::queue*>(queue_ptr);
    try {
        auto* inter_fp = static_cast<Fp*>(d_inter_fp);
        auto* inter_ext = static_cast<FpExt*>(d_inter_ext);
        auto* data = static_cast<Fp*>(const_cast<void*>(d_data));
        auto* accum = static_cast<Fp*>(const_cast<void*>(d_accum));
        auto* out = static_cast<Fp*>(const_cast<void*>(d_out));
        auto* mix = static_cast<Fp*>(const_cast<void*>(d_mix));
        auto* poly_mix = static_cast<FpExt*>(const_cast<void*>(d_poly_mix));
        constexpr uint32_t WG = 256, PM = 458;
        uint32_t gs = ((domain + WG - 1) / WG) * WG;
        q->submit([&](sycl::handler& h) {
            sycl::local_accessor<FpExt, 1> slm(sycl::range<1>(PM), h);
            h.parallel_for(sycl::nd_range<1>(gs, WG),
                [=](sycl::nd_item<1> item) {
                    uint32_t cycle = item.get_global_id(0);
                    uint32_t lid = item.get_local_id(0);
                    for (uint32_t i = lid; i < PM; i += WG) slm[i] = poly_mix[i];
                    sycl::group_barrier(item.get_group());
                    if (cycle >= domain) return;
                    Fp* args[4] = {accum, data, out, mix};
                    FpExt* pm = slm.get_multi_ptr<sycl::access::decorated::no>().get();
                    circuit::rv32im_v2::poly_fp_pass1(
                        (size_t)cycle, (size_t)domain, pm, args,
                        inter_fp, inter_ext, (size_t)domain);
                });
        });
        return nullptr;
    } catch (const sycl::exception& e) { return make_error(e.what()); }
    catch (const std::exception& e) { return make_error(e.what()); }
    catch (...) { return make_error("Unknown error"); }
}
'''
    else:  # pass 2
        return '''
#include <sycl/sycl.hpp>
#include <cstring>

using namespace risc0;

static const char* make_error(const char* msg) { return strdup(msg); }

extern "C" const char* risc0_circuit_rv32im_intel_eval_check_pass2(
    void* queue_ptr, void* d_check,
    const void* d_inter_fp, const void* d_inter_ext,
    const void* d_data, const void* d_accum, const void* d_out, const void* d_mix,
    const void* d_poly_mix, uint32_t rou_raw, uint32_t po2, uint32_t domain)
{
    auto* q = static_cast<sycl::queue*>(queue_ptr);
    try {
        auto* check = static_cast<Fp*>(d_check);
        auto* inter_fp = static_cast<const Fp*>(d_inter_fp);
        auto* inter_ext = static_cast<const FpExt*>(d_inter_ext);
        auto* data = static_cast<Fp*>(const_cast<void*>(d_data));
        auto* accum = static_cast<Fp*>(const_cast<void*>(d_accum));
        auto* out = static_cast<Fp*>(const_cast<void*>(d_out));
        auto* mix = static_cast<Fp*>(const_cast<void*>(d_mix));
        auto* poly_mix = static_cast<FpExt*>(const_cast<void*>(d_poly_mix));
        Fp rou_val;
        std::memcpy(&rou_val, &rou_raw, sizeof(uint32_t));
        constexpr uint32_t WG = 256, PM = 458;
        uint32_t gs = ((domain + WG - 1) / WG) * WG;
        q->submit([&](sycl::handler& h) {
            sycl::local_accessor<FpExt, 1> slm(sycl::range<1>(PM), h);
            h.parallel_for(sycl::nd_range<1>(gs, WG),
                [=](sycl::nd_item<1> item) {
                    uint32_t cycle = item.get_global_id(0);
                    uint32_t lid = item.get_local_id(0);
                    for (uint32_t i = lid; i < PM; i += WG) slm[i] = poly_mix[i];
                    sycl::group_barrier(item.get_group());
                    if (cycle >= domain) return;
                    Fp* args[4] = {accum, data, out, mix};
                    FpExt* pm = slm.get_multi_ptr<sycl::access::decorated::no>().get();
                    FpExt tot = circuit::rv32im_v2::poly_fp_pass2(
                        (size_t)cycle, (size_t)domain, pm, args,
                        inter_fp, inter_ext, (size_t)domain);
                    Fp x = Fp(3) * pow(rou_val, cycle);
                    Fp y = pow(x, uint32_t(1) << po2);
                    Fp quot = inv(y - Fp(1));
                    for (uint32_t i = 0; i < 4; i++)
                        check[i * domain + cycle] = tot.elems[i] * quot;
                });
        });
        return nullptr;
    } catch (const sycl::exception& e) { return make_error(e.what()); }
    catch (const std::exception& e) { return make_error(e.what()); }
    catch (...) { return make_error("Unknown error"); }
}
'''


def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <mono_amalg.cpp> <pass1_output.cpp> <pass2_output.cpp>", file=sys.stderr)
        sys.exit(1)

    mono_path = sys.argv[1]
    pass1_path = sys.argv[2]
    pass2_path = sys.argv[3]

    with open(mono_path) as f:
        content = f.read()

    # Extract the header (before namespace) + namespace constant
    ns_open = 'namespace risc0::circuit::rv32im_v2 {\n'
    ns_idx = content.find(ns_open)
    header = content[:ns_idx + len(ns_open)]
    # Include kInvRate constant (needed by the generated code)
    header += 'constexpr size_t kInvRate = 4;\n'

    # Extract namespace body
    ns_close = '} // namespace risc0::circuit::rv32im_v2\n'
    ns_close_idx = content.find(ns_close)
    ns_body = content[ns_idx + len(ns_open):ns_close_idx]

    # =============================================
    # Generate PASS 1 amalgamation
    # =============================================
    print("  Generating pass1 amalgamation...", file=sys.stderr)
    pass1 = header

    # Add index table
    pass1 += gen_index_table() + "\n"

    # Forward declarations for _pass1 functions
    variants = []
    for fn_idx in PASS1_FNS:
        variant = make_pass1_variant(content, fn_idx)
        if variant:
            variants.append(variant)
            sig = variant.split('{')[0].strip().replace('__attribute__((noinline)) ', '')
            pass1 += sig + ";\n"

    # poly_fp_pass1 forward declaration
    pass1_fn = make_poly_fp_pass1(content)
    sig = pass1_fn.split('{')[0].strip().replace('__attribute__((noinline)) ', '')
    pass1 += sig + ";\n\n"

    # Function definitions
    for variant in variants:
        pass1 += variant + "\n"
    pass1 += pass1_fn + "\n"

    pass1 += ns_close
    pass1 += gen_kernel_wrapper(1)

    with open(pass1_path, 'w') as f:
        f.write(pass1)
    print(f"  Written pass1: {pass1_path} ({pass1.count(chr(10))} lines)", file=sys.stderr)

    # =============================================
    # Generate PASS 2 amalgamation
    # =============================================
    print("  Generating pass2 amalgamation...", file=sys.stderr)
    pass2 = header

    # Add index table
    pass2 += gen_index_table() + "\n"

    # Forward declarations for v9..v0 (original functions, needed for the chain)
    for fn_idx in PASS2_FNS:
        fn_name = f'rv32im_v2_{fn_idx}'
        # Find the declaration in the amalgamation
        decl_pattern = f'FpExt {fn_name}(size_t'
        pos = content.find(decl_pattern)
        if pos >= 0:
            end = content.index(';', pos) + 1
            decl = content[pos:end].strip()
            if '__attribute__' not in decl:
                pass2 += decl + "\n"

    # poly_fp_pass2 forward declaration
    pass2_fn = make_poly_fp_pass2(content)
    sig = pass2_fn.split('{')[0].strip().replace('__attribute__((noinline)) ', '')
    pass2 += sig + ";\n\n"

    # Function definitions for v9..v0 (the ORIGINAL functions, with noinline)
    for fn_idx in PASS2_FNS:
        fn_name = f'rv32im_v2_{fn_idx}'
        result = find_function(content, fn_name)
        if result:
            _, end, sig, body = result
            pass2 += f'{sig} {{\n{body}\n}}\n\n'

    # poly_fp_pass2
    pass2 += pass2_fn + "\n"

    pass2 += ns_close
    pass2 += gen_kernel_wrapper(2)

    with open(pass2_path, 'w') as f:
        f.write(pass2)
    print(f"  Written pass2: {pass2_path} ({pass2.count(chr(10))} lines)", file=sys.stderr)
    print(f"  Cross-boundary: {len(CROSS_X33_INDICES)} Fp + 7 FpExt = {len(CROSS_X33_INDICES)*4 + 7*16} bytes/WI", file=sys.stderr)


if __name__ == '__main__':
    main()
