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

# --- v3.1.0 rebase: multipass re-derived --------------------------------
# The cross-boundary constants below (CROSS_X33_INDICES, BOUNDARY_BYVAL_EXTS,
# CROSS_X34_INDICES) and PM=484 were re-derived for the risc0 v3.1.0 rv32im
# circuit DAG by a dataflow analysis of the mono amalgamation
# (intel_eval_check_mono.cpp). The poly_fp accumulator chain threads two local
# arrays -- x33 (Fp[1018]) and x34 (FpExt[125]) -- plus the running FpExt total
# (passed by value) down through rv32im_v2_19..0. Splitting at the v10->v9 call
# means pass2 (v9..v0) must see exactly the x33/x34 slots and by-value args that
# v10..v19 produced. Because each vN renumbers its locals, the identity of the
# x33/x34 arrays is followed by propagating the parameter position through every
# vN->v(N-1) call from poly_fp's v19 call (where they are literally x33/x34):
#   CROSS_X33 = {k : x33[k] written by v10..v19} ∩ {k : x33[k] read by v9..v0}.
#   CROSS_X34 = the same intersection on the x34[] FpExt array.
# A slot read by v9..v0 and NOT written by the preamble, by v10..v19, or by
# v9..v0 itself would be an uninitialized read -- the completeness invariant
# (verified: no such slot exists) is what makes this set provably sufficient.
#   BOUNDARY_BYVAL_EXTS = the 7 by-value FpExt args at the v10->v9 call, in
#   storage order; pass2's v9 call reorders them to v9's parameter order.
#   Buffer sizes are mirrored in rv32im/src/prove/hal/intel.rs
#   (n_inter_fp = len(CROSS_X33), n_inter_ext = 7 + len(CROSS_X34)) -- keep in sync.
#   Validated bit-exact vs mono and CPU by the eval_check_*_multipass tests (PO2=4).
#   Multipass stays opt-in (RISC0_MULTIPASS), set nowhere in production; mono
#   RISC0_INTEL_EVAL_FAST remains the shipping Intel path.
MULTIPASS_SUPPORTED = True


def gen_stub(pass_num):
    """A pass1/pass2 translation unit that defines only the extern "C" entry
    point and returns a clear error, so the crate still links and the mono
    path is unaffected, but setting RISC0_MULTIPASS fails loudly rather than
    proving with a mis-derived kernel."""
    msg = (
        "multipass eval_check is unavailable: its cross-boundary derivation in "
        "gen_multipass.py has not been re-done for the risc0 v3.1.0 circuit. "
        "Unset RISC0_MULTIPASS to use the production mono RISC0_INTEL_EVAL_FAST "
        "kernel."
    )
    if pass_num == 1:
        sig = (
            'extern "C" const char* risc0_circuit_rv32im_intel_eval_check_pass1(\n'
            '    void*, void*, void*, const void*, const void*, const void*, '
            'const void*,\n    const void*, uint32_t)'
        )
    else:
        sig = (
            'extern "C" const char* risc0_circuit_rv32im_intel_eval_check_pass2(\n'
            '    void*, void*, const void*, const void*, const void*, const void*, '
            'const void*,\n    const void*, const void*, uint32_t, uint32_t, uint32_t)'
        )
    return (
        "// Stub: multipass eval_check deferred for the v3.1.0 circuit\n"
        "// (gen_multipass.py MULTIPASS_SUPPORTED=False).\n"
        "#include <cstdint>\n#include <cstring>\n\n"
        f"{sig} {{\n"
        f'    return strdup("{msg}");\n'
        "}\n"
    )


# Cross-boundary x33 indices (196): x33[k] written in pass1 (v10..v19) and read
# in pass2 (v9..v0), from the dataflow analysis of the v3.1.0 mono described above.
CROSS_X33_INDICES = [
    124, 129, 132, 135, 137, 183, 197, 206, 238, 253, 270, 271, 272, 274, 285, 299,
    300, 305, 320, 344, 362, 365, 376, 377, 385, 387, 388, 403, 404, 405, 406, 409,
    414, 416, 422, 428, 430, 434, 435, 441, 443, 444, 445, 446, 447, 448, 449, 450,
    451, 452, 453, 463, 464, 465, 466, 468, 474, 481, 504, 505, 506, 507, 508, 538,
    544, 545, 546, 547, 548, 549, 550, 569, 572, 577, 589, 597, 601, 605, 609, 610,
    629, 630, 631, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645,
    646, 650, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 708, 709, 710,
    711, 712, 713, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728,
    729, 730, 731, 732, 733, 734, 737, 742, 743, 745, 746, 747, 748, 749, 750, 751,
    752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767,
    768, 769, 770, 771, 772, 773, 774, 867, 869, 870, 876, 960, 969, 981, 983, 984,
    985, 986, 987, 988, 992, 994, 995, 996, 997, 999, 1001, 1002, 1003, 1004, 1006, 1008,
    1009, 1010, 1015, 1017,
]

PASS1_FNS = list(range(19, 9, -1))  # 19..10
PASS2_FNS = list(range(9, -1, -1))   # 9..0

# The by-value FpExt args at the v10→v9 boundary
BOUNDARY_BYVAL_EXTS = ['x1396', 'x745', 'x736', 'arg23', 'arg24', 'arg25', 'arg26']

# Cross-boundary x34 (FpExt) indices: arg2[k] written by pass1 (v10..v19) and
# read by pass2 (v9..v0). v3.1.0: x34[1] is written by v13 and read by v9;
# x34[3] is written by v12 and read by v9. Neither is written by the poly_fp
# preamble, so both must cross (a pass2 that omits x34[1] reads an
# uninitialized FpExt slot). Derived by the dataflow analysis described above.
CROSS_X34_INDICES = [1, 3]


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
        replacement += "    // Write x33 cross-boundary indices\n"
        replacement += "    for (uint32_t _i = 0; _i < CROSS_X33_COUNT; _i++) {\n"
        replacement += "        d_inter_fp[_i * domain + cycle] = arg0[CROSS_X33_INDICES[_i]];\n"
        replacement += "    }\n"
        replacement += "    // Write 7 FpExt by-value args\n"
        for i, ext_name in enumerate(BOUNDARY_BYVAL_EXTS):
            replacement += f"    d_inter_ext[{i} * domain + cycle] = {ext_name};\n"
        replacement += "    // Write x34 cross-boundary indices (arg2 is x34 in v10)\n"
        for i, x34_idx in enumerate(CROSS_X34_INDICES):
            slot = len(BOUNDARY_BYVAL_EXTS) + i
            replacement += f"    d_inter_ext[{slot} * domain + cycle] = arg2[{x34_idx}];\n"
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


def extract_hoisted_params(content, fn_name):
    """Extract CSE-hoisted `_h_data_*` parameter names from a function's
    declaration (or definition signature). Returns [] if CSE isn't applied.
    The CSE pass (poly_fp_cse.py) appends `Fp _h_data_<col>_<back>` params
    to v9 when it hoists shared `args[buf][col*steps+back]` reads into the
    parent `poly_fp`. These names ARE in scope at the pass2 v9 call site
    because pass2's preamble carries them through verbatim from poly_fp."""
    decl_re = re.compile(
        rf'FpExt\s+{re.escape(fn_name)}\s*\((?P<params>[^)]*)\)\s*[;{{]'
    )
    m = decl_re.search(content)
    if not m:
        return []
    params = m.group('params')
    names = []
    for piece in params.split(','):
        piece = piece.strip()
        # Match `Fp _h_data_<digits>_<digits>` (the CSE pass's name pattern).
        nm = re.match(r'Fp\s+(_h_data_\d+_\d+)\s*$', piece)
        if nm:
            names.append(nm.group(1))
    return names


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
    new_body += "    // Load x33 cross-boundary indices\n"
    new_body += "    for (uint32_t _i = 0; _i < CROSS_X33_COUNT; _i++) {\n"
    new_body += "        x33[CROSS_X33_INDICES[_i]] = d_inter_fp[_i * domain + cycle];\n"
    new_body += "    }\n"

    new_body += "    // Load 7 FpExt by-value args\n"
    for i in range(7):
        new_body += f"    FpExt _ext{i} = d_inter_ext[{i} * domain + cycle];\n"

    new_body += "    // Load x34 cross-boundary indices into x34[]\n"
    for i, x34_idx in enumerate(CROSS_X34_INDICES):
        slot = 7 + i
        new_body += f"    x34[{x34_idx}] = d_inter_ext[{slot} * domain + cycle];\n"

    # v9 sig: (cycle, steps, poly_mix, Fp* arg0, FpExt arg1, FpExt* arg2,
    #          FpExt arg3..arg8, Fp* arg9, Fp* arg10, Fp* arg11)
    # Mapping: arg0=x33, arg1=x1368, arg2=x34, arg3=x708,
    #          arg4=arg23, arg5=arg24, arg6=x699, arg7=arg25, arg8=arg26,
    #          arg9=data(args[1]), arg10=accum(args[0]), arg11=mix(args[3])
    # CSE-aware v9 call. If poly_fp_cse.py hoisted `_h_data_*` params into
    # v9's signature, append them to the call (their definitions live in
    # `preamble`, which is the slice of poly_fp before rv32im_v2_19 — the
    # CSE pass injects hoisted reads BEFORE that call, so they're in scope
    # here too).
    hoisted = extract_hoisted_params(content, 'rv32im_v2_9')
    new_body += "\n    auto _result = rv32im_v2_9(cycle, steps, poly_mix,\n"
    new_body += "        x33, _ext0, x34, _ext1, _ext3, _ext4, _ext2, _ext5, _ext6,\n"
    if hoisted:
        new_body += "        /*data=*/args[1], /*accum=*/args[0], /*mix=*/args[3],\n"
        # 4-per-line for readability
        for i in range(0, len(hoisted), 4):
            chunk = hoisted[i:i+4]
            comma = "," if i + 4 < len(hoisted) else ");"
            new_body += "        " + ", ".join(chunk) + comma + "\n"
    else:
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
        constexpr uint32_t WG = 256, PM = 484;  // v3.1.0: poly_mix is 484 wide (see MULTIPASS_SUPPORTED)
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
        constexpr uint32_t WG = 256, PM = 484;  // v3.1.0: poly_mix is 484 wide (see MULTIPASS_SUPPORTED)
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

    if not MULTIPASS_SUPPORTED:
        with open(pass1_path, 'w') as f:
            f.write(gen_stub(1))
        with open(pass2_path, 'w') as f:
            f.write(gen_stub(2))
        print(
            "  gen_multipass: MULTIPASS_SUPPORTED=False -- wrote stub pass1/pass2; "
            "multipass deferred for the v3.1.0 circuit, mono RISC0_INTEL_EVAL_FAST "
            "is the production path",
            file=sys.stderr,
        )
        return

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
