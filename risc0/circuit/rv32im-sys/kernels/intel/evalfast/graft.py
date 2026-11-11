# graft.py <variant_amalg.cpp> <current_untransformed_amalg.cpp> <out.cpp> [--absorb]
# Take the device-code head of a round-4 variant and the CURRENT kernel wrapper tail (A1 slab/tiling).
# --absorb: re-apply the allm8 wrapper edits (SLM Q-table + rk_hoist prologue).
import sys
k = '// Intel SYCL eval_check kernel'
v = open(sys.argv[1]).read(); c = open(sys.argv[2]).read()
tail = c[c.find(k):]
if '--absorb' in sys.argv:
    a = 'sycl::local_accessor<FpExt, 1> slm(sycl::range<1>(POLY_MIX_COUNT), h);'
    assert tail.count(a) == 1
    tail = tail.replace(a, 'sycl::local_accessor<FpExt, 1> slm(sycl::range<1>(POLY_MIX_COUNT + circuit::rv32im_v2::RK_QN + circuit::rv32im_v2::RK_HN), h);')
    b = 'for (uint32_t i = lid; i < POLY_MIX_COUNT; i += wgsz) slm[i] = poly_mix[i];\n'
    assert tail.count(b) == 1
    ind = '                    '
    tail = tail.replace(b, b + ind + 'if (lid == 0) circuit::rv32im_v2::rk_hoist(poly_mix, out, mix, slm.get_multi_ptr<sycl::access::decorated::no>().get() + POLY_MIX_COUNT + circuit::rv32im_v2::RK_QN);\n'
                        + ind + 'for (uint32_t i = lid; i < circuit::rv32im_v2::RK_QN; i += wgsz) slm[POLY_MIX_COUNT + i] = poly_mix[circuit::rv32im_v2::rk_qa[i]] * poly_mix[circuit::rv32im_v2::rk_qb[i]];\n')
open(sys.argv[3], 'w').write(v[:v.find(k)] + tail)
