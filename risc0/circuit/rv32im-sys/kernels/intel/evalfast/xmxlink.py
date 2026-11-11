#!/usr/bin/env python3
# xmxlink.py in.cpp out.cpp [LMIN]: evaluate the lazy poly_mix linking chains
# (lazy2.py's `RkAcc x = rk_link(base, c, poly_mix[k])` ... `rk_fin(...)`) on
# the XMX units with u8 x u8 -> s32 DPAS. Exact:
#   * each Fp coefficient c (canonical, c < P < 2^31) is split into 4 bytes
#     c = sum_b c_b 2^(8b); the B operand is 8 terms' c words as they are;
#   * the A operand holds the bytes of pm * 2^(8a) mod P (per FpExt component),
#     so one DPAS pair accumulates sum_terms c_b * byte(pm 2^(8a) mod P) into
#     s32 lanes; the per-chain S_j = sum_a C[j,a] 2^(8a) is congruent to
#     sum c * pm mod P (times the Montgomery factor), and one Montgomery REDC
#     per component (linear mod P) gives the same field value as rk_link/rk_fin.
#   * bounds: a lane accumulates <= 32 products of bytes per DPAS, i.e.
#     <= 32 * 255^2 < 2^21 per block; MAX_BLOCKS blocks per chain keep every
#     s32 lane < 2^31 and S < P * 2^32 (one REDC suffices).
# The A table (RK_XG aligned groups of 8 poly_mix entries x 512 B) is built by
# two small pre-kernels per launch and copied into SLM after the poly_mix /
# QN / HN entries. The always-zero term `(b0|..|b7) >> 31` in rk_xmad keeps IGC
# from hoisting the A loads out of the chain (which costs ~70 extra GRFs);
# it is 0 because every c is a canonical Fp (the kernel's loads and Fp
# arithmetic are exact only for canonical inputs anyway).
import re, sys

src, dst = sys.argv[1], sys.argv[2]
LMIN = int(sys.argv[3]) if len(sys.argv) > 3 else 4
PMBASE = 458          # global poly_mix entries (rewrite.py PMBASE)
MAX_BLOCKS = 512      # 512 * 32 * 255^2 < 2^31

L = open(src).read().split('\n')
txt = '\n'.join(L)
link = re.compile(r'^(\s*)RkAcc (x\d+) = (rk_link(?:_nf)?)\((\w+), (\w+), poly_mix\[(\d+)\]\);$')
finl = re.compile(r'^(\s*)FpExt (x\d+) = rk_fin\((rk_link(?:_nf)?)\((\w+), (\w+), poly_mix\[(\d+)\]\)\);$')
fstart = re.compile(r'^(?:__attribute__\(\(noinline\)\) )?FpExt (rv32im_v2_\d+|poly_fp)\(.*\) \{$')
QN = int(re.search(r'constexpr uint32_t RK_QN = (\d+);', txt).group(1))
HN = int(re.search(r'constexpr uint32_t RK_HN = (\d+);', txt).group(1))
PMF = PMBASE + QN + HN

chains = []; cur = {}; fn = None
for i, l in enumerate(L):
    if fstart.match(l):
        fn = i; cur = {}
    if fn is None:
        continue
    m = link.match(l) or finl.match(l)
    if m:
        ind, d, kind, base, c, k = m.groups()
        ch = cur.pop(base, None)
        if ch is None:
            ch = dict(base=base, terms=[]); chains.append(ch)
        ch['terms'].append((i, c, int(k)))
        if finl.match(l):
            ch['end'] = (i, d)
        else:
            cur[d] = ch
assert chains, 'xmxlink: no lazy poly_mix chains found (lazy2.py output shape changed?)'
assert all('end' in c for c in chains), 'xmxlink: unterminated rk_link chain'
sel = [c for c in chains if len(c['terms']) >= LMIN]
maxk = max(k for c in sel for (_, _, k) in c['terms'])
assert maxk < PMF, 'xmxlink: poly_mix[%d] outside the PMF=%d table' % (maxk, PMF)
XG = maxk // 8 + 1

rep = {}; nblk = 0
for n, ch in enumerate(sel):
    T = ch['terms']; nm = '_xc%d' % n
    filled = set(); cblk = 0
    for j, (i, c, k) in enumerate(T):
        ind = re.match(r'^(\s*)', L[i]).group(1)
        out = []
        if j == 0:
            out.append('%sRkX %s; rk_xinit(%s, %s);' % (ind, nm, nm, ch['base']))
        g = k // 8
        if k % 8 in filled:   # slot collision inside the open block: flush first
            for s in range(8):
                if s not in filled:
                    out.append('%s%s.b[%d] = 0u;' % (ind, nm, s))
            out.append('%srk_xmad(%s, poly_mix, %du);' % (ind, nm, g)); nblk += 1; cblk += 1; filled = set()
        out.append('%srk_xput(%s, %d, %s, poly_mix[%d]);' % (ind, nm, k % 8, c, k)); filled.add(k % 8)
        last = j == len(T) - 1
        if last or T[j + 1][2] // 8 != g:
            for s in range(8):
                if s not in filled:
                    out.append('%s%s.b[%d] = 0u;' % (ind, nm, s))
            out.append('%srk_xmad(%s, poly_mix, %du);' % (ind, nm, g)); nblk += 1; cblk += 1; filled = set()
        if last:
            out.append('%sFpExt %s = rk_xfin(%s);' % (ind, ch['end'][1], nm))
        rep[i] = '\n'.join(out)
    assert cblk <= MAX_BLOCKS, 'xmxlink: chain with %d DPAS blocks (> %d) could overflow s32' % (cblk, MAX_BLOCKS)
for i, r in rep.items():
    L[i] = r
out = '\n'.join(L)

helpers = r'''
// ---- xmxlink.py: poly_mix chains on XMX (DPAS u8*u8->s32) ----
typedef unsigned short rk_us8 __attribute__((ext_vector_type(8)));
typedef unsigned int rk_u8v __attribute__((ext_vector_type(8)));
typedef int rk_i8v __attribute__((ext_vector_type(8)));
#ifdef __SYCL_DEVICE_ONLY__
// cl_intel_subgroup_matrix_multiply_accumulate builtins (resolved by IGC at
// AOT finalization; joint_matrix is unavailable on this runtime for BMG).
SYCL_EXTERNAL rk_i8v intel_sub_group_u8_u8_matrix_mad_k32(rk_us8 a, rk_u8v b, rk_i8v acc);
SYCL_EXTERNAL rk_us8 intel_sub_group_block_read_us8(const __attribute__((opencl_local)) unsigned short* p);
#define RK_LPTR(p) ((const __attribute__((opencl_local)) unsigned short*)(p))
#else
static inline rk_i8v intel_sub_group_u8_u8_matrix_mad_k32(rk_us8, rk_u8v, rk_i8v acc) { return acc; }
static inline rk_us8 intel_sub_group_block_read_us8(const unsigned short*) { return rk_us8(0); }
#define RK_LPTR(p) (p)
#endif
'''
body = r'''
// ---- xmxlink.py helpers ----
constexpr uint32_t RK_XG = %d;                // aligned pm groups in the SLM A table
constexpr uint32_t RK_XBASE = %d + %d + %d;    // FpExt slot where the A table starts (after poly_mix, QN, HN)
struct RkX { rk_i8v c0, c1; rk_u8v b; RkAcc side; FpExt base; };
__attribute__((always_inline)) static inline void rk_xinit(RkX& x, const FpExt& base) { x.c0 = 0; x.c1 = 0; x.base = base; for (int j = 0; j < 4; j++) { x.side.lo[j] = 0; x.side.hi[j] = 0; } }
__attribute__((always_inline)) static inline void rk_xinit(RkX& x, const RkAcc& base) { x.c0 = 0; x.c1 = 0; x.base = FpExt(0); x.side = base; }
__attribute__((always_inline)) static inline void rk_xput(RkX& x, int s, Fp c, const FpExt&) { x.b[s] = c.asRaw(); }
__attribute__((always_inline)) static inline void rk_xput(RkX& x, int s, FpExt c, const FpExt& pm) { x.b[s] = 0u; x.side = rk_link(x.side, c, pm); }
__attribute__((always_inline)) static inline void rk_xmad(RkX& x, const FpExt* poly_mix, uint32_t g) {
  const uint32_t z = (x.b[0] | x.b[1] | x.b[2] | x.b[3] | x.b[4] | x.b[5] | x.b[6] | x.b[7]) >> 31;   // 0 (c < P): keeps IGC from hoisting the A loads
  const unsigned short* A = reinterpret_cast<const unsigned short*>(poly_mix + RK_XBASE) + g * 256u + z * 16u;
  const rk_us8 a0 = intel_sub_group_block_read_us8(RK_LPTR(A));
  const rk_us8 a1 = intel_sub_group_block_read_us8(RK_LPTR(A + 128));
  x.c0 = intel_sub_group_u8_u8_matrix_mad_k32(a0, x.b, x.c0);
  x.c1 = intel_sub_group_u8_u8_matrix_mad_k32(a1, x.b, x.c1);
}
__attribute__((always_inline)) static inline FpExt rk_xfin(const RkX& x) {
  uint32_t r[4];
  for (int j = 0; j < 4; j++) {
    const uint32_t o0 = (uint32_t)(j < 2 ? x.c0[(j & 1) * 4 + 0] : x.c1[(j & 1) * 4 + 0]);
    const uint32_t o1 = (uint32_t)(j < 2 ? x.c0[(j & 1) * 4 + 1] : x.c1[(j & 1) * 4 + 1]);
    const uint32_t o2 = (uint32_t)(j < 2 ? x.c0[(j & 1) * 4 + 2] : x.c1[(j & 1) * 4 + 2]);
    const uint32_t o3 = (uint32_t)(j < 2 ? x.c0[(j & 1) * 4 + 3] : x.c1[(j & 1) * 4 + 3]);
    const uint64_t S = (uint64_t)o0 + ((uint64_t)o1 << 8) + ((uint64_t)o2 << 16) + ((uint64_t)o3 << 24);
    const uint32_t lo = (uint32_t)S, hi = (uint32_t)(S >> 32);
    const uint32_t m = lo * Fp::M; const uint32_t t = hi - rk_mulhi(m, Fp::P); const uint32_t u = t + Fp::P;
    r[j] = t < u ? t : u;
  }
  return FpExt(Fp::fromRaw(r[0]), Fp::fromRaw(r[1]), Fp::fromRaw(r[2]), Fp::fromRaw(r[3])) + rk_fin(x.side) + x.base;
}
''' % (XG, PMBASE, QN, HN)
ns = 'namespace risc0::circuit::rv32im_v2 {'
assert ns in out
p = out.index(ns); out = out[:p] + helpers + out[p:]
anchor = '// ---- algebraic rewrite helpers (ec-algebra) ----'
assert anchor in out, 'xmxlink: helper anchor missing'
p = out.index(anchor); out = out[:p] + body + out[p:]

# Wrapper: pre-kernels build the full poly_mix table (incl. QN products and
# hoisted values) and the DPAS A table in global memory; each work-group of
# the main kernel copies the A table into SLM. rk_hoist is called with the
# global poly_mix exactly like the main kernel (rewrite.py asserts it only
# reads entries < PMBASE); a different address space here breaks IGC's
# static address-space resolution for the whole module.
pre = r'''        // ---- xmxlink: full poly_mix table and the aligned-group DPAS A table (global, copied to SLM per WG) ----
        static uint16_t* g_atab = nullptr; static FpExt* g_pmfull = nullptr;
        {
            namespace cr = circuit::rv32im_v2;
            constexpr uint32_t PMF = POLY_MIX_COUNT + cr::RK_QN + cr::RK_HN;
            if (!g_atab) {
                g_atab = sycl::malloc_device<uint16_t>(size_t(cr::RK_XG) * 256, *q);
                g_pmfull = sycl::malloc_device<FpExt>(PMF, *q);
                if (!g_atab || !g_pmfull) return make_error("eval_check: xmxlink table allocation failed");
            }
            FpExt* pmf = g_pmfull; uint16_t* atab = g_atab;
            q->submit([&](sycl::handler& h) {
                sycl::local_accessor<FpExt, 1> slm(sycl::range<1>(PMF), h);
                h.parallel_for(sycl::nd_range<1>(512, 512), [=](sycl::nd_item<1> item) {
                    uint32_t lid = item.get_local_id(0);
                    for (uint32_t i = lid; i < POLY_MIX_COUNT; i += 512) slm[i] = poly_mix[i];
                    if (lid == 0) cr::rk_hoist(poly_mix, out, mix, slm.get_multi_ptr<sycl::access::decorated::no>().get() + POLY_MIX_COUNT + cr::RK_QN);
                    for (uint32_t i = lid; i < cr::RK_QN; i += 512) slm[POLY_MIX_COUNT + i] = poly_mix[cr::rk_qa[i]] * poly_mix[cr::rk_qb[i]];
                    sycl::group_barrier(item.get_group());
                    for (uint32_t i = lid; i < PMF; i += 512) pmf[i] = slm[i];
                });
            });
            q->parallel_for(sycl::range<1>(size_t(cr::RK_XG) * 256), [=](sycl::id<1> ii) {
                const uint32_t i = ii[0], g = i / 256, w = i % 256, half = w / 128, r = (w % 128) / 16, l = w % 16;
                const uint32_t j = half * 2 + r / 4, b = r % 4, d = l >> 1, a0 = 2 * (l & 1);
                const uint32_t k = g * 8 + d;
                uint16_t v = 0;
                if (k < PMF) {
                    const uint64_t x = pmf[k].elems[j].asRaw();
                    const uint32_t q0 = uint32_t((x << (8 * a0)) % 0x78000001ull), q1 = uint32_t((x << (8 * (a0 + 1))) % 0x78000001ull);
                    v = uint16_t(((q0 >> (8 * b)) & 255u) | (((q1 >> (8 * b)) & 255u) << 8));
                }
                atab[i] = v;
            });
        }
        const uint16_t* atab_g = g_atab;
'''
# Insert right after the slab lock: the pre-kernels rewrite the shared
# g_pmfull/g_atab tables that the main kernel reads, so the lock must cover
# them and the tile loop together (two concurrent eval_check calls would
# otherwise interleave their tables on the queue).
anchor = '#ifdef RK_NWORDS\n        std::lock_guard<std::mutex> slab_lock(g_slab_mutex);\n'
assert anchor in out, 'xmxlink: wrapper anchor (slab lock) missing'
i = out.index(anchor) + len(anchor); out = out[:i] + '#endif\n' + pre + '#ifdef RK_NWORDS\n' + out[i:]
old = 'sycl::local_accessor<FpExt, 1> slm(sycl::range<1>(POLY_MIX_COUNT + circuit::rv32im_v2::RK_QN + circuit::rv32im_v2::RK_HN), h);'
assert old in out, 'xmxlink: SLM accessor anchor missing'
out = out.replace(old, 'sycl::local_accessor<FpExt, 1> slm(sycl::range<1>(POLY_MIX_COUNT + circuit::rv32im_v2::RK_QN + circuit::rv32im_v2::RK_HN + circuit::rv32im_v2::RK_XG * 32), h);')
old2 = '                    sycl::group_barrier(item.get_group());\n#ifdef RK_NWORDS\n                    if (lrow >= rows) return;'
assert old2 in out, 'xmxlink: SLM fill anchor missing'
out = out.replace(old2, '''                    { uint32_t* sa = reinterpret_cast<uint32_t*>(slm.get_multi_ptr<sycl::access::decorated::no>().get() + circuit::rv32im_v2::RK_XBASE);
                      const uint32_t* ga = reinterpret_cast<const uint32_t*>(atab_g);
                      for (uint32_t i = lid; i < circuit::rv32im_v2::RK_XG * 128; i += wgsz) sa[i] = ga[i]; }
''' + old2)
open(dst, 'w').write(out)
print('xmxlink: %d chains (>= %d terms) of %d, %d blocks, %d groups, %d terms'
      % (len(sel), LMIN, len(chains), nblk, XG, sum(len(c['terms']) for c in sel)), file=sys.stderr)
