#!/usr/bin/env python3
"""mm_chain.py IN.metal OUT.metal --funcs f1,f2|all [--mode tensor|nolink] [--wmin N]

N1 probe transform of the generated evalfast TU (not a production pass). In the named chain
functions, every poly_mix chain with >= wmin Fp-term links is rewritten:
  tensor: the RDNA4 WMMA scheme (cuda/evalfast/wmma_hip.py) on Metal 4 matmul2d. Each Fp link
    stores its c raw word in the simdgroup's threadgroup staging (slot = link index in the block
    of 32); every 32 Fp links and at the chain end n1_flush builds the Q tile of the block from
    the pm rows in the kernel (per simdgroup; production would precompute Q per proof), runs one
    uint8 x uint8 -> int32 matmul2d (16 x 32 points x 128) and folds T_j = sum_b D[(j,b)] << 8b
    into the chain's RkAcc. FpExt-term links stay scalar. Bit-exact with the scalar chain.
  nolink: the same Fp links become one xor per element (wrong results): the chain-arithmetic
    share of those functions.
The split kernel gets 6 KB of threadgroup memory per simdgroup (staging 4 KB, Q tile 2 KB).
Compile with -std=metal4.0 -DN1_MPP.
"""
import argparse, re, sys

ap = argparse.ArgumentParser()
ap.add_argument('inp'); ap.add_argument('out')
ap.add_argument('--funcs', required=True)
ap.add_argument('--mode', choices=['tensor', 'nolink'], default='tensor')
ap.add_argument('--wmin', type=int, default=8)
ap.add_argument('--acc', action='store_true', help='accumulate D of a whole chain in the cooperative tensor; one combine per chain')
a = ap.parse_args()
src = open(a.inp).read()

LINK = re.compile(r'^  RkAcc (x\d+) = (rk_link(?:_nf)?_k)\(ctx, (\w+), (\w+), (\d+)u\);$')
FIN = re.compile(r'^  FpExt (x\d+) = rk_fin\((rk_link(?:_nf)?_k)\(ctx, (\w+), (\w+), (\d+)u\)\);$')
FHEAD = re.compile(r'^EF_NOINLINE FpExt (\w+)\(.*\) \{$')

def types_of(body):
    t = {}
    for m in re.finditer(r'\b(Fp|FpExt|RkAcc)\s+(\w+)\s*[,)=(]', body): t[m.group(2)] = m.group(1)
    for m in re.finditer(r'auto (x\d+) = (.*);', body):
        e = m.group(2)
        ext = bool(re.search(r'FpExt\(|rkw_fin\(|rk_xmul\(|EF_PM\(|rk_fin\(|RK_LD4\(|rv32im_v2_\d+\(', e)) or \
            any(t.get(v) == 'FpExt' for v in re.findall(r'\b(?:x|arg)\w+', e))
        t[m.group(1)] = 'FpExt' if ext else 'Fp'
    return t

lines = src.split('\n')
want = None if a.funcs == 'all' else set(a.funcs.split(','))
kt = []          # per block: 32 table rows (0xffff = unused slot)
nblk = 0; nrew = 0; nlinks = 0; tmp = 0
i = 0
out = []
while i < len(lines):
    m = FHEAD.match(lines[i])
    if not m or (want is not None and m.group(1) not in want):
        out.append(lines[i]); i += 1; continue
    j = i
    while lines[j] != '}': j += 1
    body = lines[i:j + 1]
    ty = types_of('\n'.join(body))
    # chains: acc var -> chain id; links in order
    chain_of = {}; chains = []
    for n, L in enumerate(body):
        mm = LINK.match(L) or FIN.match(L)
        if not mm: continue
        outv, fn, base, c, k = mm.groups()
        cid = chain_of.pop(base, None)
        if cid is None:
            cid = len(chains); chains.append([])
        chains[cid].append((n, bool(FIN.match(L)), outv, fn, base, c, int(k), ty.get(c)))
        if not FIN.match(L): chain_of[outv] = cid
    assert not chain_of, f'{m.group(1)}: chains without rk_fin: {list(chain_of)[:4]}'
    for ch in chains:
        assert all(t in ('Fp', 'FpExt') for *_, t in ch), ch
    # check each RkAcc var is used exactly once (as the next link's base)
    text = '\n'.join(body)
    for ch in chains:
        for (_, fin, outv, *_r) in ch:
            if not fin:
                assert len(re.findall(r'\b%s\b' % outv, text)) == 2, outv
    repl = {}
    for ch in chains:
        nfp = sum(1 for *_, t in ch if t == 'Fp')
        if nfp < a.wmin: continue
        nrew += 1; nlinks += nfp
        slot = 0; ks = []; first = True
        for idx, (n, fin, outv, fn, base, c, k, t) in enumerate(ch):
            b0 = f'rk_acc({base})' if first else base
            if a.mode == 'nolink':
                if t == 'Fp':
                    call = f'n1_xl({b0}, {c}, {k}u)'
                    repl[n] = f'  FpExt {outv} = rk_fin({call});' if fin else f'  RkAcc {outv} = {call};'
                first = False
                continue
            if t == 'Fp':
                ks.append(k)
                cdv = f'n1_cd{nrew}'
                pre = f'  auto {cdv} = n1_cd0(); ' if (a.acc and first) else '  '
                if fin:
                    tv = f'n1_t{tmp}'; tmp += 1
                    blk = nblk; nblk += 1; kt.append(ks + [0xffff] * (32 - len(ks))); ks = []
                    if a.acc:
                        repl[n] = (f'{pre}RkAcc {tv} = {b0}; n1_st(ctx, {slot}u, {c}); n1_mma(ctx, {cdv}, {blk}u);'
                                   f' n1_comb(ctx, {tv}, {cdv}); FpExt {outv} = rk_fin({tv});')
                    else:
                        repl[n] = (f'  RkAcc {tv} = {b0}; n1_st(ctx, {slot}u, {c}); n1_flush(ctx, {tv}, {blk}u);'
                                   f' FpExt {outv} = rk_fin({tv});')
                else:
                    s = f'{pre}RkAcc {outv} = {b0}; n1_st(ctx, {slot}u, {c});'
                    slot += 1
                    if slot == 32:
                        blk = nblk; nblk += 1; kt.append(ks); ks = []; slot = 0
                        s += f' n1_mma(ctx, {cdv}, {blk}u);' if a.acc else f' n1_flush(ctx, {outv}, {blk}u);'
                    repl[n] = s
            else:  # FpExt term: scalar link; flush pending terms first at the chain end
                if fin and (ks or a.acc):
                    tv = f'n1_t{tmp}'; tmp += 1; cdv = f'n1_cd{nrew}'
                    if a.acc:
                        mma = ''
                        if ks:
                            blk = nblk; nblk += 1; kt.append(ks + [0xffff] * (32 - len(ks))); ks = []; slot = 0
                            mma = f' n1_mma(ctx, {cdv}, {blk}u);'
                        repl[n] = (f'  RkAcc {tv} = {b0};{mma} n1_comb(ctx, {tv}, {cdv});'
                                   f' FpExt {outv} = rk_fin(rk_link_k(ctx, {tv}, {c}, {k}u));')
                    else:
                        blk = nblk; nblk += 1; kt.append(ks + [0xffff] * (32 - len(ks))); ks = []; slot = 0
                        repl[n] = (f'  RkAcc {tv} = {b0}; n1_flush(ctx, {tv}, {blk}u);'
                                   f' FpExt {outv} = rk_fin(rk_link_k(ctx, {tv}, {c}, {k}u));')
                elif first and a.acc:
                    repl[n] = f'  auto n1_cd{nrew} = n1_cd0();' + body[n][1:]
            first = False
        assert not ks or a.mode == 'nolink'
    for n, L in enumerate(body):
        out.append(repl.get(n, L))
    i = j + 1

src = '\n'.join(out)
# helpers + table after the linkext helpers (rk_link_k); context fields; kernel staging
anchor = '// ---- linkext.py: rk_link with an FpExt term, products accumulated lazily ----\n'
assert src.count(anchor) == 1
helpers = r'''// ---- N1 probe (mm_chain.py): poly_mix chains on the M5 matrix units (matmul2d) ----
template <typename A> inline RkAcc n1_xl(thread const A& base, Fp c, uint32_t k) {
  RkAcc s = rk_acc(base); const uint32_t cr = c.asRaw();
  s.lo[0] ^= cr ^ k; s.lo[1] ^= cr; s.lo[2] ^= cr + k; s.lo[3] ^= cr; return s;
}
#ifdef N1_TC
#ifndef N1_QMODE
#define N1_QMODE 1
#endif
constant ushort n1_kt[] = {%(kt)s};
inline void n1_st(thread const EfCtx& ctx, uint32_t slot, Fp c) { ctx.n1s[ctx.n1lane * 32u + slot] = c.asRaw(); }
inline uint32_t n1_qw(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3, uint32_t b) {
  return ((v0 >> (8u * b)) & 255u) | (((v1 >> (8u * b)) & 255u) << 8) | (((v2 >> (8u * b)) & 255u) << 16) |
         (((v3 >> (8u * b)) & 255u) << 24);
}
#if N1_QMODE == 1
// v * 256 mod P as a Montgomery product with 256 R mod P (canonical): no 64-bit division
inline uint32_t n1_m256(uint32_t v) {
  const uint32_t c = 0xffffddeu;  // 256 * 2^32 mod P
  const uint32_t lo = v * c, hi = metal::mulhi(v, c), m = lo * Fp::M, t = hi - metal::mulhi(m, Fp::P);
  return rk_umin(t, t + Fp::P);
}
#else
inline uint32_t n1_m256(uint32_t v) { return uint32_t((uint64_t(v) << 8) %% uint64_t(Fp::P)); }
#endif
#define N1_QJ(j, x) { const uint32_t v0 = (x), v1 = n1_m256(v0), v2 = n1_m256(v1), v3 = n1_m256(v2); \
  qt[(4u * j + 0u) * 32u + lane] = n1_qw(v0, v1, v2, v3, 0u); qt[(4u * j + 1u) * 32u + lane] = n1_qw(v0, v1, v2, v3, 1u); \
  qt[(4u * j + 2u) * 32u + lane] = n1_qw(v0, v1, v2, v3, 2u); qt[(4u * j + 3u) * 32u + lane] = n1_qw(v0, v1, v2, v3, 3u); }
#define N1_T(j) (uint64_t(uint32_t(st[(4u * j + 0u) * 32u + lane])) + (uint64_t(uint32_t(st[(4u * j + 1u) * 32u + lane])) << 8) + \
                 (uint64_t(uint32_t(st[(4u * j + 2u) * 32u + lane])) << 16) + (uint64_t(uint32_t(st[(4u * j + 3u) * 32u + lane])) << 24))
inline uint64_t n1_add(uint32_t lo, uint32_t hi, uint64_t t) {
  const uint64_t s = ((uint64_t(hi) << 32) | lo) + t; const uint32_t h = uint32_t(s >> 32);
  return (uint64_t(rk_umin(h, h - Fp::P)) << 32) | uint32_t(s);
}
#ifdef N1_NOBAR  // timing ablation: no simdgroup barriers (wrong results)
#define N1_BAR()
#else
#define N1_BAR() simdgroup_barrier(metal::mem_flags::mem_threadgroup)
#endif
inline void n1_flush(thread const EfCtx& ctx, thread RkAcc& s, uint32_t blk) {
  threadgroup uint* st = ctx.n1s; threadgroup uint* qt = ctx.n1s + 1024u; const uint32_t lane = ctx.n1lane;
  const uint32_t k = n1_kt[blk * 32u + lane];
  const FpExt r = EF_PM(k < 0xffffu ? k : 0u);
  const bool on = k < 0xffffu;
#if N1_QMODE != 2  // 2: no Q tile build (timing ablation, wrong results)
  N1_QJ(0u, on ? r.elems[0].asRaw() : 0u) N1_QJ(1u, on ? r.elems[1].asRaw() : 0u)
  N1_QJ(2u, on ? r.elems[2].asRaw() : 0u) N1_QJ(3u, on ? r.elems[3].asRaw() : 0u)
#else
  (void)r; (void)on;
#endif
  N1_BAR();
  using namespace mpp::tensor_ops;
  constexpr auto desc = matmul2d_descriptor(16, 32, 128, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, metal::execution_simdgroup> op;
  using EQ = metal::extents<int32_t, 128, 16>; using EB = metal::extents<int32_t, 128, 32>; using ED = metal::extents<int32_t, 32, 16>;
  metal::tensor<threadgroup uint8_t, EQ, metal::tensor_inline> tQ((threadgroup uint8_t*)qt, EQ());
  metal::tensor<threadgroup uint8_t, EB, metal::tensor_inline> tB((threadgroup uint8_t*)st, EB());
  auto cD = op.template get_destination_cooperative_tensor<decltype(tQ), decltype(tB), int32_t>();
#ifndef N1_NOMM  // N1_NOMM: no matmul (timing ablation, wrong results)
  op.run(tQ, tB, cD);
  N1_BAR();
  metal::tensor<threadgroup int32_t, ED, metal::tensor_inline> tD((threadgroup int32_t*)st, ED());
  cD.store(tD);
#else
  (void)cD;
#endif
  N1_BAR();
#ifdef N1_NOCOMB  // timing ablation: staging stores only, no combine (wrong results)
  (void)s; (void)st;
#else
  const uint64_t s0 = n1_add(s.lo[0], s.hi[0], N1_T(0u)), s1 = n1_add(s.lo[1], s.hi[1], N1_T(1u));
  const uint64_t s2 = n1_add(s.lo[2], s.hi[2], N1_T(2u)), s3 = n1_add(s.lo[3], s.hi[3], N1_T(3u));
  N1_BAR();
  s = RkAcc{{uint32_t(s0), uint32_t(s1), uint32_t(s2), uint32_t(s3)},
            {uint32_t(s0 >> 32), uint32_t(s1 >> 32), uint32_t(s2 >> 32), uint32_t(s3 >> 32)}};
#endif
}
// --acc: one destination cooperative tensor per chain (exact in int32 up to 8256 links),
// multiply_accumulate per block, one store + combine at the chain end.
constexpr constant auto n1_desc_acc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 128, false, true, false,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
using N1AccOp = mpp::tensor_ops::matmul2d<n1_desc_acc, metal::execution_simdgroup>;
using N1EQ = metal::extents<int32_t, 128, 16>; using N1EB = metal::extents<int32_t, 128, 32>; using N1ED = metal::extents<int32_t, 32, 16>;
using N1TQ = metal::tensor<threadgroup uint8_t, N1EQ, metal::tensor_inline>;
using N1TB = metal::tensor<threadgroup uint8_t, N1EB, metal::tensor_inline>;
using N1CD = N1AccOp::cooperative_tensor_destination_t<N1TQ, N1TB, int32_t>;
inline N1CD n1_cd0() {
  N1AccOp op; N1CD c = op.get_destination_cooperative_tensor<N1TQ, N1TB, int32_t>();
  #pragma unroll
  for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0;
  return c;
}
inline void n1_mma(thread const EfCtx& ctx, thread N1CD& cd, uint32_t blk) {
  threadgroup uint* st = ctx.n1s; threadgroup uint* qt = ctx.n1s + 1024u; const uint32_t lane = ctx.n1lane;
  const uint32_t k = n1_kt[blk * 32u + lane];
  const FpExt r = EF_PM(k < 0xffffu ? k : 0u);
  const bool on = k < 0xffffu;
#if N1_QMODE != 2
  N1_QJ(0u, on ? r.elems[0].asRaw() : 0u) N1_QJ(1u, on ? r.elems[1].asRaw() : 0u)
  N1_QJ(2u, on ? r.elems[2].asRaw() : 0u) N1_QJ(3u, on ? r.elems[3].asRaw() : 0u)
#else
  (void)r; (void)on;
#endif
  N1_BAR();
  N1AccOp op;
  N1TQ tQ((threadgroup uint8_t*)qt, N1EQ()); N1TB tB((threadgroup uint8_t*)st, N1EB());
  op.run(tQ, tB, cd);
  N1_BAR();
}
inline void n1_comb(thread const EfCtx& ctx, thread RkAcc& s, thread N1CD& cd) {
  threadgroup uint* st = ctx.n1s; const uint32_t lane = ctx.n1lane;
  metal::tensor<threadgroup int32_t, N1ED, metal::tensor_inline> tD((threadgroup int32_t*)st, N1ED());
  cd.store(tD);
  N1_BAR();
  const uint64_t s0 = n1_add(s.lo[0], s.hi[0], N1_T(0u)), s1 = n1_add(s.lo[1], s.hi[1], N1_T(1u));
  const uint64_t s2 = n1_add(s.lo[2], s.hi[2], N1_T(2u)), s3 = n1_add(s.lo[3], s.hi[3], N1_T(3u));
  N1_BAR();
  s = RkAcc{{uint32_t(s0), uint32_t(s1), uint32_t(s2), uint32_t(s3)},
            {uint32_t(s0 >> 32), uint32_t(s1 >> 32), uint32_t(s2 >> 32), uint32_t(s3 >> 32)}};
}
#endif
''' % {'kt': ', '.join(str(x) for b in kt for x in b) or '0'}
src = src.replace(anchor, anchor + helpers)
ctx_old = '  thread Fp* slab;\n};\n'
assert src.count(ctx_old) == 1
src = src.replace(ctx_old, '  thread Fp* slab;\n#ifdef N1_TC\n  threadgroup uint* n1s;\n  uint n1lane;\n#endif\n};\n')
inc_old = '#include <metal_stdlib>\n'
src = src.replace(inc_old, inc_old + '#ifdef N1_MPP\n#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n#endif\n', 1)
k_old = '  ctx.slab = reinterpret_cast<thread Fp*>(lslab);\n'
assert src.count(k_old) == 1
src = src.replace(k_old, k_old + '#ifdef N1_TC\n  threadgroup uint n1_tg[(EF_THREADS / 32) * 1536];\n'
                  '  ctx.n1s = n1_tg + (tid >> 5) * 1536u;\n  ctx.n1lane = tid & 31u;\n#endif\n')
open(a.out, 'w').write(src)
print(f'mm_chain: {a.mode}: {nrew} chains, {nlinks} Fp links rewritten, {nblk} matmul blocks', file=sys.stderr)
