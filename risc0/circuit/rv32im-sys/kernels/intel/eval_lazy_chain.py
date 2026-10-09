#!/usr/bin/env python3
# Intel mono eval_check post-pass (RISC0_INTEL_EVAL_FAST): rewrite the
# `FpExt X = B + C * poly_mix[k];` accumulation chains of the generated
# poly_fp into a lazy split-word Montgomery accumulator (RkAcc). Each link
# adds the raw 64-bit product c*pm (as lo/hi words with carry, hi kept < P by
# one fold); single-use internal links stay unreduced and only chain ends do a
# REDC. Exact: REDC(sum) equals the sum of the reduced products. Measured
# -63 ms per po2=20 segment on top of the inlined FpExt multiply.
#
# usage: eval_lazy_chain.py <in.cpp> <out.cpp>
import re,sys,collections
src=open(sys.argv[1]).read()
cut=src.find('// Intel SYCL eval_check kernel')
body,wrapper=src[:cut],src[cut:]
link=re.compile(r'^(\s*)FpExt (x\d+) = (\w+) \+ (\w+) \* poly_mix\[(\d+)\];$')
out=[]; stats=collections.Counter()
lines=body.split('\n')
# split into functions
starts=[i for i,l in enumerate(lines) if re.match(r'^(__attribute__\(\([a-z_]+\)\) )?FpExt (rv32im_v2_\d+|poly_fp)\(.*\{$',l)]
starts.append(len(lines))
res=lines[:starts[0]]
for fi in range(len(starts)-1):
    F=lines[starts[fi]:starts[fi+1]]
    uses=collections.Counter(); links={}
    for i,l in enumerate(F):
        m=link.match(l)
        if m: links[m.group(2)]=(i,m)
        code=l.split('//')[0]
        if '=' in code and not code.strip().startswith('//'):
            lhs,rhs=code.split('=',1)
            for t in re.findall(r'\bx\d+\b',rhs): uses[t]+=1
            # array element writes: index tokens on lhs are literals, ignore
        elif 'return' in code:
            for t in re.findall(r'\bx\d+\b',code): uses[t]+=1
        else:
            for t in re.findall(r'\bx\d+\b',code): uses[t]+=1  # call statements etc.
    # B-use map
    usedAsB=collections.Counter(m.group(3) for (_,m) in links.values())
    internal=set(x for x in links if uses[x]==1 and usedAsB[x]==1)
    for x in links:
        i,m=links[x]
        ind,X,B,C,k=m.groups()
        e=f"rk_link({B}, {C}, poly_mix[{k}])"
        if X in internal:
            F[i]=f"{ind}RkAcc {X} = {e};"; stats['internal']+=1
        else:
            F[i]=f"{ind}FpExt {X} = rk_fin({e});"; stats['end']+=1
    res+=F
helper=r'''
// ---- lazy poly_mix accumulator (RK_LAZYCHAIN) ----
struct RkAcc { uint32_t lo[4]; uint32_t hi[4]; };   // S_j = hi*2^32+lo, invariant hi < P
__attribute__((always_inline)) static inline uint32_t rk_mulhi(uint32_t a, uint32_t b) { return uint32_t((uint64_t(a) * uint64_t(b)) >> 32); }
__attribute__((always_inline)) static inline uint32_t rk_umin(uint32_t a, uint32_t b) { return a < b ? a : b; }
__attribute__((always_inline)) static inline RkAcc rk_acc(FpExt b) { RkAcc s; for (int j = 0; j < 4; j++) { s.lo[j] = 0; s.hi[j] = b.elems[j].asRaw(); } return s; }
__attribute__((always_inline)) static inline RkAcc rk_acc(const RkAcc& s) { return s; }
template <typename A> __attribute__((always_inline)) static inline RkAcc rk_link(const A& base, Fp c, const FpExt& pm) {
  RkAcc s = rk_acc(base); uint32_t cr = c.asRaw();
  for (int j = 0; j < 4; j++) {
    uint32_t p = pm.elems[j].asRaw();
    uint32_t pl = cr * p, ph = rk_mulhi(cr, p);
    uint32_t n = s.lo[j] + pl;
    uint32_t h = s.hi[j] + ph + (n < pl ? 1u : 0u);   // < P + 0.47P + 1 < 2P
    s.lo[j] = n; s.hi[j] = rk_umin(h, h - Fp::P);
  }
  return s;
}
template <typename A> __attribute__((always_inline)) static inline RkAcc rk_link(const A& base, FpExt c, const FpExt& pm) {
  RkAcc s = rk_acc(base); FpExt p = c * pm;
  for (int j = 0; j < 4; j++) { uint32_t h = s.hi[j] + p.elems[j].asRaw(); s.hi[j] = rk_umin(h, h - Fp::P); }
  return s;
}
__attribute__((always_inline)) static inline FpExt rk_fin(const RkAcc& s) {
  uint32_t r[4];
  for (int j = 0; j < 4; j++) {
    uint32_t m = s.lo[j] * Fp::M;
    uint32_t t = s.hi[j] - rk_mulhi(m, Fp::P);
    r[j] = rk_umin(t, t + Fp::P);
  }
  return FpExt(Fp::fromRaw(r[0]), Fp::fromRaw(r[1]), Fp::fromRaw(r[2]), Fp::fromRaw(r[3]));
}
'''
txt='\n'.join(res)
anchor='namespace risc0::circuit::rv32im_v2 {\nconstexpr size_t kInvRate = 4;\n'
assert anchor in txt
txt=txt.replace(anchor,anchor+helper,1)
open(sys.argv[2],'w').write(txt+wrapper)
print('eval_lazy_chain:', dict(stats), file=sys.stderr)
if stats['internal'] == 0:
    sys.exit('eval_lazy_chain: no accumulation chains matched; generator output changed?')
