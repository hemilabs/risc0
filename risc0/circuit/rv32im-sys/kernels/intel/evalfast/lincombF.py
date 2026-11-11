#!/usr/bin/env python3
# lincomb.py in.cpp out.cpp   (final post-slab amalgamation, after swapuni/idx32)
# Fuse FpExt sum trees  S = +-(A*B) +- (f_i*E_i) +- V_j  (A,B,E,V FpExt; f Fp; every
# intermediate single-use) into one lazily-reduced 64-bit accumulation per component:
# the FpExt*FpExt product keeps its exact 64-bit dot products (no REDC), each Fp*FpExt
# product is added as (f or P-f)*E_j into the same accumulator, plain terms as hi += v (or P-v),
# and a single Montgomery REDC per component finishes. The result is the canonical field
# element, bit-identical to the unfused evaluation.
import re, sys
MINFF = int(sys.argv[3]) if len(sys.argv) > 3 else 2
src = open(sys.argv[1]).read()
k = '// Intel SYCL eval_check kernel'
head, tail = src[:src.find(k)], src[src.find(k):]
L = head.split('\n')
FD = re.compile(r'^(__attribute__\(\([a-z_]+\)\) )?(inline )?FpExt (rv32im_v2_\d+|poly_fp)\((.*)\) \{$')
DEF = re.compile(r'^(\s*)(?:auto|FpExt|Fp|constexpr Fp|constexpr FpExt|const FpExt|const Fp|RkAcc) (x\d+|arg\d+)\s*(=|\()\s*(.*?)\)?;\s*$')

HELPER = r'''
// ---- lincombF: lazily reduced Fp linear combinations ----
struct RkW1 { uint32_t lo; uint32_t hi; };
__attribute__((always_inline)) static inline uint32_t rkw1_mulhi(uint32_t a, uint32_t b) { return uint32_t((uint64_t(a) * uint64_t(b)) >> 32); }
__attribute__((always_inline)) static inline uint32_t rkw1_fold(uint32_t h) { uint32_t f = h - Fp::P; return h < f ? h : f; }
__attribute__((always_inline)) static inline RkW1 rkw1_ff0(Fp a, Fp b, bool neg) {
  const uint32_t x = neg ? Fp::P - a.asRaw() : a.asRaw(), y = b.asRaw();
  RkW1 w; w.lo = x * y; w.hi = rkw1_mulhi(x, y); return w;
}
__attribute__((always_inline)) static inline void rkw1_ff(RkW1& w, Fp a, Fp b, bool neg) {
  const uint32_t x = neg ? Fp::P - a.asRaw() : a.asRaw(), y = b.asRaw();
  uint32_t p = x * y, n = w.lo + p;
  w.hi = rkw1_fold(w.hi + rkw1_mulhi(x, y) + (n < p ? 1u : 0u));   // hi < P invariant: < 1.47P+1 before the fold
  w.lo = n;
}
__attribute__((always_inline)) static inline void rkw1_v(RkW1& w, Fp v, bool neg) {
  const uint32_t r = v.asRaw(); w.hi = rkw1_fold(w.hi + (neg ? Fp::P - r : r));
}
__attribute__((always_inline)) static inline Fp rkw1_fin(const RkW1& w) {
  const uint32_t h = w.hi;
  uint32_t t = h - rkw1_mulhi(w.lo * Fp::M, Fp::P); uint32_t u = t + Fp::P; return Fp::fromRaw(t < u ? t : u);
}
'''

def main():
    funcs = []; cur = None
    for i, l in enumerate(L):
        m = FD.match(l)
        if m: cur = dict(start=i, lines=[], params=m.group(4)); funcs.append(cur); continue
        if cur is not None and l.startswith('}'): cur = None; continue
        if cur is not None: cur['lines'].append(i)
    out = list(L); nfused = 0; nef = 0; nv = 0; nee = 0
    for f in funcs:
        ty = {}
        for p in f['params'].split(','):
            p = p.strip(); mm = re.match(r'^(Fp|FpExt)\s+(\w+)$', p)
            if mm: ty[mm.group(2)] = mm.group(1)
        defs = {}; uses = {}
        for i in f['lines']:
            code = re.sub(r'//.*', '', L[i])
            m = DEF.match(code)
            if m:
                v, rhs = m.group(2), m.group(4); st = code.strip()
                if st.startswith('RkAcc'): t = 'RkAcc'
                elif st.startswith(('constexpr Fp ', 'Fp ', 'const Fp ')): t = 'Fp'
                elif st.startswith(('constexpr FpExt', 'FpExt', 'const FpExt')): t = 'FpExt'
                elif rhs.startswith('RK_LD1('): t = 'Fp'
                elif rhs.startswith('RK_LD4(') or rhs.startswith('poly_mix['): t = 'FpExt'
                elif re.match(r'^(/\*\w+=\*/)?arg\d+\[', rhs): t = 'Fp'
                else:
                    toks = re.findall(r'\b(?:x\d+|arg\d+(?:_e\d)?)\b', rhs)
                    ts = [ty.get(x) for x in toks]
                    if 'rk_' in rhs: t = 'FpExt' if ('rk_fin' in rhs or 'rkw_fin' in rhs) else ('RkAcc' if 'rk_link' in rhs else None)
                    elif ts and all(x == 'Fp' for x in ts) and re.match(r'^[x\d\s+\-*()_ae]+$', rhs): t = 'Fp'
                    elif 'FpExt' in ts and re.match(r'^-?[x\d\s+\-*()_ae]+$', rhs): t = 'FpExt'
                    else: t = None
                ty[v] = t; defs[v] = (i, rhs, m.group(1))
                body = rhs
            else:
                body = code
            for x in re.findall(r'\bx\d+\b', body):
                uses.setdefault(x, []).append(i)
        def single(v): return len(uses.get(v, [])) == 1
        consumed = set()
        def expand(v, sign, terms):
            # returns False if v cannot be part of the tree
            if v in defs and single(v) and ty.get(v) == 'Fp' and v not in consumed:
                i, rhs, _ = defs[v]
                m = re.match(r'^(x\d+) ([+-]) (x\d+)$', rhs)
                if m and ty.get(m.group(1)) == 'Fp' and ty.get(m.group(3)) == 'Fp':
                    consumed.add(v)
                    expand(m.group(1), sign, terms)
                    expand(m.group(3), sign if m.group(2) == '+' else -sign, terms)
                    return
                m = re.match(r'^(x\d+) \* (x\d+)$', rhs)
                if m:
                    a, b = m.group(1), m.group(2)
                    if ty.get(a) == 'Fp' and ty.get(b) == 'Fp' and a != b:
                        consumed.add(v); terms.append(('ff', sign, a, b)); return
            terms.append(('v', sign, v, None))
        roots = []
        for v, (i, rhs, ind) in defs.items():
            if ty.get(v) != 'Fp': continue
            m = re.match(r'^(x\d+) ([+-]) (x\d+)$', rhs)
            if not m or ty.get(m.group(1)) != 'Fp' or ty.get(m.group(3)) != 'Fp': continue
            # root: not itself absorbed into a parent add/sub (its single user is not an FpExt add/sub)
            us = uses.get(v, [])
            if len(us) == 1:
                pm = DEF.match(re.sub(r'//.*', '', L[us[0]]))
                if pm and re.match(r'^(x\d+) ([+-]) (x\d+)$', pm.group(4)) and ty.get(pm.group(2)) == 'Fp':
                    continue
            roots.append(v)
        for v in sorted(roots, key=lambda x: defs[x][0]):
            i, rhs, ind = defs[v]
            saved = set(consumed)
            terms = []
            m = re.match(r'^(x\d+) ([+-]) (x\d+)$', rhs)
            expand(m.group(1), 1, terms); expand(m.group(3), 1 if m.group(2) == '+' else -1, terms)
            ffs = [t for t in terms if t[0] == 'ff']
            if len(ffs) < MINFF:
                consumed.clear(); consumed.update(saved); continue
            e = ffs[0]; rest = [t for t in terms if t is not e]
            code = ['RkW1 _w%s = rkw1_ff0(%s, %s, %s);' % (v, e[2], e[3], 'true' if e[1] < 0 else 'false')]
            for t in rest:
                if t[0] == 'ff': code.append('rkw1_ff(_w%s, %s, %s, %s);' % (v, t[2], t[3], 'true' if t[1] < 0 else 'false')); nef += 1
                elif t[0] == 'v': code.append('rkw1_v(_w%s, %s, %s);' % (v, t[2], 'true' if t[1] < 0 else 'false')); nv += 1
            code.append('auto %s = rkw1_fin(_w%s);' % (v, v))
            out[i] = ind + ' '.join(code)
            nfused += 1; nee += 1
        # drop the consumed intermediate definitions
        for c in consumed:
            out[defs[c][0]] = defs[c][2] + '// lincombF: fused into a lazy Fp linear combination'
    text = '\n'.join(out)
    anchor = '// ---- lazy poly_mix accumulator (RK_LAZYCHAIN) ----'
    assert text.count(anchor) == 1
    text = text.replace(anchor, HELPER + anchor)
    open(sys.argv[2], 'w').write(text + tail)
    print('lincombF: %d Fp trees fused (%d, %d extra products, %d plain terms)' % (nfused, nee, nef, nv), file=sys.stderr)

if __name__ == '__main__':
    main()
