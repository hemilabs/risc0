#!/usr/bin/env python3
# lincomb.py in.cpp out.cpp   (final post-slab amalgamation, after swapuni/idx32)
# Fuse FpExt sum trees  S = +-(A*B) +- (f_i*E_i) +- V_j  (A,B,E,V FpExt; f Fp; every
# intermediate single-use) into one lazily-reduced 64-bit accumulation per component:
# the FpExt*FpExt product keeps its exact 64-bit dot products (no REDC), each Fp*FpExt
# product is added as (f or P-f)*E_j into the same accumulator, plain terms as hi += v (or P-v),
# and a single Montgomery REDC per component finishes. The result is the canonical field
# element, bit-identical to the unfused evaluation.
import re, sys
MODE = int(sys.argv[3]) if len(sys.argv) > 3 else 1
src = open(sys.argv[1]).read()
k = '// Intel SYCL eval_check kernel'
head, tail = src[:src.find(k)], src[src.find(k):]
L = head.split('\n')
FD = re.compile(r'^(__attribute__\(\([a-z_]+\)\) )?(inline )?FpExt (rv32im_v2_\d+|poly_fp)\((.*)\) \{$')
DEF = re.compile(r'^(\s*)(?:auto|FpExt|Fp|constexpr Fp|constexpr FpExt|const FpExt|const Fp|RkAcc) (x\d+|arg\d+)\s*(=|\()\s*(.*?)\)?;\s*$')

HELPER = r'''
// ---- lincomb.py: lazily reduced FpExt linear combinations ----
struct RkW { uint32_t lo[4]; uint32_t hi[4]; };   // S_j = hi*2^32 + lo, hi < P between steps
__attribute__((always_inline)) static inline uint32_t rkw_mulhi(uint32_t a, uint32_t b) { return uint32_t((uint64_t(a) * uint64_t(b)) >> 32); }
__attribute__((always_inline)) static inline uint32_t rkw_fold(uint32_t h) { uint32_t f = h - Fp::P; return h < f ? h : f; }
__attribute__((always_inline)) static inline void rkw_mac(RkW& w, int j, uint32_t x, uint32_t y) {
  uint32_t p = x * y, n = w.lo[j] + p;
  w.hi[j] = w.hi[j] + rkw_mulhi(x, y) + (n < p ? 1u : 0u);
  w.lo[j] = n;
}
// w = a*b (exact 64-bit dot products, NBETA folded into b), hi folded below P
__attribute__((always_inline)) static inline RkW rkw_ee(FpExt a, FpExt b) {
  const uint32_t a0 = a.elems[0].asRaw(), a1 = a.elems[1].asRaw(), a2 = a.elems[2].asRaw(), a3 = a.elems[3].asRaw();
  const uint32_t b0 = b.elems[0].asRaw(), b1 = b.elems[1].asRaw(), b2 = b.elems[2].asRaw(), b3 = b.elems[3].asRaw();
  const uint32_t b1n = (b.elems[1] * Fp(Fp::P - 11)).asRaw();
  const uint32_t b2n = (b.elems[2] * Fp(Fp::P - 11)).asRaw();
  const uint32_t b3n = (b.elems[3] * Fp(Fp::P - 11)).asRaw();
  RkW w;
  const uint32_t X[4][4] = {{b0, b3n, b2n, b1n}, {b1, b0, b3n, b2n}, {b2, b1, b0, b3n}, {b3, b2, b1, b0}};
  for (int j = 0; j < 4; j++) {
    w.lo[j] = a0 * X[j][0]; w.hi[j] = rkw_mulhi(a0, X[j][0]);
    rkw_mac(w, j, a1, X[j][1]); rkw_mac(w, j, a2, X[j][2]); rkw_mac(w, j, a3, X[j][3]);
    w.hi[j] = rkw_fold(w.hi[j]);
  }
  return w;
}
// w += (neg ? -1 : 1) * a*b  (fold after every 2 products: hi < P + 2*(0.47P+1) < 2^32)
__attribute__((always_inline)) static inline void rkw_ee_acc(RkW& w, FpExt a, FpExt b, bool neg) {
  uint32_t a0 = a.elems[0].asRaw(), a1 = a.elems[1].asRaw(), a2 = a.elems[2].asRaw(), a3 = a.elems[3].asRaw();
  if (neg) { a0 = Fp::P - a0; a1 = Fp::P - a1; a2 = Fp::P - a2; a3 = Fp::P - a3; }
  const uint32_t b0 = b.elems[0].asRaw(), b1 = b.elems[1].asRaw(), b2 = b.elems[2].asRaw(), b3 = b.elems[3].asRaw();
  const uint32_t b1n = (b.elems[1] * Fp(Fp::P - 11)).asRaw();
  const uint32_t b2n = (b.elems[2] * Fp(Fp::P - 11)).asRaw();
  const uint32_t b3n = (b.elems[3] * Fp(Fp::P - 11)).asRaw();
  const uint32_t X[4][4] = {{b0, b3n, b2n, b1n}, {b1, b0, b3n, b2n}, {b2, b1, b0, b3n}, {b3, b2, b1, b0}};
  for (int j = 0; j < 4; j++) {
    rkw_mac(w, j, a0, X[j][0]); rkw_mac(w, j, a1, X[j][1]); w.hi[j] = rkw_fold(w.hi[j]);
    rkw_mac(w, j, a2, X[j][2]); rkw_mac(w, j, a3, X[j][3]); w.hi[j] = rkw_fold(w.hi[j]);
  }
}
// w = (neg ? -f : f) * e  (initial term)
__attribute__((always_inline)) static inline RkW rkw_ef0(Fp f, FpExt e, bool neg) {
  const uint32_t fr = neg ? Fp::P - f.asRaw() : f.asRaw();
  RkW w;
  for (int j = 0; j < 4; j++) { const uint32_t x = e.elems[j].asRaw(); w.lo[j] = fr * x; w.hi[j] = rkw_mulhi(fr, x); }
  return w;
}
// w += (neg ? -f : f) * e
__attribute__((always_inline)) static inline void rkw_ef(RkW& w, Fp f, FpExt e, bool neg) {
  const uint32_t fr = neg ? Fp::P - f.asRaw() : f.asRaw();
  for (int j = 0; j < 4; j++) { rkw_mac(w, j, fr, e.elems[j].asRaw()); w.hi[j] = rkw_fold(w.hi[j]); }
}
// w += (neg ? -v : v)   (v*R as a product-domain value: hi += v)
__attribute__((always_inline)) static inline void rkw_v(RkW& w, FpExt v, bool neg) {
  for (int j = 0; j < 4; j++) { const uint32_t r = v.elems[j].asRaw(); w.hi[j] = rkw_fold(w.hi[j] + (neg ? Fp::P - r : r)); }
}
__attribute__((always_inline)) static inline FpExt rkw_fin(const RkW& w) {
  uint32_t r[4];
  for (int j = 0; j < 4; j++) { uint32_t t = w.hi[j] - rkw_mulhi(w.lo[j] * Fp::M, Fp::P); uint32_t u = t + Fp::P; r[j] = t < u ? t : u; }
  return FpExt(Fp::fromRaw(r[0]), Fp::fromRaw(r[1]), Fp::fromRaw(r[2]), Fp::fromRaw(r[3]));
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
            if v in defs and single(v) and ty.get(v) == 'FpExt' and v not in consumed:
                i, rhs, _ = defs[v]
                m = re.match(r'^(x\d+) ([+-]) (x\d+)$', rhs)
                if m and ty.get(m.group(1)) == 'FpExt' and ty.get(m.group(3)) == 'FpExt':
                    consumed.add(v)
                    expand(m.group(1), sign, terms)
                    expand(m.group(3), sign if m.group(2) == '+' else -sign, terms)
                    return
                m = re.match(r'^(x\d+) \* (x\d+)$', rhs)
                if m:
                    a, b = m.group(1), m.group(2)
                    if ty.get(a) == 'FpExt' and ty.get(b) == 'FpExt':
                        consumed.add(v); terms.append(('ee', sign, a, b)); return
                    if ty.get(a) == 'Fp' and ty.get(b) == 'FpExt':
                        consumed.add(v); terms.append(('ef', sign, a, b)); return
                    if ty.get(a) == 'FpExt' and ty.get(b) == 'Fp':
                        consumed.add(v); terms.append(('ef', sign, b, a)); return
            terms.append(('v', sign, v, None))
        roots = []
        for v, (i, rhs, ind) in defs.items():
            if ty.get(v) != 'FpExt': continue
            m = re.match(r'^(x\d+) ([+-]) (x\d+)$', rhs)
            if not m or ty.get(m.group(1)) != 'FpExt' or ty.get(m.group(3)) != 'FpExt': continue
            # root: not itself absorbed into a parent add/sub (its single user is not an FpExt add/sub)
            us = uses.get(v, [])
            if len(us) == 1:
                pm = DEF.match(re.sub(r'//.*', '', L[us[0]]))
                if pm and re.match(r'^(x\d+) ([+-]) (x\d+)$', pm.group(4)) and ty.get(pm.group(2)) == 'FpExt':
                    continue
            roots.append(v)
        for v in sorted(roots, key=lambda x: defs[x][0]):
            i, rhs, ind = defs[v]
            saved = set(consumed)
            terms = []
            m = re.match(r'^(x\d+) ([+-]) (x\d+)$', rhs)
            expand(m.group(1), 1, terms); expand(m.group(3), 1 if m.group(2) == '+' else -1, terms)
            ees = [t for t in terms if t[0] == 'ee']
            efs = [t for t in terms if t[0] == 'ef']
            posee = [t for t in ees if t[1] > 0]
            ok = len(terms) >= 2 and ((MODE >= 1 and len(posee) == 1 and len(ees) == 1) or
                                      (MODE >= 2 and len(posee) >= 1 and len(ees) >= 2) or
                                      (MODE >= 3 and not ees and len(efs) >= 3))
            if not ok:
                consumed.clear(); consumed.update(saved); continue
            if posee:
                e = posee[0]; rest = [t for t in terms if t is not e]
                code = ['RkW _w%s = rkw_ee(%s, %s);' % (v, e[2], e[3])]
            else:
                e = efs[0]; rest = [t for t in terms if t is not e]
                code = ['RkW _w%s = rkw_ef0(%s, %s, %s);' % (v, e[2], e[3], 'true' if e[1] < 0 else 'false')]
            for t in rest:
                if t[0] == 'ee': code.append('rkw_ee_acc(_w%s, %s, %s, %s);' % (v, t[2], t[3], 'true' if t[1] < 0 else 'false'))
                elif t[0] == 'ef': code.append('rkw_ef(_w%s, %s, %s, %s);' % (v, t[2], t[3], 'true' if t[1] < 0 else 'false')); nef += 1
                elif t[0] == 'v': code.append('rkw_v(_w%s, %s, %s);' % (v, t[2], 'true' if t[1] < 0 else 'false')); nv += 1
            code.append('auto %s = rkw_fin(_w%s);' % (v, v))
            out[i] = ind + ' '.join(code)
            nfused += 1; nee += 1
        # drop the consumed intermediate definitions
        for c in consumed:
            out[defs[c][0]] = defs[c][2] + '// lincomb: fused into a lazy linear combination'
    text = '\n'.join(out)
    anchor = '// ---- lazy poly_mix accumulator (RK_LAZYCHAIN) ----'
    assert text.count(anchor) == 1
    text = text.replace(anchor, HELPER + anchor)
    open(sys.argv[2], 'w').write(text + tail)
    print('lincomb: %d trees fused (%d EE, %d EF, %d plain terms)' % (nfused, nee, nef, nv), file=sys.stderr)

if __name__ == '__main__':
    main()
