# batchinv.py <in.cpp> <out.cpp> [M]
# Montgomery batch inversion of the FpExt inv_0 calls inside each mux arm of
# exec_TopAccum. Denominator slices (pure: data/mix loads + arithmetic) are
# moved up to just before the first use of their chunk; each chunk of M
# inverses shares one Fp inversion of the product of the per-element norms c.
# Bit-exact: every inverse is the unique field inverse, computed with the same
# b0/b2/c formulas as fpext.h inv(); a zero norm maps to 0 exactly as inv(0)=0.
import re, sys
src = open(sys.argv[1]).read()
M = int(sys.argv[3]) if len(sys.argv) > 3 else 8
fs = src.index('ComponentStruct exec_TopAccum(ExecContext& ctx')
fs = src.rindex('\n', 0, fs) + 1
fe = src.index('\n__attribute__((noinline))', fs)
func = src[fs:fe].split('\n')
arm_re = re.compile(r'^(\} else )?if \(to_size_t\(LOAD\(LAYOUT_LOOKUP\(LAYOUT_SUBSCRIPT\(LAYOUT_LOOKUP\(arg0, instResult._selector\)')
starts = [i for i, l in enumerate(func) if arm_re.match(l)]
end = max(i for i, l in enumerate(func) if l.startswith('} else {'))
starts.append(end)
def_re = re.compile(r'^(ExtVal|Val) (x\d+) = (.*);$')
var_re = re.compile(r'\bx\d+\b')
out = func[:starts[0]]
stats = []
ctr = [0]
def emit_chunk(dens, tag):
    L = []
    n = len(dens)
    for i, d in enumerate(dens):
        p = f'_bi{tag}_{i}'
        L.append(f'Fp {p}b0 = {d}.elems[0] * {d}.elems[0] + Fp(11) * ({d}.elems[1] * ({d}.elems[3] + {d}.elems[3]) - {d}.elems[2] * {d}.elems[2]);')
        L.append(f'Fp {p}b2 = {d}.elems[0] * ({d}.elems[2] + {d}.elems[2]) - {d}.elems[1] * {d}.elems[1] + Fp(11) * ({d}.elems[3] * {d}.elems[3]);')
        L.append(f'Fp {p}c = {p}b0 * {p}b0 + Fp(11) * {p}b2 * {p}b2;')
        L.append(f'bool {p}z = ({p}c == Fp(0));')
        L.append(f'Fp {p}cs = {p}z ? Fp(1) : {p}c;')
        if i == 0:
            L.append(f'Fp {p}pp = {p}cs;')
        else:
            L.append(f'Fp {p}pp = _bi{tag}_{i-1}pp * {p}cs;')
    L.append(f'Fp _bi{tag}_r = inv_chain(_bi{tag}_{n-1}pp);')
    for i in range(n - 1, -1, -1):
        p = f'_bi{tag}_{i}'
        if i > 0:
            L.append(f'Fp {p}ic = {p}z ? Fp(0) : (_bi{tag}_r * _bi{tag}_{i-1}pp);')
            L.append(f'_bi{tag}_r = _bi{tag}_r * {p}cs;')
        else:
            L.append(f'Fp {p}ic = {p}z ? Fp(0) : _bi{tag}_r;')
    for i, d in enumerate(dens):
        p = f'_bi{tag}_{i}'
        L.append(f'{p}b0 *= {p}ic; {p}b2 *= {p}ic;')
        L.append(f'ExtVal _iv_{d} = ExtVal({d}.elems[0] * {p}b0 + Fp(11) * {d}.elems[2] * {p}b2, '
                 f'-{d}.elems[1] * {p}b0 + Fp(Fp::P - 11) * {d}.elems[3] * {p}b2, '
                 f'-{d}.elems[0] * {p}b2 + {d}.elems[2] * {p}b0, '
                 f'{d}.elems[1] * {p}b2 - {d}.elems[3] * {p}b0);')
    return L
for a, (s, e) in enumerate(zip(starts, starts[1:])):
    out.append(func[s])
    body = func[s + 1:e]
    defs = {}
    for i, l in enumerate(body):
        m = def_re.match(l)
        if m: defs[m.group(2)] = (i, m.group(1), m.group(3))
    # inverses in order of first use
    order = []; first_use = {}
    for i, l in enumerate(body):
        for v in re.findall(r'inv_0\((x\d+)\)', l):
            if v not in first_use:
                first_use[v] = i; order.append(v)
    def slice_of(v, acc):
        if v not in defs or v in acc: return
        i, t, ex = defs[v]
        acc.add(v)
        for u in var_re.findall(ex):
            slice_of(u, acc)
    ok_inv = []
    for v in order:
        if v not in defs or defs[v][1] != 'ExtVal':
            continue
        sl = set(); slice_of(v, sl)
        bad = any(('layout1' in defs[u][2]) or ('inv_0' in defs[u][2]) or ('STORE' in defs[u][2]) for u in sl)
        if bad: continue
        ok_inv.append((v, sl))
    # chunk
    chunks = [ok_inv[i:i + M] for i in range(0, len(ok_inv), M)]
    insert_at = {}   # body index -> list of lines to insert before
    moved = set()
    for c in chunks:
        if len(c) < 2: continue
        pos = min(first_use[v] for v, _ in c)
        sl = set().union(*[x for _, x in c])
        need = sorted((defs[u][0] for u in sl if defs[u][0] >= pos and u not in moved))
        # also any slice stmt defined after pos must move; ones before pos stay
        lines = []
        for idx in need:
            lines.append(body[idx]); moved.add(var_re.findall(def_re.match(body[idx]).group(2))[0])
        tag = ctr[0]; ctr[0] += 1
        lines += emit_chunk([v for v, _ in c], tag)
        insert_at.setdefault(pos, []).extend(lines)
        for v, _ in c:
            pass
    batched = set(v for c in chunks if len(c) >= 2 for v, _ in c)
    moved_idx = set(defs[u][0] for u in moved)
    for i, l in enumerate(body):
        if i in insert_at: out.extend(insert_at[i])
        if i in moved_idx: continue
        for v in batched:
            l = l.replace(f'inv_0({v})', f'_iv_{v}')
        out.append(l)
    stats.append((a, len(order), len(batched)))
out.extend(func[end:])
res = '\n'.join(out)
assert res.count('inv_0(') == sum(n - b for _, n, b in stats), (res.count('inv_0('),)
src = src[:fs] + res + src[fe:]
open(sys.argv[2], 'w').write(src)
print('M', M, 'arms', stats, 'remaining inv_0 in TopAccum', res.count('inv_0('))
