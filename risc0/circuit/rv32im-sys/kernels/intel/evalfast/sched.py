# Parse the fast eval_check amalgamation, compute source-order liveness per
# function, and optionally rewrite statement order (sink loads / min-live).
# usage: sched.py stats <amalg.cpp>
#        sched.py sink  <in.cpp> <out.cpp> [mode]   mode: loads | loads+h | all
import re, sys, collections

IDRE = re.compile(r'\b(x\d+|arg\d+|_h_data_\d+_\d+)\b')
FRE = re.compile(r'^(__attribute__\(\([a-z_]+\)\) )?FpExt (rv32im_v2_\d+|poly_fp)\((.*)\) \{$')

def load(path):
    src = open(path).read()
    cut = src.find('// Intel SYCL eval_check kernel')
    lines = src[:cut].split('\n')
    funcs = []
    for i, l in enumerate(lines):
        m = FRE.match(l)
        if not m: continue
        e = i + 1
        while lines[e] != '}': e += 1
        params = {}
        for p in m.group(3).split(','):
            p = p.strip(); mm = re.match(r'(.+?)\s*(\*?)\s*(\w+)$', p)
            params[mm.group(3)] = (mm.group(1).strip() + mm.group(2)).replace(' ', '')
        funcs.append(dict(name=m.group(2), start=i, end=e, params=params))
    return src, cut, lines, funcs

WORDS = {'Fp': 1, 'FpExt': 4, 'RkAcc': 8, 'Fp*': 2, 'FpExt*': 2, 'Fp**': 2, 'size_t': 2, 'ptr': 2}

class Stmt:
    __slots__ = ('idx', 'text', 'kind', 'defs', 'uses', 'typ', 'store_slot', 'load_slot', 'arr')

def parse_func(lines, f):
    """Return list of Stmt (comments dropped; preamble kept as kind='fixed')."""
    types = dict(f['params'])
    types['poly_mix'] = 'FpExt*'
    stmts = []
    body = lines[f['start'] + 1:f['end']]
    for t in body:
        s = t.strip()
        if not s or s.startswith('//'):
            continue
        st = Stmt(); st.idx = len(stmts); st.text = t; st.defs = []; st.uses = []
        st.typ = None; st.store_slot = None; st.load_slot = None; st.arr = None
        m = re.match(r'^(auto|RkAcc|FpExt|Fp|constexpr Fp|constexpr FpExt|size_t)\s+(\w+)\s*(=|\()(.*);$', s)
        if s.startswith('alignas'):
            # private array decl: alignas(N) uint32_t x33_raw[N]; Fp* x33 = ...
            nm = re.search(r'(Fp\*|FpExt\*) (x\d+) =', s)
            st.kind = 'fixed'; st.defs = [nm.group(2)]; types[nm.group(2)] = nm.group(1)
        elif s.startswith('return'):
            st.kind = 'ret'; st.uses = IDRE.findall(s)
        elif s.startswith('constexpr'):
            st.kind = 'const'; st.defs = [m.group(2)]
            types[m.group(2)] = 'ConstExt' if 'FpExt' in m.group(1) else 'Const'
        elif s.startswith('size_t mask'):
            st.kind = 'fixed'; st.defs = ['mask']
        elif m:
            name = m.group(2); rhs = m.group(4)
            st.defs = [name]
            us = [u for u in IDRE.findall(rhs)]
            st.uses = us
            if 'rv32im_v2_' in rhs:
                st.kind = 'call'; types[name] = 'FpExt'
            elif re.match(r'^\s*(/\*data=\*/)?(args\[\d+\]|arg\d+)\[\d+ \* steps', rhs):
                st.kind = 'gload'; types[name] = 'Fp'
            elif re.match(r'^\s*_h_data_\d+_\d+\s*$', rhs):
                st.kind = 'hcopy'; types[name] = 'Fp'
            else:
                ma = re.match(r'^\s*(x\d+|arg\d+)\[(\d+)\]\s*$', rhs)
                if ma:
                    st.kind = 'pload'; st.arr = ma.group(1); st.load_slot = int(ma.group(2))
                    at = types.get(ma.group(1), 'Fp*')
                    types[name] = at[:-1]
                else:
                    st.kind = 'arith'
                    if m.group(1) == 'RkAcc': types[name] = 'RkAcc'
                    elif m.group(1) == 'FpExt': types[name] = 'FpExt'
                    else:
                        tt = 'Fp'
                        for u in us:
                            if types.get(u) in ('FpExt', 'RkAcc', 'ConstExt'): tt = 'FpExt'
                        if 'poly_mix' in rhs: tt = 'FpExt'
                        types[name] = tt
        else:
            ms = re.match(r'^(x\d+|arg\d+)\[(\d+)\] = (x\d+);$', s)
            if ms:
                st.kind = 'pstore'; st.arr = ms.group(1); st.store_slot = int(ms.group(2))
                st.uses = [ms.group(3), ms.group(1)]
            elif s.startswith('auto') or s.startswith('FpExt'):
                raise Exception('unparsed ' + s)
            else:
                raise Exception('unparsed ' + s)
        stmts.append(st)
    return stmts, types

def words(types, v):
    t = types.get(v)
    if t is None: return 1
    if t in ('Const', 'ConstExt'): return 0
    return WORDS.get(t, 1)

def liveness(stmts, types, ignore=()):
    last = {}
    for st in stmts:
        for u in st.uses: last[u] = st.idx
    live = collections.Counter(); cur = 0; peak = 0; peak_at = 0
    # params live from start until last use
    defined = set()
    for st in stmts:
        for d in st.defs: defined.add(d)
    for v, li in last.items():
        if v not in defined and v not in ignore:
            cur += words(types, v)
    cur0 = cur
    order = sorted(stmts, key=lambda s: s.idx)
    pos = {s.idx: k for k, s in enumerate(order)}
    lastpos = {}
    for k, st in enumerate(order):
        for u in st.uses: lastpos[u] = k
    cur = cur0; peak = cur
    curve = []
    for k, st in enumerate(order):
        for d in st.defs:
            if d in lastpos and lastpos[d] > k:
                cur += words(types, d)
        for u in set(st.uses):
            if lastpos.get(u) == k and u not in ignore:
                cur -= words(types, u)
        curve.append(cur)
        if cur > peak: peak, peak_at = cur, k
    return peak, peak_at, curve

def reorder(stmts, mode):
    """Return new statement order. Sinks movable defs to just before first use."""
    n = len(stmts)
    users = collections.defaultdict(list)
    for st in stmts:
        for u in st.uses: users[u].append(st.idx)
    movable = set()
    for st in stmts:
        if st.kind == 'gload' and mode in ('loads', 'loads+h', 'all'): movable.add(st.idx)
        if st.kind == 'hcopy' and mode in ('loads+h', 'all'): movable.add(st.idx)
        if st.kind == 'const': movable.add(st.idx)
    if mode in ('loads+h+p', 'loads+h+p+u'):
        # private / mix loads: movable if no store to the same slot between the load and its first use
        stores = collections.defaultdict(list)
        for st in stmts:
            if st.kind == 'pstore': stores[(st.arr, st.store_slot)].append(st.idx)
        for st in stmts:
            if st.kind == 'pload' and st.defs:
                us = users.get(st.defs[0], [])
                if not us: continue
                fu = min(us)
                if any(st.idx < j < fu for j in stores.get((st.arr, st.load_slot), [])):
                    continue
                movable.add(st.idx)
        for st in stmts:
            if st.kind == 'hcopy': movable.add(st.idx)
            if st.kind == 'gload': movable.add(st.idx)
    # attach movable stmt before its first user (recursively for chains)
    attach = collections.defaultdict(list)
    first_user = {}
    for st in stmts:
        if st.idx in movable and st.defs:
            us = users.get(st.defs[0], [])
            if not us:
                continue
            first_user[st.idx] = min(us)
    # resolve: if first user is itself movable, follow chain
    def target(i):
        seen = 0
        while i in first_user and seen < 1000:
            i = first_user[i]; seen += 1
            if i not in movable: return i
        return i
    out = []
    K = int(__import__('os').environ.get('SINK_LEAD', '0'))
    fixed_seq = [st.idx for st in stmts if not (st.idx in movable and st.idx in first_user)]
    fpos = {i: k for k, i in enumerate(fixed_seq)}
    for st in stmts:
        if st.idx in movable and st.idx in first_user:
            t = target(st.idx)
            if K and t in fpos:
                k = fpos[t] - K
                # do not move above the original position
                while k < fpos[t] and fixed_seq[max(k, 0)] < st.idx: k += 1
                t = fixed_seq[max(k, 0)]
            attach[t].append(st)
    placed = set()
    def emit(st):
        # movable deps first, in original order
        for m in sorted(attach.get(st.idx, []), key=lambda s: s.idx):
            emit(m)
        out.append(st)
    for st in stmts:
        if st.idx in movable and st.idx in first_user:
            continue
        emit(st)
    assert len(out) == n, (len(out), n)
    return out

def main():
    cmd = sys.argv[1]
    src, cut, lines, funcs = load(sys.argv[2])
    if cmd == 'stats':
        mode = sys.argv[3] if len(sys.argv) > 3 else None
        tot = 0
        for f in funcs:
            stmts, types = parse_func(lines, f)
            p0, a0, _ = liveness(stmts, types)
            row = '%-14s n=%5d peak=%4d@%4d' % (f['name'], len(stmts), p0, a0)
            for md in ('loads', 'loads+h'):
                o = reorder(stmts, md)
                for k, s in enumerate(o): pass
                o2 = []
                for k, s in enumerate(o):
                    s2 = Stmt();
                    for a in Stmt.__slots__: setattr(s2, a, getattr(s, a))
                    s2.idx = k; o2.append(s2)
                p1, a1, _ = liveness(o2, types)
                row += '  %s=%4d@%4d' % (md, p1, a1)
            print(row)
    elif cmd == 'sink':
        out_path = sys.argv[3]; mode = sys.argv[4] if len(sys.argv) > 4 else 'loads'
        new_lines = list(lines)
        # rebuild each function body
        repl = {}
        for f in funcs:
            stmts, types = parse_func(lines, f)
            o = reorder(stmts, mode)
            repl[f['start']] = (f['end'], [s.text for s in o])
        res = []; i = 0
        while i < len(lines):
            if i in repl:
                res.append(lines[i]); e, body = repl[i]; res.extend(body); res.append('}'); i = e + 1
            else:
                res.append(lines[i]); i += 1
        open(out_path, 'w').write('\n'.join(res) + src[cut:])

if __name__ == "__main__" and sys.argv[1] in ("stats", "sink"):
    main()

# ---------------- min-live greedy list scheduler ----------------
def minlive_order(stmts, types, look=0):
    import heapq
    n = len(stmts)
    head = [s for s in stmts if s.kind in ('const', 'fixed')]
    tail = [s for s in stmts if s.kind in ('call', 'ret')]
    body = [s for s in stmts if s.kind not in ('const', 'fixed', 'call', 'ret')]
    defidx = {}
    for s in body:
        for d in s.defs: defidx[d] = s.idx
    preds = collections.defaultdict(set); succs = collections.defaultdict(set)
    for s in body:
        for u in s.uses:
            if u in defidx and defidx[u] != s.idx:
                preds[s.idx].add(defidx[u]); succs[defidx[u]].add(s.idx)
    # memory ordering per (array, slot)
    lastst = {}; loads_since = collections.defaultdict(list)
    for s in body:
        if s.kind == 'pstore':
            key = (s.arr, s.store_slot)
            if key in lastst:
                preds[s.idx].add(lastst[key]); succs[lastst[key]].add(s.idx)
            for l in loads_since[key]:
                preds[s.idx].add(l); succs[l].add(s.idx)
            loads_since[key] = []; lastst[key] = s.idx
        elif s.kind == 'pload':
            key = (s.arr, s.load_slot)
            if key in lastst:
                preds[s.idx].add(lastst[key]); succs[lastst[key]].add(s.idx)
            loads_since[key].append(s.idx)
    # remaining uses count per value (within body + tail)
    remaining = collections.Counter()
    for s in body + tail:
        for u in set(s.uses): remaining[u] += 1
    byidx = {s.idx: s for s in body}
    npred = {s.idx: len(preds[s.idx]) for s in body}
    ready = set(i for i in npred if npred[i] == 0)
    out = []
    def delta(s):
        d = 0
        for v in s.defs:
            if remaining.get(v, 0) > 0: d += words(types, v)
        for u in set(s.uses):
            if remaining.get(u, 0) == 1: d -= words(types, u)
        return d
    while ready:
        best = None
        for i in ready:
            s = byidx[i]
            key = (delta(s), i)
            if best is None or key < best[0]: best = (key, i)
        i = best[1]; s = byidx[i]
        ready.discard(i); out.append(s)
        for u in set(s.uses): remaining[u] -= 1
        for j in succs[i]:
            npred[j] -= 1
            if npred[j] == 0: ready.add(j)
    assert len(out) == len(body), (len(out), len(body))
    return head + out + tail

def renumber(order):
    o2 = []
    for k, s in enumerate(order):
        s2 = Stmt()
        for a in Stmt.__slots__: setattr(s2, a, getattr(s, a))
        s2.idx = k; o2.append(s2)
    return o2

def main2():
    cmd = sys.argv[1]
    src, cut, lines, funcs = load(sys.argv[2])
    if cmd == 'mlstats':
        for f in funcs:
            stmts, types = parse_func(lines, f)
            p0, _, _ = liveness(stmts, types)
            o = renumber(minlive_order(stmts, types))
            p1, a1, _ = liveness(o, types)
            print('%-14s orig=%4d minlive=%4d@%d' % (f['name'], p0, p1, a1))
    elif cmd == 'minlive':
        out_path = sys.argv[3]
        only = set(sys.argv[4].split(',')) if len(sys.argv) > 4 else None
        repl = {}
        for f in funcs:
            if only and f['name'] not in only: continue
            stmts, types = parse_func(lines, f)
            o = minlive_order(stmts, types)
            repl[f['start']] = (f['end'], [s.text for s in o])
        res = []; i = 0
        while i < len(lines):
            if i in repl:
                res.append(lines[i]); e, body = repl[i]; res.extend(body); res.append('}'); i = e + 1
            else:
                res.append(lines[i]); i += 1
        open(out_path, 'w').write('\n'.join(res) + src[cut:])

if __name__ == '__main__' and sys.argv[1] in ('mlstats', 'minlive'):
    main2()

def build_deps(body):
    defidx = {}
    for s in body:
        for d in s.defs: defidx[d] = s.idx
    preds = collections.defaultdict(list)
    for s in body:
        for u in s.uses:
            if u in defidx and defidx[u] != s.idx and defidx[u] not in preds[s.idx]:
                preds[s.idx].append(defidx[u])
    lastst = {}; loads_since = collections.defaultdict(list)
    for s in body:
        if s.kind == 'pstore':
            key = (s.arr, s.store_slot)
            if key in lastst: preds[s.idx].append(lastst[key])
            preds[s.idx].extend(loads_since[key]); loads_since[key] = []; lastst[key] = s.idx
        elif s.kind == 'pload':
            key = (s.arr, s.load_slot)
            if key in lastst: preds[s.idx].append(lastst[key])
            loads_since[key].append(s.idx)
    return preds

def dfs_order(stmts, types, child='orig'):
    import sys as _s
    _s.setrecursionlimit(100000)
    head = [s for s in stmts if s.kind in ('const', 'fixed')]
    tail = [s for s in stmts if s.kind in ('call', 'ret')]
    body = [s for s in stmts if s.kind not in ('const', 'fixed', 'call', 'ret')]
    byidx = {s.idx: s for s in body}
    preds = build_deps(body)
    users = collections.Counter()
    for s in body + tail:
        for u in s.uses: users[u] += 1
    # subtree size (words-needed estimate) for SU ordering
    need = {}
    def su(i):
        if i in need: return need[i]
        ps = [su(p) for p in preds[i]]
        ps.sort(reverse=True)
        r = max([p + k for k, p in enumerate(ps)] + [1])
        need[i] = r
        return r
    done = set(); out = []
    def emit(i):
        if i in done: return
        done.add(i)
        ps = list(preds[i])
        if child == 'su': ps.sort(key=lambda p: -su(p))
        else: ps.sort()
        for p in ps: emit(p)
        out.append(byidx[i])
    roots = [s.idx for s in body if s.kind == 'pstore' or not any(users[d] for d in s.defs)]
    # demand from tail too: values used by call/ret
    tail_uses = set()
    for s in tail: tail_uses.update(s.uses)
    for s in body:
        if s.kind == 'pstore' or not any(users[d] for d in s.defs):
            emit(s.idx)
        elif any(d in tail_uses for d in s.defs):
            emit(s.idx)
    for s in body: emit(s.idx)
    assert len(out) == len(body)
    return head + out + tail

def pointer_ids(lines, funcs):
    """map func -> {param: 'x33'|'x34'|'data'|'accum'|'mix'|'global'} by tracing calls."""
    byname = {f['name']: f for f in funcs}
    ids = {'poly_fp': {'x33': 'x33', 'x34': 'x34'}}
    order = ['poly_fp']
    seen = set()
    while order:
        fn = order.pop(0)
        if fn in seen: continue
        seen.add(fn)
        f = byname[fn]
        for t in lines[f['start'] + 1:f['end']]:
            m = re.search(r'= (rv32im_v2_\d+(?:_p\d+)?)\((.*)\);', t)
            if not m: continue
            callee = m.group(1); args = [a.strip() for a in m.group(2).split(',')]
            cf = byname[callee]
            pnames = list(cf['params'].keys())
            mp = {}
            for pn, a in zip(pnames, args):
                a2 = re.sub(r'/\*(\w+)=\*/args\[\d+\]', r'\1', a)
                if a2 in ('data', 'accum', 'mix', 'global'): mp[pn] = a2
                elif a2 in ids[fn]: mp[pn] = ids[fn][a2]
            ids[callee] = mp; order.append(callee)
    return ids

def uniform_set(stmts, types, pids):
    uni = set()
    for s in stmts:
        if s.kind == 'const': uni.update(s.defs); continue
        if s.kind == 'pload':
            if pids.get(s.arr) in ('mix', 'global'): uni.update(s.defs)
            continue
        if s.kind == 'arith':
            us = [u for u in s.uses]
            if all(u in uni for u in us):
                uni.update(s.defs)
    return uni

def best_orders(lines, funcs, choices=('orig', 'sink', 'dfs', 'dfs_su'), report=None):
    pids = pointer_ids(lines, funcs)
    res = {}
    for f in funcs:
        st, ty = parse_func(lines, f)
        uni = uniform_set(st, ty, pids.get(f['name'], {}))
        ty2 = dict(ty)
        for u in uni: ty2[u] = 'Const'
        cands = {}
        if 'orig' in choices: cands['orig'] = st
        if 'sink' in choices: cands['sink'] = reorder(st, 'loads+h')
        if 'dfs' in choices: cands['dfs'] = dfs_order(st, ty, 'orig')
        if 'dfs_su' in choices: cands['dfs_su'] = dfs_order(st, ty, 'su')
        sc = {k: liveness(renumber(o), ty2)[0] for k, o in cands.items()}
        k = min(sc, key=lambda x: (sc[x], x != 'orig'))
        if report is not None: report.append('%s %s %s' % (f['name'], k, sc))
        res[f['start']] = (f['end'], [s.text for s in cands[k]])
    return res

def write_with(lines, src, cut, repl, out_path):
    res = []; i = 0
    while i < len(lines):
        if i in repl:
            res.append(lines[i]); e, body = repl[i]; res.extend(body); res.append('}'); i = e + 1
        else:
            res.append(lines[i]); i += 1
    open(out_path, 'w').write('\n'.join(res) + src[cut:])

if __name__ == '__main__' and sys.argv[1] == 'best':
    src, cut, lines, funcs = load(sys.argv[2])
    rep = []
    repl = best_orders(lines, funcs, report=rep)
    write_with(lines, src, cut, repl, sys.argv[3])
    print('\n'.join(rep))

def remat_loads(stmts, types, kinds=('gload', 'pload'), min_gap=0):
    """Re-issue load statements before each use (new SSA names) when the use is
    more than min_gap statements after the previous (re)load. Returns new stmt list
    (texts rewritten) - caller should then apply sink reorder."""
    stores = collections.defaultdict(list)
    for st in stmts:
        if st.kind == 'pstore': stores[(st.arr, st.store_slot)].append(st.idx)
    loadst = {}
    for st in stmts:
        if st.kind in kinds and st.defs: loadst[st.defs[0]] = st
    out = []
    cur_name = {}   # orig name -> (current name, stmt idx of last (re)load)
    cnt = collections.Counter()
    for st in stmts:
        if st.defs and st.defs[0] in loadst and st is loadst[st.defs[0]]:
            cur_name[st.defs[0]] = (st.defs[0], len(out))
            out.append(st); continue
        newtext = st.text; newuses = list(st.uses); pre = []
        for u in set(st.uses):
            if u not in loadst: continue
            ld = loadst[u]
            nm, at = cur_name[u]
            gap = len(out) - at
            if gap <= min_gap: 
                if nm != u:
                    newtext = re.sub(r'\b%s\b' % u, nm, newtext); newuses = [nm if x == u else x for x in newuses]
                continue
            if ld.kind == 'pload' and any(ld.idx < j < st.idx for j in stores.get((ld.arr, ld.load_slot), [])):
                if nm != u:
                    newtext = re.sub(r'\b%s\b' % u, nm, newtext); newuses = [nm if x == u else x for x in newuses]
                continue
            cnt[u] += 1
            nn = '%s_r%d' % (u, cnt[u])
            r = Stmt()
            for a in Stmt.__slots__: setattr(r, a, getattr(ld, a))
            r.text = re.sub(r'\b%s\b' % u, nn, ld.text, count=1)
            r.defs = [nn]; r.idx = -1
            types[nn] = types.get(u)
            pre.append(r)
            cur_name[u] = (nn, len(out) + len(pre) - 1)
            newtext = re.sub(r'\b%s\b' % u, nn, newtext); newuses = [nn if x == u else x for x in newuses]
        out.extend(pre)
        s2 = Stmt()
        for a in Stmt.__slots__: setattr(s2, a, getattr(st, a))
        s2.text = newtext; s2.uses = newuses
        out.append(s2)
    return renumber(out)

if __name__ == '__main__' and sys.argv[1] == 'remat':
    # sched.py remat in out kinds gap fn1,fn2,...|all   (others get plain sink)
    src, cut, lines, funcs = load(sys.argv[2])
    kinds = tuple(sys.argv[4].split('+')); gap = int(sys.argv[5])
    sel = None if sys.argv[6] == 'all' else set(sys.argv[6].split(','))
    repl = {}
    for f in funcs:
        st, ty = parse_func(lines, f)
        if sel is None or f['name'] in sel:
            r = remat_loads(st, ty, kinds, gap)
            o = reorder(r, 'loads+h')
        else:
            o = reorder(st, 'loads+h')
        repl[f['start']] = (f['end'], [s.text for s in o])
    write_with(lines, src, cut, repl, sys.argv[3])

if __name__ == '__main__' and sys.argv[1] == 'unhoist':
    # replace `auto X = _h_data_C_B;` in sub-functions by a direct data load, then sink
    src, cut, lines, funcs = load(sys.argv[2])
    pids = pointer_ids(lines, funcs)
    new = list(lines)
    n = 0
    for f in funcs:
        if f['name'] == 'poly_fp': continue
        dp = [p for p, v in pids.get(f['name'], {}).items() if v == 'data']
        assert len(dp) == 1, (f['name'], dp)
        for i in range(f['start'] + 1, f['end']):
            m = re.match(r'^(\s*auto x\d+ = )_h_data_(\d+)_(\d+);$', new[i])
            if m:
                new[i] = '%s%s[%s * steps + ((cycle - kInvRate * %s) & mask)];' % (m.group(1), dp[0], m.group(2), m.group(3)); n += 1
    open(sys.argv[3] + '.tmp', 'w').write('\n'.join(new) + src[cut:])
    src, cut, lines, funcs = load(sys.argv[3] + '.tmp')
    repl = {}
    for f in funcs:
        st, ty = parse_func(lines, f)
        o = reorder(st, 'loads+h')
        repl[f['start']] = (f['end'], [s.text for s in o])
    write_with(lines, src, cut, repl, sys.argv[3])
    print('unhoisted', n)

if __name__ == '__main__' and sys.argv[1] == 'mixorder':
    # sched.py mixorder in out fn=order,...   (default order: sink)
    src, cut, lines, funcs = load(sys.argv[2])
    spec = dict(x.split('=') for x in sys.argv[4].split(',')) if len(sys.argv) > 4 else {}
    repl = {}
    for f in funcs:
        st, ty = parse_func(lines, f)
        k = spec.get(f['name'], 'sink')
        o = {'sink': lambda: reorder(st, 'loads+h'), 'dfs': lambda: dfs_order(st, ty, 'orig'),
             'dfs_su': lambda: dfs_order(st, ty, 'su'), 'orig': lambda: st}[k]()
        repl[f['start']] = (f['end'], [s.text for s in o])
    write_with(lines, src, cut, repl, sys.argv[3])

def profitable_sink(stmts, types, iters=3):
    """Move arith statements down to just before their first use when this does not
    extend any operand's live range (all operands are used again at/after the first use)."""
    order = list(stmts)
    for _ in range(iters):
        changed = 0
        pos = {id(s): k for k, s in enumerate(order)}
        users = collections.defaultdict(list)
        for k, s in enumerate(order):
            for u in s.uses: users[u].append(k)
        new = []
        moved = collections.defaultdict(list)
        skip = set()
        for k, s in enumerate(order):
            if s.kind == 'arith' and s.defs:
                us = users.get(s.defs[0], [])
                if us:
                    fu = min(us)
                    if fu > k + 1:
                        ok = True
                        for u in set(s.uses):
                            lu = max(users[u]) if users[u] else -1
                            if lu < fu: ok = False; break
                        if ok:
                            moved[fu].append(s); skip.add(k); changed += 1
        for k, s in enumerate(order):
            if k in skip: continue
            new.extend(moved.get(k, []))
            new.append(s)
        order = new
        if not changed: break
    return order
