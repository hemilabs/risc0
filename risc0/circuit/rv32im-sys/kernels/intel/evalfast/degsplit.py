#!/usr/bin/env python3
# degsplit.py in.cpp out_hi.cpp out_lo.cpp [DMAX]: split the constraint polynomial poly_fp of an
# eval_check amalgamation (build.rs's intel_eval_check_amalg.cpp) into two: out_hi keeps the
# linked terms of degree > DMAX (default 3), out_lo those of degree <= DMAX, each with the code
# only the other needed removed. out_hi + out_lo == in, for any inputs.
#
# Why: every linked term (a constraint, times its selectors) vanishes on the trace domain H
# (|H| = N = 2^po2) for a valid witness, so the quotient of the degree-<=DMAX part by Z_H = x^N - 1
# has degree < (DMAX - 1) * N. With DMAX = 3 that is < 2N: eval_check can evaluate it on the even
# half of its 4N-point coset and extend it to the odd half (see eval_check.cpp and intel.rs),
# instead of evaluating it on all 4N points.
#
# Degrees are in units of N: a trace load (data, accum, through a hoisted _h param or a per-row
# array slot) is 1, a global, mix or constant 0, a product adds, a sum takes the max. A link
# `B + C * poly_mix[k]` adds the term C (degree deg C); `B + C * D * poly_mix[k]` adds C * D, and
# when one of C, D is itself a sum of linked terms (an accumulator), its terms, each multiplied
# by the other factor. A term's degree is the one it has in the final result.
import re, sys, collections
from dag import build_all

src_path, out_hi, out_lo = sys.argv[1], sys.argv[2], sys.argv[3]
DMAX = int(sys.argv[4]) if len(sys.argv) > 4 else 3
src, lines, funcs, wrapper, Fs = build_all(src_path)
FN = {F.name: F for F in Fs}
fmeta = {f['name']: f for f in funcs}

# ---- symbolic execution: one global node per (function, node) (each function is called once) ----
calls = collections.Counter()
for F in Fs:
    for (i, k, p) in F.stmts:
        if k == 'call':
            calls[F.nodes[p]['fn']] += 1
assert all(v == 1 for v in calls.values()), 'degsplit: a function is called more than once'
assert set(calls) == set(FN) - {'poly_fp'}, 'degsplit: unreachable function'

deg = {}      # gid -> degree (plain values) or None (accumulator)
terms = {}    # gid -> list of (degree, leaf gid) for accumulators
deps = collections.defaultdict(list)   # gid -> gids it reads (for liveness)
alias_param = {}   # (fn, param node) -> caller gid
arrays = {}   # array id -> {idx: store gid}
store_val = {}     # store gid -> value gid
store_line = {}    # store gid -> (fn, line)
ret_of = {}        # fn -> ret gid
call_of = {}       # callee fn -> call gid in caller


def val(g):
    return g


def is_acc(g):
    return deg.get(g) is None and g in terms


def acc_terms(g, shift):
    return [(d + shift, leaf) for d, leaf in terms[g]]


def run(fname, argmap):
    F = FN[fname]
    N = F.nodes
    ptr = {}   # pointer param / local array name -> ('arr', id) | ('buf', name)
    for (t, pn) in F.params:
        if pn in argmap and t.endswith('*'):
            ptr[pn] = argmap[pn]
    events = sorted([(n.get('line', -1), 0, i) for i, n in enumerate(N)] +
                    [(ln, 1, (k, p)) for (ln, k, p) in F.stmts], key=lambda e: (e[0], e[1]))
    for ln, kind, p in events:
        if kind == 0:
            i = p; n = N[i]; g = (fname, i); op = n['op']
            if op == 'param':
                src_g = argmap.get(n['name'])
                if src_g is None:
                    raise Exception('unbound param %s.%s' % (fname, n['name']))
                alias_param[g] = src_g
                deps[g].append(src_g)
                if is_acc(src_g):
                    deg[g] = None; terms[g] = terms[src_g]
                else:
                    deg[g] = deg[src_g]
            elif op in ('const', 'econst', 'pmread'):
                deg[g] = 0
            elif op == 'zeroE':
                deg[g] = None; terms[g] = []
            elif op == 'load':
                deg[g] = 1
            elif op == 'aload':
                s = n['src']
                if s.startswith('args['):
                    assert s in ('args[2]', 'args[3]'), s
                    deg[g] = 0
                else:
                    kind_, ref = ptr[s]
                    if kind_ == 'buf':
                        assert ref in ('args[2]', 'args[3]'), (fname, s, ref)
                        deg[g] = 0
                    else:
                        st = arrays[ref].get(n['idx'])
                        assert st is not None, 'load before store %s %s[%d]' % (fname, s, n['idx'])
                        v = store_val[st]
                        deps[g].append(st)
                        if is_acc(v):
                            deg[g] = None; terms[g] = terms[v]
                        else:
                            deg[g] = deg[v]
            elif op in ('+', '-', '*', 'neg'):
                a = [(fname, x) for x in n['args']]
                deps[g].extend(a)
                accs = [x for x in a if is_acc(x)]
                if not accs:
                    deg[g] = sum(deg[x] for x in a) if op == '*' else max(deg[x] for x in a)
                elif op == '*':
                    assert len(accs) == 1, 'acc * acc in %s' % fname
                    other = [x for x in a if x != accs[0]][0]
                    deg[g] = None; terms[g] = acc_terms(accs[0], deg[other])
                else:
                    assert all(is_acc(x) for x in a), 'plain value added to an accumulator in %s' % fname
                    deg[g] = None; terms[g] = [t for x in a for t in terms[x]]
            elif op in ('link', 'clink'):
                a = [(fname, x) for x in n['args']]
                deps[g].extend(a)
                base, fac = a[0], a[1:]
                assert is_acc(base), 'link base is not an accumulator in %s' % fname
                accs = [x for x in fac if is_acc(x)]
                t = list(terms[base])
                if not accs:
                    t.append((sum(deg[x] for x in fac), g))
                else:
                    assert len(accs) == 1
                    other = [x for x in fac if x != accs[0]]
                    t += acc_terms(accs[0], sum(deg[x] for x in other))
                deg[g] = None; terms[g] = t
            elif op == 'call':
                callee = n['fn']
                cF = FN[callee]
                cargs = {}
                at = n['argtext'][3:]
                cparams = [pn for (_, pn) in cF.params][3:]
                assert len(at) == len(cparams), callee
                for txt, (pt, pn) in zip(at, cF.params[3:]):
                    m = re.match(r'/\*\w+=\*/(args\[\d\])$', txt)
                    if pt.endswith('*'):
                        if m:
                            cargs[pn] = ('buf', m.group(1))
                        else:
                            cargs[pn] = ptr[txt]
                    else:
                        cargs[pn] = (fname, F.var[txt])
                call_of[callee] = g
                r = run(callee, cargs)
                deps[g].append(r)
                # A kept call passes all its arguments, used or not.
                deps[g].extend((fname, x) for x in n['args'])
                if is_acc(r):
                    deg[g] = None; terms[g] = terms[r]
                else:
                    deg[g] = deg[r]
            else:
                raise Exception('degsplit: op %s in %s' % (op, fname))
        else:
            k, pl = p
            if k == 'decl':
                aid = (fname, pl); arrays[aid] = {}; ptr[pl] = ('arr', aid)
            elif k == 'store':
                name, idx, vnode = pl
                sg = (fname, 'store', ln)
                vg = (fname, vnode)
                kind_, ref = ptr[name]
                assert kind_ == 'arr'
                arrays[ref][idx] = sg
                store_val[sg] = vg; store_line[sg] = (fname, ln)
                deps[sg].append(vg)
                deg[sg] = deg[vg]
                if is_acc(vg):
                    terms[sg] = terms[vg]
            elif k == 'ret':
                ret_of[fname] = (fname, pl)
                return (fname, pl)
            elif k == 'call':
                pass
    raise Exception('no return in ' + fname)


root = run('poly_fp', {})
assert is_acc(root)
# A sub-sum may reach the result more than once (a block's constraints under several selectors);
# a term is removed from every use at once, so it goes to out_lo only if all its uses have degree
# <= DMAX (out_hi is evaluated on every point and takes any degree).
uses = collections.defaultdict(set)
for d, leaf in terms[root]:
    uses[leaf].add(d)
mixed = [leaf for leaf, ds in uses.items() if len({d > DMAX for d in ds}) > 1]
leaf_deg = {leaf: max(ds) for leaf, ds in uses.items()}
sys.stderr.write('degsplit: %d terms used at degrees on both sides of %d (kept in hi)\n' % (len(mixed), DMAX))
hist = collections.Counter(d for d, _ in terms[root])
sys.stderr.write('degsplit: %d terms, by degree %s\n' % (len(leaf_deg), dict(sorted(hist.items()))))
# every link node is a leaf or a nested link (whose factor is an accumulator)
leaf_set = set(leaf_deg)


def variant(keep_hi):
    removed = {g for g, d in leaf_deg.items() if (d > DMAX) != keep_hi}
    # zero accumulators and link aliases, in execution order (gids were created in that order)
    zero = {}
    alias = {}

    def res(g):
        while g in alias:
            g = alias[g]
        return g
    order = list(deps.keys())
    for g in [x for x in deg]:   # deg was filled in execution order
        if len(g) == 2:
            n = FN[g[0]].nodes[g[1]]
            op = n['op']
        else:
            op = 'store'
        if not is_acc(g):
            continue
        if op == 'zeroE':
            zero[g] = True
        elif op in ('link', 'clink'):
            a = [(g[0], x) for x in n['args']]
            base = a[0]
            fac_accs = [x for x in a[1:] if is_acc(x)]
            dead_term = (g in removed) or (fac_accs and zero[fac_accs[0]])
            if dead_term:
                alias[g] = base
                zero[g] = zero[base]
            else:
                zero[g] = False
        elif op == 'param':
            zero[g] = zero[alias_param[g]]
        elif op == 'aload':
            zero[g] = zero[deps[g][0]]
        elif op == 'store':
            zero[g] = zero[deps[g][0]]
        elif op == 'call':
            zero[g] = zero[deps[g][0]]
        elif op in ('+', '-', 'neg'):
            zero[g] = all(zero[x] for x in deps[g] if is_acc(x))
        elif op == '*':
            zero[g] = any(zero[x] for x in deps[g] if is_acc(x))
        else:
            raise Exception(op)
    # liveness from the result; a function with anything live (e.g. a store a later function
    # reads) needs its call, and the call its arguments
    live = set()
    stack = [res(root)]
    while stack:
        while stack:
            g = stack.pop()
            g = res(g)
            if g in live:
                continue
            live.add(g)
            for d in deps[g]:
                stack.append(res(d))
        live_fn = {g[0] for g in live}
        stack = [call_of[f] for f in live_fn if f in call_of and call_of[f] not in live]
    # emit
    out = list(lines)
    drop = set()
    rename = {}   # (fn, varname) -> varname
    for F in Fs:
        inv = collections.defaultdict(list)
        for name, i in F.var.items():
            inv[i].append(name)
        nm = {}
        for i in range(len(F.nodes)):
            g = (F.name, i)
            if g in alias:
                t = res(g)
                assert t[0] == F.name
                # the target's own name: the one defined on its line
                tl = F.nodes[t[1]].get('line')
                tn = [x for x in inv[t[1]] if re.search(r'\b%s\b = ' % re.escape(x), lines[tl])] if tl is not None else []
                tn = tn[0] if tn else inv[t[1]][0]
                for x in inv[i]:
                    nm[x] = tn
        rename[F.name] = nm
        for i, n in enumerate(F.nodes):
            g = (F.name, i)
            ln = n.get('line')
            if ln is None:
                continue
            if g in alias or g not in live:
                if n['op'] in ('const', 'econst'):
                    continue   # harmless, the compiler drops unused constants
                drop.add(ln)
        for (ln, k, p) in F.stmts:
            if k == 'store' and (F.name, 'store', ln) not in live:
                drop.add(ln)
            if k == 'call' and F.nodes[p]['fn'] not in live_fn:
                drop.add(ln)
    tok = re.compile(r'\bx\d+\b')
    for F in Fs:
        nm = rename[F.name]
        f = fmeta[F.name]
        for i in range(f['start'] + 1, f['end']):
            if i in drop:
                out[i] = None
            elif nm and out[i]:
                l = out[i]
                lhs, eq, rhs = l.partition(' = ')
                if eq:
                    out[i] = lhs + eq + tok.sub(lambda m: nm.get(m.group(0), m.group(0)), rhs)
                else:
                    out[i] = tok.sub(lambda m: nm.get(m.group(0), m.group(0)), l)
    body = '\n'.join(x for x in out if x is not None)
    kept = sum(1 for g in leaf_set if (g in leaf_set and g not in removed))
    return body + wrapper, kept, len(live)


for path, hi in ((out_hi, True), (out_lo, False)):
    text, kept, nlive = variant(hi)
    open(path, 'w').write(text)
    sys.stderr.write('degsplit: %s: %d terms, %d live nodes\n' % ('hi' if hi else 'lo', kept, nlive))
print('degsplit: ok')
