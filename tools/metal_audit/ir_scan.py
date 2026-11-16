#!/usr/bin/env python3
"""R17 audit: scan optimized Metal LLVM IR (from emit_ir.sh) for the Apple GPU private-array defect shape.

The defect (R17 minrepro, rv32im witgen exec_ToBits_5__W): a private (alloca) array is initialised,
then a loop that is NOT unrolled calls a function that stays a call (noinline or not inlined) and stores
into the array; after the loop, the element stored in the last iteration reads back its initial value.
In the IR of the reproducer the initialisation is the C++ constructor loop over the array (a loop whose
induction variable is a POINTER phi into the alloca); the same kernel with an index-loop initialisation
(i32 phi) is correct, as is the kernel with the store before the call.

For every function the scanner finds natural loops (dominator-based back edges) and reports:
  tier A: alloca X has a pointer-phi (constructor) init loop, a later loop L calls a user function and
          stores into X after the call, and X is loaded outside L          (exact reproducer shape)
          R17b: also without the loop L (straight line): the constructor init (a pointer-phi loop
          over X, or a call of a function that runs such a loop over its pointer argument, e.g.
          an out-of-line array/struct constructor at -O0) dominates a user call that dominates a
          store into X, and X is loaded after that store (the unrolled-map/mux shape:
          default-constructed aggregate, call, element stores, read)
  tier B: as A (loop form) but X is initialised some other way (stores, memset, index loop) or not
          at all
  tier C: a loop with a user call stores into some alloca (no load of it outside the loop seen)
  tier S: (R17b, the second trigger, R17 fix round 2) an out-of-line function with an aggregate
          by-value parameter (>= 2 scalar leaves, e.g. FpExt) spills it to an alloca X, then
          writes X after or through a user call (a store after the call, or a call that takes a
          pointer into X, e.g. `lhs *= rhs` in `operator*(FpExt lhs, Fp rhs) { lhs *= rhs;
          return lhs; }`), and reads X afterwards (S/param); or such a function writes a local
          aggregate X, calls, writes X again and reads it (S/local)
Plus, per function, the number of loops that contain a user call ("call loops"); a user call is a call
to anything but llvm.* / air.* intrinsics. "Dominates" is per instruction (block dominance, or
earlier in the same block), so the arms of an if/else do not combine into a shape.

usage: ir_scan.py [--json out.json] [--tiers ABCS] [--fail-tiers AB] <dir-or-.ll>...
  --fail-tiers T: exit status 1 if any finding of a tier in T is reported (a test gate).
Pure stdlib; run with python3 -I.
"""
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

DEF_RE = re.compile(r'^define\b.*?@("?[^\s(]+"?)\(')
LABEL_RE = re.compile(r'^([\w.$-]+):')
VAL_RE = re.compile(r'%[\w.$-]+')
CALL_RE = re.compile(r'\b(?:tail |musttail |notail )?call\b.*?@("?[\w.$-]+"?)\(')
DBG_RE = re.compile(r'!dbg !(\d+)')


def parse_metadata(lines):
    md = {}
    for ln in lines:
        if ln.startswith('!') and ' = ' in ln:
            k, v = ln.split(' = ', 1)
            if k[1:].isdigit():
                md[int(k[1:])] = v
    return md


def md_field(s, name):
    m = re.search(r'\b' + name + r': (!?\d+|"[^"]*")', s)
    return m.group(1) if m else None


def loc_of(md, n, cache):
    """(file, line) of DILocation !n (innermost), or None."""
    if n in cache:
        return cache[n]
    res = None
    s = md.get(n, '')
    if 'DILocation' in s:
        line = md_field(s, 'line')
        scope = md_field(s, 'scope')
        f = None
        seen = 0
        while scope and seen < 50:
            seen += 1
            ss = md.get(int(scope[1:]), '')
            fl = md_field(ss, 'file')
            if fl:
                fs = md.get(int(fl[1:]), '')
                fn = md_field(fs, 'filename')
                f = fn.strip('"') if fn else None
                break
            scope = md_field(ss, 'scope')
        res = (f, int(line) if line else None)
    cache[n] = res
    return res


class Fn:
    def __init__(self, name, header=''):
        self.name = name
        self.header = header    # the `define ... {` line
        self.blocks = []        # [label]
        self.insts = {}         # label -> [line]
        self.succ = {}


def parse_functions(lines):
    fns = []
    cur = None
    label = None
    for ln in lines:
        if cur is None:
            m = DEF_RE.match(ln)
            if m and ln.rstrip().endswith('{'):
                cur = Fn(m.group(1).strip('"'), ln)
                label = '%entry'
                cur.blocks.append(label)
                cur.insts[label] = []
            continue
        if ln.startswith('}'):
            fns.append(cur)
            cur = None
            continue
        m = LABEL_RE.match(ln)
        if m and not ln.startswith(' '):
            label = '%' + m.group(1)
            cur.blocks.append(label)
            cur.insts[label] = []
            continue
        s = ln.strip()
        if s and not s.startswith(';'):
            cur.insts[label].append(s)
    for f in fns:
        for b in f.blocks:
            term = f.insts[b][-1] if f.insts[b] else ''
            succ = []
            if term.startswith('br ') or term.startswith('switch ') or term.startswith('indirectbr'):
                succ = re.findall(r'label (%[\w.$-]+)', term)
            elif term.startswith('switch'):
                pass
            # switch spans several lines in LLVM text: collect labels from all trailing lines
            if not succ:
                for s in reversed(f.insts[b][-64:]):
                    succ += re.findall(r'label (%[\w.$-]+)', s)
                    if s.startswith('switch'):
                        break
                else:
                    succ = [x for s in f.insts[b][-1:] for x in re.findall(r'label (%[\w.$-]+)', s)]
            f.succ[b] = list(dict.fromkeys(succ))
    return fns


def dominators(f):
    blocks = f.blocks
    idx = {b: i for i, b in enumerate(blocks)}
    preds = defaultdict(list)
    for b in blocks:
        for s in f.succ[b]:
            if s in idx:
                preds[s].append(b)
    # reverse postorder
    order, seen = [], set()
    stack = [(blocks[0], iter(f.succ[blocks[0]]))]
    seen.add(blocks[0])
    while stack:
        b, it = stack[-1]
        nxt = next(it, None)
        if nxt is None:
            order.append(b)
            stack.pop()
        elif nxt in idx and nxt not in seen:
            seen.add(nxt)
            stack.append((nxt, iter(f.succ[nxt])))
    rpo = list(reversed(order))
    rpo_i = {b: i for i, b in enumerate(rpo)}
    idom = {rpo[0]: rpo[0]}

    def intersect(a, b):
        while a != b:
            while rpo_i[a] > rpo_i[b]:
                a = idom[a]
            while rpo_i[b] > rpo_i[a]:
                b = idom[b]
        return a
    changed = True
    while changed:
        changed = False
        for b in rpo[1:]:
            ps = [p for p in preds[b] if p in idom]
            if not ps:
                continue
            new = ps[0]
            for p in ps[1:]:
                new = intersect(p, new)
            if idom.get(b) != new:
                idom[b] = new
                changed = True
    return idom, preds, rpo_i


def dominates(idom, a, b):
    while True:
        if a == b:
            return True
        nb = idom.get(b)
        if nb is None or nb == b:
            return False
        b = nb


def natural_loops(f):
    idom, preds, rpo_i = dominators(f)
    loops = defaultdict(set)
    for b in f.blocks:
        if b not in idom:
            continue
        for s in f.succ[b]:
            if s in idom and dominates(idom, s, b):
                body = loops[s]
                body.add(s)
                work = [b]
                while work:
                    x = work.pop()
                    if x in body:
                        continue
                    body.add(x)
                    work.extend(p for p in preds[x] if p in idom)
    return loops


def user_call(s):
    m = CALL_RE.search(s)
    if not m:
        return None
    name = m.group(1).strip('"')
    if name.startswith('llvm.') or name.startswith('air.'):
        return None
    return name


def lhs(s):
    if s.startswith('%') and ' = ' in s:
        return s.split(' = ', 1)[0]
    return None


def loop_md(f, body, md):
    """llvm.loop hints (strings) on the loop's latch branches."""
    out = set()
    for b in body:
        for s in f.insts[b][-1:]:
            m = re.search(r'!llvm\.loop !(\d+)', s)
            if not m:
                continue
            node = md.get(int(m.group(1)), '')
            for ref in re.findall(r'!(\d+)', node):
                for st in re.findall(r'!"([^"]+)"', md.get(int(ref), '')):
                    out.add(st.replace('llvm.loop.', ''))
    return sorted(out)


TYPE_RE = re.compile(r'^(%[\w."$:<>, -]+|%"[^"]+") = type (.*)$')


def parse_types(lines):
    """%name -> its body text (`{ ... }`, `<{ ... }>` or `opaque`)."""
    out = {}
    for ln in lines:
        m = TYPE_RE.match(ln)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def split_type_list(t):
    """'i32, [4 x %class.Fp], { i32 }' -> top-level items."""
    out, depth, cur = [], 0, ''
    for c in t:
        if c in '[{<(':
            depth += 1
        elif c in ']}>)':
            depth -= 1
        if c == ',' and depth == 0:
            out.append(cur.strip())
            cur = ''
        else:
            cur += c
    if cur.strip():
        out.append(cur.strip())
    return out


def leaves(t, types, depth=0):
    """Scalar leaves of an IR type (pointers count 1; 0 for unknown/opaque)."""
    t = t.strip()
    if depth > 40:
        return 0
    if t.endswith('*'):
        return 1
    m = re.match(r'^\[(\d+) x (.*)\]$', t)
    if m:
        return int(m.group(1)) * leaves(m.group(2), types, depth + 1)
    m = re.match(r'^<?\{(.*)\}>?$', t)
    if m:
        return sum(leaves(x, types, depth + 1) for x in split_type_list(m.group(1)))
    if t.startswith('%'):
        body = types.get(t)
        return leaves(body, types, depth + 1) if body and body != 'opaque' else 0
    if re.match(r'^(i\d+|half|float|double|<\d+ x [^>]+>)$', t):
        return 1
    return 0


def fn_params(header):
    """[(type, '%N')] of a `define` line (attributes dropped)."""
    m = re.search(r'@"?[^\s(]+"?\((.*)\)[^)]*\{\s*$', header)
    if not m:
        return []
    out = []
    for item in split_type_list(m.group(1)):
        nm = re.search(r'(%[\w.$-]+)\s*$', item)
        if not nm:
            continue
        ty = item[:nm.start()].strip()
        # drop parameter attributes (noundef, byval(...), ...) after the type
        tm = re.match(r'^((?:\[[^\]]*\]|<?\{[^}]*\}>?|%"[^"]+"|%[\w.$:-]+|i\d+|float|half|double)'
                      r'(?:\s*addrspace\(\d+\))?\**)', ty)
        out.append((tm.group(1) if tm else ty.split(' ')[0], nm.group(1), ty))
    return out


def fn_ret(header):
    m = re.match(r'^define\s+((?:\w+\s+)*)(.*?)\s+@', header)
    if not m:
        return ''
    ret = m.group(2)
    for kw in ('noundef', 'nonnull', 'zeroext', 'signext', 'inreg', 'fastcc', 'dso_local'):
        ret = re.sub(r'\b%s\b' % kw, '', ret)
    ret = re.sub(r'\b(?:align|dereferenceable(?:_or_null)?)\(?\s*\d+\)?', '', ret)
    return ret.strip()


def call_args(s):
    """Value tokens passed in a call's argument list."""
    m = CALL_RE.search(s)
    if not m:
        return []
    o = s.find('(', m.end() - 1)
    return VAL_RE.findall(s[o:])


def derive(f, roots, through_calls=False):
    """value -> root for every pointer derived (gep/bitcast/phi/select/addrspacecast) from roots;
    also through a pointer slot (-O0: `store %p, %slot` then `load %slot`) and, with
    through_calls, a pointer returned by a call that takes a derived pointer (-O0:
    `r0::array::operator[](this, i)`)."""
    derived = {r: r for r in roots}
    defs = {}
    slots = {}
    for b in f.blocks:
        for s in f.insts[b]:
            v = lhs(s)
            if v:
                defs[v] = s
    stores = [s for b in f.blocks for s in f.insts[b] if s.startswith('store ') and '*' in s]
    changed = True
    while changed:
        changed = False
        for s in stores:
            parts = split_type_list(s[len('store '):].split(', align')[0])
            if len(parts) >= 2 and parts[0].rstrip().endswith('*') is False and '*' in parts[0]:
                vt, pt = VAL_RE.findall(parts[0]), VAL_RE.findall(parts[1])
                if vt and pt and vt[-1] in derived and pt[-1] not in slots and \
                        ' = alloca ' in defs.get(pt[-1], ''):
                    slots[pt[-1]] = derived[vt[-1]]
                    changed = True
        for v, s in defs.items():
            if v in derived:
                continue
            rhs = s.split(' = ', 1)[1]
            op = rhs.split(' ', 1)[0]
            if op in ('getelementptr', 'bitcast', 'phi', 'select', 'addrspacecast'):
                for t in VAL_RE.findall(rhs):
                    if t in derived:
                        derived[v] = derived[t]
                        changed = True
                        break
            elif op == 'load' and '*' in rhs.split(',', 1)[0]:
                for t in VAL_RE.findall(rhs):
                    if t in slots:
                        derived[v] = slots[t]
                        changed = True
                        break
            elif through_calls and CALL_RE.search(s) and user_call(s):
                rt = rhs.split('@', 1)[0]
                if '*' in rt:
                    for t in call_args(s):
                        if t in derived:
                            derived[v] = derived[t]
                            changed = True
                            break
    return derived


def writer_functions(fns):
    """Functions that write memory reached from a pointer parameter (a store or memcpy/memset
    destination derived from one, or such a pointer passed to a writer), e.g. FpExt::operator*=
    or Fp::operator=; not r0::array::operator[] (returns a pointer, writes nothing)."""
    info = {}
    for f in fns:
        params = [n for t, n, _ in fn_params(f.header) if t.endswith('*')]
        if not params:
            info[f.name] = (False, [])
            continue
        d = derive(f, params)
        own, calls = False, []
        for b in f.blocks:
            for s in f.insts[b]:
                if s.startswith('store '):
                    parts = split_type_list(s[len('store '):].split(', align')[0])
                    if len(parts) >= 2 and any(t in d for t in VAL_RE.findall(parts[1])):
                        own = True
                elif 'llvm.memcpy' in s or 'llvm.memmove' in s or 'llvm.memset' in s:
                    toks = VAL_RE.findall(s.split('(', 1)[1] if '(' in s else '')
                    if toks and toks[0] in d:
                        own = True
                else:
                    c = user_call(s)
                    if c and any(t in d for t in call_args(s)):
                        calls.append(c)
        info[f.name] = (own, calls)
    out = {n for n, (own, _) in info.items() if own}
    changed = True
    while changed:
        changed = False
        for n, (own, calls) in info.items():
            if n not in out and any(c in out or c not in info for c in calls):
                out.add(n)
                changed = True
    return out, set(info)


def init_functions(fns):
    """R17b: functions that run a pointer-phi loop over (memory reached from) a pointer
    parameter, e.g. an out-of-line array or struct constructor at -O0, or call one with such a
    pointer; their call on an alloca counts as a constructor-loop init (tier A)."""
    info = {}
    for f in fns:
        params = [n for t, n, _ in fn_params(f.header) if t.endswith('*')]
        if not params:
            continue
        d = derive(f, params)
        loops = natural_loops(f) if any(' = phi ' in s for b in f.blocks for s in f.insts[b]) else {}
        inloop = set(b for body in loops.values() for b in body)
        own = False
        calls = []
        for b in f.blocks:
            for s in f.insts[b]:
                v = lhs(s)
                if v and v in d and ' = phi ' in s and b in inloop and \
                        '*' in s.split(' = phi ', 1)[1].split('[')[0]:
                    own = True
                c = user_call(s)
                if c and any(t in d for t in call_args(s)):
                    calls.append(c)
        info[f.name] = (own, calls)
    out = {n for n, (own, _) in info.items() if own}
    changed = True
    while changed:
        changed = False
        for n, (own, calls) in info.items():
            if n not in out and any(c in out for c in calls):
                out.add(n)
                changed = True
    return out


class DomTree:
    def __init__(self, f):
        self.idom, self.preds, _ = dominators(f)
        kids = defaultdict(list)
        for b, p in self.idom.items():
            if b != p:
                kids[p].append(b)
        self.pre, self.post = {}, {}
        root = f.blocks[0]
        n = 0
        stack = [(root, 0)]
        while stack:
            b, st = stack.pop()
            if st == 0:
                self.pre[b] = n
                n += 1
                stack.append((b, 1))
                for k in kids[b]:
                    stack.append((k, 0))
            else:
                self.post[b] = n
                n += 1
        self.succ = f.succ
        self._reach = {}

    def dom(self, a, b):
        """block a dominates block b (reflexive)."""
        if a not in self.pre or b not in self.pre:
            return False
        return self.pre[a] <= self.pre[b] and self.post[b] <= self.post[a]

    def before(self, p, q):
        """instruction p=(block, i) dominates instruction q (strictly before it)."""
        if p[0] == q[0]:
            return p[1] < q[1]
        return self.dom(p[0], q[0])

    def reach_blocks(self, b):
        """blocks reachable from the successors of b."""
        r = self._reach.get(b)
        if r is None:
            r, work = set(), list(self.succ.get(b, []))
            while work:
                x = work.pop()
                if x in r:
                    continue
                r.add(x)
                work.extend(self.succ.get(x, []))
            self._reach[b] = r
        return r

    def reaches(self, p, q):
        """some path runs from instruction p to instruction q."""
        if p[0] == q[0] and p[1] < q[1]:
            return True
        return q[0] in self.reach_blocks(p[0])


def straight_line(f, md, types, init_fns, loop_seen, writers=None, defined=None):
    """R17b: tier A without a loop (constructor init -> user call -> store -> load, each
    dominating the next, the load reachable) and tier S (the second trigger shape)."""
    allocas = {}
    for b in f.blocks:
        for s in f.insts[b]:
            v = lhs(s)
            if v and ' = alloca ' in s:
                allocas[v] = s.split(' = alloca ', 1)[1].split(',')[0].strip()
    if not allocas:
        return []
    params = fn_params(f.header)
    agg_params = {n: t for t, n, _ in params if not t.endswith('*') and leaves(t, types) >= 2}
    byval_params = {n: t for t, n, full in params if 'byval' in full}
    ret = fn_ret(f.header)
    agg_sig = bool(agg_params or byval_params) or (leaves(ret, types) >= 2 and not ret.endswith('*'))
    d = derive(f, list(allocas), through_calls=True)
    loops = natural_loops(f) if any(' = phi ' in s for b in f.blocks for s in f.insts[b]) else {}
    loop_of = {}
    for h, body in loops.items():
        for b in body:
            loop_of.setdefault(b, set()).add(h)
    ev = defaultdict(lambda: {'init': [], 'write': [], 'cwrite': [], 'read': [], 'pinit': [],
                              'whole': []})
    writers = writers or set()
    defined = defined or set()
    calls = []   # (pos, callee, set(roots it takes))
    for b in f.blocks:
        for i, s in enumerate(f.insts[b]):
            pos = (b, i)
            v = lhs(s)
            if v and v in d and ' = phi ' in s and '*' in s.split(' = phi ', 1)[1].split('[')[0] \
                    and b in loop_of:
                ev[d[v]]['init'].append(pos)
            if s.startswith('store '):
                rest = s[len('store '):]
                parts = split_type_list(rest.split(', align')[0].split(', !')[0])
                if len(parts) >= 2:
                    ptoks = [t for t in VAL_RE.findall(parts[1]) if t in d]
                    if ptoks:
                        root = d[ptoks[-1]]
                        ev[root]['write'].append(pos)
                        vt = VAL_RE.findall(parts[0])
                        if vt and vt[-1] in agg_params:
                            ev[root]['pinit'].append(pos)
                        elif leaves(parts[0].rsplit(' ', 1)[0], types) >= 2:
                            ev[root]['whole'].append(pos)
                continue
            if ' = load ' in s:
                for t in VAL_RE.findall(s.split(' = load ', 1)[1]):
                    if t in d:
                        ev[d[t]]['read'].append(pos)
                        break
                continue
            if 'llvm.memcpy' in s or 'llvm.memmove' in s or 'llvm.memset' in s:
                toks = VAL_RE.findall(s.split('(', 1)[1] if '(' in s else '')
                if toks and toks[0] in d:
                    ev[d[toks[0]]]['write'].append(pos)
                    if len(toks) > 1 and toks[1] in byval_params and 'llvm.memset' not in s:
                        ev[d[toks[0]]]['pinit'].append(pos)
                    elif toks[0] == d[toks[0]] or ' = bitcast ' in s:
                        ev[d[toks[0]]]['whole'].append(pos)
                if len(toks) > 1 and toks[1] in d and 'llvm.memset' not in s:
                    ev[d[toks[1]]]['read'].append(pos)
                continue
            c = user_call(s)
            if c:
                roots = {d[t] for t in call_args(s) if t in d}
                for r in roots:
                    if c in init_fns:
                        ev[r]['init'].append(pos)
                    elif c in writers or c not in defined:
                        ev[r]['cwrite'].append(pos)
                    else:
                        ev[r]['read'].append(pos)
                calls.append((pos, c, roots))
    if not ev:
        return []
    dt = DomTree(f)
    calls_by_block = defaultdict(list)
    for pos, c, roots in calls:
        calls_by_block[pos[0]].append((pos[1], c, roots))

    def call_between(p, q, x):
        """a user call not taking x, dominated by p and dominating q (p dominates q)."""
        b = q[0]
        while True:
            for i, c, roots in calls_by_block.get(b, []):
                if x in roots:
                    continue
                if (b != q[0] or i < q[1]) and (b != p[0] or i > p[1]):
                    return (b, i), c
            if b == p[0]:
                return None
            nb = dt.idom.get(b)
            if nb is None or nb == b:
                return None
            b = nb
    out = []
    for x, e in ev.items():
        if x not in allocas:
            continue
        t = allocas[x]
        found = None
        # tier A, straight line: init -> call -> store/memcpy/call-write -> read
        if x not in loop_seen:
            for ip in e['init']:
                for wp in e['write'] + e['cwrite']:
                    if wp == ip or not dt.before(ip, wp):
                        continue
                    cb = call_between(ip, wp, x)
                    if not cb:
                        continue
                    lp = next((r for r in e['read'] if dt.reaches(wp, r)), None)
                    if lp:
                        found = ('A', 'ctor-loop' if ip[0] in loop_of and
                                 ip not in [(pp[0], pp[1]) for pp in e['cwrite']] else 'ctor-call',
                                 ip, cb, wp, lp)
                        break
                if found:
                    break
        # tier S: an aggregate-signature function writes a (param-spilled) aggregate after or
        # through a call and reads it back
        if not found and agg_sig and leaves(t, types) >= 2:
            firsts = [(p, 'param') for p in e['pinit']] + \
                     [(p, 'local') for p in e['whole'] + e['init']]
            for fp, kind in firsts:
                cand = [(w, None) for w in e['cwrite'] if dt.before(fp, w)]
                for w in e['write']:
                    if w != fp and dt.before(fp, w):
                        cb = call_between(fp, w, x)
                        if cb:
                            cand.append((w, cb))
                for w, cb in cand:
                    lp = next((r for r in e['read'] if dt.reaches(w, r)), None)
                    if lp:
                        found = ('S', kind, fp, cb or (w, 'write-call'), w, lp)
                        break
                if found:
                    break
        if not found:
            continue
        tier, init, ip, cb, wp, lp = found
        sline = f.insts[wp[0]][wp[1]]
        out.append({
            'loop_md': [], 'loop_blocks': 0, 'tier': tier, 'fn': f.name, 'loop': '-',
            'alloca': x, 'type': t, 'init': init, 'store_after_call': True, 'var_index': False,
            'read_after_loop': True, 'calls': [cb[1]] if isinstance(cb[1], str) and cb[1] != 'write-call'
            else [user_call(sline) or 'write-call'], 'store': sline,
            'n_stores': len(e['write']) + len(e['cwrite']), 'shape': 'straight-line',
        })
    return out


def analyse(f, md=None):
    # alloca-derived pointers
    allocas = {}
    derived = {}
    defs = {}
    for b in f.blocks:
        for i, s in enumerate(f.insts[b]):
            v = lhs(s)
            if v:
                defs[v] = (b, i, s)
                if ' = alloca ' in s:
                    allocas[v] = s.split(' = alloca ', 1)[1].split(',')[0]
                    derived[v] = v
    changed = True
    while changed:
        changed = False
        for v, (b, i, s) in defs.items():
            if v in derived:
                continue
            rhs = s.split(' = ', 1)[1]
            op = rhs.split(' ', 1)[0]
            if op in ('getelementptr', 'bitcast', 'phi', 'select', 'addrspacecast'):
                for t in VAL_RE.findall(rhs):
                    if t in derived:
                        derived[v] = derived[t]
                        changed = True
                        break
    if not allocas:
        return [], 0
    loops = natural_loops(f)
    # per alloca: init shape, ptr-phi (ctor) loops
    ctor_loop = set()
    memset = set()
    for v, (b, i, s) in defs.items():
        if v in derived and ' = phi ' in s and '*' in s.split(' = phi ', 1)[1].split('[')[0]:
            ctor_loop.add(derived[v])
    for b in f.blocks:
        for s in f.insts[b]:
            if 'llvm.memset' in s:
                for t in VAL_RE.findall(s):
                    if t in derived:
                        memset.add(derived[t])
    # loads/stores per alloca with location
    findings = []
    ncall_loops = 0
    for h, body in loops.items():
        calls = []
        for b in body:
            for i, s in enumerate(f.insts[b]):
                c = user_call(s)
                if c:
                    calls.append((b, i, c))
        if not calls:
            continue
        ncall_loops += 1
        call_blocks = defaultdict(list)
        for b, i, c in calls:
            call_blocks[b].append(i)
        stores = defaultdict(list)
        for b in body:
            for i, s in enumerate(f.insts[b]):
                if s.startswith('store '):
                    ptr = s.split(', ', 1)[1] if ', ' in s else ''
                    toks = VAL_RE.findall(ptr.split(',')[0] if ptr else '')
                    ptok = [t for t in toks if t in derived]
                    if not ptok:
                        continue
                    a = derived[ptok[-1]]
                    after = (b not in call_blocks) or any(ci < i for ci in call_blocks[b])
                    pdef = defs.get(ptok[-1], ('', 0, ''))[2]
                    varidx = (' = phi ' in pdef) or bool(re.search(r'i(?:32|64) %', pdef.split(' = ', 1)[-1]))
                    stores[a].append((b, i, s, after, varidx))
                elif 'llvm.memcpy' in s or 'llvm.memmove' in s:
                    toks = VAL_RE.findall(s.split('(', 1)[1] if '(' in s else '')
                    if toks and toks[0] in derived:
                        a = derived[toks[0]]
                        stores[a].append((b, i, s, True, False))
        for a, sts in stores.items():
            # loads of a outside the loop
            read_after = False
            for b in f.blocks:
                if b in body:
                    continue
                for s in f.insts[b]:
                    if (' = load ' in s or 'llvm.memcpy' in s) and any(
                            t in derived and derived[t] == a for t in VAL_RE.findall(s)):
                        read_after = True
                        break
                if read_after:
                    break
            after = any(x[3] for x in sts)
            if a in ctor_loop and after and read_after:
                tier = 'A'
            elif after and read_after:
                tier = 'B'
            else:
                tier = 'C'
            init = 'ctor-loop' if a in ctor_loop else ('memset' if a in memset else 'other')
            findings.append({
                'loop_md': loop_md(f, body, md or {}), 'loop_blocks': len(body),
                'tier': tier, 'fn': f.name, 'loop': h, 'alloca': a, 'type': allocas[a],
                'init': init, 'store_after_call': after, 'var_index': any(x[4] for x in sts),
                'read_after_loop': read_after, 'calls': sorted({c for _, _, c in calls}),
                'store': sts[0][2], 'n_stores': len(sts),
            })
    return findings, ncall_loops


def demangle(names):
    if not names:
        return {}
    for tool in ('llvm-cxxfilt', 'c++filt'):
        try:
            p = subprocess.run([tool], input='\n'.join(names), capture_output=True, text=True,
                               check=True)
            out = p.stdout.splitlines()
            if len(out) == len(names):
                return dict(zip(names, out))
        except (OSError, subprocess.CalledProcessError):
            continue
    return {n: n for n in names}


def scan_file(path):
    with open(path) as fh:
        lines = fh.read().split('\n')
    md = parse_metadata(lines)
    cache = {}
    fns = parse_functions(lines)
    types = parse_types(lines)
    init_fns = init_functions(fns)
    writers, defined = writer_functions(fns)
    out = []
    call_loops = 0
    for f in fns:
        fd, n = analyse(f, md)
        call_loops += n
        fd += straight_line(f, md, types, init_fns, {x['alloca'] for x in fd}, writers, defined)
        for d in fd:
            m = DBG_RE.search(d['store'])
            d['src'] = loc_of(md, int(m.group(1)), cache) if m else None
            d['store'] = DBG_RE.sub('', d['store'])[:200]
            out.append(d)
    return out, len(fns), call_loops


def main(argv):
    js = None
    tiers = 'ABCS'
    fail = ''
    paths = []
    it = iter(argv)
    for a in it:
        if a == '--json':
            js = next(it)
        elif a == '--tiers':
            tiers = next(it)
        elif a == '--fail-tiers':
            fail = next(it)
        else:
            paths.append(a)
    files = []
    for p in paths:
        if os.path.isdir(p):
            for r, _, fs in os.walk(p):
                files += [os.path.join(r, x) for x in fs if x.endswith('.ll')]
        else:
            files.append(p)
    allf = []
    summary = []
    for path in sorted(files):
        fd, nf, ncl = scan_file(path)
        for d in fd:
            d['file'] = path
        allf += fd
        cnt = {t: sum(1 for d in fd if d['tier'] == t) for t in 'ABCS'}
        summary.append((path, nf, ncl, cnt))
    dm = demangle(sorted({d['fn'] for d in allf} | {c for d in allf for c in d['calls']}))
    print('%-70s %6s %6s %4s %4s %4s %4s' % ('file', 'fns', 'cloops', 'A', 'B', 'C', 'S'))
    for path, nf, ncl, cnt in summary:
        print('%-70s %6d %6d %4d %4d %4d %4d' % (os.path.relpath(path)[-70:], nf, ncl, cnt['A'],
                                                 cnt['B'], cnt['C'], cnt['S']))
    for d in allf:
        if d['tier'] not in tiers:
            continue
        src = '%s:%s' % d['src'] if d['src'] else '?'
        print('\n[%s] %s\n  fn    %s\n  src   %s\n  alloca %s %s init=%s var_index=%s n_stores=%d loop=%s%s\n'
              '  calls %s\n  store %s' % (
                  d['tier'], os.path.relpath(d['file']), dm.get(d['fn'], d['fn'])[:200], src,
                  d['alloca'], d['type'], d['init'], d['var_index'], d['n_stores'], ','.join(d['loop_md']) or '-',
                  ' shape=' + d['shape'] if d.get('shape') else '',
                  ', '.join(dm.get(c, c)[:80] for c in d['calls'][:4]), d['store']))
    if js:
        with open(js, 'w') as fh:
            json.dump(allf, fh, indent=1, default=str)
    bad = [d for d in allf if d['tier'] in fail]
    if bad:
        print('\nir_scan: %d finding(s) of tier(s) %s' % (len(bad), fail), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
