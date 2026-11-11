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
  tier B: as A but X is initialised some other way (stores, memset, index loop) or not at all
  tier C: a loop with a user call stores into some alloca (no load of it outside the loop seen)
Plus, per function, the number of loops that contain a user call ("call loops"); a user call is a call
to anything but llvm.* / air.* intrinsics.

usage: ir_scan.py [--json out.json] [--tiers ABC] <dir-or-.ll>...
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
    def __init__(self, name):
        self.name = name
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
                cur = Fn(m.group(1).strip('"'))
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
    out = []
    call_loops = 0
    for f in fns:
        fd, n = analyse(f, md)
        call_loops += n
        for d in fd:
            m = DBG_RE.search(d['store'])
            d['src'] = loc_of(md, int(m.group(1)), cache) if m else None
            d['store'] = DBG_RE.sub('', d['store'])[:200]
            out.append(d)
    return out, len(fns), call_loops


def main(argv):
    js = None
    tiers = 'ABC'
    paths = []
    it = iter(argv)
    for a in it:
        if a == '--json':
            js = next(it)
        elif a == '--tiers':
            tiers = next(it)
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
        cnt = {t: sum(1 for d in fd if d['tier'] == t) for t in 'ABC'}
        summary.append((path, nf, ncl, cnt))
    dm = demangle(sorted({d['fn'] for d in allf} | {c for d in allf for c in d['calls']}))
    print('%-70s %6s %6s %4s %4s %4s' % ('file', 'fns', 'cloops', 'A', 'B', 'C'))
    for path, nf, ncl, cnt in summary:
        print('%-70s %6d %6d %4d %4d %4d' % (os.path.relpath(path)[-70:], nf, ncl, cnt['A'], cnt['B'],
                                             cnt['C']))
    for d in allf:
        if d['tier'] not in tiers:
            continue
        src = '%s:%s' % d['src'] if d['src'] else '?'
        print('\n[%s] %s\n  fn    %s\n  src   %s\n  alloca %s %s init=%s var_index=%s n_stores=%d loop=%s\n'
              '  calls %s\n  store %s' % (
                  d['tier'], os.path.relpath(d['file']), dm.get(d['fn'], d['fn'])[:200], src,
                  d['alloca'], d['type'], d['init'], d['var_index'], d['n_stores'], ','.join(d['loop_md']) or '-',
                  ', '.join(dm.get(c, c)[:80] for c in d['calls'][:4]), d['store']))
    if js:
        with open(js, 'w') as fh:
            json.dump(allf, fh, indent=1, default=str)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
