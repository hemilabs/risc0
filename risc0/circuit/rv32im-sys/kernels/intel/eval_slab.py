#!/usr/bin/env python3
# Intel mono eval_check post-pass (RISC0_INTEL_EVAL_FAST), run after
# eval_nozero.py and eval_lazy_chain.py: move poly_fp's per-row private arrays
# (x33 Fp[1007], x34 FpExt[144]) into a slot-major global slab.
#
# Private memory is laid out per lane (lane-contiguous), so each SIMD16 access
# to one array slot touched 16 cache lines. Here every u32 word w of row r
# lives at slab[(r / 512) * NWORDS * 512 + w * 512 + r % 512]: a SIMD16 access
# is one contiguous 64-byte line. Array slots are also renumbered by liveness
# (interval colouring over the program-ordered access trace through the
# rv32im_v2_* call chain), so the pool shrinks from 1583 to ~356 words per row.
# Measured: eval_check 1514 -> 620 ms per po2=20 segment, bit-identical.
#
# The kernel wrapper (eval_check.cpp, under #ifdef RK_NWORDS) allocates the
# slab and tiles the launch; rows only ever read words they wrote themselves,
# and tiles reuse the slab in order on the single in-order queue.
#
# The assumptions (straight-line code, literal indices, plain stores, one
# call per sub-function, write-before-read, one write per slot) are asserted;
# the build fails rather than miscompiling.
#
# usage: eval_slab.py <in.cpp> <out.cpp>
import heapq
import re
import sys
from collections import Counter, defaultdict

STRIDE = 512  # rows per slab block == eval_check work-group size

FDEF = re.compile(r'^(?:__attribute__\(\(noinline\)\) )?FpExt (rv32im_v2_\d+|poly_fp)\((.*)\) \{')
CALL = re.compile(r'(rv32im_v2_\d+)\(cycle, steps, poly_mix, (.*)\);')
CALL_STMT = re.compile(r'^\s*auto \w+ = rv32im_v2_\d+\(cycle, steps, poly_mix, [\w\s,/*=\[\]]*\);\s*$')
ACC = re.compile(r'\b(\w+)\[(\d+)\]')
# The analysis treats each function body as straight-line code.
CONTROL = re.compile(r'\b(if|else|for|while|do|switch|case|goto)\b|\?|[{}]')
# Anything after a subscript other than a read or a plain `=` store
# (compound assignment, increment, member access) is unsupported.
BAD_AFTER = re.compile(r'^(\+\+|--|[-+*/%&|^]=|<<=|>>=|\.|->)')
# Declarations as emitted by eval_nozero.py.
DECL = re.compile(r'^(\s*)alignas\(16\) uint32_t (x\d+)_raw\[(\d+)\]; (Fp|FpExt)\* \2 = reinterpret_cast<\4\*>\(\2_raw\);$')


def fail(msg):
    sys.exit('eval_slab: ' + msg)


def parse(src):
    funcs, cur = {}, None
    for i, l in enumerate(src):
        m = FDEF.match(l)
        if m:
            cur = m.group(1)
            if cur in funcs:
                fail('duplicate function ' + cur)
            funcs[cur] = {'params': [p.strip().split()[-1] for p in m.group(2).split(',')],
                          'start': i, 'lines': []}
        elif cur and l.startswith('}'):
            cur = None
        elif cur:
            funcs[cur]['lines'].append(i)
    if 'poly_fp' not in funcs:
        fail('poly_fp not found')
    return funcs


def trace(src, funcs, arrays):
    """Program-ordered access trace of the arrays through the call chain."""
    out, calls = [], Counter()

    def run(fname, alias):
        calls[fname] += 1
        for i in funcs[fname]['lines']:
            if i in arrays['decl_lines']:
                continue
            l = src[i]
            code = re.sub(r'//.*', '', l)
            if CONTROL.search(code):
                fail('control flow in %s at line %d (straight-line code expected): %s'
                     % (fname, i, code.strip()[:100]))
            for m in ACC.finditer(l):
                nm = m.group(1)
                if nm in alias:
                    after = l[m.end():].lstrip()
                    if BAD_AFTER.match(after):
                        fail('unsupported use of %s[%s] at line %d: %s' % (nm, m.group(2), i, l.strip()[:100]))
                    if (after.startswith('=') and not after.startswith('==')
                            and re.search(r'\b%s\[%s\]' % (re.escape(nm), m.group(2)), after[1:])):
                        # The trace records this store before the read on its
                        # right-hand side; C++ reads first.
                        fail('statement reads the slot it writes at line %d: %s' % (i, l.strip()[:100]))
                    out.append(dict(arr=alias[nm], idx=int(m.group(2)),
                                    rw='W' if after.startswith('=') and not after.startswith('==') else 'R',
                                    line=i, s=m.start(), e=m.end(), name=nm))
            for nm in alias:
                for m in re.finditer(r'\b%s\b' % re.escape(nm), l):
                    rest = l[m.end():]
                    if rest.startswith('[') and not re.match(r'\[\d+\]', rest):
                        fail('non-literal index at line %d' % i)
            c = CALL.search(l)
            if c and not CALL_STMT.match(l):
                # Reads/stores sharing a statement with a call are not ordered
                # against the callee by this trace (C++ leaves it unspecified).
                fail('call inside a larger statement at line %d: %s' % (i, l.strip()[:100]))
            if c:
                f = c.group(1)
                args = [re.sub(r'/\*.*?\*/', '', a).strip() for a in c.group(2).split(',')]
                params = funcs[f]['params'][3:]
                if len(params) != len(args):
                    fail('argument count mismatch calling %s at line %d' % (f, i))
                for a in args:
                    for nm in alias:
                        if a != nm and re.search(r'\b%s\b' % re.escape(nm), a):
                            fail('array used inside an argument expression at line %d' % i)
                run(f, {p: alias[a] for p, a in zip(params, args) if a in alias})
            elif any(re.search(r'\b%s\b(?!\[)' % re.escape(nm), l) for nm in alias):
                fail('array name used outside a call/subscript at line %d: %s' % (i, l.strip()[:100]))

    run('poly_fp', {name: name for name in arrays['names']})
    return out, calls


def main():
    src = open(sys.argv[1]).read().split('\n')
    funcs = parse(src)
    pf = funcs['poly_fp']
    arrays = {'names': {}, 'decl_lines': set()}
    for i in pf['lines']:
        m = DECL.match(src[i])
        if m:
            ind, name, words, ty = m.groups()
            width = 4 if ty == 'FpExt' else 1
            arrays['names'][name] = (ty, width, int(words) // width)
            arrays['decl_lines'].add(i)
    if not arrays['names']:
        fail('no eval_nozero.py array declarations found in poly_fp (run it first)')

    tr, calls = trace(src, funcs, arrays)
    if any(v != 1 for v in calls.values()):
        fail('a sub-function is called more than once: %s' % dict(calls))
    if not tr:
        fail('no array accesses found')

    # One live interval per slot: first write .. last read; read-before-write
    # or a second write to the same slot would break the renumbering.
    iv, opened = {}, {}
    for t, a in enumerate(tr):
        key = (a['arr'], a['idx'])
        if a['rw'] == 'W':
            if key in opened or key in iv:
                fail('slot %s written twice' % (key,))
            opened[key] = [t, t]
        else:
            if key not in opened:
                fail('slot %s read before written' % (key,))
            opened[key][1] = t
    iv.update(opened)

    # Lowest-free interval colouring in u32 words (FpExt slots take 4 words).
    events = []
    for key, (s, e) in iv.items():
        events.append((s, 0, key))  # alloc
        events.append((e, 1, key))  # free after the last read (also e == s)
    events.sort(key=lambda x: (x[0], x[1]))
    free, nxt, words = [], 0, {}
    for _, kind, key in events:
        width = arrays['names'][key[0]][1]
        if kind == 0:
            ws = []
            for _ in range(width):
                if free:
                    ws.append(heapq.heappop(free))
                else:
                    ws.append(nxt)
                    nxt += 1
            words[key] = ws
        else:
            for w in words[key]:
                heapq.heappush(free, w)
    nwords = nxt

    out = list(src)
    for acc in sorted(tr, key=lambda x: (x['line'], -x['s'])):
        ws = words[(acc['arr'], acc['idx'])]
        l = out[acc['line']]
        if acc['rw'] == 'R':
            new = ('RK_LD1(%s, %du)' % (acc['name'], ws[0]) if len(ws) == 1 else
                   'RK_LD4(%s, %du, %du, %du, %du)' % ((acc['name'],) + tuple(ws)))
            out[acc['line']] = l[:acc['s']] + new + l[acc['e']:]
        else:
            m = re.match(r'^(\s*)%s\[%d\] = (.*);\s*$' % (acc['name'], acc['idx']), l)
            if not m:
                fail('unexpected store form at line %d: %s' % (acc['line'], l.strip()[:100]))
            if len(ws) == 1:
                out[acc['line']] = '%sRK_ST1(%s, %du, %s);' % (m.group(1), acc['name'], ws[0], m.group(2))
            else:
                out[acc['line']] = '%sRK_ST4(%s, %du, %du, %du, %du, %s);' % (
                    (m.group(1), acc['name']) + tuple(ws) + (m.group(2),))
    for i in arrays['decl_lines']:
        ind, name, _, ty = DECL.match(src[i]).groups()
        out[i] = '%s%s* %s = reinterpret_cast<%s*>(args[4]);' % (ind, ty, name, ty)

    macros = '''
// ---- eval_slab.py: per-row arrays live in a slot-major global slab (args[4]) ----
#define RK_STRIDE %du
#define RK_NWORDS %du
#define RK_LD1(p, w) (static_cast<const ::risc0::Fp>(::risc0::Fp::fromRaw(reinterpret_cast<const uint32_t*>(p)[(size_t)(w) * RK_STRIDE])))
#define RK_ST1(p, w, v) (reinterpret_cast<uint32_t*>(p)[(size_t)(w) * RK_STRIDE] = ::risc0::Fp(v).asRaw())
#define RK_LD4(p, w0, w1, w2, w3) (static_cast<const ::risc0::FpExt>(::risc0::FpExt(RK_LD1(p, w0), RK_LD1(p, w1), RK_LD1(p, w2), RK_LD1(p, w3))))
#define RK_ST4(p, w0, w1, w2, w3, v) do { ::risc0::FpExt _rk_v = (v); RK_ST1(p, w0, _rk_v.elems[0]); RK_ST1(p, w1, _rk_v.elems[1]); RK_ST1(p, w2, _rk_v.elems[2]); RK_ST1(p, w3, _rk_v.elems[3]); } while (0)
''' % (STRIDE, nwords)
    proto = next((i for i, l in enumerate(out) if re.match(r'^FpExt (rv32im_v2_\d+|poly_fp)\(.*\);', l)), None)
    if proto is None:
        fail('function prototypes not found')
    out[proto] = macros + out[proto]
    text = '\n'.join(out)
    if 'Fp* args[5] = {accum, data, out, mix,' not in text:
        fail('kernel wrapper does not pass the slab row as args[4] (eval_check.cpp, #ifdef RK_NWORDS)')
    open(sys.argv[2], 'w').write(text)
    print('eval_slab: arrays %s, %d accesses, %d slots -> %d words/row (%.0f MB per 2^20 rows)'
          % (sorted(arrays['names']), len(tr), len(iv), nwords, nwords * 4.0), file=sys.stderr)


if __name__ == '__main__':
    main()
