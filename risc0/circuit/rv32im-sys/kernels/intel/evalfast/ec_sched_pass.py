#!/usr/bin/env python3
"""eval_check scheduling post-pass (run AFTER eval_nozero.py / eval_lazy_chain.py
on the fast mono amalgamation). Measured on the real kernel (harness, po2=20,
bit-exact vs production .so on seeds 0,1,2):
   production fast kernel            1513 ms
   + step 2 (sink loads)             1395 ms
   + step 4 (scalar FpExt args)      1284 ms
   + step 1 (un-hoist _h_data)       1262 ms
   + --dfs-su v2_10,v2_4             1253 ms
   + --merge 8,5                     1251 ms
Steps:
 1. un-hoist: in every rv32im_v2_K, `auto x = _h_data_C_B;` becomes a direct load
    from that function's data pointer (the 16 hoisted values were live as
    parameters through all 20 sub-functions).
 2. sink: every global data load / hcopy is moved to just before its first use.
    vISA's pre-RA scheduler skips basic blocks > 20000 instructions
    (G4_Sched.cpp LARGE_BLOCK_SIZE), and every sub-function is one such block, so
    source order is final order. Loads issued earlier (lead 8 / 32 statements)
    measured 1479 / 1524 ms, i.e. at-first-use is best.
 3. strip the now-unused _h_data_* parameters.
 4. scalar FpExt args: by-value FpExt parameters are lowered by clang's SPIR ABI to
    byval pointers into private memory; pass 4 Fp scalars instead.
usage: ec_sched_pass.py in.cpp out.cpp [--dfs-su rv32im_v2_10,rv32im_v2_4] [--merge 8,5]
"""
import os, re, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sched as S

def main():
    a = sys.argv[1:]
    inp, outp = a[0], a[1]
    dfs_su = []; merge = []
    if '--dfs-su' in a: dfs_su = a[a.index('--dfs-su') + 1].split(',')
    if '--merge' in a: merge = a[a.index('--merge') + 1].split(',')
    td = tempfile.mkdtemp()
    # 1+2: unhoist then sink (sched.py unhoist does both)
    src, cut, lines, funcs = S.load(inp)
    pids = S.pointer_ids(lines, funcs)
    new = list(lines)
    for f in funcs:
        if f['name'] == 'poly_fp': continue
        dp = [p for p, v in pids.get(f['name'], {}).items() if v == 'data']
        assert len(dp) == 1, (f['name'], dp)
        for i in range(f['start'] + 1, f['end']):
            m = re.match(r'^(\s*auto x\d+ = )_h_data_(\d+)_(\d+);$', new[i])
            if m:
                new[i] = '%s%s[%s * steps + ((cycle - kInvRate * %s) & mask)];' % (m.group(1), dp[0], m.group(2), m.group(3))
    t1 = os.path.join(td, 'a1.cpp')
    open(t1, 'w').write('\n'.join(new) + src[cut:])
    src, cut, lines, funcs = S.load(t1)
    repl = {}
    for f in funcs:
        st, ty = S.parse_func(lines, f)
        o = S.reorder(st, 'loads+h')
        # invariant: same multiset of statements
        assert sorted(s.text for s in o) == sorted(s.text for s in st), f['name']
        repl[f['start']] = (f['end'], [s.text for s in o])
    t2 = os.path.join(td, 'a2.cpp')
    S.write_with(lines, src, cut, repl, t2)
    if dfs_su:
        src, cut, lines, funcs = S.load(t2)
        repl = {}
        for f in funcs:
            if f['name'] not in dfs_su: continue
            st, ty = S.parse_func(lines, f)
            o = S.dfs_order(st, ty, 'su')
            assert sorted(x.text for x in o) == sorted(x.text for x in st), f['name']
            repl[f['start']] = (f['end'], [x.text for x in o])
        t2b = os.path.join(td, 'a2b.cpp')
        S.write_with(lines, src, cut, repl, t2b)
        t2 = t2b
    t3 = os.path.join(td, 'a3.cpp')
    subprocess.check_call([sys.executable, os.path.join(HERE, 'striph.py'), t2, t3])
    if '--no-sargs' in a:
        import shutil; shutil.copy(t3, outp)
    else:
        subprocess.check_call([sys.executable, os.path.join(HERE, 'scalarargs.py'), t3, outp])
    if merge:
        s = open(outp).read()
        for k in merge:
            s, n = re.subn(r'(?m)^__attribute__\(\(noinline\)\) FpExt (rv32im_v2_%s)\(' % k,
                           r'__attribute__((always_inline)) inline FpExt \1(', s)
            assert n == 1, k
        open(outp, 'w').write(s)
    body = open(outp).read()
    assert '_h_data' not in ''.join(re.findall(r'(?s)FpExt rv32im_v2_\d+\([^)]*\) \{.*?\n\}', body)), 'unhoist incomplete'

if __name__ == '__main__':
    main()
