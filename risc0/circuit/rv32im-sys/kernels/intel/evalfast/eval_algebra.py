#!/usr/bin/env python3
# Production driver (proposed): field-exact algebraic rewrites of the Intel eval_check
# amalgamation, to run BEFORE eval_nozero.py / eval_lazy_chain.py.
#   1. --zerolink : drop `B + Fp(0)*poly_mix[k]` links (181)
#   2. --absorb   : c*d*pm[k] with d a short zero-headed Fp chain -> links over SLM table pm[k]*pm[j] (24 clinks)
#   3. --chaindiff --cd-minsave=8 : derive mux-arm accumulation chains from an earlier chain plus the differing terms
#   4. --hoist    : row-invariant subexpressions computed once per work-group (lid 0) into SLM
# Each step is checked by evaluating poly_fp symbolically-numerically on random inputs (verify.py).
# usage: eval_algebra.py <amalg.cpp> <out.cpp>
import subprocess, sys, os
D=os.path.dirname(os.path.abspath(__file__))
src,dst=sys.argv[1],sys.argv[2]
steps=[['--zerolink'],['--absorb'],['--chaindiff','--cd-minsave=8'],['--hoist']]
cur=src
for i,s in enumerate(steps):
    nxt=dst+'.step%d.cpp'%i
    subprocess.check_call([sys.executable,os.path.join(D,'rewrite.py'),cur,nxt]+s)
    if cur!=src: os.remove(cur)
    cur=nxt
os.replace(cur,dst)
env=dict(os.environ,PYTHONHASHSEED='0')
out=subprocess.check_output([sys.executable,os.path.join(D,'verify.py'),src,dst,'2'],env=env).decode()
sys.stderr.write(out)
if 'ALL OK' not in out: sys.exit('eval_algebra: numeric verification FAILED')
