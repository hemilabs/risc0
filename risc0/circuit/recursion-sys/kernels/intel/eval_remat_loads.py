#!/usr/bin/env python3
# Sink buffer loads in poly_fp to their first use and re-load (volatile, so
# LLVM cannot CSE them back) for use groups separated by more than G statements.
# usage: remat_loads.py in.cpp out.cpp G [volatile=1]
import re,sys,collections
src=open(sys.argv[1]).read(); G=int(sys.argv[3]); VOL=int(sys.argv[4]) if len(sys.argv)>4 else 1
L=src.split('\n')
s=[i for i,l in enumerate(L) if 'FpExt poly_fp(size_t cycle' in l and '{' in l][0]
e=[i for i in range(s,len(L)) if L[i].startswith('}')][0]
body=L[s+1:e]
ld=re.compile(r'^(\s*)auto (x\d+) = (/\*\w+=\*/args\[\d\]\[[^;]*\]);\s*$')
loads={}
for i,l in enumerate(body):
    m=ld.match(l)
    if m: loads[m.group(2)]=(i,m.group(1),m.group(3))
uses=collections.defaultdict(list)
tok=re.compile(r'\bx\d+\b')
for i,l in enumerate(body):
    code=l.split('//')[0]
    if not code.strip(): continue
    rhs=code.split('=',1)[1] if re.match(r'^\s*(auto|FpExt|RkAcc|Fp|constexpr Fp) x\d+ =',code) else code
    for t in tok.findall(rhs):
        if t in loads: uses[t].append(i)
# statement index (non-comment) for gap measure
stmt=[0]*len(body); c=0
for i,l in enumerate(body):
    if l.strip() and not l.strip().startswith('//'): c+=1
    stmt[i]=c
insert=collections.defaultdict(list); rename={}  # (line, var)->newname
nre=0
for x,(i,ind,expr) in loads.items():
    body[i]=ind+'// sunk '+x
    us=sorted(set(uses[x]))
    if not us: continue
    groups=[[us[0]]]
    for u in us[1:]:
        if stmt[u]-stmt[groups[-1][-1]]>G: groups.append([u])
        else: groups[-1].append(u)
    for j,g in enumerate(groups):
        name=x if j==0 else f'{x}_r{j}'
        if j==0 or not VOL: insert[g[0]].append(f'{ind}auto {name} = {expr};')
        else: insert[g[0]].append(f'{ind}auto {name} = rk_vld(&{expr});'); nre+=1
        if j>0:
            for u in g: rename[(u,x)]=name
out=[]
for i,l in enumerate(body):
    out+=insert.get(i,[])
    for (u,x),nm in [(k,v) for k,v in rename.items() if k[0]==i]:
        pre,eq,rhs=l.partition('=') if re.match(r'^\s*(auto|FpExt|RkAcc|Fp) x\d+ =',l) else ('','',l)
        rhs=re.sub(r'\b%s\b'%x,nm,rhs)
        l=pre+eq+rhs
    out.append(l)
helper='''__attribute__((always_inline)) static inline Fp rk_vld(const Fp* p) { return Fp::fromRaw(*(const volatile uint32_t*)p); }
'''
res=L[:s]+[helper]+[L[s]]+out+L[e:]
open(sys.argv[2],'w').write('\n'.join(res))
print('loads',len(loads),'reloads',nre,file=sys.stderr)
