#!/usr/bin/env python3
# N1: count the poly_mix rk_link chains of the generated Metal evalfast TU (eval_check_ef.metal), per
# function scope: Fp-term vs FpExt-term links, chain lengths, links in chains above a size.
#   count_chains.py <eval_check_ef.metal>
import re,sys,collections
src=open(sys.argv[1]).read()
funcs=re.split(r'\n(?=\S[^\n]*\)\s*\{\s*\n)',src)
cnt=collections.Counter(); lens=[]; cl=0
link=re.compile(r'(rk_link(?:_nf)?_k)\(ctx, (\w+), (\w+), (\d+)u\)')
for f in funcs:
    types={}
    for m in re.finditer(r'\b(Fp|FpExt|RkAcc|RkW|RkW1)\s+(\w+)\s*[=(]',f): types[m.group(2)]=m.group(1)
    for m in re.finditer(r'\b(Fp|FpExt)\s+(arg\w+)\s*[,)]',f): types.setdefault(m.group(2),m.group(1))
    for m in re.finditer(r'auto (x\d+) = (.*);',f):
        e=m.group(2)
        ext = bool(re.search(r'FpExt\(|rkw_fin\(|rk_xmul\(|EF_PM\(|rk_fin\(|RK_LD4\(|rv32im_v2_\d+\(',e)) or any(types.get(v)=='FpExt' for v in re.findall(r'\b(?:x|arg)\w+',e))
        types[m.group(1)]='FpExt' if ext else 'Fp'
    acc={}
    for L in f.split('\n'):
        m=re.search(r'RkAcc (x\d+) = (rk_link(?:_nf)?_k)\(ctx, (\w+), (\w+), (\d+)u\)',L)
        g=re.search(r'rk_fin\((rk_link(?:_nf)?_k)\(ctx, (\w+), (\w+), (\d+)u\)\)',L)
        for mm in link.finditer(L): cnt[types.get(mm.group(3),'?')]+=1
        if m:
            fp=types.get(m.group(4))=='Fp'; prev=acc.pop(m.group(3),(0,0)); acc[m.group(1)]=(prev[0]+1,prev[1]+fp)
        elif g:
            fp=types.get(g.group(3))=='Fp'; prev=acc.pop(g.group(2),(0,0)); lens.append((prev[0]+1,prev[1]+fp))
    if acc: print('open chains',len(acc))
print('link c types',dict(cnt))
print('chains',len(lens),'links',sum(l for l,_ in lens),'Fp-term links',sum(p for _,p in lens))
for w in (1,4,8,16):
    print(f'chains with >= {w} Fp links: {sum(1 for l,p in lens if p>=w)}, Fp links in them {sum(p for l,p in lens if p>=w)}')
print('rk_cl_acc',len(re.findall(r'rk_cl_acc\(ctx',src)),'rk_cl0',len(re.findall(r'rk_cl0\(ctx',src)))
print('len hist (links, bucket of 8):',sorted(collections.Counter(l//8*8 for l,_ in lens).items()))
