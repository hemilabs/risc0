# Parse amalg_orig.cpp into per-function typed statement lists.
import re, sys, collections
def load(path):
    src=open(path).read()
    cut=src.find('// Intel SYCL eval_check kernel')
    body,wrapper=src[:cut],src[cut:]
    lines=body.split('\n')
    fre=re.compile(r'^(__attribute__\(\([a-z_]+\)\) )?FpExt (rv32im_v2_\d+|poly_fp)\((.*)\) \{$')
    starts=[i for i,l in enumerate(lines) if fre.match(l)]
    funcs=[]
    for s in starts:
        e=s+1
        while lines[e]!='}': e+=1
        m=fre.match(lines[s])
        params=[]
        for p in m.group(3).split(','):
            p=p.strip(); mm=re.match(r'(.+?)\s*(\*?)\s*(\w+)$',p)
            params.append((mm.group(1).strip()+mm.group(2), mm.group(3)))
        funcs.append(dict(name=m.group(2),start=s,end=e,params=params))
    return src,lines,funcs,wrapper
