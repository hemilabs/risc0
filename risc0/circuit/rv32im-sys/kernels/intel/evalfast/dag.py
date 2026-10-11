# Per-function SSA DAG over the amalgamation (restricted grammar).
import re, collections
from parse import load
LOADRE=re.compile(r'auto (x\d+) = (\w+)\[(\d+) \* steps \+ \(\(cycle - kInvRate \* (\d+)\) & mask\)\];')
class Fn:
    pass
def build(lines,f):
    F=Fn(); F.name=f['name']; F.params=f['params']
    T={}; ptr={}
    for t,n in f['params']:
        if n in ('cycle','steps','poly_mix','args'): continue
        if t.endswith('*'): ptr[n]=t[:-1]
        else: T[n]=t
    nodes=[]  # dict(op,args,t,var,line,...)
    var={}    # name -> node id
    def mk(**kw):
        nodes.append(kw); return len(nodes)-1
    for n,t in T.items(): var[n]=mk(op='param',args=(),t='E' if t=='FpExt' else 'F',name=n)
    stmts=[]  # (lineidx, kind, payload)
    for i in range(f['start']+1,f['end']):
        raw=lines[i]; l=raw.strip()
        if not l or l.startswith('//'): continue
        if l.startswith('size_t mask'): continue
        m=re.match(r'constexpr Fp (x\d+)\((\d+)\);',l)
        if m: var[m.group(1)]=mk(op='const',args=(),t='F',c=int(m.group(2)),line=i); continue
        m=re.match(r'constexpr FpExt (x\d+)\((\d+),(\d+),(\d+),(\d+)\);',l)
        if m: var[m.group(1)]=mk(op='econst',args=(),t='E',ev=tuple(int(m.group(k)) for k in range(2,6)),line=i); continue
        m=re.match(r'(Fp|FpExt) (x\d+)\[(\d+)\];',l)
        if m: ptr[m.group(2)]=m.group(1); stmts.append((i,'decl',m.group(2))); continue
        m=LOADRE.match(l)
        if m: var[m.group(1)]=mk(op='load',args=(),t='F',src=m.group(2),col=int(m.group(3)),back=int(m.group(4)),line=i); continue
        m=re.match(r'auto (\w+) = /\*\w+=\*/args\[(\d)\]\[(\d+) \* steps \+ \(\(cycle - kInvRate \* (\d+)\) & mask\)\];',l)
        if m: var[m.group(1)]=mk(op='load',args=(),t='F',src='args[%s]'%m.group(2),col=int(m.group(3)),back=int(m.group(4)),line=i); continue
        m=re.match(r'auto (x\d+) = (_h\w+);',l)
        if m: var[m.group(1)]=var[m.group(2)] if m.group(2) in var else mk(op='param',args=(),t='F',name=m.group(2)); continue
        m=re.match(r'auto (x\d+) = (\w+)\[(\d+)\];',l)
        if m:
            b=m.group(2); k=int(m.group(3))
            var[m.group(1)]=mk(op='aload',args=(),t='E' if ptr[b]=='FpExt' else 'F',src=b,idx=k,line=i); continue
        m=re.match(r'(\w+)\[(\d+)\] = (x\d+);',l)
        if m: stmts.append((i,'store',(m.group(1),int(m.group(2)),var[m.group(3)]))); continue
        m=re.match(r'auto (x\d+) = (\w+) ([-+*]) (\w+);',l)
        if m:
            a,b=var[m.group(2)],var[m.group(4)]
            t='E' if 'E' in (nodes[a]['t'],nodes[b]['t']) else 'F'
            var[m.group(1)]=mk(op=m.group(3),args=(a,b),t=t,line=i); continue
        m=re.match(r'FpExt (x\d+) = (\w+) \+ (\w+) \* poly_mix\[(\d+)\];',l)
        if m:
            var[m.group(1)]=mk(op='link',args=(var[m.group(2)],var[m.group(3)]),t='E',k=int(m.group(4)),line=i); continue
        m=re.match(r'FpExt (x\d+) = (\w+) \+ (\w+) \* (\w+) \* poly_mix\[(\d+)\];',l)
        if m:
            var[m.group(1)]=mk(op='clink',args=(var[m.group(2)],var[m.group(3)],var[m.group(4)]),t='E',k=int(m.group(5)),line=i); continue
        m=re.match(r'FpExt (x\d+) = poly_mix\[(\d+)\];',l)
        if m: var[m.group(1)]=mk(op='pmread',args=(),t='E',k=int(m.group(2)),line=i); continue
        m=re.match(r'Fp (x\d+) = poly_mix\[(\d+)\]\.elems\[0\];',l)
        if m: var[m.group(1)]=mk(op='pmread',args=(),t='F',k=int(m.group(2)),line=i); continue
        m=re.match(r'auto (x\d+) = -(\w+);',l)
        if m:
            a=var[m.group(2)]; var[m.group(1)]=mk(op='neg',args=(a,),t=nodes[a]['t'],line=i); continue
        m=re.match(r'auto (x\d+) = rk_xmul\((x\d+)\);',l)
        if m: var[m.group(1)]=mk(op='xmul',args=(var[m.group(2)],),t='E',line=i); continue
        m=re.match(r'auto (x\d+) = (x\d+) \* Fp\((\d+)u\);',l)
        if m: var[m.group(1)]=mk(op='scal',args=(var[m.group(2)],),t='E',c=int(m.group(3)),line=i); continue
        m=re.match(r'(?:auto|FpExt) (x\d+) = (\w+);',l)
        if m: var[m.group(1)]=var[m.group(2)]; continue
        m=re.match(r'FpExt (x\d+) = FpExt\(0\);',l)
        if m: var[m.group(1)]=mk(op='zeroE',args=(),t='E',line=i); continue
        m=re.match(r'auto (x\d+) = (rv32im_v2_\d+)\((.*)\);',l)
        if m:
            av=[a.strip() for a in m.group(3).split(',')]
            vals=[var[a] if a in var else None for a in av]
            id_=mk(op='call',args=tuple(v for v in vals if v is not None),t='E',fn=m.group(2),argtext=av,line=i)
            var[m.group(1)]=id_; stmts.append((i,'call',id_)); continue
        m=re.match(r'auto (_h\w+) = /\*\w+=\*/args\[(\d)\]\[(\d+) \* steps \+ \(\(cycle - kInvRate \* (\d+)\) & mask\)\];',l)
        if m: var[m.group(1)]=mk(op='load',args=(),t='F',src='args[%s]'%m.group(2),col=int(m.group(3)),back=int(m.group(4)),line=i); continue
        m=re.match(r'return (x\d+);',l)
        if m: F.ret=var[m.group(1)]; stmts.append((i,'ret',var[m.group(1)])); continue
        raise Exception('unparsed %s %d: %s'%(f['name'],i,l[:200]))
    F.nodes=nodes; F.var=var; F.stmts=stmts; F.ptr=ptr
    users=collections.defaultdict(list)
    for j,n in enumerate(nodes):
        for a in n['args']: users[a].append(j)
    for (i,k,p) in stmts:
        if k=='store': users[p[2]].append(('store',i))
        if k=='ret': users[p].append(('ret',i))
    F.users=users
    return F
def build_all(path):
    src,lines,funcs,wrapper=load(path)
    return src,lines,funcs,wrapper,[build(lines,f) for f in funcs]
