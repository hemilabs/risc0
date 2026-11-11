# Symbolic global trace of uniform (row-invariant) values -> straight-line hoist code.
import collections
class Sym:
    """value = (uniform:boolean, expr-name or None, type 'F'/'E')"""
def gen(Fs):
    Fm={F.name:F for F in Fs}
    code=[]; ctr=[0]; origin={}   # origin[(fn,node)] = hoisted name (for uniform nodes)
    loadbufs=collections.Counter()
    def fresh(): ctr[0]+=1; return 'h%d'%ctr[0]
    def run(name,argmap):
        F=Fm[name]; N=F.nodes; V={}; ptr={}
        for (t,pn) in F.params:
            if pn in argmap and t.endswith('*'): ptr[pn]=argmap[pn]
        ev=[(n.get('line',-1),0,i) for i,n in enumerate(N)]+[(ln,1,(k,p)) for (ln,k,p) in F.stmts]
        ev.sort(key=lambda e:(e[0],e[1]))
        for ln,kind,p in ev:
            if kind==0:
                i=p; n=N[i]; op=n['op']; v=None
                if op=='param': v=argmap.get(n['name'])
                elif op=='const': v=('Fp(%du)'%(n['c']%0x78000001),'F')
                elif op=='econst': v=('FpExt(Fp(%du),Fp(%du),Fp(%du),Fp(%du))'%tuple(x%0x78000001 for x in n['ev']),'E')
                elif op=='zeroE': v=('FpExt(0)','E')
                elif op=='load':
                    src=n['src']; b=src if src.startswith('args[') else ptr[src][1]
                    loadbufs[b]+=1
                    assert b in ('args[0]','args[1]'), ('cycle-indexed load from uniform buffer', b)
                    v=None
                elif op=='aload':
                    pp=ptr[n['src']]
                    if isinstance(pp,tuple):
                        assert pp[1] in ('args[2]','args[3]')
                        nmx=fresh(); code.append('  Fp %s = %s[%d];'%(nmx,{'args[2]':'out','args[3]':'mix'}[pp[1]],n['idx'])); v=(nmx,'F')
                    else: v=pp.get(n['idx'])
                elif op in ('+','-','*'):
                    a,b=V[n['args'][0]],V[n['args'][1]]
                    if a is not None and b is not None:
                        nmx=fresh(); t='E' if 'E' in (a[1],b[1]) else 'F'
                        code.append('  %s %s = %s %s %s;'%('FpExt' if t=='E' else 'Fp',nmx,a[0],op,b[0])); v=(nmx,t)
                elif op=='link':
                    a,c=V[n['args'][0]],V[n['args'][1]]
                    if a is not None and c is not None:
                        nmx=fresh(); code.append('  FpExt %s = %s + %s * poly_mix[%d];'%(nmx,a[0],c[0],n['k'])); v=(nmx,'E')
                elif op=='clink':
                    a,c,d=(V[x] for x in n['args'])
                    if a is not None and c is not None and d is not None:
                        nmx=fresh(); code.append('  FpExt %s = %s + %s * %s * poly_mix[%d];'%(nmx,a[0],c[0],d[0],n['k'])); v=(nmx,'E')
                elif op=='neg':
                    a=V[n['args'][0]]
                    if a is not None:
                        nmx=fresh(); code.append('  %s %s = -%s;'%('FpExt' if a[1]=='E' else 'Fp',nmx,a[0])); v=(nmx,a[1])
                elif op=='pmread': v=None
                elif op=='call':
                    callee=Fm[n['fn']]; am={}
                    for (t,pn),txt in zip(callee.params[3:],n['argtext'][3:]):
                        if txt.startswith('/*'): am[pn]=('buf',txt.split('*/')[1])
                        elif t.endswith('*'): am[pn]=ptr[txt]
                        else: am[pn]=V[F.var[txt]] if txt in F.var else None
                    run(n['fn'],am); v=None
                else: raise Exception(op)
                V[i]=v
                if v is not None: origin[(name,i)]=v
            else:
                k,pp=p
                if k=='decl': ptr[pp]={}
                elif k=='store': ptr[pp[0]][pp[1]]=V[pp[2]]
                elif k=='ret': return
    run('poly_fp',{})
    return code,origin,loadbufs
