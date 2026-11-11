# Interpreter over the per-function DAGs (numeric field eval, or uniformity).
import random, collections
P=0x78000001; NB=P-11
def emul(a,b):
    return ((a[0]*b[0]+NB*(a[1]*b[3]+a[2]*b[2]+a[3]*b[1]))%P,(a[0]*b[1]+a[1]*b[0]+NB*(a[2]*b[3]+a[3]*b[2]))%P,
            (a[0]*b[2]+a[1]*b[1]+a[2]*b[0]+NB*(a[3]*b[3]))%P,(a[0]*b[3]+a[1]*b[2]+a[2]*b[1]+a[3]*b[0])%P)
def lift(x): return x if isinstance(x,tuple) else (x,0,0,0)
def mul(a,b):
    if isinstance(a,int) and isinstance(b,int): return a*b%P
    if isinstance(a,int): return tuple(a*y%P for y in b)
    if isinstance(b,int): return tuple(b*y%P for y in a)
    return emul(a,b)
def add(a,b):
    if isinstance(a,int) and isinstance(b,int): return (a+b)%P
    a,b=lift(a),lift(b); return tuple((x+y)%P for x,y in zip(a,b))
def sub(a,b):
    if isinstance(a,int) and isinstance(b,int): return (a-b)%P
    a,b=lift(a),lift(b); return tuple((x-y)%P for x,y in zip(a,b))
BUFU={'args[0]':False,'args[1]':False,'args[2]':True,'args[3]':True}
NAMEMAP={'accum':'args[0]','data':'args[1]','global':'args[2]','mix':'args[3]'}
class Interp:
    def __init__(s,Fs,seed=1,mode='num',qt=()):
        s.F={F.name:F for F in Fs}; s.mode=mode; s.seed=seed
        rnd=random.Random(seed)
        s.pm=[tuple(rnd.randrange(P) for _ in range(4)) for _ in range(458)]
        for a,b in qt: s.pm.append(emul(s.pm[a],s.pm[b]))
        s.cache={}; s.vals={}   # vals[(fn,node)] -> value
    def loadv(s,buf,col,back):
        if s.mode=='u': return BUFU[buf]
        k=(buf,col,back)
        if k not in s.cache: s.cache[k]=random.Random(hash((s.seed,)+k)).randrange(P)
        return s.cache[k]
    def run(s,name,argmap):
        F=s.F[name]; N=F.nodes; V={}
        ptr={}  # name -> ('buf',bufname) or array dict
        for (t,pn) in F.params:
            if pn in argmap: 
                if t.endswith('*'): ptr[pn]=argmap[pn]
        events=[]
        for i,n in enumerate(N):
            events.append((n.get('line',-1),0,i))
        for (ln,k,p) in F.stmts: events.append((ln,1,(k,p)))
        events.sort(key=lambda e:(e[0],e[1]))
        u=(s.mode=='u')
        for ln,kind,p in events:
            if kind==0:
                i=p; n=N[i]; op=n['op']
                if op=='param': v=argmap[n['name']]
                elif op=='const': v=True if u else n['c']%P
                elif op=='econst': v=True if u else tuple(x%P for x in n['ev'])
                elif op=='zeroE': v=True if u else (0,0,0,0)
                elif op=='load':
                    src=n['src']
                    b=src if src.startswith('args[') else ptr[src][1]
                    v=s.loadv(b,n['col'],n['back'])
                elif op=='aload':
                    pp=ptr[n['src']]
                    if isinstance(pp,tuple):
                        assert pp[1] in ('args[2]','args[3]'), pp
                        v=s.loadv(pp[1],n['idx'],'u')
                    else: v=pp[n['idx']]
                elif op in '+-*':
                    a,b=V[n['args'][0]],V[n['args'][1]]
                    v=(a and b) if u else (add(a,b) if op=='+' else sub(a,b) if op=='-' else mul(a,b))
                elif op=='link':
                    a,c=V[n['args'][0]],V[n['args'][1]]
                    v=(a and c) if u else add(a,mul(c,s.pm[n['k']]))
                elif op=='clink':
                    a,c,d=(V[x] for x in n['args'])
                    v=(a and c and d) if u else add(a,mul(mul(c,d),s.pm[n['k']]))
                elif op=='call':
                    callee=s.F[n['fn']]; am={}
                    for (t,pn),txt in zip(callee.params[3:],n['argtext'][3:]):
                        if txt.startswith('/*'):
                            am[pn]=('buf',txt.split('*/')[1])
                        elif t.endswith('*'): am[pn]=ptr[txt]
                        else: am[pn]=V[F.var[txt]] if txt in F.var else None
                    v=s.run(n['fn'],am)
                elif op=='pmread':
                    v=True if u else (s.pm[n['k']] if n['t']=='E' else s.pm[n['k']][0])
                elif op=='neg':
                    a=V[n['args'][0]]; v=a if u else sub(0 if isinstance(a,int) else (0,0,0,0),a)
                elif op=='xmul':
                    a=V[n['args'][0]]; v=a if u else ((a[3]*NB)%P,a[0],a[1],a[2])
                elif op=='scal':
                    a=V[n['args'][0]]; v=a if u else mul(n['c'],a)
                elif op=='hoist':
                    v=s.hoist[n['slot']] if not u else True
                else: raise Exception(op)
                V[i]=v; s.vals[(name,i)]=v
            else:
                k,pp=p
                if k=='decl': ptr[pp]={}
                elif k=='store':
                    arr,idx,val=pp; ptr[arr][idx]=V[val]
                elif k=='ret': return V[pp]
        raise Exception('no return')
    def run_hoist(s,src):
        import re
        m=re.search(r'void rk_hoist\([^)]*\) \{\n(.*?)\n\}\n',src,re.S)
        if not m: return
        env={}
        def val(t):
            t=t.strip()
            mm=re.fullmatch(r'Fp\((\d+)u\)',t)
            if mm: return int(mm.group(1))%P
            if t=='FpExt(0)': return (0,0,0,0)
            mm=re.fullmatch(r'FpExt\(Fp\((\d+)u\),Fp\((\d+)u\),Fp\((\d+)u\),Fp\((\d+)u\)\)',t)
            if mm: return tuple(int(x)%P for x in mm.groups())
            mm=re.fullmatch(r'FpExt\((h\d+)\)',t)
            if mm: return lift(env[mm.group(1)])
            return env[t]
        TOK=r'(FpExt\(Fp\(\d+u\),Fp\(\d+u\),Fp\(\d+u\),Fp\(\d+u\)\)|FpExt\(0\)|Fp\(\d+u\)|h\d+)'
        tab={}
        for l in m.group(1).split('\n'):
            l=l.strip()
            if not l: continue
            mm=re.fullmatch(r'Fp (h\d+) = (out|mix)\[(\d+)\];',l)
            if mm: env[mm.group(1)]=s.loadv({'out':'args[2]','mix':'args[3]'}[mm.group(2)],int(mm.group(3)),'u'); continue
            mm=re.fullmatch(r'(?:Fp|FpExt) (h\d+) = '+TOK+r' \+ '+TOK+r' \* poly_mix\[(\d+)\];',l)
            if mm: env[mm.group(1)]=add(val(mm.group(2)),mul(val(mm.group(3)),s.pm[int(mm.group(4))])); continue
            mm=re.fullmatch(r'(?:Fp|FpExt) (h\d+) = '+TOK+r' \+ '+TOK+r' \* '+TOK+r' \* poly_mix\[(\d+)\];',l)
            if mm: env[mm.group(1)]=add(val(mm.group(2)),mul(mul(val(mm.group(3)),val(mm.group(4))),s.pm[int(mm.group(5))])); continue
            mm=re.fullmatch(r'(?:Fp|FpExt) (h\d+) = '+TOK+r' ([-+*]) '+TOK+r';',l)
            if mm:
                a,b=val(mm.group(2)),val(mm.group(4)); o=mm.group(3)
                env[mm.group(1)]=add(a,b) if o=='+' else sub(a,b) if o=='-' else mul(a,b); continue
            mm=re.fullmatch(r'(?:Fp|FpExt) (h\d+) = -'+TOK+r';',l)
            if mm: a=val(mm.group(2)); env[mm.group(1)]=sub(0 if isinstance(a,int) else (0,0,0,0),a); continue
            mm=re.fullmatch(r'tab\[(\d+)\] = (.*);',l)
            if mm: tab[int(mm.group(1))]=lift(val(mm.group(2))); continue
            raise Exception('hoist unparsed '+l)
        mq=re.search(r'constexpr uint32_t RK_QN = (\d+);',src)
        base=458+(int(mq.group(1)) if mq else 0)
        assert len(s.pm)==base
        for k in range(len(tab)): s.pm.append(tab[k])
    def go(s):
        return s.run('poly_fp',{})
