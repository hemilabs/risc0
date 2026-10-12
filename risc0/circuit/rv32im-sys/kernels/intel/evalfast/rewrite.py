#!/usr/bin/env python3
# Field-exact algebraic rewrites on the eval_check amalgamation (pre eval_nozero/eval_lazy_chain).
# usage: rewrite.py in.cpp out.cpp [--xshift] [--chaindiff] [--logup] ...
import sys, re, collections
from dag import build_all
args=sys.argv[3:]
src,lines,funcs,wrapper,Fs=build_all(sys.argv[1])
P=0x78000001
repl={}      # line idx -> list of replacement lines
stats=collections.Counter()
_ctr=[max([2000000]+[int(x) for x in re.findall(r'\bx(\d+)\b',src)])+1]
# existing Q table (from a previous absorb run)
QT=[]
mq=re.search(r'constexpr uint16_t rk_qa\[\] = \{([^}]*)\};\s*constexpr uint16_t rk_qb\[\] = \{([^}]*)\};',src)
if mq:
    QT=list(zip([int(v) for v in mq.group(1).split(',') if v.strip()],[int(v) for v in mq.group(2).split(',') if v.strip()]))
PMBASE=458
def fresh():
    _ctr[0]+=1; return 'x%d'%_ctr[0]
def nm(F,i):
    """source name of node i"""
    n=F.nodes[i]
    if n['op']=='param': return n['name']
    return F.names[i]
def setnames(F):
    F.names={}
    for v,i in F.var.items():
        n=F.nodes[i]
        if n['op']=='param': continue
        # prefer the defining variable (the one on the node's line)
        if i not in F.names: F.names[i]=v
        else:
            l=lines[n['line']]
            if re.search(r'\b(auto|FpExt|Fp) %s\b'%v,l): F.names[i]=v
for F in Fs: setnames(F)

# ---------------- pass: X-shift / base-constant FpExt multiplies -----------------
def p_xshift(F):
    N=F.nodes
    for i,n in enumerate(N):
        if n['op']!='*' or n['t']!='E': continue
        a,b=n['args']
        for x,y in ((a,b),(b,a)):
            if N[x]['op']=='econst' and N[y]['t']=='E':
                ev=tuple(v%P for v in N[x]['ev']); out=nm(F,i); o=nm(F,y)
                if ev==(0,1,0,0): repl[n['line']]=['  auto %s = rk_xmul(%s);'%(out,o)]; stats['xshift']+=1
                elif ev[1:]==(0,0,0) and ev[0]==1: repl[n['line']]=['  auto %s = %s;'%(out,o)]; stats['xone']+=1
                elif ev[1:]==(0,0,0): repl[n['line']]=['  auto %s = %s * Fp(%du);'%(out,o,ev[0])]; stats['xbase']+=1
                else: stats['xother']+=1
                break

# ---------------- pass: chain differencing (shared poly_mix terms) -----------------
def p_chaindiff(F, minsave=int(dict(a.split('=') for a in args if a.startswith('--cd-minsave=')).get('--cd-minsave','2'))):
    N=F.nodes
    def is_link(i): return N[i]['op'] in ('link','clink')
    def linkusers(i): return [u for u in F.users[i] if isinstance(u,int) and is_link(u) and N[u]['args'][0]==i]
    ends=[i for i,n in enumerate(N) if is_link(i) and not linkusers(i)]
    ch={}
    for e in ends:
        terms={}; x=e
        while is_link(x):
            n=N[x]; k=n['k']
            assert k not in terms
            terms[k]=('l',n['args'][1]) if n['op']=='link' else ('c',n['args'][1],n['args'][2])
            x=n['args'][0]
        ch[e]=(x,terms)
    ends.sort(key=lambda e:N[e]['line'])
    def diff(ha,ta,hb,tb):
        ops=[]; cost=0
        for k in sorted(set(ta)|set(tb)):
            A,B=ta.get(k),tb.get(k)
            if A==B: continue
            if A is None: ops.append(('add',k,B)); cost+=1
            elif B is None: ops.append(('neg',k,A)); cost+=1
            elif A[0]=='l' and B[0]=='l': ops.append(('subl',k,B[1],A[1])); cost+=1
            elif A[0]=='c' and B[0]=='c' and A[2]==B[2]: ops.append(('subc',k,B[1],A[1],B[2])); cost+=1
            else: ops.append(('add',k,B)); ops.append(('neg',k,A)); cost+=2
        if ha!=hb: cost+=0.5
        return cost,ops
    prefix='--cdprefix' in args
    walk={}; tdict={}; hd={}
    if prefix:
        order=sorted([i for i in range(len(N)) if is_link(i)], key=lambda i:N[i]['line'])
        for x in order:
            n=N[x]; bse=n['args'][0]
            t=dict(tdict[bse]) if bse in tdict else {}
            t[n['k']]=('l',n['args'][1]) if n['op']=='link' else ('c',n['args'][1],n['args'][2])
            tdict[x]=t; hd[x]=hd[bse] if bse in hd else bse
        for e in ends:
            w=[]; x=e
            while is_link(x): w.append(x); x=N[x]['args'][0]
            walk[e]=w
    rewritten=set(); pinned=set()
    for j,b in enumerate(ends):
        hb,tb=ch[b]; best=None
        cands=[(a,ch[a][0],ch[a][1],0) for a in ends[:j]]
        if prefix:
            seen=set(a for a,_,_,_ in cands)
            for e2 in ends:
                if e2 in rewritten or e2==b: continue
                for x in walk[e2]:
                    if x in seen or N[x]['line']>=N[b]['line'] or x in walk[b]: continue
                    seen.add(x)
                    extra=0 if len(F.users[x])>1 else 1
                    cands.append((x,hd[x],tdict[x],extra))
        for a,ha,ta,extra in cands:
            c,ops=diff(ha,ta,hb,tb); c+=extra
            if best is None or c<best[0]: best=(c,a,ops,ha)
        if best is None or best[0] > len(tb)-minsave or b in pinned: continue
        if prefix:
            a=best[1]
            if a not in ends:
                for e2 in ends:
                    if a in walk[e2]: pinned.add(e2)
            rewritten.add(b)
        c,a,ops,ha=best
        out=[]; cur=nm(F,a)
        if ha!=hb:
            t=fresh(); out.append('  auto %s = %s - %s;'%(t,cur,nm(F,ha))); cur=t
            t=fresh(); out.append('  auto %s = %s + %s;'%(t,cur,nm(F,hb))); cur=t
        for idx,op in enumerate(ops):
            last=(idx==len(ops)-1)
            res=nm(F,b) if last else fresh()
            kind=op[0]; k=op[1]
            if kind=='add':
                T=op[2]
                if T[0]=='l': out.append('  FpExt %s = %s + %s * poly_mix[%d];'%(res,cur,nm(F,T[1]),k))
                else: out.append('  FpExt %s = %s + %s * %s * poly_mix[%d];'%(res,cur,nm(F,T[1]),nm(F,T[2]),k))
            elif kind=='neg':
                T=op[2]; t=fresh(); out.append('  auto %s = -%s;'%(t,nm(F,T[1])))
                if T[0]=='l': out.append('  FpExt %s = %s + %s * poly_mix[%d];'%(res,cur,t,k))
                else: out.append('  FpExt %s = %s + %s * %s * poly_mix[%d];'%(res,cur,t,nm(F,T[2]),k))
            elif kind=='subl':
                t=fresh(); out.append('  auto %s = %s - %s;'%(t,nm(F,op[2]),nm(F,op[3])))
                out.append('  FpExt %s = %s + %s * poly_mix[%d];'%(res,cur,t,k))
            elif kind=='subc':
                t=fresh(); out.append('  auto %s = %s - %s;'%(t,nm(F,op[2]),nm(F,op[3])))
                out.append('  FpExt %s = %s + %s * %s * poly_mix[%d];'%(res,cur,t,nm(F,op[4]),k))
            cur=res
        if not ops:
            out.append('  FpExt %s = %s;'%(nm(F,b),cur))
        repl[N[b]['line']]=out
        stats['chaindiff_chains']+=1; stats['chaindiff_links_before']+=len(tb); stats['chaindiff_cost_after']+=c

# ---------------- pass: absorb c*d*pm[k] with d a short zero-headed Fp chain -----------------
ZV=None
def p_absorb(F, maxlen=int(dict(a.split('=') for a in args if a.startswith('--absorb-max=')).get('--absorb-max','24'))):
    global ZV
    if ZV is None:
        from interp import Interp
        ZV=[]
        for sd in (11,12):
            I=Interp(Fs,seed=sd); I.go(); ZV.append(I.vals)
    N=F.nodes
    qidx={q:i for i,q in enumerate(QT)}
    for i,n in enumerate(N):
        if n['op']!='clink': continue
        b,c,d=n['args']
        if N[c]['t']=='E': c,d=d,c
        if N[c]['t']!='F' or len(F.users[d])!=1: continue
        x=d; terms=[]; ok=True
        while N[x]['op'] in ('link','clink'):
            if N[x]['op']!='link' or N[N[x]['args'][1]]['t']!='F' or (x!=d and len(F.users[x])!=1) or N[x]['k']>=PMBASE: ok=False; break
            terms.append((N[x]['args'][1],N[x]['k'])); x=N[x]['args'][0]
        if not ok or not terms or len(terms)>maxlen: continue
        if not all(V.get((F.name,x))==(0,0,0,0) for V in ZV): continue
        k=n['k']; out=[]; cur=nm(F,b)
        terms.reverse()
        for j,(e,kj) in enumerate(terms):
            key=(min(k,kj),max(k,kj))
            if key not in qidx: qidx[key]=len(QT); QT.append(key)
            t=fresh(); out.append('  auto %s = %s * %s;'%(t,nm(F,c),nm(F,e)))
            res=nm(F,i) if j==len(terms)-1 else fresh()
            out.append('  FpExt %s = %s + %s * poly_mix[%d];'%(res,cur,t,PMBASE+qidx[key])); cur=res
        repl[n['line']]=out
        stats['absorb_clinks']+=1; stats['absorb_terms']+=len(terms)

from factor_pass import make_pass
p_factor=make_pass(nm,fresh,repl,stats)
from factorF_pass import make_pass_F
p_factorF=make_pass_F(nm,fresh,repl,stats)
HOIST=None
def p_hoist_global():
    global HOIST
    from hoist_gen import gen
    code,origin,_=gen(Fs)
    ARITH=('+','-','*','link','clink','neg')
    slots=[]; 
    hb=PMBASE+len(QT)   # absorb table must be final before hoist (run absorb in an earlier invocation)
    for F in Fs:
        N=F.nodes
        U=lambda i:(F.name,i) in origin
        for i,n in enumerate(N):
            if n['op'] not in ARITH or not U(i): continue
            root=False
            for u in F.users[i]:
                if not isinstance(u,int): root=True; break
                if N[u]['op']=='call' or not U(u) or N[u]['op'] not in ARITH: root=True; break
            if not root: continue
            h,t=origin[(F.name,i)]
            slot=len(slots); slots.append((h,t))
            if t=='E': repl[n['line']]=['  FpExt %s = poly_mix[%d];'%(nm(F,i),hb+slot)]
            else: repl[n['line']]=['  Fp %s = poly_mix[%d].elems[0];'%(nm(F,i),hb+slot)]
            stats['hoist_roots_'+t]+=1
    body='\n'.join(code)+'\n'+'\n'.join('  tab[%d] = %s;'%(k,h if t=='E' else 'FpExt(%s)'%h) for k,(h,t) in enumerate(slots))
    # rk_hoist runs on the GLOBAL poly_mix array, which holds only the first
    # PMBASE entries; the Q-table products and hoist slots (>= PMBASE) exist
    # only in SLM. verify.py's interpreter has them all, so a hoisted read of
    # one would pass verification and read garbage on the device.
    bad=sorted({int(k) for k in re.findall(r'poly_mix\[(\d+)\]',body) if int(k)>=PMBASE})
    assert not bad, 'rk_hoist reads poly_mix entries that are not in the global array: %s'%bad[:8]
    HOIST=(len(slots),body)
def p_zerolink(F):
    N=F.nodes
    for i,n in enumerate(N):
        if n['op'] in ('link','clink') and any(N[a]['op']=='const' and N[a]['c']%P==0 for a in n['args'][1:]):
            repl[n['line']]=['  FpExt %s = %s;'%(nm(F,i),nm(F,n['args'][0]))]; stats['zerolink']+=1
PASSES=[]
if '--zerolink' in args: PASSES.append(p_zerolink)
if '--absorb' in args: PASSES.append(p_absorb)
if '--factor' in args: PASSES.append(p_factor)
if '--factorF' in args: PASSES.append(p_factorF)
if '--xshift' in args: PASSES.append(p_xshift)
if '--chaindiff' in args: PASSES.append(p_chaindiff)
for F in Fs:
    for p in PASSES: p(F)
if '--hoist' in args: p_hoist_global()

# ---------------- emit + DCE -----------------
out=list(lines)
for ln,new in repl.items(): out[ln]='\n'.join(new)
# DCE per function
defre=re.compile(r'^\s*(?:auto|FpExt|Fp|constexpr Fp|constexpr FpExt) (x\d+)\b')
removed=0
for f in funcs:
    s,e=f['start'],f['end']
    stl=[]  # (outidx, subline idx) flattened statements
    for li in range(s+1,e):
        if out[li] is None: continue
        for sj,t in enumerate(out[li].split('\n')): stl.append([li,sj,t])
    # alias propagation: `auto|FpExt X = Y;` (Y an x-name or param) -> substitute X by Y downstream
    alias={}
    arex=re.compile(r'^\s*(?:auto|FpExt|Fp) (x\d+) = (\w+);\s*$')
    for st in stl:
        t=st[2]
        if alias:
            code,sep,com=t.partition('//')
            code=re.sub(r'\b(x\d+)\b',lambda m:alias.get(m.group(1),m.group(1)),code)
            t=code+sep+com; st[2]=t
        m=arex.match(t)
        if m and not m.group(2).startswith('_h') and m.group(2)!='steps':
            alias[m.group(1)]=m.group(2); st[2]=None; stats['alias_removed']+=1
    while True:
        uses=collections.Counter()
        for li,sj,t in stl:
            if t is None: continue
            code=t.split('//')[0]
            m=defre.match(code)
            rhs=code.split('=',1)[1] if (m and '=' in code) else code
            if m and '(' in code and '=' not in code: rhs=''  # constexpr Fp x(5);
            for tok in re.findall(r'\bx\d+\b',rhs): uses[tok]+=1
        ch=0
        for st in stl:
            t=st[2]
            if t is None: continue
            code=t.split('//')[0]
            m=defre.match(code)
            if m and 'rv32im_v2_' not in code and uses[m.group(1)]==0 and not re.match(r'^\s*(Fp|FpExt) x\d+\[',code):
                st[2]=None; ch+=1
        removed+=ch
        if not ch: break
    byline=collections.defaultdict(list)
    for li,sj,t in stl:
        if t is not None: byline[li].append(t)
    for li in range(s+1,e): out[li]='\n'.join(byline[li]) if byline[li] else None
# drop comment-only lines preceding removed statements is cosmetic; keep simple
txt='\n'.join(l for l in out if l is not None)
helper='''
// ---- algebraic rewrite helpers (ec-algebra) ----
__attribute__((always_inline)) static inline FpExt rk_xmul(const FpExt& a) {
  return FpExt(a.elems[3] * Fp(0x78000001u - 11u), a.elems[0], a.elems[1], a.elems[2]);
}
// ---- end helpers ----
'''
anchor='namespace risc0::circuit::rv32im_v2 {\nconstexpr size_t kInvRate = 4;\n'
assert anchor in txt
if 'static inline FpExt rk_xmul' not in txt: txt=txt.replace(anchor,anchor+helper,1)
txt=re.sub(r'constexpr uint32_t RK_QN = \d+;\n.*?\n.*?\n','',txt)
if QT:
    tab='constexpr uint32_t RK_QN = %d;\nconstexpr uint16_t rk_qa[] = {%s};\nconstexpr uint16_t rk_qb[] = {%s};\n'%(len(QT),','.join(str(a) for a,b in QT),','.join(str(b) for a,b in QT))
    txt=txt.replace('// ---- end helpers ----\n','// ---- end helpers ----\n'+tab,1)
if QT and 'RK_QN' not in wrapper:
    w=wrapper
    a1='sycl::local_accessor<FpExt, 1> slm(sycl::range<1>(POLY_MIX_COUNT), h);'
    a2='for (uint32_t i = lid; i < POLY_MIX_COUNT; i += wgsz) slm[i] = poly_mix[i];'
    assert a1 in w and a2 in w, 'wrapper anchors'
    w=w.replace(a1,'sycl::local_accessor<FpExt, 1> slm(sycl::range<1>(POLY_MIX_COUNT + circuit::rv32im_v2::RK_QN), h);')
    w=w.replace(a2,a2+'\n                    for (uint32_t i = lid; i < circuit::rv32im_v2::RK_QN; i += wgsz) slm[POLY_MIX_COUNT + i] = poly_mix[circuit::rv32im_v2::rk_qa[i]] * poly_mix[circuit::rv32im_v2::rk_qb[i]];')
    wrapper=w
if HOIST:
    assert 'rk_hoist' not in txt
    fn='constexpr uint32_t RK_HN = %d;\n__attribute__((noinline)) void rk_hoist(const FpExt* poly_mix, const Fp* out, const Fp* mix, FpExt* tab) {\n%s\n}\n'%HOIST
    txt=txt.replace('} // namespace risc0::circuit::rv32im_v2\n', fn+'} // namespace risc0::circuit::rv32im_v2\n',1)
    w=wrapper
    qn='circuit::rv32im_v2::RK_QN' if 'RK_QN' in txt else '0'
    a1=re.search(r'sycl::local_accessor<FpExt, 1> slm\(sycl::range<1>\(([^)]*)\), h\);',w)
    w=w.replace(a1.group(0),'sycl::local_accessor<FpExt, 1> slm(sycl::range<1>(%s + circuit::rv32im_v2::RK_HN), h);'%a1.group(1))
    a2='for (uint32_t i = lid; i < POLY_MIX_COUNT; i += wgsz) slm[i] = poly_mix[i];'
    assert a2 in w
    w=w.replace(a2,a2+'\n                    if (lid == 0) circuit::rv32im_v2::rk_hoist(poly_mix, out, mix, slm.get_multi_ptr<sycl::access::decorated::no>().get() + POLY_MIX_COUNT + %s);'%qn)
    wrapper=w
cut=txt.find('// Intel SYCL eval_check kernel')
open(sys.argv[2],'w').write(txt+wrapper)
print('rewrite:',dict(stats),'dce_removed',removed,file=sys.stderr)
