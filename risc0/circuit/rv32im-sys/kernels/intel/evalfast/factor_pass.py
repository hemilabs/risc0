# Greedy common-E-factor extraction over maximal +/- trees of products (field-exact).
import collections
def make_pass(nm, fresh, repl, stats):
    def p_factor(F):
        N=F.nodes; users=F.users
        def nusers(i): return len(users[i])
        def isaddsub(i): return N[i]['op'] in ('+','-')
        def ismul(i): return N[i]['op']=='*'
        def isE(i): return N[i]['t']=='E'
        roots=[i for i,n in enumerate(N) if isaddsub(i) and isE(i) and not (nusers(i)==1 and isinstance(users[i][0],int) and isaddsub(users[i][0]))]
        for r in roots:
            terms=[]
            def walk(i,sg):
                if isaddsub(i) and (i==r or nusers(i)==1):
                    a,b=N[i]['args']; walk(a,sg); walk(b, sg if N[i]['op']=='+' else -sg)
                else: terms.append((sg,i))
            walk(r,1)
            def flat(i,top):
                if ismul(i) and (top or nusers(i)==1):
                    a,b=N[i]['args']; return flat(a,False)+flat(b,False)
                return [i]
            fl=[(sg,flat(t,True)) for sg,t in terms]
            def eecost_term(fs):
                e=sum(1 for f in fs if isE(f)); return max(e-1,0)
            before=sum(eecost_term(f) for s,f in fl)
            # greedy tree
            def solve(fl):
                cnt=collections.Counter()
                for s,f in fl:
                    for a in set(f):
                        if isE(a): cnt[a]+=1
                if not cnt or cnt.most_common(1)[0][1]<2:
                    return ('sum',[('t',s,f) for s,f in fl])
                # deterministic tie-break: highest count, then lowest node id
                best=max(cnt.items(), key=lambda kv:(kv[1],-kv[0]))[0]
                inside=[];outside=[]
                for s,f in fl:
                    if best in f:
                        f2=list(f); f2.remove(best); inside.append((s,f2))
                    else: outside.append((s,f))
                return ('sum',[('g',best,solve(inside))]+([('s',solve(outside))] if outside else []))
            tree=solve(fl)
            def cost(tr):
                # returns (EE count, isE)
                tot=0; anyE=False
                for it in tr[1]:
                    if it[0]=='t':
                        tot+=eecost_term(it[2]); anyE|=any(isE(x) for x in it[2])
                    elif it[0]=='g':
                        c,e=cost(it[2]); tot+=c+(1 if e else 0); anyE=True
                    else:
                        c,e=cost(it[1]); tot+=c; anyE|=e
                return tot,anyE
            after=cost(tree)[0]
            if after>=before: continue
            out=[]
            def emit_prod(fs):
                Fs_=[f for f in fs if not isE(f)]; Es=[f for f in fs if isE(f)]
                order=Fs_+Es
                if not order: raise Exception('empty product')
                cur=nm(F,order[0])
                for f in order[1:]:
                    t=fresh(); out.append('  auto %s = %s * %s;'%(t,cur,nm(F,f))); cur=t
                return cur
            def emit_sum(tr):
                items=[]  # (sign, name)
                for it in tr[1]:
                    if it[0]=='t': items.append((it[1],emit_prod(it[2])))
                    elif it[0]=='g':
                        sg,inner=emit_sum(it[2]); t=fresh(); out.append('  auto %s = %s * %s;'%(t,inner,nm(F,it[1]))); items.append((sg,t))
                    else: items.append(emit_sum(it[1]))
                items.sort(key=lambda x:-x[0])
                s0,cur=items[0]
                flip = s0<0          # all negative -> emit sum of magnitudes, return sign -1
                for sg,nmx in items[1:]:
                    eff = sg if not flip else -sg
                    t=fresh(); out.append('  auto %s = %s %s %s;'%(t,cur,'+' if eff>0 else '-',nmx)); cur=t
                return (-1 if flip else 1), cur
            sg,res=emit_sum(tree)
            if sg<0:
                out.append('  auto %s = -%s;'%(nm(F,r),res)); stats['factor_topneg']+=1
            else:
                out.append('  auto %s = %s;'%(nm(F,r),res))
            repl[N[r]['line']]=out
            stats['factor_roots']+=1; stats['factor_EE_before']+=before; stats['factor_EE_after']+=after
    return p_factor
