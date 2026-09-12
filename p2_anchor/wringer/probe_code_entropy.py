"""g9 碼流無損壓縮界:一階熵 / 條件熵(左鄰、上鄰、群內平均幅度)/ T1,T2 分面。純 CPU,碼不動。"""
import torch, numpy as np, json, sys, glob
def H(c):
    p=c[c>0]/c.sum(); return float(-(p*np.log2(p)).sum())
def Hcond(j):  # j: [ctx, sym] counts -> H(sym|ctx)
    j=j[j.sum(1)>0]; pc=j.sum(1)/j.sum()
    return float(sum(pc[i]*H(j[i]) for i in range(len(j))))
tot={'n':0,'H0':0,'Hleft':0,'Hup':0,'Hgrp':0,'HT1':0,'HT2':0,'HT2|T1':0,'Hleft2':0}
per_mod={}; per_layer={}
files=sorted(glob.glob('data/qs_g9/layer*.pt'))
for f in files:
    st=torch.load(f,map_location='cpu'); L=int(f.split('layer')[-1][:2])
    for name,v in st.items():
        if not isinstance(v,dict) or 'T1' not in v: continue
        s=(v['T1'].numpy().astype(np.int64)+1)*3+(v['T2'].numpy().astype(np.int64)+1)  # 0..8
        R,G,B=s.shape; flat=s.reshape(R,G*B); n=flat.size
        c0=np.bincount(flat.ravel(),minlength=9)
        jl=np.zeros((9,9)); np.add.at(jl,(flat[:,:-1].ravel(),flat[:,1:].ravel()),1)
        ju=np.zeros((9,9)); np.add.at(ju,(flat[:-1].ravel(),flat[1:].ravel()),1)
        jl2=np.zeros((81,9)); np.add.at(jl2,((flat[:,:-2]*9+flat[:,1:-1]).ravel(),flat[:,2:].ravel()),1)
        mag=np.abs(s-4).mean(-1,keepdims=True); mb=np.minimum((mag*4).astype(np.int64),15)  # 群幅度 16 桶(側資訊 4 b/群=0.125 b/w)
        jg=np.zeros((16,9)); np.add.at(jg,(np.broadcast_to(mb,s.shape).ravel(),s.ravel()),1)
        t1=v['T1'].numpy().astype(np.int64)+1; t2=v['T2'].numpy().astype(np.int64)+1
        jt=np.zeros((3,3)); np.add.at(jt,(t1.ravel(),t2.ravel()),1)
        r={'n':n,'H0':H(c0),'Hleft':Hcond(jl),'Hup':Hcond(ju),'Hgrp':Hcond(jg),'HT1':H(jt.sum(1)),'HT2':H(jt.sum(0)),'HT2|T1':Hcond(jt),'Hleft2':Hcond(jl2)}
        m=name.split('.')[-1]
        for d in (tot,per_mod.setdefault(m,{k:0 for k in tot}),per_layer.setdefault(L,{k:0 for k in tot})):
            d['n']+=n
            for k in r:
                if k!='n': d[k]+=r[k]*n
        if L in (0,10,31) and m=='down_proj': print(L,m,'hist',np.round(c0/c0.sum(),3).tolist(),flush=True)
def fin(d): return {k:(round(v/d['n'],4) if k!='n' else v) for k,v in d.items()}
out={'total':fin(tot),'per_module':{m:fin(d) for m,d in per_mod.items()},'per_layer':{L:fin(d) for L,d in per_layer.items()}}
print(json.dumps(out,ensure_ascii=False,indent=1)); json.dump(out,open('evidence/p1_grouping/corkscrew/g9_code_entropy.json','w'),ensure_ascii=False,indent=1)
print('G9_ENTROPY_DONE')
