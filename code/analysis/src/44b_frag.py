import numpy as np, pandas as pd, rasterio
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr, pearsonr
B="__PROJECT_ROOT__"
G=f"{B}/work/grid"; NR,NC=17400,43200; PIX=1/120
cm=pd.read_csv(f"{B}/work/certified_many.csv")
cdf=pd.read_csv(f"{G}/countries.csv"); idxmap=dict(zip(cdf.iso3,cdf.idx))
pop=np.fromfile(f"{G}/pop2025.f32",dtype=np.float32).reshape(NR,NC)
CTRY=np.fromfile(f"{G}/country.i16",dtype=np.int16).reshape(NR,NC)
rows=[]
for iso in cm.iso:
    cid=int(idxmap[iso])
    m=(CTRY==cid)&(pop>0)
    if m.sum()<10: continue
    rr,cc=np.nonzero(m)
    r0,r1,c0,c1=rr.min(),rr.max()+1,cc.min(),cc.max()+1
    sub=np.where(CTRY[r0:r1,c0:c1]==cid,pop[r0:r1,c0:c1],0.0)
    # coarse 0.2 deg
    bs=int(0.2/PIX); H=(sub.shape[0]//bs); W=(sub.shape[1]//bs)
    if H<2 or W<2: continue
    co=sub[:H*bs,:W*bs].reshape(H,bs,W,bs).sum(axis=(1,3))
    on=co>50
    if on.sum()<2:
        rows.append({"iso":iso,"F":0.0,"clusters":1}); continue
    ids=np.arange(H*W).reshape(H,W)
    rs,cs=[],[]
    for dr,dc in ((0,1),(-1,0)):
        a=ids[max(0,-dr):H-max(0,dr), max(0,-dc):W-max(0,dc)]
        b=ids[max(0,dr):H-max(0,-dr), max(0,dc):W-max(0,-dc)]
        o=on[max(0,-dr):H-max(0,dr), max(0,-dc):W-max(0,dc)] & on[max(0,dr):H-max(0,-dr), max(0,dc):W-max(0,-dc)]
        rs.append(a[o]); cs.append(b[o])
    if not rs: rows.append({"iso":iso,"F":0.0,"clusters":1}); continue
    g=sparse.coo_matrix((np.ones(sum(len(x) for x in rs)),(np.concatenate(rs),np.concatenate(cs))),shape=(H*W,H*W)).tocsr()
    g=g.maximum(g.T)
    ncomp,lab=connected_components(g,directed=False)
    fl=lab.ravel(); fp=co.ravel(); have=fp>0
    cp=np.bincount(fl[have],weights=fp[have])
    F=1.0-cp.max()/cp.sum()
    rows.append({"iso":iso,"F":float(F),"clusters":int((cp>0).sum())})
f=pd.DataFrame(rows).merge(cm,on="iso")
rho=spearmanr(f.F,f.width)
print(f"n={len(f)}")
print(f"width median {f.width.median()*100:.2f} p90 {f.width.quantile(.9)*100:.2f} max {f.width.max()*100:.2f}")
print(f">7.4pp: {(f.width>0.074).sum()}/{len(f)}")
print(f"F: median {f.F.median():.3f} max {f.F.max():.3f}")
print(f"Spearman F vs width: rho={rho.statistic:.3f} p={rho.pvalue:.4f}")
print(f.Pearson if False else "")
print(f[["iso","n","width","F","clusters"]].sort_values("F",ascending=False).head(12).round(3).to_string(index=False))
f.to_csv(f"{B}/work/fragmentation2.csv",index=False)