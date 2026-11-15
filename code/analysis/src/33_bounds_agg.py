import numpy as np, pandas as pd, json
G="__PROJECT_ROOT__/work/grid"
NR,NC=17400,43200
ctry=np.fromfile(f"{G}/country.i16",dtype=np.int16).reshape(NR,NC).ravel()
cdf=pd.read_csv(f"{G}/countries.csv")
bra=int(cdf.loc[cdf.iso3=="BRA","idx"].iloc[0])
pop=np.fromfile(f"{G}/pop2025.f32",dtype=np.float32).reshape(NR,NC).ravel()
urban=np.fromfile(f"{G}/urban.u8",dtype=np.uint8).reshape(NR,NC).ravel().astype(bool)
rural=np.fromfile(f"{G}/rural.u8",dtype=np.uint8).reshape(NR,NC).ravel().astype(bool)
m=(ctry==bra)&(pop>0); pv=pop[m]
rows=[]
for name,f in [("OSM hospital","tt_D1.f32"),("rural-first (u=0)","tt_D2u_u000.f32"),
               ("urban-first (u=1)","tt_D2u_u100.f32"),("greedy max","tt_BRA_gmax.f32"),
               ("greedy min","tt_BRA_gmin.f32"),("activity 2021","tt_BRA_act2021.f32")]:
    t=np.fromfile(f"{G}/{f}",dtype=np.float32).reshape(NR,NC).ravel()[m]
    for th in (30,120):
        a=np.where(t<=th,pv,0.0)
        rows.append({"layer":name,"thr":th,"all":float(a.sum()/pv.sum()),
                     "rural":float(a[rural[m]].sum()/pv[rural[m]].sum())})
df=pd.DataFrame(rows)
p=df.pivot_table(index="layer",columns="thr",values="all")
print("Brazil coverage (%):")
print((p*100).round(2).to_string())
print()
p30=p[30]*100
print("30-min: greedy max %.2f%%, greedy min %.2f%%, width %.2f pp"%(p30["greedy max"],p30["greedy min"],p30["greedy max"]-p30["greedy min"]))
print("greedy max vs activity truth: %+.2f pp"%(p30["greedy max"]-p30["activity 2021"]))
print("greedy min vs activity truth: %+.2f pp"%(p30["greedy min"]-p30["activity 2021"]))
df.to_csv("__PROJECT_ROOT__/work/brazil_bounds_coverage.csv",index=False)
json.dump(p.to_dict(),open("__PROJECT_ROOT__/work/brazil_bounds_coverage.json","w"),indent=1,default=float)
print("DONE")