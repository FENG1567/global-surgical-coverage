#!/usr/bin/env python
"""20_aggregate_spatial.py — coverage under four spatial allocations of the same p_c."""
import json, os
import numpy as np
import pandas as pd

BASE = "__PROJECT_ROOT__"
G = f"{BASE}/work/grid"
NR, NC = 17400, 43200

ctry = np.fromfile(f"{G}/country.i16", dtype=np.int16).reshape(NR, NC).ravel()
cdf = pd.read_csv(f"{G}/countries.csv")
NCY = int(cdf.idx.max()) + 1
pop = np.fromfile(f"{G}/pop2025.f32", dtype=np.float32).reshape(NR, NC).ravel()
urban = np.fromfile(f"{G}/urban.u8", dtype=np.uint8).reshape(NR, NC).ravel().astype(bool)
rural = np.fromfile(f"{G}/rural.u8", dtype=np.uint8).reshape(NR, NC).ravel().astype(bool)
base = (ctry >= 0) & (pop > 0)
idx = ctry[base].astype(np.int64); pv = pop[base]; u = urban[base]; r = rural[base]
pb = np.bincount(idx, weights=pv, minlength=NCY)

S = [("random", "tt_D2sc_rand.f32"), ("urban-first", "tt_D2sc_urban.f32"),
     ("rural-first", "tt_D2sc_rural.f32"), ("clustered", "tt_D2sc_cluster.f32")]

rows = []
cs = {}
for name, f in S:
    t = np.fromfile(f"{G}/{f}", dtype=np.float32).reshape(NR, NC).ravel()[base]
    for th in (15, 30, 60, 120):
        av = np.where(t <= th, pv, 0.0)
        rows.append({"alloc": name, "thr": th, "all": float(av.sum() / pv.sum()),
                     "urban": float(av[u].sum() / pv[u].sum()),
                     "rural": float(av[r].sum() / pv[r].sum())})
        if th == 30:
            ab = np.bincount(idx, weights=av, minlength=NCY)
            cs[name] = pd.Series(ab / np.where(pb == 0, np.nan, pb), index=range(NCY))

df = pd.DataFrame(rows)
df.to_csv(f"{BASE}/work/spatial_scenarios_access.csv", index=False)
p = df.pivot_table(index="alloc", columns="thr", values="all").reindex([s[0] for s in S])
pr = df.pivot_table(index="alloc", columns="thr", values="rural").reindex([s[0] for s in S])
print("=== coverage (%) by allocation, same p_c (86.6% of hospitals) ===")
print((p * 100).round(2).to_string())
print("\n=== rural only (%) ===")
print((pr * 100).round(2).to_string())
for th in (30, 120):
    spread = (p[th].max() - p[th].min()) * 100
    print(f"\n{th} min: range {p[th].min()*100:.2f}%–{p[th].max()*100:.2f}%  spread {spread:.2f} pp")

cc = pd.DataFrame(cs).dropna()
cc.to_csv(f"{BASE}/work/spatial_country30.csv")
out = {"table": df.to_dict("records"),
       "spread_30_pp": float((p[30].max() - p[30].min()) * 100),
       "spread_120_pp": float((p[120].max() - p[120].min()) * 100),
       "spread_rural30_pp": float((pr[30].max() - pr[30].min()) * 100)}
json.dump(out, open(f"{BASE}/work/spatial_scenarios.json", "w"), indent=1, default=float)
print("\nSPATIAL_DONE")
