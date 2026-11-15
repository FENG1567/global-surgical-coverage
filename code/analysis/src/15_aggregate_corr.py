#!/usr/bin/env python
"""15_aggregate_corr.py — effect of OSM hospital misclassification on 2h access."""
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

SURF = [("D1", "tt_D1.f32"), ("D1_p85", "tt_D1_p85.f32"), ("D1_p75", "tt_D1_p75.f32"),
        ("D1_p65", "tt_D1_p65.f32"), ("D1_p85r", "tt_D1_p85rand.f32"),
        ("D1_p75r", "tt_D1_p75rand.f32"), ("D1_p65r", "tt_D1_p65rand.f32")]

rows = []
for key, f in SURF:
    p = f"{G}/{f}"
    if not os.path.exists(p):
        print("skip", key); continue
    tt = np.fromfile(p, dtype=np.float32).reshape(NR, NC).ravel()
    t = tt[base]; del tt
    for thr, tag in ((120, "2h"), (180, "3h")):
        av = np.where(t <= thr, pv, 0.0)
        rows.append({"defn": key, "thr": tag, "stratum": "all", "pop": float(pv.sum()),
                     "acc": float(av.sum()), "share": float(av.sum() / pv.sum())})
        for nm, m in (("urban", u), ("rural", r)):
            ps, ac = float(pv[m].sum()), float(av[m].sum())
            rows.append({"defn": key, "thr": tag, "stratum": nm, "pop": ps, "acc": ac, "share": ac / ps})
df = pd.DataFrame(rows)
piv = df[df.thr == "2h"].pivot_table(index="defn", columns="stratum", values="share")
print(piv.round(4).to_string())
d1 = float(piv.loc["D1", "all"])
print("\ndelta vs D1 (2h, global):")
for k in piv.index:
    print(f"  {k:10s} {piv.loc[k,'all']*100:6.2f}%   delta {(d1-piv.loc[k,'all'])*100:+.2f} pp")
df.to_csv(f"{BASE}/work/osm_correction_access.csv", index=False)
json.dump({k: {c: float(piv.loc[k, c]) for c in piv.columns} for k in piv.index},
          open(f"{BASE}/work/osm_correction.json", "w"), indent=1)
print("CORR_DONE")
