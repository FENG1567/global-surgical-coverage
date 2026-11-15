#!/usr/bin/env python
"""
21_tail_validation.py — does the global agreement hide errors in the tail?

1. Recompute-vs-published travel time, stratified by urban/rural and income group,
   at 30 / 60 / 120 min (not only the global 2 h figure).
2. Population-weighted travel-time percentiles by stratum.
"""
import json
import numpy as np
import pandas as pd
import rasterio

BASE = "__PROJECT_ROOT__"
G = f"{BASE}/work/grid"
NR, NC = 17400, 43200
ROW0, NRC = 120, 17280          # published grid starts at 84N

ctry = np.fromfile(f"{G}/country.i16", dtype=np.int16).reshape(NR, NC)
cdf = pd.read_csv(f"{G}/countries.csv")
inc = dict(zip(cdf.idx, cdf.income))
urban = np.fromfile(f"{G}/urban.u8", dtype=np.uint8).reshape(NR, NC).astype(bool)
rural = np.fromfile(f"{G}/rural.u8", dtype=np.uint8).reshape(NR, NC).astype(bool)
pop = np.fromfile(f"{G}/pop2025.f32", dtype=np.float32).reshape(NR, NC)

mine = np.fromfile(f"{G}/tt_D0.f32", dtype=np.float32).reshape(NR, NC)[ROW0:ROW0 + NRC]
with rasterio.open(f"{BASE}/data/worldpop/global_TT_HC_M_fWS_v1.tif") as ds:
    pub = ds.read(1).astype(np.float32)
    nod = ds.nodata
print("shapes", mine.shape, pub.shape, "nodata", nod)

valid = np.isfinite(mine) & (mine < 1e8) & (pub > -1000) & (pub < 1e8)
print("valid cells", int(valid.sum()))

sub_ctry = ctry[ROW0:ROW0 + NRC]
sub_pop = pop[ROW0:ROW0 + NRC]
sub_u = urban[ROW0:ROW0 + NRC]
sub_r = rural[ROW0:ROW0 + NRC]

rows = []
for name, mask in (("all", np.ones_like(valid)),
                   ("urban", sub_u), ("rural", sub_r)):
    m = valid & mask
    w = sub_pop[m].astype(np.float64)
    if w.sum() == 0:
        continue
    for th in (30, 60, 120):
        agree = ((mine[m] <= th) == (pub[m] <= th))
        rows.append({"stratum": name, "thr": th,
                     "agree_cells": float(agree.mean()),
                     "agree_pop": float((w * agree).sum() / w.sum()),
                     "mine_share": float((w * (mine[m] <= th)).sum() / w.sum()),
                     "pub_share": float((w * (pub[m] <= th)).sum() / w.sum())})
    d = np.abs(mine[m] - pub[m])
    rows.append({"stratum": name, "thr": "absdiff",
                 "median": float(np.median(d)), "p90": float(np.percentile(d, 90)),
                 "p99": float(np.percentile(d, 99))})

# income strata
for g in ("LIC", "LMC", "UMC", "HIC"):
    ids = [k for k, v in inc.items() if v == g]
    if not ids:
        continue
    m = valid & np.isin(sub_ctry, ids)
    w = sub_pop[m].astype(np.float64)
    if w.sum() == 0:
        continue
    for th in (30, 120):
        agree = ((mine[m] <= th) == (pub[m] <= th))
        rows.append({"stratum": g, "thr": th,
                     "agree_cells": float(agree.mean()),
                     "agree_pop": float((w * agree).sum() / w.sum()),
                     "mine_share": float((w * (mine[m] <= th)).sum() / w.sum()),
                     "pub_share": float((w * (pub[m] <= th)).sum() / w.sum())})

df = pd.DataFrame(rows)
df.to_csv(f"{BASE}/work/tail_validation.csv", index=False)
print(df.to_string(index=False))

# ---- population-weighted travel time percentiles (D1) ----
tt = np.fromfile(f"{G}/tt_D1.f32", dtype=np.float32).reshape(NR, NC)
base = (ctry >= 0) & (pop > 0)
pct = {}
for name, mask in (("all", np.ones_like(base)), ("urban", urban), ("rural", rural)):
    m = base & mask
    t = tt[m].astype(np.float64); w = pop[m].astype(np.float64)
    ok = np.isfinite(t) & (t < 1e8)
    t, w = t[ok], w[ok]
    order = np.argsort(t)
    t, w = t[order], w[order]
    cw = np.cumsum(w) / w.sum()
    pct[name] = {p: float(t[np.searchsorted(cw, p / 100.0)]) for p in (50, 75, 90, 95, 99)}
print("\npopulation-weighted travel time to nearest OSM hospital (minutes):")
print(pd.DataFrame(pct).round(1).to_string())
json.dump({"tail": df.to_dict("records"), "percentiles": pct},
          open(f"{BASE}/work/tail_validation.json", "w"), indent=1, default=float)
print("TAIL_DONE")
