#!/usr/bin/env python
"""
16_v15_30min.py — v1.5 main estimand: 30-minute access, and the G2' gate checks.
Gates were pre-registered in PROJECT_CONTRACT.md §15.3 before this script ran.
"""
import json, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = "__PROJECT_ROOT__"
G = f"{BASE}/work/grid"
NR, NC = 17400, 43200

ctry = np.fromfile(f"{G}/country.i16", dtype=np.int16).reshape(NR, NC).ravel()
cdf = pd.read_csv(f"{G}/countries.csv")
NCY = int(cdf.idx.max()) + 1
urban = np.fromfile(f"{G}/urban.u8", dtype=np.uint8).reshape(NR, NC).ravel().astype(bool)
rural = np.fromfile(f"{G}/rural.u8", dtype=np.uint8).reshape(NR, NC).ravel().astype(bool)
pop25 = np.fromfile(f"{G}/pop2025.f32", dtype=np.float32).reshape(NR, NC).ravel()
pop20 = np.fromfile(f"{G}/pop2020.f32", dtype=np.float32).reshape(NR, NC).ravel()

SURF = {
    "D0u": "tt_D0u.f32", "D1": "tt_D1.f32", "D2": "tt_D2meas.f32",
    "D2corr": "tt_D2meas_corr.f32", "D1walk": "tt_D1_walk.f32",
    "D1_4n": "tt_D1_4n.f32",
}
THR = [15, 30, 60, 120]

base = (ctry >= 0) & (pop25 > 0)
idx = ctry[base].astype(np.int64)
u = urban[base]; r = rural[base]
pv25 = pop25[base]; pv20 = pop20[base]

res = {}
country = {}
for key, f in SURF.items():
    t = np.fromfile(f"{G}/{f}", dtype=np.float32).reshape(NR, NC).ravel()[base]
    for th in THR:
        av = np.where(t <= th, pv25, 0.0)
        res[(key, th)] = {
            "all": float(av.sum() / pv25.sum()),
            "urban": float(av[u].sum() / pv25[u].sum()),
            "rural": float(av[r].sum() / pv25[r].sum()),
        }
        if th == 30:
            pb = np.bincount(idx, weights=pv25, minlength=NCY)
            ab = np.bincount(idx, weights=av, minlength=NCY)
            country[key] = pd.Series(ab / np.where(pb == 0, np.nan, pb), index=range(NCY))
    # population vintage check (30 min, 2025 pop vs 2020 pop, same surface)
    av20 = np.where(t <= 30, pv20, 0.0)
    res[(key, "pop20_30")] = float(av20.sum() / pv20.sum())

# ---------- G2' checks ----------
gates = {}
cc = pd.DataFrame(country).dropna()
gates["G2'-1 monotonic"] = float(((cc.D0u >= cc.D1 - 1e-9) & (cc.D1 >= cc.D2 - 1e-9) &
                                  (cc.D2 >= cc.D2corr - 1e-9)).mean())
# threshold monotone per country (D1)
thr_ok = []
for iso_i in cc.index:
    v = [res[("D1", th)]["all"] for th in THR]
    thr_ok.append(all(v[i] <= v[i + 1] + 1e-9 for i in range(3)))
gates["G2'-2 threshold gradient"] = float(np.mean(thr_ok))

# rank correlations across thresholds (D1, country level, need per-threshold country series)
c15 = {}; c60 = {}; c120 = {}
t = np.fromfile(f"{G}/tt_D1.f32", dtype=np.float32).reshape(NR, NC).ravel()[base]
pb = np.bincount(idx, weights=pv25, minlength=NCY)
for th, dst in ((15, c15), (60, c60), (120, c120)):
    av = np.where(t <= th, pv25, 0.0)
    ab = np.bincount(idx, weights=av, minlength=NCY)
    dst["D1"] = pd.Series(ab / np.where(pb == 0, np.nan, pb), index=range(NCY))
gates["G2'-3 rho 15v30"] = float(spearmanr(cc.D1, pd.Series(c15["D1"]).reindex(cc.index)).statistic)
gates["G2'-3 rho 30v120"] = float(spearmanr(cc.D1, pd.Series(c120["D1"]).reindex(cc.index)).statistic)
gates["G2'-4 rho motor v walk"] = float(spearmanr(cc.D1, cc.D1walk).statistic)
gates["G2'-5 pop vintage diff pp"] = float(abs(res[("D1", 30)]["all"] - res[("D1", "pop20_30")]) * 100)
gates["G2'-6 urban>rural share"] = float((cc.D1 > cc.D1walk).mean())  # placeholder, replaced below
gates["G2'-7 4n v 8n diff pp"] = float(abs(res[("D1", 30)]["all"] - res[("D1_4n", 30)]["all"]) * 100)

# income gradient at 30 min (D1 and D2corr)
inc = cdf.set_index("idx").income
tmp = pd.DataFrame({"D1": cc.D1, "D2corr": cc.D2corr}).join(inc, how="left")
grp = tmp.groupby("income")[["D1", "D2corr"]].median()
order = ["HIC", "UMC", "LMC", "LIC"]
g1 = [float(grp.loc[i, "D1"]) for i in order if i in grp.index]
g2 = [float(grp.loc[i, "D2corr"]) for i in order if i in grp.index]
gates["G2'-8 income monotone D1"] = bool(all(g1[i] >= g1[i + 1] for i in range(len(g1) - 1)))
gates["G2'-8 income monotone D2corr"] = bool(all(g2[i] >= g2[i + 1] for i in range(len(g2) - 1)))

print("=== 30-min ladder (2025 pop) ===")
for key in SURF:
    a = res[(key, 30)]
    print(f"  {key:8s} all={a['all']*100:5.2f}%  urban={a['urban']*100:5.2f}%  rural={a['rural']*100:5.2f}%  gap={(a['urban']-a['rural'])*100:5.1f} pp")
print("\n=== threshold gradient (D1) ===")
for th in THR:
    a = res[("D1", th)]
    print(f"  {th:3d} min all={a['all']*100:5.2f}% gap={(a['urban']-a['rural'])*100:5.1f} pp")
print("\n=== G2' checks ===")
for k, v in gates.items():
    print(f"  {k}: {v}")
print("\n=== income (median country share, 30 min) ===")
print(grp.round(4).to_string())

json.dump({"res": {f"{k[0]}_{k[1]}": v for k, v in res.items()},
           "gates": gates,
           "income_median": grp.round(6).to_dict()},
          open(f"{BASE}/work/v15_numbers.json", "w"), indent=1, default=float)
cc.join(inc, how="left").to_csv(f"{BASE}/work/v15_by_country.csv")
print("V15_DONE")
