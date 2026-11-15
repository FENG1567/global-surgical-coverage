#!/usr/bin/env python
"""
18_brazil_validation.py — three-layer facility validation for Brazil.

Layer 1  OSM `amenity=hospital`            (crowd-sourced)
Layer 2  CNES registered hospitals          (national registry)
Layer 3a CNES with surgical centre          (structural capability)
Layer 3b CNES with observed surgery         (>=1 cholecystectomy in SIH 2024)

Question: how much does the measured population coverage change as the facility
definition moves from "mapped hospital" to "hospital that actually operated"?
"""
import json
import numpy as np
import pandas as pd

BASE = "__PROJECT_ROOT__"
G = f"{BASE}/work/grid"
NR, NC = 17400, 43200

ctry = np.fromfile(f"{G}/country.i16", dtype=np.int16).reshape(NR, NC).ravel()
cdf = pd.read_csv(f"{G}/countries.csv")
bra = int(cdf.loc[cdf.iso3 == "BRA", "idx"].iloc[0])
pop = np.fromfile(f"{G}/pop2025.f32", dtype=np.float32).reshape(NR, NC).ravel()
urban = np.fromfile(f"{G}/urban.u8", dtype=np.uint8).reshape(NR, NC).ravel().astype(bool)
rural = np.fromfile(f"{G}/rural.u8", dtype=np.uint8).reshape(NR, NC).ravel().astype(bool)

m = (ctry == bra) & (pop > 0)
pv = pop[m]; u = urban[m]; r = rural[m]

LAYERS = [
    ("OSM hospital", "tt_D1.f32", 7785),
    ("CNES registered", "tt_BRA_cnes_all.f32", 8444),
    ("CNES surgical centre", "tt_D2brazil.f32", 8000),
    ("CNES chole >=1", "tt_BRA_act.f32", 2303),
    ("CNES chole >=10", "tt_BRA_act10.f32", 2004),
]

rows = []
for name, f, n in LAYERS:
    t = np.fromfile(f"{G}/{f}", dtype=np.float32).reshape(NR, NC).ravel()[m]
    for th in (15, 30, 60, 120):
        av = np.where(t <= th, pv, 0.0)
        rows.append({"layer": name, "n_fac": n, "thr": th,
                     "all": float(av.sum() / pv.sum()),
                     "urban": float(av[u].sum() / pv[u].sum()),
                     "rural": float(av[r].sum() / pv[r].sum())})

df = pd.DataFrame(rows)
df.to_csv(f"{BASE}/work/brazil_validation.csv", index=False)
piv = df.pivot_table(index="layer", columns="thr", values="all")
piv = piv.reindex([l[0] for l in LAYERS])
print("Brazil population-weighted access by facility layer (%):")
print((piv * 100).round(2).to_string())
print("\nrural (%):")
pr = df.pivot_table(index="layer", columns="thr", values="rural").reindex([l[0] for l in LAYERS])
print((pr * 100).round(2).to_string())

base30 = float(piv.loc["OSM hospital", 30])
act30 = float(piv.loc["CNES chole >=1", 30])
base120 = float(piv.loc["OSM hospital", 120])
act120 = float(piv.loc["CNES chole >=1", 120])
print(f"\nOSM -> observed-activity: 30min {base30*100:.2f}% -> {act30*100:.2f}% "
      f"({(base30-act30)*100:+.2f} pp); 120min {base120*100:.2f}% -> {act120*100:.2f}% "
      f"({(base120-act120)*100:+.2f} pp)")

json.dump({"table": df.to_dict("records"),
           "delta_30_pp": (base30 - act30) * 100,
           "delta_120_pp": (base120 - act120) * 100},
          open(f"{BASE}/work/brazil_validation.json", "w"), indent=1, default=float)
print("BRA_DONE")
