#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""59_export_selection.py — 导出两方案的选中设施 ID 与逐国 k_c 清单"""
import numpy as np
import pandas as pd

B = "__PROJECT_ROOT__"
G = f"{B}/work/grid"
NR, NC = 17400, 43200
PIX = 1.0 / 120.0

h = pd.read_parquet(f"{B}/data/osm/osm_hospitals_all.parquet").reset_index(drop=True)
pop = np.fromfile(f"{G}/pop2025.f32", dtype=np.float32).reshape(NR, NC)
cap = pd.read_csv(f"{B}/outputs/hospital_capability.csv")
pcmap = dict(zip(cap.iso3, cap.p_c))

row = np.floor((85.0 - h.lat.values) / PIX).astype(np.int64)
col = np.floor((h.lon.values + 180.0) / PIX).astype(np.int64)
ok = (row >= 1) & (row < NR - 1) & (col >= 1) & (col < NC - 1)
dens = np.zeros(len(h))
r, c = row[ok], col[ok]
dens[ok] = (pop[r - 1, c - 1] + pop[r - 1, c] + pop[r - 1, c + 1] +
            pop[r, c - 1] + pop[r, c] + pop[r, c + 1] +
            pop[r + 1, c - 1] + pop[r + 1, c] + pop[r + 1, c + 1])
h["local_pop"] = dens
h["p_c"] = h.iso3.map(pcmap).fillna(0.80)
h["is_urban"] = h.local_pop > h.groupby("iso3").local_pop.transform("median")
h["facility_idx"] = np.arange(len(h))

# 逐国预算表
per = []
for iso, g in h.groupby("iso3"):
    n_c = len(g)
    k = max(1, int(round(n_c * float(g.p_c.iloc[0]))))
    per.append({"iso3": iso, "n_country": n_c, "k_c": k,
                "p_c": float(g.p_c.iloc[0])})
per = pd.DataFrame(per)

for name in ("rand", "cluster"):
    ids = np.load(f"{B}/work/scenario_ids_{name}.npy")
    sub = h.iloc[ids][["facility_idx", "iso3", "lat", "lon", "local_pop", "is_urban"]].copy()
    sub["scenario"] = name
    sub.to_csv(f"{B}/work/selection_{name}.csv", index=False)
    chk = sub.groupby("iso3").size().rename("n_selected").reset_index()
    m = per.merge(chk, on="iso3", how="left").fillna({"n_selected": 0})
    m["n_selected"] = m.n_selected.astype(int)
    m["violation"] = m.n_selected != m.k_c
    m.to_csv(f"{B}/work/selection_{name}_budget.csv", index=False)
    print(f"{name}: selected={len(sub)}  countries={len(m)}  violations={int(m.violation.sum())}")
    print(f"   u_global={sub.is_urban.mean():.4f}  wrote selection_{name}.csv / selection_{name}_budget.csv")

per.to_csv(f"{B}/work/country_budget_reference.csv", index=False)
print(f"\ncountry_budget_reference.csv: {len(per)} countries, sum k_c = {int(per.k_c.sum())}")
print("EXPORT_DONE")
