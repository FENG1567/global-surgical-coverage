#!/usr/bin/env python
"""
24_flow_validation_lagged.py — break the circularity.

Capability is defined from 2021 SIH activity (hospitals that performed >=1
cholecystectomy in 2021). The outcome is 2024 SIH flows. A three-year gap means
the predictor cannot mechanically reproduce the outcome.
"""
import json
import numpy as np
import pandas as pd

BASE = "__PROJECT_ROOT__"
G = f"{BASE}/work/grid"
NR, NC = 17400, 43200
PIX = 1.0 / 120.0
SD = "__BRAZIL_SOURCE_ROOT__"

# ---- 2021 activity by CNES ----
s21 = pd.read_parquet("__SIH_PARQUET_ROOT__/sih_2021_cholec.parquet",
                      columns=["CNES"])
s21["CNES"] = pd.to_numeric(s21.CNES, errors="coerce")
cnt = s21.dropna().groupby("CNES").size().rename("n2021").reset_index()
print("hospitals with chole in 2021:", len(cnt), "cases:", int(cnt.n2021.sum()))

cnes = pd.read_csv(f"{SD}/03_outputs/hospital_points.csv")
cnes = cnes[cnes.coord_ok == True].dropna(subset=["lat", "lon"])
cnes["cnes"] = pd.to_numeric(cnes.cnes, errors="coerce")
m = cnes.merge(cnt, left_on="cnes", right_on="CNES", how="inner")
print("2021-active hospitals with coordinates:", len(m))

row = np.floor((85.0 - m.lat.values) / PIX).astype(np.int64)
col = np.floor((m.lon.values + 180.0) / PIX).astype(np.int64)
ok = (row >= 0) & (row < NR) & (col >= 0) & (col < NC)
mask = np.zeros(NR * NC, dtype=np.uint8)
mask[row[ok] * NC + col[ok]] = 1
mask.tofile(f"{G}/src_BRA_act2021.u8")
print("cells:", int(mask.sum()))
json.dump({"n_hosp": int(len(m)), "cases": int(cnt.n2021.sum())},
          open(f"{BASE}/work/act2021.json", "w"))
print("MASK_DONE")
