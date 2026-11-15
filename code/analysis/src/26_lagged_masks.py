#!/usr/bin/env python
"""
26_lagged_masks.py — capability defined from pre-COVID (2017-2019) and from 2023,
to be validated against 2024 flows. Longer lag + different period than the
2021 definition, which the external review asked for.
"""
import json
import numpy as np
import pandas as pd

BASE = "__PROJECT_ROOT__"
G = f"{BASE}/work/grid"
NR, NC = 17400, 43200
PIX = 1.0 / 120.0
SD = "__BRAZIL_SOURCE_ROOT__"
SIHDIR = "__SIH_PARQUET_ROOT__"

cnes = pd.read_csv(f"{SD}/03_outputs/hospital_points.csv")
cnes = cnes[cnes.coord_ok == True].dropna(subset=["lat", "lon"])
cnes["cnes"] = pd.to_numeric(cnes.cnes, errors="coerce")

out = {}
for tag, years in (("pre", [2017, 2018, 2019]), ("y2023", [2023])):
    frames = []
    for y in years:
        d = pd.read_parquet(f"{SIHDIR}/sih_{y}_cholec.parquet", columns=["CNES"])
        d["CNES"] = pd.to_numeric(d.CNES, errors="coerce")
        frames.append(d.dropna())
    cnt = pd.concat(frames).groupby("CNES").size().rename("n").reset_index()
    m = cnes.merge(cnt, left_on="cnes", right_on="CNES", how="inner")
    row = np.floor((85.0 - m.lat.values) / PIX).astype(np.int64)
    col = np.floor((m.lon.values + 180.0) / PIX).astype(np.int64)
    ok = (row >= 0) & (row < NR) & (col >= 0) & (col < NC)
    mask = np.zeros(NR * NC, dtype=np.uint8)
    mask[row[ok] * NC + col[ok]] = 1
    mask.tofile(f"{G}/src_BRA_act_{tag}.u8")
    out[tag] = {"hospitals": int(len(m)), "cells": int(mask.sum()),
                "cases": int(cnt.n.sum()), "years": years}
    print(tag, out[tag])

json.dump(out, open(f"{BASE}/work/lagged_masks.json", "w"), indent=1)
print("LAG_MASKS_DONE")
