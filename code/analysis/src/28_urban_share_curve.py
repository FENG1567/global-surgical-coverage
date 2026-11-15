#!/usr/bin/env python
"""
28_urban_share_curve.py — C as a function of the urban share of surgical capacity.

Instead of four discrete allocations, hold p_c fixed and vary u = share of the
retained hospitals that are the most urban. u is in principle observable
(national facility inventories report urban/rural location), so C(u) turns the
non-identifiability into a one-dimensional sensitivity that a data system could
actually pin down.
"""
import json
import numpy as np
import pandas as pd

BASE = "__PROJECT_ROOT__"
G = f"{BASE}/work/grid"
NR, NC = 17400, 43200
PIX = 1.0 / 120.0

h = pd.read_parquet(f"{BASE}/data/osm/osm_hospitals_all.parquet")
cap = pd.read_csv(f"{BASE}/outputs/hospital_capability.csv")
pcmap = dict(zip(cap.iso3, cap.p_c))
pop = np.fromfile(f"{G}/pop2025.f32", dtype=np.float32).reshape(NR, NC)

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

out = {}
for u in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
    keep = np.zeros(len(h), dtype=bool)
    for iso, g in h.groupby("iso3"):
        k = max(1, int(round(len(g) * float(g.p_c.iloc[0]))))
        n_urb = int(round(k * u))
        gs = g.sort_values("local_pop", ascending=False)
        idx_urb = gs.index[:n_urb]
        idx_rur = gs.index[::-1][: k - n_urb]
        keep[idx_urb] = True
        keep[idx_rur] = True
    m = np.zeros(NR * NC, dtype=np.uint8)
    m[row[keep] * NC + col[keep]] = 1
    tag = f"u{int(u*100):03d}"
    m.tofile(f"{G}/src_D2u_{tag}.u8")
    out[tag] = {"u": u, "hospitals": int(keep.sum()), "cells": int(m.sum())}
    print(tag, out[tag])

json.dump(out, open(f"{BASE}/work/urban_share_masks.json", "w"), indent=1)
print("USHARE_MASKS_DONE")
