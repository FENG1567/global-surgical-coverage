#!/usr/bin/env python
"""
30_conditional_set.py — the identified set CONDITIONAL on the urban share u.

At fixed p_c and fixed u, how much freedom is left? We draw several random
arrangements that respect u (half the capable hospitals urban, half rural) and
measure the spread. If the spread is small, then reporting u — which facility
inventories can do — nearly identifies coverage; if it is large, u is not enough.
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

# urban/rural label: local population above/below the national median
h["is_urban"] = h.local_pop > h.groupby("iso3").local_pop.transform("median")

out = {}
for seed in (1, 2, 3):
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(h), dtype=bool)
    for iso, g in h.groupby("iso3"):
        k = max(1, int(round(len(g) * float(g.p_c.iloc[0]))))
        nu = k // 2
        urb = g[g.is_urban].index.values
        rur = g[~g.is_urban].index.values
        nu = min(nu, len(urb)); nr = min(k - nu, len(rur))
        sel = np.r_[rng.choice(urb, nu, replace=False), rng.choice(rur, nr, replace=False)]
        keep[sel] = True
    m = np.zeros(NR * NC, dtype=np.uint8)
    m[row[keep] * NC + col[keep]] = 1
    tag = f"c{seed}"
    m.tofile(f"{G}/src_D2c_{tag}.u8")
    out[tag] = {"hospitals": int(keep.sum()), "cells": int(m.sum()), "seed": seed}
    print(tag, out[tag])

json.dump(out, open(f"{BASE}/work/conditional_masks.json", "w"), indent=1)
print("COND_MASKS_DONE")
