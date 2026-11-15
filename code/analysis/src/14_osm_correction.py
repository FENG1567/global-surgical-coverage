#!/usr/bin/env python
"""
14_osm_correction.py — bound the effect of OSM hospital misclassification.

Measured (13_osm_precision.py): at 1 km, 60-72% of OSM hospitals match a
national registry hospital; at 2-3 km, 69-85%. So roughly 15-30% of OSM
"hospitals" are not hospitals, and 15-30% of real hospitals are missing.

We thin the OSM layer by 15 / 25 / 35 %, removing the *least urban* hospitals
first (false positives are expected to be small rural facilities), and compare
the resulting 2 h access with D1.
"""
import json, os
import numpy as np
import pandas as pd

BASE = "__PROJECT_ROOT__"
G = f"{BASE}/work/grid"
NR, NC = 17400, 43200
PIX = 1.0 / 120.0


def main():
    h = pd.read_parquet(f"{BASE}/data/osm/osm_hospitals_all.parquet")
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

    rng = np.random.default_rng(20261110)
    for keep in (0.85, 0.75, 0.65):
        tag = f"p{int(keep*100)}"
        # rural-first removal: keep the most urban
        keepm = np.zeros(len(h), dtype=bool)
        for iso, g in h.groupby("iso3"):
            k = max(1, int(round(len(g) * keep)))
            keepm[g.nlargest(k, "local_pop").index] = True
        m = np.zeros(NR * NC, dtype=np.uint8)
        m[row[keepm] * NC + col[keepm]] = 1
        m.tofile(f"{G}/src_D1_{tag}.u8")
        # random removal
        keepr = np.zeros(len(h), dtype=bool)
        for iso, g in h.groupby("iso3"):
            k = max(1, int(round(len(g) * keep)))
            keepr[rng.choice(g.index.values, size=k, replace=False)] = True
        m2 = np.zeros(NR * NC, dtype=np.uint8)
        m2[row[keepr] * NC + col[keepr]] = 1
        m2.tofile(f"{G}/src_D1_{tag}rand.u8")
        print(f"{tag}: urban-first kept {int(keepm.sum())} cells {int(m.sum())}; "
              f"random kept {int(keepr.sum())} cells {int(m2.sum())}")
    print("MASK_DONE")


if __name__ == "__main__":
    main()
