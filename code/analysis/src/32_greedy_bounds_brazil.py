#!/usr/bin/env python
"""
32_greedy_bounds_brazil.py 鈥?feasible coverage bounds on a national scale.

Exact max/min-coverage over k facilities is infeasible globally but tractable
for one country with ~8k facilities. We run greedy max-coverage (which carries
a 1-1/e guarantee, so OPT <= greedy/(1-1/e) ~ 1.58 x greedy) and greedy
min-coverage (heuristic, no guarantee), then evaluate both on the real 1 km
Dijkstra engine.

Output: two masks (gmax / gmin) for Brazil + a JSON with the bound arithmetic.
"""
import json
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

BASE = "__PROJECT_ROOT__"
G = f"{BASE}/work/grid"
NR, NC = 17400, 43200
PIX = 1.0 / 120.0
CG = 0.05
# Brazil window
LON0, LON1, LAT0, LAT1 = -74.5, -33.5, 5.5, -34.0
NCX = int(round((LON1 - LON0) / CG))
NCY = int(round((LAT0 - LAT1) / CG))

h = pd.read_parquet(f"{BASE}/data/osm/osm_hospitals_all.parquet")
br = h[h.iso3 == "BRA"].reset_index(drop=True)
print("Brazil OSM hospitals:", len(br))
k = int(round(len(br) * 0.80))
print("k (p_c=0.80):", k)

# coarse population for Brazil window
pop = np.fromfile(f"{G}/pop2025.f32", dtype=np.float32).reshape(NR, NC)
bs = int(CG / PIX)
r0 = int(round((85.0 - LAT0) / PIX)); c0 = int(round((LON0 + 180) / PIX))
sub = pop[r0:r0 + NCY * bs, c0:c0 + NCX * bs]
sub = sub[: NCY * bs, : NCX * bs].reshape(NCY, bs, NCX, bs).sum(axis=(1, 3))
print("coarse pop sum: %.1f M" % (sub.sum() / 1e6))

fi = np.clip(((LAT0 - br.lat.values) / CG).astype(int), 0, NCY - 1)
fj = np.clip(((br.lon.values - LON0) / CG).astype(int), 0, NCX - 1)

K = 5  # ~5 coarse cells = 25 km window (generous 30-min catchment proxy)


def greedy(order):
    uncovered = sub.astype(np.float64).copy()
    chosen = []
    for _ in range(k):
        gain_r = uniform_filter(uncovered, size=K, mode="constant") * (K * K)
        gains = gain_r[fi, fj]
        gains[chosen] = -np.inf if order == "max" else np.inf
        j = int(np.argmax(gains) if order == "max" else np.argmin(gains))
        chosen.append(j)
        # mark neighbourhood covered
        a = max(0, fi[j] - K // 2); b = min(NCY, fi[j] + K // 2 + 1)
        c = max(0, fj[j] - K // 2); d = min(NCX, fj[j] + K // 2 + 1)
        uncovered[a:b, c:d] = 0.0
        if (_ + 1) % 1000 == 0:
            print(f"  {order}: {_+1}/{k} covered={1-uncovered.sum()/sub.sum():.3f}")
    return np.array(chosen)


print("greedy max ...")
idx_max = greedy("max")
print("greedy min ...")
idx_min = greedy("min")

row = np.floor((85.0 - br.lat.values) / PIX).astype(np.int64)
col = np.floor((br.lon.values + 180.0) / PIX).astype(np.int64)
out = {}
for tag, idx in (("gmax", idx_max), ("gmin", idx_min)):
    keep = np.zeros(len(br), dtype=bool)
    keep[idx] = True
    m = np.zeros(NR * NC, dtype=np.uint8)
    m[row[keep] * NC + col[keep]] = 1
    m.tofile(f"{G}/src_BRA_{tag}.u8")
    out[tag] = int(keep.sum())
    print(tag, "kept", int(keep.sum()), "cells", int(m.sum()))

json.dump({"k": k, "n": len(br), "masks": out,
           "max_gap_bound": "OPT_max <= greedy_max/(1-1/e) = 1.582 x greedy_max"},
          open(f"{BASE}/work/brazil_bounds.json", "w"), indent=1)
print("BRA_BOUNDS_MASKS_DONE")
