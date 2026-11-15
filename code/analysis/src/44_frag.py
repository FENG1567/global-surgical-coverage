#!/usr/bin/env python
"""
44_fragmentation.py 鈥?does population fragmentation predict the certified width?

The two-centre construction says the identified set is wide when capable
facilities can be confined to a subset of population centres. If that is the
mechanism, the certified width should track how much of a country's population
sits outside its largest connected landmass.

Index:  F = 1 - (population in the largest connected component of populated
                 cells) / (total populated population)
"""
import json
import numpy as np
import pandas as pd
import rasterio
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr, pearsonr

BASE = "__PROJECT_ROOT__"
G = f"{BASE}/work/grid"
NR, NC = 17400, 43200
PIX = 1.0 / 120.0
FRIC = f"{BASE}/data/friction/2020_motorized_friction_surface.geotiff"

cm = pd.read_csv(f"{BASE}/work/certified_many.csv")
cdf = pd.read_csv(f"{G}/countries.csv")
idxmap = dict(zip(cdf.iso3, cdf.idx))
h_all = pd.read_parquet(f"{BASE}/data/osm/osm_hospitals_all.parquet")
pop_glob = np.fromfile(f"{G}/pop2025.f32", dtype=np.float32).reshape(NR, NC)
CTRY = np.fromfile(f"{G}/country.i16", dtype=np.int16).reshape(NR, NC)
with rasterio.open(FRIC) as ds:
    FR = ds.read(1)

rows = []
for iso in cm.iso:
    h = h_all[h_all.iso3 == iso]
    r0 = int(max(0, (85.0 - (h.lat.max() + 0.3)) / PIX)); r1 = int(min(NR, (85.0 - (h.lat.min() - 0.3)) / PIX))
    c0 = int(max(0, (h.lon.min() - 0.3 + 180) / PIX)); c1 = int(min(NC, (h.lon.max() + 0.3 + 180) / PIX))
    if r1 - r0 < 5 or c1 - c0 < 5:
        continue
    fr = FR[r0:r1, c0:c1]
    pp = np.where(CTRY[r0:r1, c0:c1] == int(idxmap[iso]), pop_glob[r0:r1, c0:c1], 0.0)
    H, W = fr.shape
    if H * W > 4_000_000:
        continue
    idx = np.arange(H * W).reshape(H, W)
    valid = (fr > 0) & (fr < 1000)
    rows_, cols_, data_ = [], [], []
    for dr, dc in ((0, 1), (-1, 0)):
        rr = slice(max(0, -dr), H - max(0, dr)); cc = slice(max(0, -dc), W - max(0, dc))
        tr = slice(max(0, dr), H - max(0, -dr)); tc = slice(max(0, dc), W - max(0, -dc))
        a, b = idx[rr, cc], idx[tr, tc]
        ok = valid[rr, cc] & valid[tr, tc]
        rows_.append(a[ok]); cols_.append(b[ok]); data_.append(np.ones(ok.sum()))
    g = sparse.coo_matrix((np.concatenate(data_), (np.concatenate(rows_), np.concatenate(cols_))),
                          shape=(H * W, H * W)).tocsr()
    g = g.maximum(g.T)
    ncomp, lab = connected_components(g, directed=False)
    flat_pop = pp.ravel(); flat_lab = lab
    have = flat_pop > 0
    if not have.any():
        continue
    # population per component
    comp_pop = np.bincount(flat_lab[have], weights=flat_pop[have])
    F = 1.0 - comp_pop.max() / comp_pop.sum()
    rows.append({"iso": iso, "F": float(F), "components_with_pop": int((comp_pop > 0).sum())})

fdf = pd.DataFrame(rows).merge(cm, on="iso", how="inner")
fdf.to_csv(f"{BASE}/work/fragmentation.csv", index=False)
print(f"n = {len(fdf)}")
print(f"width: median {fdf.width.median()*100:.2f} pp, p90 {fdf.width.quantile(.9)*100:.2f}, "
      f"max {fdf.width.max()*100:.2f} ({fdf.loc[fdf.width.idxmax(),'iso']})")
print(f"countries with width > 7.4 pp: {(fdf.width > 0.074).sum()} / {len(fdf)}")
rho = spearmanr(fdf.F, fdf.width)
r = pearsonr(fdf.F, fdf.width)
print(f"\nfragmentation F vs width: Spearman rho = {rho.statistic:.3f} (p={rho.pvalue:.4f}); "
      f"Pearson r = {r[0]:.3f} (p={r[1]:.4f})")
print("\ntop widths:")
print(fdf.nlargest(8, "width")[["iso", "n", "observed", "width", "F"]].round(3).to_string(index=False))
print("\nzero width:")
print(fdf[fdf.width < 0.005][["iso", "n", "observed", "width", "F"]].round(4).to_string(index=False))
json.dump({"n": int(len(fdf)), "median": float(fdf.width.median()),
           "p90": float(fdf.width.quantile(.9)), "max": float(fdf.width.max()),
           "n_above_7.4": int((fdf.width > 0.074).sum()),
           "spearman": float(rho.statistic), "spearman_p": float(rho.pvalue),
           "pearson": float(r[0])},
          open(f"{BASE}/work/fragmentation.json", "w"), indent=1)
print("FRAG_DONE")
