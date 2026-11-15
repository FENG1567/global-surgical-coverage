#!/usr/bin/env python
"""
27_flow_holdout.py — spatial holdout and incremental validity.

External review asked for: spatial block holdout, incremental value over
baseline covariates, and a capability definition that is not contemporaneous
with the outcome.
"""
import json
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
import geopandas as gpd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr

BASE = "__PROJECT_ROOT__"
G = f"{BASE}/work/grid"
NR, NC = 17400, 43200
PIX = 1.0 / 120.0
SD = "__BRAZIL_SOURCE_ROOT__"

LAYERS = {"D1_osm": "tt_D1.f32", "D2_act_pre": "tt_BRA_act_pre.f32",
          "D2_act_2023": "tt_BRA_act_y2023.f32", "D2_act_2021": "tt_BRA_act2021.f32"}

g = gpd.read_file(f"{SD}/river/data/official/ibge_malha/extracted/BR_Municipios_2022.shp")
code_col = [c for c in g.columns if c.upper().startswith("CD_MUN")][0]
g["m6"] = g[code_col].astype(str).str[:6].astype(np.int64)
minx, miny, maxx, maxy = g.total_bounds
c0 = int(np.floor((minx + 180) / PIX)); c1 = int(np.ceil((maxx + 180) / PIX))
r0 = int(np.floor((85 - maxy) / PIX)); r1 = int(np.ceil((85 - miny) / PIX))
W, H = c1 - c0, r1 - r0
transform = rasterio.transform.from_origin(-180 + c0 * PIX, 85 - r0 * PIX, PIX, PIX)
muni = features.rasterize(((geom, int(m)) for geom, m in zip(g.geometry, g.m6)),
                          out_shape=(H, W), transform=transform, fill=0, dtype="int64")
flat = muni.ravel(); valid = flat > 0
idx = flat[valid]
codes, inv = np.unique(idx, return_inverse=True)

pop = np.fromfile(f"{G}/pop2025.f32", dtype=np.float32).reshape(NR, NC)[r0:r1, c0:c1].ravel()[valid]
urban = np.fromfile(f"{G}/urban.u8", dtype=np.uint8).reshape(NR, NC)[r0:r1, c0:c1].ravel()[valid]
pb = np.bincount(inv, weights=pop, minlength=len(codes))
ub = np.bincount(inv, weights=urban.astype(float), minlength=len(codes))

acc = {"pop": pd.Series(pb, index=codes), "urban_share": pd.Series(ub / np.where(pb == 0, np.nan, pb), index=codes)}
for name, f in LAYERS.items():
    t = np.fromfile(f"{G}/{f}", dtype=np.float32).reshape(NR, NC)[r0:r1, c0:c1].ravel()[valid]
    a = np.where(t <= 30, pop, 0.0)
    ab = np.bincount(inv, weights=a, minlength=len(codes))
    acc[name] = pd.Series(ab / np.where(pb == 0, np.nan, pb), index=codes)
A = pd.DataFrame(acc); A.index.name = "m6"

# outcome
sih = pd.read_parquet("__SIH_PARQUET_ROOT__/sih_2024_cholec.parquet",
                      columns=["MUNIC_RES", "MUNIC_MOV"])
sih["r"] = pd.to_numeric(sih.MUNIC_RES, errors="coerce"); sih["m"] = pd.to_numeric(sih.MUNIC_MOV, errors="coerce")
sih = sih.dropna()
sih["local"] = (sih.r == sih.m)
agg = sih.groupby(sih.r.astype(np.int64)).agg(n=("local", "size"), local=("local", "sum"))
agg["outflow"] = 1 - agg.local / agg.n
agg.index.name = "m6"

df = A.join(agg, how="inner")
df = df[df.n >= 10].copy()
df["has_local"] = (df.outflow < 1).astype(int)
mm = pd.read_csv(f"{SD}/01_data/interim/municipios_master.csv")
mm["m6"] = pd.to_numeric(mm.codigo_ibge.astype(str).str[:6], errors="coerce")
uf = pd.to_numeric(mm.codigo_uf, errors="coerce")
def to_region(u):
    if 11 <= u <= 17: return "N"
    if 21 <= u <= 29: return "NE"
    if 31 <= u <= 35: return "SE"
    if 41 <= u <= 43: return "S"
    if 50 <= u <= 53: return "CO"
    return None
mm["region"] = uf.map(to_region)
df = df.merge(mm[["m6", "region"]].drop_duplicates("m6"), on="m6", how="left")
print("municipalities:", len(df), "with local surgery:", int(df.has_local.sum()))

res = {}
print("\n=== AUC by layer (all municipalities) ===")
for name in LAYERS:
    d = df.dropna(subset=[name])
    res[name] = {"auc": float(roc_auc_score(d.has_local, d[name])),
                 "rho": float(spearmanr(d[name], d.outflow).statistic), "n": int(len(d))}
    print(f"  {name:14s} AUC={res[name]['auc']:.3f}  rho={res[name]['rho']:+.3f}")

print("\n=== spatial holdout: AUC by macro-region ===")
tab = {}
for r in sorted(df.region.dropna().unique()):
    d = df[(df.region == r)].dropna(subset=["D1_osm", "D2_act_pre"])
    if len(d) < 80:
        continue
    row = {"n": len(d)}
    for name in ("D1_osm", "D2_act_pre"):
        if d[name].nunique() > 1 and d.has_local.nunique() > 1:
            row[name] = float(roc_auc_score(d.has_local, d[name]))
    tab[r] = row
print(pd.DataFrame(tab).T.round(3).to_string())

print("\n=== incremental validity (logistic, 5-fold CV) ===")
from sklearn.model_selection import cross_val_predict
d = df.dropna(subset=["D1_osm", "D2_act_pre", "pop", "urban_share"]).copy()
d["logpop"] = np.log1p(d["pop"])
y = d.has_local.values
base = d[["logpop", "urban_share"]].values
add_osm = d[["logpop", "urban_share", "D1_osm"]].values
add_pre = d[["logpop", "urban_share", "D2_act_pre"]].values
out = {}
for name, X in (("baseline", base), ("baseline+OSM", add_osm), ("baseline+2017-19 activity", add_pre)):
    p = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=5, method="predict_proba")[:, 1]
    out[name] = float(roc_auc_score(y, p))
    print(f"  {name:28s} AUC={out[name]:.3f}")

res["holdout_by_region"] = tab
res["incremental"] = out
json.dump(res, open(f"{BASE}/work/flow_holdout.json", "w"), indent=1, default=float)
df.to_csv(f"{BASE}/work/flow_holdout.csv")
print("\nHOLDOUT_DONE")
