#!/usr/bin/env python
"""
23_flow_validation.py — does 30-minute access predict where surgery actually happens?

For each Brazilian municipality:
  access_30  = population-weighted share of residents within 30 min of a facility
  outflow    = 1 - (cholecystectomies performed in the municipality of residence)
                   / (all cholecystectomies among residents)

If access_30 has external validity, low-access municipalities should show higher
outflow. We test this for four facility layers, to see whether the *definition*
matters for predictive validity, not just for the coverage number.
"""
import json, os
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
import geopandas as gpd

BASE = "__PROJECT_ROOT__"
G = f"{BASE}/work/grid"
SD = "__BRAZIL_SOURCE_ROOT__"
NR, NC = 17400, 43200
PIX = 1.0 / 120.0

LAYERS = {
    "D1_osm": "tt_D1.f32",
    "D2_surg": "tt_D2meas.f32",
    "D2corr": "tt_D2meas_corr.f32",
    "D2act": "tt_BRA_act.f32",
    "D2act2021": "tt_BRA_act2021.f32",
}

shp = f"{SD}/river/data/official/ibge_malha/extracted/BR_Municipios_2022.shp"
g = gpd.read_file(shp)
code_col = [c for c in g.columns if c.upper().startswith("CD_MUN")][0]
g["m6"] = g[code_col].astype(str).str[:6].astype(np.int64)
print("municipalities:", len(g), "code col:", code_col)

minx, miny, maxx, maxy = g.total_bounds
c0 = int(np.floor((minx + 180) / PIX)); c1 = int(np.ceil((maxx + 180) / PIX))
r0 = int(np.floor((85 - maxy) / PIX)); r1 = int(np.ceil((85 - miny) / PIX))
W, H = c1 - c0, r1 - r0
print(f"window rows {r0}-{r1} cols {c0}-{c1} = {H}x{W}")

transform = rasterio.transform.from_origin(-180 + c0 * PIX, 85 - r0 * PIX, PIX, PIX)
muni = features.rasterize(
    ((geom, int(m)) for geom, m in zip(g.geometry, g.m6)),
    out_shape=(H, W), transform=transform, fill=0, dtype="int64")
print("rasterised; unique municipalities:", len(np.unique(muni)) - 1)

pop = np.fromfile(f"{G}/pop2025.f32", dtype=np.float32).reshape(NR, NC)[r0:r1, c0:c1]
flat = muni.ravel()
valid = flat > 0
idx = flat[valid]
pv = pop.ravel()[valid]

# map municipality code -> compact index
codes, inv = np.unique(idx, return_inverse=True)
print("municipalities with cells:", len(codes))

rows = {}
for name, f in LAYERS.items():
    t = np.fromfile(f"{G}/{f}", dtype=np.float32).reshape(NR, NC)[r0:r1, c0:c1].ravel()[valid]
    acc = np.where(t <= 30, pv, 0.0)
    pb = np.bincount(inv, weights=pv, minlength=len(codes))
    ab = np.bincount(inv, weights=acc, minlength=len(codes))
    rows[name] = pd.Series(ab / np.where(pb == 0, np.nan, pb), index=codes)
acc_df = pd.DataFrame(rows)
acc_df["pop"] = pd.Series(pb, index=codes)
acc_df.index.name = "m6"

# ---- SIH outflow ----
sih = pd.read_parquet("__SIH_PARQUET_ROOT__/sih_2024_cholec.parquet",
                      columns=["MUNIC_RES", "MUNIC_MOV"])
sih["MUNIC_RES"] = pd.to_numeric(sih.MUNIC_RES, errors="coerce")
sih["MUNIC_MOV"] = pd.to_numeric(sih.MUNIC_MOV, errors="coerce")
sih = sih.dropna()
sih["m6"] = sih.MUNIC_RES.astype(np.int64)          # SIH already uses the 6-digit code
sih["m6_mov"] = sih.MUNIC_MOV.astype(np.int64)
sih["local"] = (sih.m6 == sih.m6_mov)
agg = sih.groupby("m6").agg(n=("local", "size"), local=("local", "sum"))
agg["outflow"] = 1 - agg.local / agg.n
print("SIH municipalities:", len(agg), "cases:", int(agg.n.sum()))

df = acc_df.join(agg, how="inner")
df = df[df.n >= 10]
print("municipalities with >=10 cases:", len(df))

res = {}
from sklearn.metrics import roc_auc_score
for name in LAYERS:
    d = df.dropna(subset=[name, "outflow"])
    if len(d) < 50:
        continue
    # AUC: does low access predict above-median outflow?
    y = (d.outflow > d.outflow.median()).astype(int)
    auc = roc_auc_score(y, -d[name])          # lower access -> higher outflow
    from scipy.stats import spearmanr
    rho = spearmanr(d[name], d.outflow).statistic
    # outflow by access decile
    d = d.copy()
    d["bin"] = pd.qcut(d[name], 5, labels=False, duplicates="drop")
    b = d.groupby("bin").agg(acc=(name, "mean"), out=("outflow", "mean"), n=("n", "sum"))
    res[name] = {"auc": float(auc), "spearman": float(rho), "n": int(len(d)),
                 "bins": b.round(4).to_dict("index")}
    print(f"\n{name}: AUC={auc:.3f}  rho={rho:.3f}  n={len(d)}")
    print(b.round(3).to_string())

json.dump(res, open(f"{BASE}/work/flow_validation.json", "w"), indent=1, default=float)
df.to_csv(f"{BASE}/work/flow_validation.csv")
print("\nFLOW_DONE")
