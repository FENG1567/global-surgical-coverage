#!/usr/bin/env python
"""
47_spatial_auc.py 鈥?spatial-block validation with uncertainty for the AUC increment.

The point estimate (baseline 0.853 -> +OSM 0.854 -> +activity 0.870) came from
random 5-fold CV, which leaks spatial structure: neighbouring municipalities
share facilities, referral patterns and unobserved determinants. Here we

  1. assign every municipality to a 2-degree spatial block,
  2. repeatedly partition the blocks into 5 folds (50 random partitions),
  3. fit and predict out-of-fold for three nested models,
  4. report the distribution of AUC and of the AUC *difference* across
     partitions, which is the spatial uncertainty of the increment.
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BASE = "__PROJECT_ROOT__"
W = f"{BASE}/work"

df = pd.read_csv(f"{W}/flow_holdout.csv")
print("rows", len(df), "cols", [c for c in df.columns][:14])

# spatial block: 2 degree cells from the municipality coordinates
cent = pd.read_csv("__BRAZIL_SOURCE_ROOT__/01_data/interim/municipios_master.csv")
cent["m6"] = pd.to_numeric(cent.codigo_ibge.astype(str).str[:6], errors="coerce")
cent = cent[["m6", "latitude", "longitude"]].dropna().drop_duplicates("m6")
df = df.merge(cent, on="m6", how="left")
df = df.dropna(subset=["latitude", "longitude", "D1_osm", "D2_act_pre", "pop", "urban_share"])
df["logpop"] = np.log1p(df["pop"])
df["blk"] = (np.floor(df.latitude / 2).astype(int) * 1000 +
             np.floor(df.longitude / 2).astype(int))
print("blocks:", df.blk.nunique(), " municipalities:", len(df))

y = df.has_local.values
X = {
    "baseline": df[["logpop", "urban_share"]].values,
    "baseline+OSM": df[["logpop", "urban_share", "D1_osm"]].values,
    "baseline+activity": df[["logpop", "urban_share", "D2_act_pre"]].values,
}
blocks = df.blk.values
ub = np.unique(blocks)
rng = np.random.default_rng(20261111)

NREP = 50
scores = {k: [] for k in X}
for rep in range(NREP):
    perm = rng.permutation(ub)
    fold_of_block = {b: i % 5 for i, b in enumerate(perm)}
    fold = np.array([fold_of_block[b] for b in blocks])
    oof = {k: np.zeros(len(y)) for k in X}
    for f in range(5):
        tr = fold != f; te = fold == f
        if te.sum() < 50 or len(np.unique(y[tr])) < 2:
            continue
        for k, Xm in X.items():
            m = LogisticRegression(max_iter=2000)
            m.fit(Xm[tr], y[tr])
            oof[k][te] = m.predict_proba(Xm[te])[:, 1]
    for k in X:
        if len(np.unique(y)) == 2:
            scores[k].append(roc_auc_score(y, oof[k]))

res = {}
print(f"\nspatial-block CV, {NREP} random block partitions:")
for k, v in scores.items():
    v = np.array(v)
    res[k] = {"mean": float(v.mean()), "lo": float(np.percentile(v, 2.5)),
              "hi": float(np.percentile(v, 97.5)), "n": int(len(v))}
    print(f"  {k:20s} AUC = {v.mean():.3f}  [{np.percentile(v,2.5):.3f}, {np.percentile(v,97.5):.3f}]")

d_osm = np.array(scores["baseline+OSM"]) - np.array(scores["baseline"])
d_act = np.array(scores["baseline+activity"]) - np.array(scores["baseline"])
d_act_osm = np.array(scores["baseline+activity"]) - np.array(scores["baseline+OSM"])
for nm, d in (("+OSM - baseline", d_osm), ("+activity - baseline", d_act),
              ("+activity - +OSM", d_act_osm)):
    print(f"  delta {nm:22s} {d.mean():+.4f}  [{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}]  "
          f"P(<=0)={float((d<=0).mean()):.2f}")
    res[nm] = {"mean": float(d.mean()), "lo": float(np.percentile(d, 2.5)),
               "hi": float(np.percentile(d, 97.5)), "p_le0": float((d <= 0).mean())}

json.dump(res, open(f"{W}/spatial_auc.json", "w"), indent=1)
print("\nSPATIAL_AUC_DONE")
