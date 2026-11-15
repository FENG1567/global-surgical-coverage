#!/usr/bin/env python
"""
25_workforce_and_ranking.py — two additions requested by the external review.

A. Workforce constraint: does the "~2,045 surgical facilities" implied by the
   2015 5-billion figure survive an independent check against the global
   surgical workforce?
B. Priority ranking: does the facility definition change WHICH places are
   ranked as priorities (not just the coverage level)?
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = "__PROJECT_ROOT__"
G = f"{BASE}/work/grid"

# ---------------- A. workforce ----------------
wf = pd.read_csv(f"{BASE}/data/wb/surgical_workforce_latest.csv")
cdf = pd.read_csv(f"{G}/countries.csv")

# population by country (2025) from the earlier aggregation
acc = pd.read_csv(f"{BASE}/work/v12_by_country.csv")
pop = acc[(acc.year == 2025) & (acc.thr == "2h") & (acc.defn == "D1")][["iso3", "pop"]]
pop = pop.rename(columns={"pop": "pop25"})

w = wf.merge(pop, on="iso3", how="inner")
w["surgeons"] = w.density / 1e5 * w.pop25
tot_surg = float(w.surgeons.sum())
print(f"countries with workforce + population: {len(w)}")
print(f"implied global surgical workforce: {tot_surg/1e6:.2f} million")
print(f"population covered: {w.pop25.sum()/1e9:.2f} billion")

cap = pd.read_csv(f"{BASE}/outputs/hospital_capability.csv")
n_osm = int(cap.n_osm_hospitals.sum())
n_meas = float((cap.n_osm_hospitals * cap.p_c).sum())
print(f"OSM hospitals: {n_osm:,};  p_c-weighted surgical facilities: {n_meas:,.0f}")

implied = 2045
print(f"\n5-billion implies ~{implied:,} surgical facilities")
print(f"  surgeons per facility at that count: {tot_surg/implied:,.0f}")
print(f"  surgeons per facility if every OSM hospital could operate: {tot_surg/n_osm:,.1f}")
print(f"  surgeons per facility at p_c-weighted count: {tot_surg/n_meas:,.1f}")

# how many facilities can the workforce plausibly staff?  (k surgeons per facility)
for k in (2, 4, 8):
    print(f"  max facilities at {k} surgeons/facility: {tot_surg/k:,.0f}")

# ---------------- B. ranking stability ----------------
print("\n=== country ranking under different definitions (30 min) ===")
v15 = pd.read_csv(f"{BASE}/work/v15_by_country.csv", index_col=0)
v15.index.name = "idx"
v15 = v15.reset_index()
v15 = v15.merge(cdf[["idx", "iso3"]], on="idx", how="left")
v15 = v15.merge(pop, on="iso3", how="left")
big = v15[v15.pop25 > 1e6].dropna(subset=["D1", "D2corr"]).copy()
print(f"countries with >1M population: {len(big)}")
for a, b in (("D1", "D2corr"), ("D1", "D2"), ("D2", "D2corr")):
    rho = spearmanr(big[a], big[b]).statistic
    print(f"  rho({a}, {b}) = {rho:.3f}")

big["rank_D1"] = big.D1.rank(ascending=False)
big["rank_D2corr"] = big.D2corr.rank(ascending=False)
big["rank_shift"] = big.rank_D2corr - big.rank_D1
print("\nlargest rank shifts (D1 -> D2corr), + means falls down the priority list:")
print(big.reindex(big.rank_shift.abs().sort_values(ascending=False).index)
      .head(12)[["iso3", "income", "D1", "D2corr", "rank_D1", "rank_D2corr", "rank_shift"]]
      .round(3).to_string(index=False))

# top-20 priority (lowest access) overlap
for k in (10, 20, 30):
    a = set(big.nsmallest(k, "D1").iso3)
    b = set(big.nsmallest(k, "D2corr").iso3)
    print(f"  top-{k} worst-access countries: overlap {len(a&b)}/{k}")

# ---------------- Brazil states ----------------
print("\n=== Brazil: state ranking, OSM vs observed activity ===")
fl = pd.read_csv(f"{BASE}/work/flow_validation.csv")
mm = pd.read_csv("__BRAZIL_SOURCE_ROOT__/01_data/interim/municipios_master.csv")
ufcol = [c for c in mm.columns if c.lower() in ("codigo_uf", "uf", "sigla_uf")]
print("municipio master cols:", list(mm.columns)[:12], "uf col:", ufcol)
key = [c for c in mm.columns if c.lower() in ("codigo_ibge", "m6", "codigo_municipio")]
print("key col:", key)
if ufcol and key:
    mm2 = mm[[key[0], ufcol[0]]].rename(columns={key[0]: "m6", ufcol[0]: "uf"})
    mm2["m6"] = pd.to_numeric(mm2.m6.astype(str).str[:6], errors="coerce")
    fl = fl.merge(mm2, on="m6", how="left")
    g2 = fl.dropna(subset=["uf", "D1_osm", "D2act2021"]).copy()
    g2["w_osm"] = g2.D1_osm * g2["n"]
    g2["w_act"] = g2.D2act2021 * g2["n"]
    st = g2.groupby("uf").agg(n=("n", "sum"), w_osm=("w_osm", "sum"), w_act=("w_act", "sum"))
    st["acc_osm"] = st.w_osm / st["n"]
    st["acc_act"] = st.w_act / st["n"]
    st["delta"] = st.acc_act - st.acc_osm
    st["rank_osm"] = st.acc_osm.rank(ascending=False)
    st["rank_act"] = st.acc_act.rank(ascending=False)
    st["shift"] = st.rank_act - st.rank_osm
    print(st.round(3).sort_values("delta").to_string())
    print(f"\n  rho(state ranking) = {spearmanr(st.acc_osm, st.acc_act).statistic:.3f}")
    st.to_csv(f"{BASE}/work/brazil_state_ranking.csv")

json.dump({"total_surgeons": tot_surg, "n_osm": n_osm, "n_pc": n_meas,
           "implied_2045_surgeons_each": tot_surg / implied},
          open(f"{BASE}/work/workforce_check.json", "w"), indent=1)
print("\nWF_RANK_DONE")
