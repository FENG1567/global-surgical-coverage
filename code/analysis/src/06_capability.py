#!/usr/bin/env python
"""
06_capability.py — build hospital_capability.csv and the D2 scenario masks.

There is no global facility-level registry of operating theatres. Direct
evidence (OSM surgery tags, national registries, observed surgical volume)
covers a tiny share of OSM hospitals, so the global D2 estimate must be a
scenario. We bracket it:

  D2_urban : within each country keep the p_c fraction of hospitals with the
             highest local population density (surgical capacity concentrates
             in urban referral hospitals)  -> LOWER bound on access
  D2_rand  : keep a random p_c fraction (surgical capability independent of
             location)                      -> UPPER bound on access

p_c by World Bank income group is an explicit assumption, anchored where a
national measurement exists (Brazil).
"""
import json, os
import numpy as np
import pandas as pd

BASE = "__PROJECT_ROOT__"
G = f"{BASE}/work/grid"
NR, NC = 17400, 43200
PIX = 1.0 / 120.0

# explicit priors: share of hospitals able to provide surgery
P_C = {"HIC": 0.85, "UMC": 0.60, "LMC": 0.35, "LIC": 0.20, "NA": 0.35}
# low/high sensitivity band around the prior
P_LO = {"HIC": 0.70, "UMC": 0.45, "LMC": 0.22, "LIC": 0.10, "NA": 0.22}
P_HI = {"HIC": 0.95, "UMC": 0.75, "LMC": 0.50, "LIC": 0.35, "NA": 0.50}


def main():
    h = pd.read_parquet(f"{BASE}/data/osm/osm_hospitals_all.parquet")
    pop = np.fromfile(f"{G}/pop2025.f32", dtype=np.float32).reshape(NR, NC)
    cdf = pd.read_csv(f"{G}/countries.csv")
    inc = dict(zip(cdf.iso3, cdf.income))
    nm = dict(zip(cdf.iso3, cdf.name))

    # local population density around each hospital (3x3 cells, ~3 km)
    row = np.floor((85.0 - h.lat.values) / PIX).astype(np.int64)
    col = np.floor((h.lon.values + 180.0) / PIX).astype(np.int64)
    ok = (row >= 1) & (row < NR - 1) & (col >= 1) & (col < NC - 1)
    dens = np.zeros(len(h), dtype=np.float64)
    r, c = row[ok], col[ok]
    dens[ok] = (pop[r - 1, c - 1] + pop[r - 1, c] + pop[r - 1, c + 1] +
                pop[r, c - 1] + pop[r, c] + pop[r, c + 1] +
                pop[r + 1, c - 1] + pop[r + 1, c] + pop[r + 1, c + 1])
    # fallback for edge cells
    h["local_pop"] = dens
    h["income"] = h.iso3.map(inc).fillna("NA")
    h["p_c"] = h.income.map(P_C)
    h["p_c_low"] = h.income.map(P_LO)
    h["p_c_high"] = h.income.map(P_HI)

    # ---- capability table ----
    h["has_surgery_tag"] = h.healthcare_speciality.fillna("").str.contains(
        "surgery|surgical", case=False, regex=True)
    h["has_emergency"] = h.healthcare_speciality.fillna("").str.contains(
        "emergency", case=False, regex=True)
    cap = h.groupby("iso3").agg(
        n_hospitals=("id", "size"),
        n_surgery_tag=("has_surgery_tag", "sum"),
        n_emergency_tag=("has_emergency", "sum"),
    ).reset_index()
    cap["income"] = cap.iso3.map(inc).fillna("NA")
    cap["country"] = cap.iso3.map(nm).fillna(cap.iso3)
    cap["p_c_assumed"] = cap.income.map(P_C)
    cap["p_c_low"] = cap.income.map(P_LO)
    cap["p_c_high"] = cap.income.map(P_HI)
    cap["direct_evidence_share"] = cap.n_surgery_tag / cap.n_hospitals
    cap["evidence_source"] = "income-group prior (no national registry)"
    # Brazil: measured from CNES
    bz = pd.read_csv("__BRAZIL_SOURCE_ROOT__/03_outputs/hospital_points.csv")
    bz = bz[bz.coord_ok == True]
    cap.loc[cap.iso3 == "BRA", "p_c_assumed"] = float(bz.has_surg_center.mean())
    cap.loc[cap.iso3 == "BRA", "p_c_low"] = float(bz.cap_chole_behavior.mean())
    cap.loc[cap.iso3 == "BRA", "p_c_high"] = 1.0
    cap.loc[cap.iso3 == "BRA", "evidence_source"] = "CNES 2025 measured (structural; low=behavioural chole)"
    cap = cap.sort_values("n_hospitals", ascending=False)
    cap.to_csv(f"{BASE}/outputs/hospital_capability.csv", index=False)
    print("hospital_capability.csv rows:", len(cap))
    print(cap.head(10)[["iso3", "n_hospitals", "income", "p_c_assumed", "direct_evidence_share"]].to_string())

    # ---- D2 masks ----
    rng = np.random.default_rng(20261108)
    for tag, mode, pcol in (("D2urban", "urban", "p_c"),
                            ("D2rand", "rand", "p_c"),
                            ("D2urban_lo", "urban", "p_c_low"),
                            ("D2urban_hi", "urban", "p_c_high")):
        keep = np.zeros(len(h), dtype=bool)
        for iso, g in h.groupby("iso3"):
            k = max(1, int(round(len(g) * float(g[pcol].iloc[0]))))
            if mode == "urban":
                sel = g.nlargest(k, "local_pop").index
            else:
                sel = rng.choice(g.index.values, size=k, replace=False)
            keep[sel] = True
        m = np.zeros(NR * NC, dtype=np.uint8)
        idx = row[keep] * NC + col[keep]
        m[idx] = 1
        m.tofile(f"{G}/src_{tag}.u8")
        print(f"src_{tag}.u8: hospitals kept {int(keep.sum())} / {len(h)} "
              f"({keep.mean()*100:.1f}%), cells {int(m.sum())}")

    print("CAP_DONE")


if __name__ == "__main__":
    main()
