#!/usr/bin/env python
"""
11_sara_pc.py — measured p_c from SARA/HHFA reports + revised D2 scenario.

p_c = share of *designated hospitals* that offer comprehensive/basic surgical
services, read from the national SARA/HHFA reports (data/who/sara/).

Every value below is quoted from a specific table/figure; see sara_p_c.csv
for the page and the exact wording. This is a facility-stratum statistic, NOT
a share of OSM `amenity=hospital` (OSM over-includes lower-level facilities).
"""
import json, os
import numpy as np
import pandas as pd

BASE = "__PROJECT_ROOT__"
G = f"{BASE}/work/grid"
NR, NC = 17400, 43200
PIX = 1.0 / 120.0

# ---- measured p_c (hospitals offering surgery) ----
MEASURED = {
    # iso3: (year, p_c, basis, n_hospitals_in_survey, source_page)
    "TZA": (2012, 0.79, "basic surgery, hospitals", 52, "Table 4.17.1, p.68"),
    "ETH": (2018, 0.97, "comprehensive surgery, hospitals", 303, "Figure 3.10.2.1, p.83"),
    "KEN": (2018, 0.68, "comprehensive surgery, hospitals", 411, "Figure 178, p.198"),
    "GHA": (2023, 0.85, "major surgical procedures, hospitals", None, "text p.~90"),
    "MLI": (2018, 0.58, "comprehensive surgery, hospitals", None, "text IV.1, p.164"),
    "ZWE": (2015, 0.80, "comprehensive surgery, hospitals", 91, "Figure 59, p.127"),
    "COD": (2014, 0.98, "basic surgery, hospitals", None, "Tableau 21, p.137"),
    "NER": (2015, 0.90, "comprehensive surgery, hospitals", None, "text, chirurgie complete"),
    "MOZ": (2018, 0.93, "comprehensive surgery, hospitals", 68, "text p.~118, N=68"),
    "UGA": (2023, 0.92, "essential surgical procedures, hospitals", 187, "Table 13, facility level"),
    "TCD": (2015, 0.794, "comprehensive surgery, hospitals", 97, "Graphique 47, p.119"),
    "BFA": (2016, 0.65, "comprehensive surgery, hospitals", 17, "section 8.1.1, p.162"),
}
# revised priors for countries without a measured value (anchored on the
# measured range 0.58-0.97; median of the six 'comprehensive' values = 0.825)
P_C = {"HIC": 0.95, "UMC": 0.88, "LMC": 0.80, "LIC": 0.75, "NA": 0.80}
P_LO = {"HIC": 0.85, "UMC": 0.70, "LMC": 0.58, "LIC": 0.50, "NA": 0.58}
P_HI = {"HIC": 0.99, "UMC": 0.95, "LMC": 0.92, "LIC": 0.90, "NA": 0.92}


def main():
    h = pd.read_parquet(f"{BASE}/data/osm/osm_hospitals_all.parquet")
    pop = np.fromfile(f"{G}/pop2025.f32", dtype=np.float32).reshape(NR, NC)
    cdf = pd.read_csv(f"{G}/countries.csv")
    inc = dict(zip(cdf.iso3, cdf.income))
    nm = dict(zip(cdf.iso3, cdf.name))

    row = np.floor((85.0 - h.lat.values) / PIX).astype(np.int64)
    col = np.floor((h.lon.values + 180.0) / PIX).astype(np.int64)
    ok = (row >= 1) & (row < NR - 1) & (col >= 1) & (col < NC - 1)
    dens = np.zeros(len(h))
    r, c = row[ok], col[ok]
    dens[ok] = (pop[r - 1, c - 1] + pop[r - 1, c] + pop[r - 1, c + 1] +
                pop[r, c - 1] + pop[r, c] + pop[r, c + 1] +
                pop[r + 1, c - 1] + pop[r + 1, c] + pop[r + 1, c + 1])
    h["local_pop"] = dens
    h["income"] = h.iso3.map(inc).fillna("NA")
    h["p_c"] = [MEASURED[i][1] if i in MEASURED else P_C[inc.get(i, "NA")] for i in h.iso3]
    h["p_c_lo"] = [MEASURED[i][1] if i in MEASURED else P_LO[inc.get(i, "NA")] for i in h.iso3]
    h["p_c_hi"] = [MEASURED[i][1] if i in MEASURED else P_HI[inc.get(i, "NA")] for i in h.iso3]
    h["p_c_source"] = ["measured" if i in MEASURED else "prior" for i in h.iso3]

    # ---- table ----
    rows = []
    for iso, g in h.groupby("iso3"):
        rows.append({"iso3": iso, "country": nm.get(iso, iso), "income": g.income.iloc[0],
                     "n_osm_hospitals": len(g), "p_c": g.p_c.iloc[0],
                     "p_c_lo": g.p_c_lo.iloc[0], "p_c_hi": g.p_c_hi.iloc[0],
                     "p_c_source": g.p_c_source.iloc[0],
                     "survey_year": MEASURED[iso][0] if iso in MEASURED else None,
                     "basis": MEASURED[iso][2] if iso in MEASURED else None,
                     "n_survey_hospitals": MEASURED[iso][3] if iso in MEASURED else None,
                     "source": MEASURED[iso][4] if iso in MEASURED else None})
    cap = pd.DataFrame(rows).sort_values("n_osm_hospitals", ascending=False)
    cap.to_csv(f"{BASE}/outputs/hospital_capability.csv", index=False)
    cap[cap.p_c_source == "measured"].to_csv(f"{BASE}/outputs/sara_p_c_measured.csv", index=False)
    print(cap[cap.p_c_source == "measured"][["iso3", "income", "n_osm_hospitals", "p_c",
                                             "basis", "n_survey_hospitals"]].to_string(index=False))
    print(f"\nmeasured countries: {(cap.p_c_source=='measured').sum()} / {len(cap)}")

    # ---- masks ----
    rng = np.random.default_rng(20261109)
    for tag, mode, pcol in (("D2meas", "urban", "p_c"),
                            ("D2meas_lo", "urban", "p_c_lo"),
                            ("D2meas_hi", "urban", "p_c_hi"),
                            ("D2meas_rand", "rand", "p_c")):
        keep = np.zeros(len(h), dtype=bool)
        for iso, g in h.groupby("iso3"):
            k = max(1, int(round(len(g) * float(g[pcol].iloc[0]))))
            if mode == "urban":
                sel = g.nlargest(k, "local_pop").index
            else:
                sel = rng.choice(g.index.values, size=k, replace=False)
            keep[sel] = True
        m = np.zeros(NR * NC, dtype=np.uint8)
        m[row[keep] * NC + col[keep]] = 1
        m.tofile(f"{G}/src_{tag}.u8")
        print(f"src_{tag}.u8: hospitals {int(keep.sum())}/{len(h)} ({keep.mean()*100:.1f}%), cells {int(m.sum())}")
    print("SARA_PC_DONE")


if __name__ == "__main__":
    main()
