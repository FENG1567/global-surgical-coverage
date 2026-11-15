#!/usr/bin/env python
"""12_aggregate_meas.py — aggregate the measured-p_c D2 surfaces and update the ladder."""
import json, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = "__PROJECT_ROOT__"
G = f"{BASE}/work/grid"
NR, NC = 17400, 43200

ctry = np.fromfile(f"{G}/country.i16", dtype=np.int16).reshape(NR, NC).ravel()
cdf = pd.read_csv(f"{G}/countries.csv")
NCY = int(cdf.idx.max()) + 1
urban = np.fromfile(f"{G}/urban.u8", dtype=np.uint8).reshape(NR, NC).ravel().astype(bool)
rural = np.fromfile(f"{G}/rural.u8", dtype=np.uint8).reshape(NR, NC).ravel().astype(bool)

NEW = [("D2meas", "tt_D2meas.f32"), ("D2meas_lo", "tt_D2meas_lo.f32"),
       ("D2meas_hi", "tt_D2meas_hi.f32"), ("D2meas_rand", "tt_D2meas_rand.f32")]

pop = np.fromfile(f"{G}/pop2025.f32", dtype=np.float32).reshape(NR, NC).ravel()
base = (ctry >= 0) & (pop > 0)
idx = ctry[base].astype(np.int64)
pv = pop[base]
u = urban[base]; r = rural[base]

rows = []
for key, f in NEW:
    tt = np.fromfile(f"{G}/{f}", dtype=np.float32).reshape(NR, NC).ravel()
    t = tt[base]
    del tt
    for thr, tag in ((120, "2h"), (180, "3h")):
        av = np.where(t <= thr, pv, 0.0)
        rows.append({"defn": key, "year": 2025, "thr": tag, "stratum": "all",
                     "pop": float(pv.sum()), "acc": float(av.sum()), "share": float(av.sum() / pv.sum())})
        for nm, m in (("urban", u), ("rural", r)):
            ps, ac = float(pv[m].sum()), float(av[m].sum())
            rows.append({"defn": key, "year": 2025, "thr": tag, "stratum": nm,
                         "pop": ps, "acc": ac, "share": ac / ps})
        pb = np.bincount(idx, weights=pv, minlength=NCY)
        ab = np.bincount(idx, weights=av, minlength=NCY)
        for i in np.nonzero(pb > 0)[0]:
            rows.append({"defn": key, "year": 2025, "thr": tag, "stratum": "country",
                         "idx": int(i), "pop": pb[i], "acc": ab[i], "share": ab[i] / pb[i]})

new = pd.DataFrame(rows)
old = pd.read_csv(f"{BASE}/work/v12_ladder.csv")
old = old.rename(columns={"year": "year"})
old_s = old[["defn", "year", "thr", "stratum", "pop", "acc", "share"]].copy()
allad = pd.concat([old_s, new[new.stratum != "country"]], ignore_index=True)
allad.to_csv(f"{BASE}/work/v13_ladder.csv", index=False)

piv = allad[(allad.year == 2025) & (allad.thr == "2h")].pivot_table(
    index="defn", columns="stratum", values="share")
print("=== ladder 2025, 2h ===")
print(piv.round(4).to_string())

# country-level D1 -> D2meas
oldc = pd.read_csv(f"{BASE}/work/v12_by_country.csv")
newc = new[new.stratum == "country"].copy()
newc = newc.merge(cdf[["idx", "iso3", "name", "income"]], on="idx", how="left")
newc["year"] = 2025
newc.to_csv(f"{BASE}/work/v13_by_country_meas.csv", index=False)
p = oldc[(oldc.year == 2025) & (oldc.thr == "2h")].pivot_table(index="iso3", columns="defn", values="share")
q = newc[newc.thr == "2h"].pivot_table(index="iso3", columns="defn", values="share")
m = p.join(q, how="inner")
m["dD1_D2meas"] = m.D1 - m.D2meas
print(f"\nD1 -> D2meas: global share {piv.loc['D1','all']:.4f} -> {piv.loc['D2meas','all']:.4f} "
      f"(delta {(piv.loc['D1','all']-piv.loc['D2meas','all'])*100:.2f} pp)")
print(f"country median delta {m.dD1_D2meas.median()*100:.2f} pp; "
      f"p10 {m.dD1_D2meas.quantile(.1)*100:.2f}; p90 {m.dD1_D2meas.quantile(.9)*100:.2f}")
print(f"countries where delta >= 5 pp: {(m.dD1_D2meas>=0.05).sum()} / {len(m)}")

# income gradient under D2meas
inc = new[new.stratum == "all"].groupby(["defn", "thr"], as_index=False)[["pop", "acc"]].sum()
# need income split: recompute from country rows
cn = newc.groupby(["iso3", "income", "thr"], as_index=False)[["pop", "acc"]].sum()
cn["share"] = cn.acc / cn["pop"]
print("\n=== income, D2meas, 2h ===")
print(cn[cn.thr == "2h"].pivot_table(index="income", columns="defn", values="share").round(4).to_string())

out = {"ladder_2025_2h": {k: {c: float(piv.loc[k, c]) for c in piv.columns} for k in piv.index},
       "delta_D1_D2meas_pp": float((piv.loc["D1", "all"] - piv.loc["D2meas", "all"]) * 100),
       "country_delta_median_pp": float(m.dD1_D2meas.median() * 100),
       "countries_ge5pp": int((m.dD1_D2meas >= 0.05).sum()), "n_countries": int(len(m))}
json.dump(out, open(f"{BASE}/work/v13_numbers.json", "w"), indent=1, default=float)
m.to_csv(f"{BASE}/work/v13_country_effect.csv")
print("\nMEAS_DONE")
