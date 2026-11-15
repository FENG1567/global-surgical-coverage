#!/usr/bin/env python
"""
29_framework_analytic.py — the analytic core of the identifiability paper.

(a) A two-centre construction with a closed-form identified set: shows that the
    width of the set is governed by how unequally facilities are distributed
    across population centres, not by p_c.
(b) Per-country sensitivity of coverage to the urban share u, from the real
    surfaces: which countries' estimates are most fragile.
"""
import json
import numpy as np
import pandas as pd

BASE = "__PROJECT_ROOT__"
G = f"{BASE}/work/grid"
NR, NC = 17400, 43200

# ---------------- (a) two-centre construction ----------------
# Two population centres A and B, equal population, farther apart than the
# catchment radius. n_A facilities in A, n_B in B (n_A > n_B). k = p_c*(n_A+n_B).
# If all k capable facilities are in A: A covered, B not -> C = 1/2.
# If k > n_A: A is saturated, the surplus must go to B -> C = 1.
def two_centre(nA, nB, pc):
    k = round(pc * (nA + nB))
    # worst case: put as many as possible in A (A saturates at nA, rest spill to B)
    inA = min(k, nA)
    inB = k - inA
    C_worst = 0.5 * (1 if inA > 0 else 0) + 0.5 * (1 if inB > 0 else 0)
    # best case: spread so that both are covered
    C_best = 0.5 * (1 if k > 0 else 0) + 0.5 * (1 if k > 1 else 0)
    # more informative: coverage fraction given catchment capacity
    return {"nA": nA, "nB": nB, "pc": pc, "k": k,
            "C_min": C_worst, "C_max": C_best, "width": C_best - C_worst}


rows = []
for nA, nB in ((100, 10), (60, 50), (50, 50)):
    for pc in (0.2, 0.5, 0.8):
        rows.append(two_centre(nA, nB, pc))
syn = pd.DataFrame(rows)
print("=== two-centre construction (equal populations, separated beyond radius) ===")
print(syn.to_string(index=False))

# ---------------- (b) per-country sensitivity to u ----------------
ctry = np.fromfile(f"{G}/country.i16", dtype=np.int16).reshape(NR, NC).ravel()
cdf = pd.read_csv(f"{G}/countries.csv")
NCY = int(cdf.idx.max()) + 1
pop = np.fromfile(f"{G}/pop2025.f32", dtype=np.float32).reshape(NR, NC).ravel()
base = (ctry >= 0) & (pop > 0)
idx = ctry[base].astype(np.int64)
pv = pop[base]
pb = np.bincount(idx, weights=pv, minlength=NCY)

res = {}
for tag in ("u000", "u100"):
    t = np.fromfile(f"{G}/tt_D2u_{tag}.f32", dtype=np.float32).reshape(NR, NC).ravel()[base]
    a = np.where(t <= 30, pv, 0.0)
    ab = np.bincount(idx, weights=a, minlength=NCY)
    res[tag] = pd.Series(ab / np.where(pb == 0, np.nan, pb), index=range(NCY))

d = pd.DataFrame({"C_rural": res["u000"], "C_urban": res["u100"], "pop": pb})
d = d.join(cdf.set_index("idx")[["iso3", "income"]])
d["width"] = d.C_rural - d.C_urban
d = d[d["pop"] > 1e6].dropna(subset=["width"])
print(f"\n=== per-country width of the identified set (u=0 vs u=1), n={len(d)} ===")
print(f"  median width: {d.width.median()*100:.2f} pp")
print(f"  p90: {d.width.quantile(.9)*100:.2f} pp   max: {d.width.max()*100:.2f} pp")
print("\n  most fragile countries:")
print(d.nlargest(10, "width")[["iso3", "income", "C_rural", "C_urban", "width"]].round(3).to_string(index=False))
print("\n  least fragile:")
print(d.nsmallest(5, "width")[["iso3", "income", "C_rural", "C_urban", "width"]].round(3).to_string(index=False))

d.to_csv(f"{BASE}/work/identified_set_width.csv")
syn.to_csv(f"{BASE}/work/two_centre.csv", index=False)
json.dump({"median_width_pp": float(d.width.median() * 100),
           "p90_width_pp": float(d.width.quantile(.9) * 100),
           "max_width_pp": float(d.width.max() * 100),
           "n_countries": int(len(d))},
          open(f"{BASE}/work/identified_set.json", "w"), indent=1)
print("\nANALYTIC_DONE")
