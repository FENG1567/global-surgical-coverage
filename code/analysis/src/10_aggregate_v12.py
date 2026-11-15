#!/usr/bin/env python
"""
10_aggregate_v12.py — v1.2 aggregation: nested ladder, urban/rural, 4-neighbour.
"""
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

SURF = [
    ("D0", "tt_D0.f32"), ("D0u", "tt_D0u.f32"), ("D1", "tt_D1.f32"),
    ("D1_4n", "tt_D1_4n.f32"), ("D2urban", "tt_D2urban.f32"),
    ("D2urban_lo", "tt_D2urban_lo.f32"), ("D2urban_hi", "tt_D2urban_hi.f32"),
    ("D2rand", "tt_D2rand.f32"), ("D2p01", "tt_D2p01.f32"), ("D2p05", "tt_D2p05.f32"),
    ("D0walk", "tt_D0_walk.f32"), ("D1walk", "tt_D1_walk.f32"),
]


def main():
    pop = {y: np.fromfile(f"{G}/pop{y}.f32", dtype=np.float32).reshape(NR, NC).ravel()
           for y in (2020, 2025)}
    base = (ctry >= 0) & (pop[2025] > 0)
    idx_all = ctry[base].astype(np.int64)
    u_all = urban[base]
    r_all = rural[base]

    rows, lad = [], []
    for key, f in SURF:
        p = f"{G}/{f}"
        if not os.path.exists(p):
            print("skip", key); continue
        tt = np.fromfile(p, dtype=np.float32).reshape(NR, NC).ravel()
        t = tt[base]
        del tt
        for y in (2025,):
            pv = pop[y][base]
            for thr, tag in ((120, "2h"), (180, "3h")):
                av = np.where(t <= thr, pv, 0.0)
                # global
                lad.append({"defn": key, "year": y, "thr": tag,
                            "pop": float(pv.sum()), "acc": float(av.sum()),
                            "share": float(av.sum() / pv.sum()), "stratum": "all"})
                # urban / rural
                for nm, m in (("urban", u_all), ("rural", r_all)):
                    ps, ac = float(pv[m].sum()), float(av[m].sum())
                    if ps > 0:
                        lad.append({"defn": key, "year": y, "thr": tag,
                                    "pop": ps, "acc": ac, "share": ac / ps, "stratum": nm})
                # by country
                pb = np.bincount(idx_all, weights=pv, minlength=NCY)
                ab = np.bincount(idx_all, weights=av, minlength=NCY)
                for i in np.nonzero(pb > 0)[0]:
                    rows.append((key, y, tag, int(i), pb[i], ab[i]))
    ldf = pd.DataFrame(lad)
    ldf.to_csv(f"{BASE}/work/v12_ladder.csv", index=False)

    df = pd.DataFrame(rows, columns=["defn", "year", "thr", "idx", "pop", "acc"])
    df = df.merge(cdf[["idx", "iso3", "name", "income"]], on="idx", how="left")
    df["share"] = df["acc"] / df["pop"]
    df.to_csv(f"{BASE}/work/v12_by_country.csv", index=False)

    # ---- nested monotonicity ----
    piv = df[(df.year == 2025) & (df.thr == "2h")].pivot_table(
        index="iso3", columns="defn", values="share")
    mono = ((piv.D0u >= piv.D1 - 1e-9) & (piv.D1 >= piv.D2urban - 1e-9))
    print(f"nested monotonicity: {int(mono.sum())}/{len(piv)} = {mono.mean()*100:.2f}%")

    # ---- 2h vs 3h rank ----
    d = df[(df.defn == "D1") & (df.year == 2025) & (df["pop"] > 1e6)]
    pp = d.pivot_table(index="iso3", columns="thr", values="share").dropna()
    rho = spearmanr(pp["2h"], pp["3h"]).statistic
    print(f"D1 2h vs 3h rank rho = {rho:.4f} (n={len(pp)})")

    # ---- 4n vs 8n ----
    p4 = df[(df.year == 2025) & (df.thr == "2h")].pivot_table(
        index="iso3", columns="defn", values="share")
    diff48 = (p4.D1 - p4.D1_4n).dropna()
    print(f"8n vs 4n: global D1 {p4.D1.mean():.4f} vs D1_4n {p4.D1_4n.mean():.4f}; "
          f"median country diff {diff48.median()*100:.2f} pp")

    print("\n=== v1.2 ladder ===")
    print(ldf[ldf.thr == "2h"].pivot_table(index="defn", columns="stratum",
                                           values="share").round(4).to_string())

    # ---- income ----
    inc = df[(df.year == 2025) & (df.thr == "2h")].groupby(
        ["defn", "income"], as_index=False)[["pop", "acc"]].sum()
    inc["share"] = inc.acc / inc["pop"]
    print("\n=== income, 2h, 2025 ===")
    print(inc.pivot_table(index="income", columns="defn", values="share").round(4).to_string())

    out = {"nested_mono_share": float(mono.mean()),
           "rho_2h_3h_D1": float(rho),
           "d1_8n": float(p4.D1.mean()), "d1_4n": float(p4.D1_4n.mean()),
           "ladder": ldf.to_dict("records")}
    json.dump(out, open(f"{BASE}/work/v12_numbers.json", "w"), indent=1, default=float)
    print("V12_DONE")


if __name__ == "__main__":
    main()
