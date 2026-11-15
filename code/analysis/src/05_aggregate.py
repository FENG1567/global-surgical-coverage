#!/usr/bin/env python
"""
05_aggregate.py — population-weighted access for every facility definition.

Optimised: restrict to cells with population > 0 (a few % of the grid) before
any bincount, and slice the travel-time array once per surface.
"""
import json, os
import numpy as np
import pandas as pd

BASE = "__PROJECT_ROOT__"
G = f"{BASE}/work/grid"
NR, NC = 17400, 43200

ctry = np.fromfile(f"{G}/country.i16", dtype=np.int16).reshape(NR, NC).ravel()
cdf = pd.read_csv(f"{G}/countries.csv")
NCY = int(cdf.idx.max()) + 1
idx_of = dict(zip(cdf.iso3, cdf.idx))

SURFACES = [
    ("D0", "tt_D0.f32", None),
    ("D1", "tt_D1.f32", None),
    ("D2brazil", "tt_D2brazil.f32", "BRA"),
    ("D2urban", "tt_D2urban.f32", None),
    ("D2rand", "tt_D2rand.f32", None),
    ("D2urban_lo", "tt_D2urban_lo.f32", None),
    ("D2urban_hi", "tt_D2urban_hi.f32", None),
    ("D2p01", "tt_D2p01.f32", None),
    ("D2p02", "tt_D2p02.f32", None),
    ("D2p05", "tt_D2p05.f32", None),
    ("D2p10", "tt_D2p10.f32", None),
    ("D0walk", "tt_D0_walk.f32", None),
    ("D1walk", "tt_D1_walk.f32", None),
]


def main():
    pop = {y: np.fromfile(f"{G}/pop{y}.f32", dtype=np.float32).reshape(NR, NC).ravel()
           for y in (2020, 2025)}
    print("GHSL totals:", {k: round(float(v.sum(dtype=np.float64)) / 1e9, 4) for k, v in pop.items()},
          flush=True)
    for y in pop:
        print(f"  pop{y} with country: {float(pop[y][ctry >= 0].sum(dtype=np.float64))/1e9:.4f} B",
              flush=True)

    recs = []
    for key, fname, restrict in SURFACES:
        p = f"{G}/{fname}"
        if not os.path.exists(p):
            print("skip", key, flush=True)
            continue
        tt = np.fromfile(p, dtype=np.float32).reshape(NR, NC).ravel()
        m = (ctry == idx_of[restrict]) if restrict else (ctry >= 0)
        # keep only populated cells
        keep = m & (pop[2025] > 0)
        idx = ctry[keep].astype(np.int64)
        t = tt[keep]
        del tt
        print(f"{key}: populated cells={int(keep.sum())} reachable={int((t<1e8).sum())}", flush=True)
        for y in (2020, 2025):
            pv = pop[y][keep]
            for thr, tag in ((120, "2h"), (180, "3h")):
                av = np.where(t <= thr, pv, 0.0)
                pb = np.bincount(idx, weights=pv, minlength=NCY)
                ab = np.bincount(idx, weights=av, minlength=NCY)
                nz = np.nonzero(pb > 0)[0]
                for i in nz:
                    recs.append((key, y, tag, int(i), pb[i], ab[i], restrict or "ALL"))

    df = pd.DataFrame(recs, columns=["defn", "year", "thr", "idx", "pop", "acc", "restricted_to"])
    df = df.merge(cdf[["idx", "iso3", "name", "income"]], on="idx", how="left")
    df["share"] = df["acc"] / df["pop"]
    df.to_csv(f"{BASE}/work/access_by_country.csv", index=False)

    lad = df.groupby(["defn", "year", "thr", "restricted_to"], as_index=False)[["pop", "acc"]].sum()
    lad["share"] = lad["acc"] / lad["pop"]
    lad.to_csv(f"{BASE}/work/definition_ladder.csv", index=False)

    inc = df[df.restricted_to == "ALL"].groupby(
        ["defn", "year", "thr", "income"], as_index=False)[["pop", "acc"]].sum()
    inc = inc[inc.income != "NA"]
    inc["share"] = inc["acc"] / inc["pop"]
    inc.to_csv(f"{BASE}/work/access_by_income.csv", index=False)

    print("\n=== definition ladder (2025) ===", flush=True)
    print(lad[lad.year == 2025].pivot_table(index="defn", columns="thr", values="share").round(4).to_string())
    print("\n=== ladder 2020 ===", flush=True)
    print(lad[lad.year == 2020].pivot_table(index="defn", columns="thr", values="share").round(4).to_string())
    print("\n=== income, 2025, 2h ===", flush=True)
    print(inc.query("year==2025 and thr=='2h'").pivot_table(index="income", columns="defn",
                                                            values="share").round(4).to_string())
    print("AGG_DONE", flush=True)


if __name__ == "__main__":
    main()
