#!/usr/bin/env python
"""
04_validate.py — G0 pipeline check.

Compares the locally recomputed travel time (D0 = all WorldPop facilities)
against the published WorldPop TT_HC v1 rasters.

  * classified raster (always available): agreement of the 2h/3h classes
  * continuous raster (if downloaded): Spearman rho, median |diff|

Restricted to cells inside the published grid (lat 84..-60) and, for the
continuous comparison, to cells >50 km from an international border
(WorldPop computed travel time per country; the local run is globally
connected, so borders differ by construction).
"""
import json, os
import numpy as np
import rasterio
from scipy import stats
from scipy.ndimage import binary_erosion

BASE = "__PROJECT_ROOT__"
G = f"{BASE}/work/grid"
NR, NC = 17400, 43200
TT = f"{G}/tt_D0.f32"
CLS = f"{BASE}/data/worldpop/global_TT_HC_M_fWS_cls_v1.tif"
TTFULL = f"{BASE}/data/worldpop/global_TT_HC_M_fWS_v1.tif"
POP = f"{G}/pop2025.f32"
CTRY = f"{G}/country.i16"

# published cls grid: top=84.0 -> our row 120
ROW0 = 120
NR_C = 17280


def main():
    tt = np.fromfile(TT, dtype=np.float32).reshape(NR, NC)
    sub = tt[ROW0:ROW0 + NR_C, :]
    pop = np.fromfile(POP, dtype=np.float32).reshape(NR, NC)[ROW0:ROW0 + NR_C, :]
    ctry = np.fromfile(CTRY, dtype=np.int16).reshape(NR, NC)[ROW0:ROW0 + NR_C, :]

    with rasterio.open(CLS) as ds:
        cls = ds.read(1)
    valid = (cls != 8888) & np.isfinite(sub) & (sub < 1e8)
    print(f"cells: total={cls.size} valid={int(valid.sum())} "
          f"({valid.mean()*100:.1f}%)")

    mine_le120 = sub <= 120.0
    pub_le120 = (cls >= 1) & (cls <= 3)
    agree = (mine_le120 == pub_le120)[valid].mean()
    print(f"2h class agreement (all valid cells): {agree*100:.2f}%")

    # population-weighted agreement
    w = pop[valid]
    ok = (mine_le120 == pub_le120)[valid]
    print(f"2h class agreement (population-weighted): {(w*ok).sum()/w.sum()*100:.2f}%")

    print(f"2h share  mine={(mine_le120[valid]*w).sum()/w.sum()*100:.2f}%  "
          f"published={(pub_le120[valid]*w).sum()/w.sum()*100:.2f}%")

    mine_le180 = sub <= 180.0
    pub_le180 = (cls >= 1) & (cls <= 4)
    print(f"3h class agreement (population-weighted): "
          f"{((mine_le180==pub_le180)[valid]*w).sum()/w.sum()*100:.2f}%")

    # confusion
    tp = int((mine_le120 & pub_le120 & valid).sum())
    fp = int((mine_le120 & ~pub_le120 & valid).sum())
    fn = int((~mine_le120 & pub_le120 & valid).sum())
    print(f"confusion  TP={tp} FP={fp} FN={fn}")

    res = {"agreement_all": float(agree),
           "agreement_pop": float(((mine_le120 == pub_le120)[valid] * w).sum() / w.sum()),
           "share2h_mine": float((mine_le120[valid] * w).sum() / w.sum()),
           "share2h_pub": float((pub_le120[valid] * w).sum() / w.sum())}

    # ---- continuous comparison if available ----
    if os.path.exists(TTFULL) and os.path.getsize(TTFULL) > 700_000_000:
        try:
            with rasterio.open(TTFULL) as ds:
                pub = ds.read(1).astype(np.float32)
            print("full TT opened", pub.shape, ds.nodata)
            v2 = valid & np.isfinite(pub) & (pub >= 0) & (pub < 1e8)
            # border mask: erode "same country" regions
            same = np.ones_like(ctry, dtype=bool)
            same[1:, :] &= (ctry[1:, :] == ctry[:-1, :])
            same[:-1, :] &= (ctry[:-1, :] == ctry[1:, :])
            same[:, 1:] &= (ctry[:, 1:] == ctry[:, :-1])
            same[:, :-1] &= (ctry[:, :-1] == ctry[:, 1:])
            interior = binary_erosion(same, iterations=50)  # ~50 cells ~ 50 km
            sel = v2 & interior
            print("interior cells:", int(sel.sum()))
            a = sub[sel].astype(np.float64)
            b = pub[sel].astype(np.float64)
            rho = stats.spearmanr(a, b).statistic
            d = np.abs(a - b)
            print(f"Spearman rho={rho:.4f}  median|diff|={np.median(d):.2f} min  "
                  f"p90|diff|={np.percentile(d,90):.2f} min")
            res.update({"spearman": float(rho), "median_abs_diff": float(np.median(d)),
                        "p90_abs_diff": float(np.percentile(d, 90))})
        except Exception as e:
            print("continuous comparison failed:", type(e).__name__, str(e)[:200])

    json.dump(res, open(f"{G}/validation.json", "w"), indent=1)
    print("VALIDATE_DONE")


if __name__ == "__main__":
    main()
