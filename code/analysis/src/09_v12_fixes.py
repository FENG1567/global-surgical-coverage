#!/usr/bin/env python
"""
09_v12_fixes.py — v1.2 additions requested after the pilot:

  A. urban/rural mask from GHS_SMOD 2020 (classes 21/22/23/30 = urban,
     11/12/13 = rural), shifted onto the analysis grid exactly like GHSL pop.
  B. strictly nested D0:  D0u = D0 (WorldPop facilities) UNION D1 (OSM hospitals)
     so that D0u >= D1 >= D2 holds by construction (the v1.0 ladder used two
     different point sources and was therefore not nested).
"""
import json, os
import numpy as np
import rasterio

BASE = "__PROJECT_ROOT__"
G = f"{BASE}/work/grid"
NR, NC = 17400, 43200
PIX = 1.0 / 120.0

SMOD = f"{BASE}/data/ghsl/GHS_SMOD_E2020_GLOBE_R2023A_4326_30ss_V2_0.tif"
TOP_F, LEFT_F = 85.0, -180.0

meta = json.load(open(f"{G}/meta.json"))


def main():
    with rasterio.open(SMOD) as ds:
        t = ds.transform
        gh, gw = ds.height, ds.width
        shift_r = int(round((t.f - TOP_F) / t.e))     # src -> dest row offset
        shift_c = int(round((t.c - LEFT_F) / t.a))
        print(f"SMOD {gh}x{gw} shift_row={shift_r} shift_col={shift_c}", flush=True)
        out = np.full((NR, NC), -200, dtype=np.int16)
        gc0, gc1 = max(0, -shift_c), min(gw, NC - shift_c)
        fc0, fc1 = gc0 + shift_c, gc1 + shift_c
        block = 2000
        for R0 in range(0, NR, block):
            R1 = min(R0 + block, NR)
            r0, r1 = R0 - shift_r, R1 - shift_r
            rr0, rr1 = max(r0, 0), min(r1, gh)
            if rr1 <= rr0:
                continue
            a = ds.read(1, window=((rr0, rr1), (gc0, gc1)))
            out[rr0 + shift_r:rr1 + shift_r, fc0:fc1] = a

    urban = np.isin(out, [21, 22, 23, 30])
    rural = np.isin(out, [11, 12, 13])
    water = out == 10
    print(f"cells urban={int(urban.sum())} rural={int(rural.sum())} water={int(water.sum())}",
          flush=True)
    np.packbits(urban.reshape(-1)).tofile(f"{G}/urban.pack")
    np.packbits(rural.reshape(-1)).tofile(f"{G}/rural.pack")
    # store raw too (simpler for later reads)
    urban.astype(np.uint8).tofile(f"{G}/urban.u8")
    rural.astype(np.uint8).tofile(f"{G}/rural.u8")

    pop = np.fromfile(f"{G}/pop2025.f32", dtype=np.float32).reshape(NR, NC)
    print(f"pop urban={pop[urban].sum(dtype=np.float64)/1e9:.3f}B "
          f"rural={pop[rural].sum(dtype=np.float64)/1e9:.3f}B "
          f"other={pop[~(urban|rural)].sum(dtype=np.float64)/1e9:.3f}B", flush=True)

    # ---- strictly nested D0 ----
    d0 = np.fromfile(f"{G}/src_D0_facility.u8", dtype=np.uint8)
    d1 = np.fromfile(f"{G}/src_D1_hospital.u8", dtype=np.uint8)
    d0u = np.maximum(d0, d1)
    d0u.tofile(f"{G}/src_D0u.u8")
    print(f"src_D0u: D0 cells={int(d0.sum())} D1 cells={int(d1.sum())} "
          f"union={int(d0u.sum())}", flush=True)
    print("SMOD_DONE", flush=True)


if __name__ == "__main__":
    main()
