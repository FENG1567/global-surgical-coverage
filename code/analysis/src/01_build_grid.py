#!/usr/bin/env python
"""
01_build_grid.py — build the common 0.01 deg analysis grid.

Outputs (all in work/grid/):
  friction.f32   float32  nrows*ncols   minutes per metre (MAP/Weiss 2020 motorised)
  pop2020.f32    float32  nrows*ncols   GHSL R2023A 1km, persons per cell
  pop2025.f32    float32  nrows*ncols
  country.i16    int16    nrows*ncols   index into countries.csv (-1 = none)
  countries.csv  idx,iso3,name,income,region
  meta.json

Grid = friction surface grid (north-up, 0.01 deg, lon -180..180, lat 85..-89).
GHSL is on the same 0.01 deg lattice but offset; it is shifted by an exact
integer number of cells (no resampling error).
"""
import json, os, sys
import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd

BASE = "__PROJECT_ROOT__"
OUT = f"{BASE}/work/grid"
os.makedirs(OUT, exist_ok=True)

FRIC = f"{BASE}/data/friction/2020_motorized_friction_surface.geotiff"
GHSL = {2020: f"{BASE}/data/ghsl/GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif",
        2025: f"{BASE}/data/ghsl/GHS_POP_E2025_GLOBE_R2023A_4326_30ss_V1_0.tif"}
NE = f"{BASE}/data/boundaries/ne10m/ne_10m_admin_0_countries.shp"
WB = f"{BASE}/data/wb/wb_countries.json"


def main():
    with rasterio.open(FRIC) as ds:
        nrows, ncols = ds.height, ds.width
        T = ds.transform
        fric = ds.read(1).astype(np.float32)
        print("friction", nrows, ncols, ds.transform, ds.nodata, flush=True)
        meta = {"nrows": nrows, "ncols": ncols,
                "left": T.c, "top": T.f, "dx": T.a, "dy": T.e,
                "friction_nodata": ds.nodata,
                "friction_file": os.path.basename(FRIC)}
    # keep nodata as -9999, everything else as is
    fric.tofile(f"{OUT}/friction.f32")
    print("wrote friction.f32", fric.nbytes / 1e9, "GB", flush=True)
    del fric

    # ---------- GHSL -> friction grid (integer shift) ----------
    for year, path in GHSL.items():
        with rasterio.open(path) as ds:
            gT = ds.transform
            gh, gw = ds.height, ds.width
            shift_c = int(round((gT.c - T.c) / T.a))
            shift_r = int(round((gT.f - T.f) / T.e))
            print(f"GHSL {year}: {gh}x{gw} origin=({gT.c:.3f},{gT.f:.2f}) "
                  f"shift_row={shift_r} shift_col={shift_c}", flush=True)
            out = np.zeros((nrows, ncols), dtype=np.float32)
            # column mapping: dest C = src c + shift_c
            gc0 = max(0, -shift_c)
            gc1 = min(gw, ncols - shift_c)
            fc0 = gc0 + shift_c
            fc1 = gc1 + shift_c
            block = 2000
            for R0 in range(0, nrows, block):          # iterate over DEST rows
                R1 = min(R0 + block, nrows)
                r0, r1 = R0 - shift_r, R1 - shift_r    # matching SRC rows
                rr0, rr1 = max(r0, 0), min(r1, gh)
                if rr1 <= rr0:
                    continue
                RR0, RR1 = rr0 + shift_r, rr1 + shift_r
                a = ds.read(1, window=((rr0, rr1), (gc0, gc1))).astype(np.float32)
                out[RR0:RR1, fc0:fc1] = a
            out.tofile(f"{OUT}/pop{year}.f32")
            tot = float(out.sum(dtype=np.float64))
            print(f"  pop{year} total = {tot/1e9:.4f} billion", flush=True)
            meta[f"pop{year}_total"] = tot
            del out

    # ---------- countries ----------
    wb = json.load(open(WB))
    inc = {r["id"]: (r["incomeLevel"]["id"], r["region"]["id"], r["name"]) for r in wb[1]}
    g = gpd.read_file(NE)
    g = g[g["ADM0_A3"].notna()].copy()
    g["iso3"] = g["ADM0_A3"].astype(str)
    # Natural Earth uses -99 for disputed; fall back to ISO_A3
    bad = g["iso3"].isin(["-99", "nan"])
    g.loc[bad, "iso3"] = g.loc[bad, "ISO_A3"].astype(str)
    g = g[~g["iso3"].isin(["-99", "nan"])].copy()
    g = g.to_crs("EPSG:4326")
    rows = []
    shapes = []
    for i, (idx, rec) in enumerate(g.iterrows(), start=1):
        iso = rec["iso3"]
        i_inc, i_reg, i_name = inc.get(iso, ("NA", "NA", rec.get("NAME", iso)))
        rows.append({"idx": i, "iso3": iso, "name": i_name,
                     "income": i_inc, "region": i_reg})
        shapes.append((rec["geometry"], i))
    import pandas as pd
    cdf = pd.DataFrame(rows)
    cdf.to_csv(f"{OUT}/countries.csv", index=False)
    print("countries:", len(cdf), flush=True)

    out = np.full((nrows, ncols), -1, dtype=np.int16)
    arr = features.rasterize(
        shapes, out_shape=(nrows, ncols), transform=T,
        fill=-1, dtype="int16", all_touched=False)
    out[:] = arr
    out.tofile(f"{OUT}/country.i16")
    print("wrote country.i16", flush=True)

    meta["countries"] = int(len(cdf))
    meta["grid"] = "friction 0.01deg"
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
    print("GRID_DONE", flush=True)


if __name__ == "__main__":
    main()
