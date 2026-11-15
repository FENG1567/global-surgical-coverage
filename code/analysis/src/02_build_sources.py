#!/usr/bin/env python
"""
02_build_sources.py — rasterise facility points onto the analysis grid.

Builds uint8 source masks (nrows*ncols, row-major, 1 = source cell):
  src_D0_facility.u8   all WorldPop health facilities (HF_pnts)  -> reproduction of TT_HF
  src_D1_hospital.u8   all OSM amenity=hospital                  -> "hospital" definition
  src_D2_brazil.u8     Brazil hospitals with CNES surgical centre (gold standard)

Grid: 43200 x 17400, 1/120 deg, lon -180..180, lat 85..-60.
"""
import json, os
import numpy as np
import pandas as pd
import geopandas as gpd

BASE = "__PROJECT_ROOT__"
G = f"{BASE}/work/grid"
meta = json.load(open(f"{G}/meta.json"))
NR, NC = meta["nrows"], meta["ncols"]
PIX = 1.0 / 120.0
LEFT, TOP = -180.0, 85.0


def to_rc(lon, lat):
    col = np.floor((np.asarray(lon, dtype=np.float64) - LEFT) / PIX).astype(np.int64)
    row = np.floor((TOP - np.asarray(lat, dtype=np.float64)) / PIX).astype(np.int64)
    return row, col


def mask_from_points(lon, lat, name):
    row, col = to_rc(lon, lat)
    ok = (row >= 0) & (row < NR) & (col >= 0) & (col < NC) & np.isfinite(lon) & np.isfinite(lat)
    m = np.zeros(NR * NC, dtype=np.uint8)
    idx = row[ok] * NC + col[ok]
    m[idx] = 1
    m.tofile(f"{G}/{name}")
    print(f"{name}: points={len(lon)} in_grid={int(ok.sum())} cells={int(m.sum())} "
          f"({m.sum()/max(len(lon),1)*100:.1f}% of points)", flush=True)
    return m


def main():
    # ---- D0: WorldPop HF points ----
    hf = gpd.read_file(f"{BASE}/data/worldpop/HF_pnts.gpkg", layer="HF_pnts")
    print("HF_pnts:", len(hf), hf.geom_type.value_counts().to_dict(), flush=True)
    lon = hf.geometry.x.values
    lat = hf.geometry.y.values
    mask_from_points(lon, lat, "src_D0_facility.u8")

    # ---- D1: OSM hospitals ----
    h = pd.read_parquet(f"{BASE}/data/osm/osm_hospitals_all.parquet")
    print("OSM hospitals:", len(h), flush=True)
    mask_from_points(h.lon.values, h.lat.values, "src_D1_hospital.u8")

    # ---- D2 Brazil: CNES hospitals with surgical centre ----
    bz = pd.read_csv(f"{BASE}/../../surgical_desert_20261108/03_outputs/hospital_points.csv") \
        if False else pd.read_csv("__BRAZIL_SOURCE_ROOT__/03_outputs/hospital_points.csv")
    bz = bz[(bz.coord_ok == True) & bz.lat.notna() & bz.lon.notna()]
    print("Brazil CNES hospitals with coords:", len(bz), flush=True)
    print("  has_surg_center:", int(bz.has_surg_center.sum()),
          " cap_chole_behavior:", int(bz.cap_chole_behavior.sum()), flush=True)
    # surgical centre OR actually performed cholecystectomy
    sel = bz[(bz.has_surg_center == True) | (bz.cap_chole_behavior == True)]
    mask_from_points(sel.lon.values, sel.lat.values, "src_D2_brazil.u8")
    sel.to_csv(f"{BASE}/work/grid/brazil_surgical_hospitals.csv", index=False)

    print("SOURCES_DONE", flush=True)


if __name__ == "__main__":
    main()
