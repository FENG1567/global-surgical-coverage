#!/usr/bin/env python
"""
13_osm_precision.py — how good is the OSM `amenity=hospital` layer?

For the three countries with a national facility registry (BRA, IDN, MEX) we
match every OSM hospital to the nearest registry hospital and report

  precision = share of OSM hospitals with a registry hospital within R km
  recall    = share of registry hospitals with an OSM hospital within R km

This is the only *measured* estimate of OSM hospital-layer quality; everything
else in the project is bounded by it.
"""
import json, os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE = "__PROJECT_ROOT__"
REG = f"{BASE}/data/registry"
R_KM = 1.0

# local equirectangular projection: x = lon*cos(lat0)*111.32, y = lat*110.57
def proj(lon, lat, lat0):
    return np.c_[lon * 111.32 * np.cos(np.deg2rad(lat0)), lat * 110.57]


def match(a_lon, a_lat, b_lon, b_lat, r_km=R_KM):
    lat0 = float(np.mean(np.r_[a_lat, b_lat]))
    A = proj(np.asarray(a_lon), np.asarray(a_lat), lat0)
    B = proj(np.asarray(b_lon), np.asarray(b_lat), lat0)
    tree = cKDTree(B)
    d, idx = tree.query(A, k=1, distance_upper_bound=r_km)
    ok = np.isfinite(d)
    return ok, idx, d


def main():
    osm = pd.read_parquet(f"{BASE}/data/osm/osm_hospitals_all.parquet")
    out = []

    # ---------- BRA ----------
    cnes = pd.read_csv("__BRAZIL_SOURCE_ROOT__/03_outputs/hospital_points.csv")
    cnes = cnes[cnes.coord_ok == True].dropna(subset=["lat", "lon"])
    o = osm[osm.iso3 == "BRA"]
    ok, idx, d = match(o.lon.values, o.lat.values, cnes.lon.values, cnes.lat.values)
    ok2, _, _ = match(cnes.lon.values, cnes.lat.values, o.lon.values, o.lat.values)
    out.append({"iso3": "BRA", "registry": "CNES 2025", "n_osm": len(o), "n_reg": len(cnes),
                "precision": ok.mean(), "recall": ok2.mean(),
                "median_dist_m": float(np.median(d[ok]) * 1000) if ok.any() else None})
    print(f"BRA  OSM={len(o)}  CNES={len(cnes)}  precision={ok.mean()*100:.1f}%  recall={ok2.mean()*100:.1f}%")

    # ---------- IDN ----------
    idn = pd.read_csv(f"{REG}/IDN/IDN_facilities_safe_extract.csv").dropna(subset=["latitude", "longitude"])
    o = osm[osm.iso3 == "IDN"]
    ok, idx, d = match(o.lon.values, o.lat.values, idn.longitude.values, idn.latitude.values)
    ok2, _, _ = match(idn.longitude.values, idn.latitude.values, o.lon.values, o.lat.values)
    out.append({"iso3": "IDN", "registry": "SIRS/RS Online", "n_osm": len(o), "n_reg": len(idn),
                "precision": ok.mean(), "recall": ok2.mean(),
                "median_dist_m": float(np.median(d[ok]) * 1000) if ok.any() else None})
    print(f"IDN  OSM={len(o)}  SIRS={len(idn)}  precision={ok.mean()*100:.1f}%  recall={ok2.mean()*100:.1f}%")

    # ---------- MEX ----------
    mex = pd.read_csv(f"{REG}/MEX/CLUES_202609/MEX_CLUES_202609_safe_extract.csv")
    mex = mex[(mex.operational_status == "EN OPERACION") &
              (mex.facility_type == "DE HOSPITALIZACIÓN")].dropna(subset=["latitude", "longitude"])
    o = osm[osm.iso3 == "MEX"]
    ok, idx, d = match(o.lon.values, o.lat.values, mex.longitude.values, mex.latitude.values)
    ok2, _, _ = match(mex.longitude.values, mex.latitude.values, o.lon.values, o.lat.values)
    out.append({"iso3": "MEX", "registry": "CLUES 2026 (operational inpatient)", "n_osm": len(o),
                "n_reg": len(mex), "precision": ok.mean(), "recall": ok2.mean(),
                "median_dist_m": float(np.median(d[ok]) * 1000) if ok.any() else None})
    print(f"MEX  OSM={len(o)}  CLUES={len(mex)}  precision={ok.mean()*100:.1f}%  recall={ok2.mean()*100:.1f}%")

    df = pd.DataFrame(out)
    df.to_csv(f"{BASE}/work/osm_precision.csv", index=False)
    print("\n", df.round(3).to_string(index=False))
    print("PRECISION_DONE")


if __name__ == "__main__":
    main()
