#!/usr/bin/env python
"""
03_osm_hospitals.py — build a global OSM hospital point layer from HDX HOTOSM
per-country health-facility exports.

Input : data/wb/wb_countries.json  (World Bank economy list)
Output: data/osm/parts/<ISO3>.parquet  (hospitals only, with tags + lat/lon)
        data/osm/osm_hospitals_all.parquet
        data/osm/osm_download_log.csv
"""
import json, os, sys, zipfile, io, traceback, time
import concurrent.futures as cf
import pandas as pd
from shapely.geometry import shape
from shapely.validation import make_valid

BASE = "__PROJECT_ROOT__"
OSM = f"{BASE}/data/osm"
RAW = f"{OSM}/raw"
PARTS = f"{OSM}/parts"
for d in (RAW, PARTS):
    os.makedirs(d, exist_ok=True)

URL = ("https://production-raw-data-api.s3.amazonaws.com/ISO3/{ISO}/health_facilities/"
       "hotosm_{iso}_health_facilities_osm_geojson.zip")

KEEP = ["id", "name", "amenity", "healthcare", "healthcare_speciality",
        "capacity_persons", "operator_type", "building", "addr_city",
        "adm0_pcode", "adm1_name", "adm2_name", "source"]


def countries():
    d = json.load(open(f"{BASE}/data/wb/wb_countries.json"))
    out = []
    for r in d[1]:
        if r["region"]["id"] == "NA":      # WB aggregates
            continue
        out.append({"iso3": r["id"], "name": r["name"],
                    "income": r["incomeLevel"]["id"], "region": r["region"]["id"]})
    return pd.DataFrame(out)


def fetch(iso3, timeout=300):
    import urllib.request
    iso = iso3.lower()
    url = URL.format(ISO=iso3, iso=iso)
    dst = f"{RAW}/{iso3}.zip"
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        return dst, "cached", 0
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "global-surg-access/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r, open(dst, "wb") as f:
            f.write(r.read())
        return dst, "ok", time.time() - t0
    except Exception as e:
        if os.path.exists(dst):
            os.remove(dst)
        return None, f"{type(e).__name__}:{getattr(e,'code','')}", time.time() - t0


def to_points(path, iso3):
    """Read the geojson inside the zip, keep hospitals, return DataFrame."""
    with zipfile.ZipFile(path) as zf:
        cand = [n for n in zf.namelist() if n.lower().endswith(".geojson")]
        if not cand:
            raise RuntimeError("no geojson in zip")
        with zf.open(cand[0]) as fh:
            d = json.load(fh)
    rows = []
    for ft in d.get("features", []):
        p = ft.get("properties") or {}
        if p.get("amenity") != "hospital" and p.get("healthcare") != "hospital":
            continue
        g = ft.get("geometry")
        if not g:
            continue
        try:
            geom = shape(g)
            if not geom.is_valid:
                geom = make_valid(geom)
            c = geom if geom.geom_type == "Point" else geom.centroid
            lon, lat = float(c.x), float(c.y)
        except Exception:
            continue
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            continue
        rec = {k: p.get(k) for k in KEEP}
        rec["iso3"] = iso3
        rec["lon"] = lon
        rec["lat"] = lat
        rows.append(rec)
    return pd.DataFrame(rows)


def work(iso3):
    try:
        path, status, secs = fetch(iso3)
        if path is None:
            return {"iso3": iso3, "status": status, "n_hosp": 0, "secs": round(secs, 1)}
        df = to_points(path, iso3)
        df.to_parquet(f"{PARTS}/{iso3}.parquet", index=False)
        return {"iso3": iso3, "status": status, "n_hosp": len(df), "secs": round(secs, 1)}
    except Exception as e:
        return {"iso3": iso3, "status": f"ERR:{type(e).__name__}:{str(e)[:80]}", "n_hosp": 0, "secs": 0}


def main():
    cs = countries()
    print(f"countries to try: {len(cs)}", flush=True)
    rows = []
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(work, r.iso3): r.iso3 for r in cs.itertuples()}
        for i, f in enumerate(cf.as_completed(futs), 1):
            rows.append(f.result())
            if i % 20 == 0:
                print(f"  {i}/{len(futs)} done", flush=True)
    log = pd.DataFrame(rows).merge(cs, on="iso3", how="left")
    log.to_csv(f"{OSM}/osm_download_log.csv", index=False)
    print(log["status"].value_counts().head(10))
    print("total hospitals:", int(log["n_hosp"].sum()))

    parts = sorted(os.listdir(PARTS))
    frames = [pd.read_parquet(f"{PARTS}/{p}") for p in parts if p.endswith(".parquet")]
    if frames:
        allh = pd.concat(frames, ignore_index=True)
        allh = allh.drop_duplicates(subset=["id"])
        allh.to_parquet(f"{OSM}/osm_hospitals_all.parquet", index=False)
        print("combined:", len(allh), "by iso3:", allh["iso3"].nunique())
    print("OSM_DONE")


if __name__ == "__main__":
    main()
