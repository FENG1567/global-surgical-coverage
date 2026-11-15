#!/usr/bin/env python
"""
22_france_prep.py — build the French facility layer from DREES SAE 2024.

BLOCS_2024 gives operating-theatre counts per FINESS establishment;
ID_2024 gives the postal address. Coordinates are obtained from the official
French geocoder (api-adresse.data.gouv.fr, bulk CSV endpoint).
"""
import csv, io, json, os, sys, time
import urllib.request, urllib.parse
import pandas as pd

BASE = "__PROJECT_ROOT__"
FR = f"{BASE}/data/france"
CSVDIR = f"{FR}/SAE 2024 Base administrative - Formats SAS-CSV/Base CSV"

blocs = pd.read_csv(f"{CSVDIR}/BLOCS_2024.csv", sep=";", dtype=str, encoding="latin-1")
ids = pd.read_csv(f"{CSVDIR}/ID_2024.csv", sep=";", dtype=str, encoding="latin-1")
print("BLOCS", blocs.shape, "ID", ids.shape)

for c in ("SALTOT", "SALTOT24", "SALCHIR", "SAL24CHIR"):
    blocs[c] = pd.to_numeric(blocs[c], errors="coerce").fillna(0)
print("estab with any theatre:", int((blocs.SALTOT > 0).sum()))
print("estab with 24h theatre:", int((blocs.SALTOT24 > 0).sum()))
print("estab with surgical room:", int((blocs.SALCHIR > 0).sum()))

m = blocs.merge(ids[["FI", "RS", "NUMVOI", "TYPVOI", "NOMVOI", "CPO", "LIBCOM", "DEP", "REG"]],
                on="FI", how="left", suffixes=("", "_id"))
m = m[m.SALTOT > 0].copy()
print("surgical establishments:", len(m))

def addr(r):
    parts = [str(r.NUMVOI) if pd.notna(r.NUMVOI) else "",
             str(r.TYPVOI) if pd.notna(r.TYPVOI) else "",
             str(r.NOMVOI) if pd.notna(r.NOMVOI) else "",
             str(r.CPO) if pd.notna(r.CPO) else "",
             str(r.LIBCOM) if pd.notna(r.LIBCOM) else ""]
    return " ".join([p for p in parts if p and p != "nan"]).strip()

m["adresse"] = m.apply(addr, axis=1)
m = m[m.adresse.str.len() > 5].reset_index(drop=True)
print("with usable address:", len(m))
m[["FI", "RS", "adresse", "SALTOT", "SALTOT24", "SALCHIR", "DEP"]].to_csv(
    f"{FR}/france_addresses.csv", index=False, sep=";")

# ---- bulk geocode ----
buf = io.StringIO()
w = csv.writer(buf)
w.writerow(["id", "adresse"])
for i, r in m.iterrows():
    w.writerow([i, r.adresse])
data = buf.getvalue().encode("utf-8")

boundary = "----dshboundary"
body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"data\"; filename=\"a.csv\"\r\n"
        f"Content-Type: text/csv\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
req = urllib.request.Request(
    "https://api-adresse.data.gouv.fr/search/csv/?columns=adresse&result_columns=latitude&result_columns=longitude&result_columns=result_score",
    data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                        "User-Agent": "research/1.0"})
try:
    with urllib.request.urlopen(req, timeout=600) as r:
        out = r.read().decode("utf-8")
    open(f"{FR}/france_geocoded.csv", "w", encoding="utf-8").write(out)
    g = pd.read_csv(io.StringIO(out))
    print("geocoded rows:", len(g), "with lat:", int(g.latitude.notna().sum()))
    print(g.head(3).to_string())
except Exception as e:
    print("BULK FAILED", type(e).__name__, str(e)[:300])
print("FR_PREP_DONE")
