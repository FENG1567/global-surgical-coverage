#!/usr/bin/env python
"""
43_certify_many.py 鈥?certified identified sets for every small country.

For each country with <= NMAX facilities, solve the two MILPs at 30 min and
record the certified width. Then we can ask what predicts the width across
countries, instead of reporting three anecdotes.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import rasterio
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.optimize import milp, LinearConstraint, Bounds

BASE = "__PROJECT_ROOT__"
G = f"{BASE}/work/grid"
NR, NC = 17400, 43200
PIX = 1.0 / 120.0
FRIC = f"{BASE}/data/friction/2020_motorized_friction_surface.geotiff"
NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 40
TAU = 30

cap = pd.read_csv(f"{BASE}/outputs/hospital_capability.csv")
pcmap = dict(zip(cap.iso3, cap.p_c))
cdf = pd.read_csv(f"{G}/countries.csv")
idxmap = dict(zip(cdf.iso3, cdf.idx))
h_all = pd.read_parquet(f"{BASE}/data/osm/osm_hospitals_all.parquet")
pop_glob = np.fromfile(f"{G}/pop2025.f32", dtype=np.float32).reshape(NR, NC)
CTRY = np.fromfile(f"{G}/country.i16", dtype=np.int16).reshape(NR, NC)
with rasterio.open(FRIC) as ds:
    FR = ds.read(1)

counts = h_all.groupby("iso3").size()
todo = [iso for iso, n in counts.items() if 8 <= n <= NMAX and iso in idxmap]
print(f"countries to certify: {len(todo)}", flush=True)

out = []
for k_i, iso in enumerate(todo, 1):
    t0 = time.time()
    try:
        h = h_all[h_all.iso3 == iso].reset_index(drop=True)
        n = len(h); k = max(1, int(round(n * float(pcmap.get(iso, 0.8)))))
        r0 = int(max(0, (85.0 - (h.lat.max() + 2.0)) / PIX))
        r1 = int(min(NR, (85.0 - (h.lat.min() - 2.0)) / PIX))
        c0 = int(max(0, (h.lon.min() - 2.0 + 180) / PIX))
        c1 = int(min(NC, (h.lon.max() + 2.0 + 180) / PIX))
        if r1 - r0 < 5 or c1 - c0 < 5:
            continue
        fr = FR[r0:r1, c0:c1].astype(np.float64)
        pp = np.where(CTRY[r0:r1, c0:c1] == int(idxmap[iso]), pop_glob[r0:r1, c0:c1], 0.0).astype(np.float64)
        H, W = fr.shape
        if H * W > 12_000_000:
            print(f"skip {iso}: window too big", flush=True); continue

        idx = np.arange(H * W).reshape(H, W)
        valid = (fr > 0) & (fr < 1000)
        rows, cols, data = [], [], []
        lat_c = 85.0 - PIX * (np.arange(r0, r1) + 0.5)
        dx = 111320.0 * PIX * np.cos(np.deg2rad(lat_c)); dy = 110574.0 * PIX
        for dr, dc in ((0, 1), (-1, 0), (-1, -1), (-1, 1)):
            rr = slice(max(0, -dr), H - max(0, dr)); cc = slice(max(0, -dc), W - max(0, dc))
            tr = slice(max(0, dr), H - max(0, -dr)); tc = slice(max(0, dc), W - max(0, -dc))
            a, b = idx[rr, cc], idx[tr, tc]
            f1 = fr[rr, cc]; f2 = fr[tr, tc]
            ok = valid[rr, cc] & valid[tr, tc]
            step_full = (np.broadcast_to(dx[:, None], (H, W)) if dr == 0 else
                         np.full((H, W), dy) if dc == 0 else
                         np.broadcast_to(np.hypot(dx[:, None], dy), (H, W)))
            w = 0.5 * (f1 + f2) * step_full[rr, cc]
            rows.append(a[ok]); cols.append(b[ok]); data.append(w[ok])
        if not data:
            continue
        g = sparse.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(H * W, H * W)).tocsr()
        g = g.maximum(g.T)

        fi = np.clip(((85.0 - h.lat.values) / PIX).astype(int) - r0, 0, H - 1)
        fj = np.clip(((h.lon.values + 180) / PIX).astype(int) - c0, 0, W - 1)
        nodes = (fi * W + fj).astype(np.int64)
        T = dijkstra(g, indices=nodes)
        pcell = pp.ravel(); keep = pcell > 0
        if keep.sum() < 50:
            continue
        A = (T[:, keep] <= TAU)
        pc = pcell[keep]; ni = A.shape[1]

        w = pc / pc.sum()
        cons = []
        M_up = sparse.hstack([-sparse.csr_matrix(A.T.astype(float)),
                              sparse.identity(ni, format="csr")])
        cons.append(LinearConstraint(M_up, -np.inf, 0.0))
        jj, ii = np.nonzero(A); nnz = len(jj)
        if nnz == 0:
            continue
        rr2 = np.repeat(np.arange(nnz), 2)
        cc2 = np.empty(2 * nnz, dtype=np.int64); cc2[0::2] = jj; cc2[1::2] = n + ii
        M_low = sparse.coo_matrix((np.tile([1.0, -1.0], nnz), (rr2, cc2)), shape=(nnz, n + ni)).tocsr()
        cons.append(LinearConstraint(M_low, -np.inf, 0.0))
        sx = sparse.hstack([sparse.csr_matrix(np.ones((1, n))), sparse.csr_matrix((1, ni))])
        cons.append(LinearConstraint(sx, k, k))
        integ = np.r_[np.ones(n), np.zeros(ni)]
        bounds = Bounds(np.r_[np.zeros(n), np.zeros(ni)], np.r_[np.ones(n), np.ones(ni)])

        vals = {}
        gaps = {}
        for tag in ("max", "min"):
            cvec = np.r_[np.zeros(n), (-w if tag == "max" else w)]
            r = milp(c=cvec, constraints=cons, integrality=integ, bounds=bounds,
                     options={"time_limit": 45, "mip_rel_gap": 1e-3})
            if r.fun is None:
                vals[tag] = None
                gaps[tag] = None
            else:
                vals[tag] = -float(r.fun) if tag == "max" else float(r.fun)
                gaps[tag] = float(getattr(r, "mip_gap", 0.0) or 0.0)
        if vals["max"] is None or vals["min"] is None:
            print(f"{k_i}/{len(todo)} {iso}: n={n} FAILED", flush=True)
            continue
        base = float(pc[A.any(axis=0)].sum() / pc.sum())
        rec = {"iso": iso, "n": n, "k": k, "observed": base,
               "min": vals["min"], "max": vals["max"],
               "width": vals["max"] - vals["min"],
               "gap_max": gaps.get("max"), "gap_min": gaps.get("min"),
               "certified": bool((gaps.get("max") is not None) and (gaps.get("min") is not None)
                                  and gaps["max"] <= 1e-4 and gaps["min"] <= 1e-4),
               "secs": round(time.time() - t0, 1)}
        out.append(rec)
        print(f"{k_i}/{len(todo)} {iso}: n={n} obs={base:.3f} min={vals['min']:.3f} "
              f"max={vals['max']:.3f} width={(vals['max']-vals['min'])*100:.1f}pp "
              f"({rec['secs']}s)", flush=True)
    except Exception as e:
        print(f"{k_i}/{len(todo)} {iso}: ERR {type(e).__name__} {str(e)[:80]}", flush=True)

pd.DataFrame(out).to_csv(f"{BASE}/work/certified_many.csv", index=False)
print("CERTIFY_MANY_DONE", len(out), flush=True)
