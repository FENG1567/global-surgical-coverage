#!/usr/bin/env python
"""
46_stress_large.py 鈥?endpoint stress test on larger countries.

The 26 certified countries all have <= 40 candidate facilities. This script
takes countries with 40 < n <= 150 and solves the same two MILPs with a time
limit, reporting the solver's own bounds rather than requiring gap = 0.

If M in [b, U_M] and m in [L_m, a] then

    b - a  <=  M - m  <=  U_M - L_m

so even an unfinished solve gives an inner and an outer bound on the width.
"""
import json, sys, time
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
LO, HI, TL = 40, 150, 60

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
todo = [c for c in ['GAB','BWA','NAM','LBR','SOM','MRT'] if c in idxmap]
_ = sorted([i for i, n in counts.items() if LO < n <= HI and i in idxmap],
              key=lambda i: counts[i])
print(f"candidates: {len(todo)} -> {todo}", flush=True)

out = []
for iso in todo:
    t0 = time.time()
    try:
        h = h_all[h_all.iso3 == iso].reset_index(drop=True)
        n = len(h); k = max(1, int(round(n * float(pcmap.get(iso, 0.8)))))
        r0 = int(max(0, (85.0 - (h.lat.max() + 0.3)) / PIX)); r1 = int(min(NR, (85.0 - (h.lat.min() - 0.3)) / PIX))
        c0 = int(max(0, (h.lon.min() - 0.3 + 180) / PIX)); c1 = int(min(NC, (h.lon.max() + 0.3 + 180) / PIX))
        if r1 - r0 < 5 or c1 - c0 < 5:
            continue
        fr = FR[r0:r1, c0:c1].astype(np.float64)
        pp = np.where(CTRY[r0:r1, c0:c1] == int(idxmap[iso]), pop_glob[r0:r1, c0:c1], 0.0).astype(np.float64)
        H, W = fr.shape
        if H * W > 3_000_000:
            print(f"skip {iso}: window {H}x{W}", flush=True); continue
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
        g = sparse.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(H * W, H * W)).tocsr()
        g = g.maximum(g.T)
        fi = np.clip(((85.0 - h.lat.values) / PIX).astype(int) - r0, 0, H - 1)
        fj = np.clip(((h.lon.values + 180) / PIX).astype(int) - c0, 0, W - 1)
        T = dijkstra(g, indices=(fi * W + fj).astype(np.int64))
        pcell = pp.ravel(); keep = pcell > 0
        A = (T[:, keep] <= 30); pc = pcell[keep]; ni = A.shape[1]

        w = pc / pc.sum()
        cons = [LinearConstraint(sparse.hstack([-sparse.csr_matrix(A.T.astype(float)),
                                                sparse.identity(ni, format="csr")]), -np.inf, 0.0)]
        jj, ii = np.nonzero(A); nnz = len(jj)
        rr2 = np.repeat(np.arange(nnz), 2)
        cc2 = np.empty(2 * nnz, dtype=np.int64); cc2[0::2] = jj; cc2[1::2] = n + ii
        cons.append(LinearConstraint(sparse.coo_matrix((np.tile([1.0, -1.0], nnz), (rr2, cc2)),
                                                       shape=(nnz, n + ni)).tocsr(), -np.inf, 0.0))
        cons.append(LinearConstraint(sparse.hstack([sparse.csr_matrix(np.ones((1, n))),
                                                    sparse.csr_matrix((1, ni))]), k, k))
        integ = np.r_[np.ones(n), np.zeros(ni)]
        bounds = Bounds(np.r_[np.zeros(n), np.zeros(ni)], np.r_[np.ones(n), np.ones(ni)])

        res = {}
        for tag in ("max", "min"):
            cvec = np.r_[np.zeros(n), (-w if tag == "max" else w)]
            r = milp(c=cvec, constraints=cons, integrality=integ, bounds=bounds,
                     options={"time_limit": TL, "mip_rel_gap": 1e-3})
            if r.fun is None:
                res[tag] = (None, None, r.status)
            else:
                inc = -float(r.fun) if tag == "max" else float(r.fun)
                db = getattr(r, "mip_dual_bound", None)
                if db is None:
                    bound = None
                else:
                    bound = -float(db) if tag == "max" else float(db)
                res[tag] = (inc, bound, r.status)
        b, U, sm = res["max"]; a, L, sn = res["min"]
        if b is None or a is None:
            print(f"{iso}: n={n} no incumbent", flush=True); continue
        rec = {"iso": iso, "n": n, "k": k,
               "inc_max": b, "bound_max": U, "inc_min": a, "bound_min": L,
               "width_inner": b - a,
               "width_outer": None if (U is None or L is None) else U - L,
               "status_max": sm, "status_min": sn, "secs": round(time.time() - t0, 1)}
        out.append(rec)
        print(f"{iso}: n={n} obs={b:.3f} width in [{rec['width_inner']*100:.1f}, "
              f"{'' if rec['width_outer'] is None else round(rec['width_outer']*100,1)}] pp "
              f"status=({sm},{sn}) {rec['secs']}s", flush=True)
    except Exception as e:
        print(f"{iso}: ERR {type(e).__name__} {str(e)[:90]}", flush=True)

pd.DataFrame(out).to_csv(f"{BASE}/work/stress_large.csv", index=False)
print("STRESS_DONE", len(out), flush=True)
