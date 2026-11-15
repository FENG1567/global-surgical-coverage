#!/usr/bin/env python
"""
41_certified_bounds.py 鈥?MILP bounds with optimality certificates on small countries.

For a country with n facilities and k = round(p_c * n), the identified set is
{C(S) : |S| = k}. We compute its true endpoints by solving two MILPs

    maximise / minimise  sum_i pop_i * y_i
    s.t.  y_i <= sum_{j : T_ij <= tau} x_j     (coverage)
          sum_j x_j = k                        (capability budget)
          x_j in {0,1},  y_i in [0,1]

with HiGHS (via scipy.optimize.milp), which reports primal and dual bounds, so
we get a certificate:  b - a  <=  diam C(k)  <=  U - L.
"""
import json, os, sys
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

COUNTRIES = {"GMB": 68.0, "DJI": 30.0}
TAUS = [30, 120]

cap = pd.read_csv(f"{BASE}/outputs/hospital_capability.csv")
pcmap = dict(zip(cap.iso3, cap.p_c))
cdf = pd.read_csv(f"{G}/countries.csv")
h_all = pd.read_parquet(f"{BASE}/data/osm/osm_hospitals_all.parquet")
pop_glob = np.fromfile(f"{G}/pop2025.f32", dtype=np.float32).reshape(NR, NC)
CTRY = np.fromfile(f"{G}/country.i16", dtype=np.int16).reshape(NR, NC)

with rasterio.open(FRIC) as ds:
    FR = ds.read(1)

results = {}
for iso in COUNTRIES:
    h = h_all[h_all.iso3 == iso]
    n = len(h)
    if n < 10:
        continue
    k = max(1, int(round(n * float(pcmap.get(iso, 0.8)))))
    lat_min, lat_max = h.lat.min() - 2.0, h.lat.max() + 2.0
    lon_min, lon_max = h.lon.min() - 2.0, h.lon.max() + 2.0
    r0 = int(max(0, (85.0 - lat_max) / PIX)); r1 = int(min(NR, (85.0 - lat_min) / PIX))
    c0 = int(max(0, (lon_min + 180) / PIX)); c1 = int(min(NC, (lon_max + 180) / PIX))
    fr = FR[r0:r1, c0:c1].astype(np.float64)
    pp = pop_glob[r0:r1, c0:c1].astype(np.float64)
    cidx = int(cdf.loc[cdf.iso3 == iso, "idx"].iloc[0])
    cmask = (CTRY[r0:r1, c0:c1] == cidx)
    pp = np.where(cmask, pp, 0.0)
    H, W = fr.shape
    print(f"\n{iso}: n={n} k={k} window {H}x{W}")

    # graph
    idx = np.arange(H * W).reshape(H, W)
    valid = (fr > 0) & (fr < 1000)
    rows, cols, data = [], [], []
    lat_c = 85.0 - PIX * (np.arange(r0, r1) + 0.5)
    dx = 111320.0 * PIX * np.cos(np.deg2rad(lat_c))
    dy = 110574.0 * PIX
    for dr, dc in ((0, 1), (-1, 0), (-1, -1), (-1, 1)):
        rr = slice(max(0, -dr), H - max(0, dr)); cc = slice(max(0, -dc), W - max(0, dc))
        tr = slice(max(0, dr), H - max(0, -dr)); tc = slice(max(0, dc), W - max(0, -dc))
        a, b = idx[rr, cc], idx[tr, tc]
        f1 = fr[rr, cc]; f2 = fr[tr, tc]
        ok = valid[rr, cc] & valid[tr, tc]
        if dr == 0:
            step_full = np.broadcast_to(dx[:, None], (H, W))
        elif dc == 0:
            step_full = np.full((H, W), dy)
        else:
            step_full = np.broadcast_to(np.hypot(dx[:, None], dy), (H, W))
        step = step_full[rr, cc]
        w = 0.5 * (f1 + f2) * step
        rows.append(a[ok]); cols.append(b[ok]); data.append(w[ok])
    g = sparse.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(H * W, H * W)).tocsr()
    g = g.maximum(g.T)

    fi = np.clip(((85.0 - h.lat.values) / PIX).astype(int) - r0, 0, H - 1)
    fj = np.clip(((h.lon.values + 180) / PIX).astype(int) - c0, 0, W - 1)
    nodes = (fi * W + fj).astype(np.int64)
    T = dijkstra(g, indices=nodes)                     # n x (H*W)

    pcell = pp.ravel()
    keep = pcell > 0
    Tp = T[:, keep]                                     # n x ncell
    pc = pcell[keep]
    print(f"  populated cells: {keep.sum()}")
    row = {"iso": iso, "n": n, "k": k}
    for tau in TAUS:
        A = (Tp <= tau)                                 # n x ncell boolean
        cov = A.any(axis=0)
        base = float(pc[cov].sum() / pc.sum())
        # MILP: variables [x (n), y (ncell)]
        ni = A.shape[1]
        cons = []
        # y_i - sum_j A[j,i] x_j <= 0
        M = sparse.hstack([-sparse.csr_matrix(A.T.astype(float)),
                           sparse.identity(ni, format="csr")])
        cons.append(LinearConstraint(M, -np.inf, 0.0))
        # lower bound: y_i >= x_j for every (i,j) with A[j,i]=1
        jj, ii = np.nonzero(A)              # A is (n, ncell)
        nnz = len(jj)
        if nnz > 0:
            rr = np.repeat(np.arange(nnz), 2)
            cc = np.empty(2 * nnz, dtype=np.int64)
            cc[0::2] = jj
            cc[1::2] = n + ii
            dd = np.tile([1.0, -1.0], nnz)
            M_low = sparse.coo_matrix((dd, (rr, cc)), shape=(nnz, n + ni)).tocsr()
            cons.append(LinearConstraint(M_low, -np.inf, 0.0))
        # sum x = k
        sx = sparse.hstack([sparse.csr_matrix(np.ones((1, n))),
                            sparse.csr_matrix((1, ni))])
        cons.append(LinearConstraint(sx, k, k))
        w = pc / pc.sum()
        integrality = np.r_[np.ones(n), np.zeros(ni)]
        bounds = Bounds(np.r_[np.zeros(n), np.zeros(ni)], np.r_[np.ones(n), np.ones(ni)])
        out = {}
        for tag in ("max", "min"):
            cvec = np.r_[np.zeros(n), (-w if tag == "max" else w)]
            r = milp(c=cvec, constraints=cons, integrality=integrality,
                     bounds=bounds, options={"time_limit": 300, "mip_rel_gap": 1e-4})
            if r.fun is None:
                print(f"    {tag}: status={r.status} msg={r.message[:120]}")
                out[tag] = {"coverage": float('nan'), "status": int(r.status),
                            "dual_bound": None, "gap": None}
                continue
            obj = float(r.fun)
            coverage = -obj if tag == "max" else obj
            out[tag] = {"coverage": coverage, "status": int(r.status),
                        "dual_bound": getattr(r, "mip_dual_bound", None),
                        "gap": getattr(r, "mip_gap", None)}
        row[f"tau{tau}"] = {
            "observed_all": base,
            "max_incumbent": out["max"]["coverage"],
            "min_incumbent": out["min"]["coverage"],
            "max_status": out["max"]["status"], "min_status": out["min"]["status"],
            "max_gap": out["max"]["gap"], "min_gap": out["min"]["gap"],
            "width_incumbent": out["max"]["coverage"] - out["min"]["coverage"],
        }
        print(f"  tau={tau}: observed(all)={base:.3f}  "
              f"max={out['max']['coverage']:.3f} (gap {out['max']['gap']})  "
              f"min={out['min']['coverage']:.3f} (gap {out['min']['gap']})  "
              f"width={out['max']['coverage']-out['min']['coverage']:.3f}")
    results[iso] = row

json.dump(results, open(f"{BASE}/work/certified_bounds.json", "w"), indent=1, default=float)
print("\nCERTIFIED_DONE")
