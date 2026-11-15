#!/usr/bin/env python
"""
42_value_of_information.py 鈥?how much does auditing q% of hospitals buy?

Setup: country with n facilities, capability budget k. Exactly k are capable but
we do not know which. We reveal the true label of a fraction q of them and
re-solve the two MILPs. The reduction in the width of C(k) is the value of that
audit.

Two audit strategies:
  random    -- reveal q*n facilities drawn at random
  targeted  -- reveal the q*n facilities with the largest catchment population
               (the ones that matter most for coverage)
"""
import json, sys
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
ISO = sys.argv[1] if len(sys.argv) > 1 else "DJI"
TAU = 30

cap = pd.read_csv(f"{BASE}/outputs/hospital_capability.csv")
pcmap = dict(zip(cap.iso3, cap.p_c))
cdf = pd.read_csv(f"{G}/countries.csv")
h_all = pd.read_parquet(f"{BASE}/data/osm/osm_hospitals_all.parquet")
pop_glob = np.fromfile(f"{G}/pop2025.f32", dtype=np.float32).reshape(NR, NC)
CTRY = np.fromfile(f"{G}/country.i16", dtype=np.int16).reshape(NR, NC)
with rasterio.open(FRIC) as ds:
    FR = ds.read(1)

h = h_all[h_all.iso3 == ISO].reset_index(drop=True)
n = len(h)
k = max(1, int(round(n * float(pcmap.get(ISO, 0.8)))))
lat_min, lat_max = h.lat.min() - 0.5, h.lat.max() + 0.5
lon_min, lon_max = h.lon.min() - 0.5, h.lon.max() + 0.5
r0 = int(max(0, (85.0 - lat_max) / PIX)); r1 = int(min(NR, (85.0 - lat_min) / PIX))
c0 = int(max(0, (lon_min + 180) / PIX)); c1 = int(min(NC, (lon_max + 180) / PIX))
fr = FR[r0:r1, c0:c1].astype(np.float64)
pp = pop_glob[r0:r1, c0:c1].astype(np.float64)
cidx = int(cdf.loc[cdf.iso3 == ISO, "idx"].iloc[0])
pp = np.where(CTRY[r0:r1, c0:c1] == cidx, pp, 0.0)
H, W = fr.shape
print(f"{ISO}: n={n} k={k} window {H}x{W}")

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
nodes = (fi * W + fj).astype(np.int64)
T = dijkstra(g, indices=nodes)
pcell = pp.ravel(); keep = pcell > 0
A = (T[:, keep] <= TAU)                       # n x ncell
pc = pcell[keep]; ni = A.shape[1]
catch = A @ pc / pc.sum()                     # per-facility catchment share
print(f"  populated cells {ni}; catchment share min/median/max "
      f"{catch.min():.3f}/{np.median(catch):.3f}/{catch.max():.3f}")

w = pc / pc.sum()
base_cons = []
M_up = sparse.hstack([-sparse.csr_matrix(A.T.astype(float)),
                      sparse.identity(ni, format="csr")])
base_cons.append(LinearConstraint(M_up, -np.inf, 0.0))
jj, ii = np.nonzero(A); nnz = len(jj)
rr = np.repeat(np.arange(nnz), 2)
ccx = np.empty(2 * nnz, dtype=np.int64); ccx[0::2] = jj; ccx[1::2] = n + ii
M_low = sparse.coo_matrix((np.tile([1.0, -1.0], nnz), (rr, ccx)), shape=(nnz, n + ni)).tocsr()
base_cons.append(LinearConstraint(M_low, -np.inf, 0.0))
sx = sparse.hstack([sparse.csr_matrix(np.ones((1, n))), sparse.csr_matrix((1, ni))])
base_cons.append(LinearConstraint(sx, k, k))
integrality = np.r_[np.ones(n), np.zeros(ni)]
lb = np.r_[np.zeros(n), np.zeros(ni)]
ub = np.r_[np.ones(n), np.ones(ni)]


def width(fixed):
    """fixed: dict j -> 0/1"""
    lo = lb.copy(); hi = ub.copy()
    for j, v in fixed.items():
        lo[j] = v; hi[j] = v
    b = Bounds(lo, hi)
    out = {}
    for tag in ("max", "min"):
        cvec = np.r_[np.zeros(n), (-w if tag == "max" else w)]
        r = milp(c=cvec, constraints=base_cons, integrality=integrality, bounds=b,
                 options={"time_limit": 120, "mip_rel_gap": 1e-4})
        if r.fun is None:
            return None
        out[tag] = -float(r.fun) if tag == "max" else float(r.fun)
    return out["max"] - out["min"]


rng = np.random.default_rng(20261111)
truth = np.zeros(n, dtype=int); truth[rng.choice(n, k, replace=False)] = 1
w0 = width({})
print(f"q=0 (nothing revealed): width = {w0*100:.2f} pp")

res = {"iso": ISO, "n": n, "k": k, "w0": w0, "curve": []}
order_pop = np.argsort(-catch)
for q in (0.1, 0.2, 0.4):
    m = int(round(q * n))
    for strat in ("random", "targeted"):
        if strat == "random":
            rev = rng.choice(n, m, replace=False)
        else:
            rev = order_pop[:m]
        fx = {int(j): int(truth[j]) for j in rev}
        wq = width(fx)
        res["curve"].append({"q": q, "strategy": strat, "revealed": m,
                             "width": wq,
                             "reduction": None if wq is None else (w0 - wq)})
        print(f"  q={q:.0%} {strat:8s}: width={None if wq is None else round(wq*100,2)} pp "
              f"reduction={None if wq is None else round((w0-wq)*100,2)} pp")

json.dump(res, open(f"{BASE}/work/voi_{ISO}.json", "w"), indent=1, default=float)
print("VOI_DONE")
