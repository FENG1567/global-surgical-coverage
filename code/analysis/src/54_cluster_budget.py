#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
54_cluster_budget.py — 预算精确的空间场景（重写版）

修正 19_spatial_scenarios.py 的两个缺陷：
  1) cluster 分支按整格累加，|S| != k_c；
  2) 首格超预算时回退成 urban。

本版每国严格保留 k_c = round(N_c * p_c)，四个场景各自独立生成掩膜。

导出：
  work/scenario_budget.csv          逐国 k_c / 实际保留 / 全局 u
  work/scenario_ids_<name>.npy      选中设施行号（审稿可核验）
  work/grid/src_D2sc_<name>_k.u8    掩膜
"""
import json
import numpy as np
import pandas as pd

B = "__PROJECT_ROOT__"
G = f"{B}/work/grid"
NR, NC = 17400, 43200
PIX = 1.0 / 120.0

h = pd.read_parquet(f"{B}/data/osm/osm_hospitals_all.parquet").reset_index(drop=True)
cap = pd.read_csv(f"{B}/outputs/hospital_capability.csv")
pcmap = dict(zip(cap.iso3, cap.p_c))
pop = np.fromfile(f"{G}/pop2025.f32", dtype=np.float32).reshape(NR, NC)

row = np.floor((85.0 - h.lat.values) / PIX).astype(np.int64)
col = np.floor((h.lon.values + 180.0) / PIX).astype(np.int64)
ok = (row >= 1) & (row < NR - 1) & (col >= 1) & (col < NC - 1)
dens = np.zeros(len(h))
r, c = row[ok], col[ok]
dens[ok] = (pop[r - 1, c - 1] + pop[r - 1, c] + pop[r - 1, c + 1] +
            pop[r, c - 1] + pop[r, c] + pop[r, c + 1] +
            pop[r + 1, c - 1] + pop[r + 1, c] + pop[r + 1, c + 1])
h["local_pop"] = dens
h["p_c"] = h.iso3.map(pcmap).fillna(0.80)
h["is_urban"] = h.local_pop > h.groupby("iso3").local_pop.transform("median")
h["cell1"] = (np.floor(85.0 - h.lat.values) * 1000 +
              np.floor(h.lon.values + 180.0)).astype(np.int64)

names = ("rand", "urban", "rural", "cluster")
rng = np.random.default_rng(20261111)
rows = []
keeps = {}
selections = {}

for name in names:
    keep = np.zeros(len(h), dtype=bool)
    sel_all = []
    for iso, g in h.groupby("iso3"):
        n_c = len(g)
        k = max(1, int(round(n_c * float(g.p_c.iloc[0]))))
        if name == "rand":
            sel = rng.choice(g.index.values, size=k, replace=False)
        elif name == "urban":
            sel = g.nlargest(k, "local_pop").index.values
        elif name == "rural":
            sel = g.nsmallest(k, "local_pop").index.values
        else:
            cell_sizes = g.groupby("cell1").size()
            order = rng.permutation(cell_sizes.index.values)
            picked, remaining = [], k
            cell_arr = g.cell1.values
            idx_arr = g.index.values
            for cell in order:
                idx = idx_arr[cell_arr == cell]
                if len(idx) <= remaining:
                    picked.extend(idx.tolist())
                    remaining -= len(idx)
                else:
                    picked.extend(rng.choice(idx, size=remaining, replace=False).tolist())
                    remaining = 0
                if remaining == 0:
                    break
            if remaining > 0:
                rest = np.setdiff1d(idx_arr, np.array(picked, dtype=np.int64))
                picked.extend(rng.choice(rest, size=remaining, replace=False).tolist())
            sel = np.array(picked, dtype=np.int64)
        assert len(sel) == k, f"{iso} {name}: {len(sel)} != {k}"
        keep[sel] = True
        sel_all.append(sel)
        rows.append({"scenario": name, "iso3": iso, "n_country": n_c, "k_c": k,
                     "n_selected": int(len(sel)),
                     "u_country": float(h.is_urban.values[sel].mean())})
    keeps[name] = keep
    selections[name] = np.concatenate(sel_all)

df = pd.DataFrame(rows)
df.to_csv(f"{B}/work/scenario_budget.csv", index=False)

for name in names:
    np.save(f"{B}/work/scenario_ids_{name}.npy", selections[name])
    kp = keeps[name]
    m = np.zeros(NR * NC, dtype=np.uint8)
    m[row[kp] * NC + col[kp]] = 1
    m.tofile(f"{G}/src_D2sc_{name}_k.u8")
    print(f"wrote {name}: kept {int(kp.sum())} cells {int(m.sum())}")

print("\n=== 预算核对 ===")
bad = df[df.n_selected != df.k_c]
print(f"  rows={len(df)} violations={len(bad)}")
print("\n=== 各场景 ===")
for name in names:
    g = df[df.scenario == name]
    ids = np.load(f"{B}/work/scenario_ids_{name}.npy")
    u = float(h.is_urban.values[ids].mean())
    print(f"  {name:8s} total={len(ids):7d} k_total={int(g.k_c.sum()):7d} "
          f"u={u:.3f} in_range={0.45 <= u <= 0.68}")
print("\nBUDGET_EXACT_DONE")
