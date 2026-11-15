import pandas as pd, numpy as np
from scipy.stats import spearmanr
B="__PROJECT_ROOT__"
d=pd.read_csv(f"{B}/work/certified_many.csv")
d["k_over_n"]=d.k/d.n
print(f"n={len(d)}; width median {d.width.median()*100:.2f} p90 {d.width.quantile(.9)*100:.2f} max {d.width.max()*100:.2f}")
print(f"n range {d.n.min()}-{d.n.max()}")
print(f"k/n range {d.k_over_n.min():.2f}-{d.k_over_n.max():.2f}")
for col in ["n","k_over_n","observed"]:
    r=spearmanr(d[col],d.width)
    print(f"width vs {col}: rho={r.statistic:+.3f} p={r.pvalue:.4f}")
# split by n
small=d[d.n<=20]; big=d[d.n>20]
print(f"\nn<=20 (n={len(small)}): median {small.width.median()*100:.2f} pp, >7.4pp {(small.width>0.074).sum()}")
print(f"n>20  (n={len(big)}): median {big.width.median()*100:.2f} pp, >7.4pp {(big.width>0.074).sum()}")
# split by observed coverage
lo=d[d.observed<0.9]; hi=d[d.observed>=0.9]
print(f"observed<0.9 (n={len(lo)}): median {lo.width.median()*100:.2f} pp")
print(f"observed>=0.9 (n={len(hi)}): median {hi.width.median()*100:.2f} pp")