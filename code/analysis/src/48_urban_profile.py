import numpy as np, pandas as pd
B="__PROJECT_ROOT__"
G=f"{B}/work/grid"; NR,NC=17400,43200; PIX=1/120
pop=np.fromfile(f"{G}/pop2025.f32",dtype=np.float32).reshape(NR,NC)
h=pd.read_parquet(f"{B}/data/osm/osm_hospitals_all.parquet")
def dens(lat,lon):
    r=np.floor((85.0-lat)/PIX).astype(int); c=np.floor((lon+180)/PIX).astype(int)
    r=np.clip(r,1,NR-2); c=np.clip(c,1,NC-2)
    return (pop[r-1,c-1]+pop[r-1,c]+pop[r-1,c+1]+pop[r,c-1]+pop[r,c]+pop[r,c+1]+pop[r+1,c-1]+pop[r+1,c]+pop[r+1,c+1])
for iso,name in (("BRA","Brazil"),("FRA","France")):
    sub=h[h.iso3==iso]
    d_all=dens(sub.lat.values,sub.lon.values)
    thr_med=np.median(d_all)
    thr_q75=np.percentile(d_all,75)
    print(f"{name}: {len(sub)} OSM facilities; median local pop {thr_med:.0f}; q75 {thr_q75:.0f}")
    if iso=="BRA":
        act=pd.read_csv(f"{B}/work/brazil_validation.csv")
        a=pd.read_csv("__BRAZIL_SOURCE_ROOT__/03_outputs/hospital_points.csv")
        a=a[(a.coord_ok==True)&(a.cap_chole_behavior==True)].dropna(subset=["lat","lon"])
        da=dens(a.lat.values,a.lon.values)
        print(f"  active (chole>=1): {len(a)}; share above OSM median = {(da>thr_med).mean():.3f}; above q75 = {(da>thr_q75).mean():.3f}")
        print(f"  median local pop of active: {np.median(da):.0f} vs all {thr_med:.0f}")
    else:
        f=pd.read_csv(f"{B}/data/france/france_geocoded.csv").dropna(subset=["latitude","longitude"])
        aa=pd.read_csv(f"{B}/data/france/france_addresses.csv",sep=";").reset_index().rename(columns={"index":"id"})
        m=f.merge(aa,on="id",how="inner"); m=m[m.SALTOT24>0]
        dm=dens(m.latitude.values,m.longitude.values)
        print(f"  24h theatre: {len(m)}; share above OSM median = {(dm>thr_med).mean():.3f}; above q75 = {(dm>thr_q75).mean():.3f}")
        print(f"  median local pop of 24h: {np.median(dm):.0f} vs all {thr_med:.0f}")