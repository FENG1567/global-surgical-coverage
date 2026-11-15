from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE=Path('__PROJECT_ROOT__')
G=BASE/'work/grid';OUT=BASE/'upgrade_20261112/derived/composition_curve'
OUT.mkdir(parents=True,exist_ok=True)
h=pd.read_parquet(BASE/'data/osm/osm_hospitals_all.parquet')
cap=pd.read_csv(BASE/'outputs/hospital_capability.csv')
pcmap=dict(zip(cap.iso3,cap.p_c))
pop=np.memmap(G/'pop2025.f32',dtype='float32',mode='r',shape=(17400,43200))
rr=np.floor((85-h.lat.to_numpy())*120).astype(int);cc=np.floor((h.lon.to_numpy()+180)*120).astype(int)
valid=(rr>=1)&(rr<17399)&(cc>=1)&(cc<43199)
density=np.zeros(len(h));r,c=rr[valid],cc[valid]
density[valid]=(pop[r-1,c-1]+pop[r-1,c]+pop[r-1,c+1]+pop[r,c-1]+pop[r,c]+pop[r,c+1]+pop[r+1,c-1]+pop[r+1,c]+pop[r+1,c+1])
h['local_pop']=density
h['above_median']=h.local_pop>h.groupby('iso3').local_pop.transform('median')
h['p_c']=h.iso3.map(pcmap).fillna(.8)
curves=pd.read_csv(BASE/'work/urban_share_curve.csv')
rows=[]
for u in [0,.2,.4,.6,.8,1]:
    keep=np.zeros(len(h),dtype=bool);budgets=[]
    for iso,g in h.groupby('iso3'):
        k=max(1,int(round(len(g)*float(g.p_c.iloc[0]))))
        high=int(round(k*u));ordered=g.sort_values('local_pop',ascending=False)
        selected=np.r_[ordered.index[:high].to_numpy(),ordered.index[::-1][:k-high].to_numpy()]
        assert len(np.unique(selected))==k
        keep[selected]=True;budgets.append(k)
    cells=np.unique(rr[keep]*43200+cc[keep])
    path=G/f'src_D2u_u{int(u*100):03d}.u8'
    mask=np.memmap(path,dtype='uint8',mode='r')
    matches=bool((mask[cells]>0).all() and np.count_nonzero(mask)==len(cells))
    coverage=curves[curves.u.eq(u)&curves.thr.eq(30)]['all'].iloc[0]
    rows.append({'nominal_mix_u':u,'actual_global_q':float(h.loc[keep,'above_median'].mean()),
                 'n_selected':int(keep.sum()),'sum_country_k':sum(budgets),
                 'reconstructed_source_mask_matches':matches,'coverage30':float(coverage)})
frame=pd.DataFrame(rows)
frame['actual_q_in_045_068']=frame.actual_global_q.between(.45,.68)
frame.to_csv(OUT/'nominal_vs_actual.csv',index=False)
summary={'all_six_source_masks_reconstructed':bool(frame.reconstructed_source_mask_matches.all()),
 'all_six_selection_counts_equal':int(frame.n_selected.nunique())==1,
 'actual_q_range':[float(frame.actual_global_q.min()),float(frame.actual_global_q.max())],
 'nominal_parameter_is_realized_q':False,
 'sampled_curve_width_with_actual_q_in_045_068_pp':float(frame.loc[frame.actual_q_in_045_068,'coverage30'].max()-frame.loc[frame.actual_q_in_045_068,'coverage30'].min())*100,
 'interpretation':'finite sampled feasible allocations, not optimized conditional bounds; use actual q rather than equating the mixing parameter to q'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(frame.to_json(orient='records'),flush=True)
print(json.dumps(summary),flush=True)
