"""Recheck facility ID -> mask -> recomputed travel time -> coverage chain."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd

BASE=Path('__PROJECT_ROOT__')
G=BASE/'work/grid'
OUT=BASE/'upgrade_20261112/derived/global_witness'
OUT.mkdir(parents=True,exist_ok=True)
nr,nc=17400,43200
h=pd.read_parquet(BASE/'data/osm/osm_hospitals_all.parquet').reset_index(drop=True)
pop=np.memmap(G/'pop2025.f32',dtype='float32',mode='r',shape=(nr,nc))
ctry=np.memmap(G/'country.i16',dtype='int16',mode='r',shape=(nr,nc))
results={}
for name in ['rand','cluster']:
    csv=pd.read_csv(BASE/f'work/selection_{name}.csv')
    ids=np.load(BASE/f'work/scenario_ids_{name}.npy')
    assert np.array_equal(np.sort(ids),np.sort(csv.facility_idx))
    chosen=h.iloc[ids]
    matched=csv.set_index('facility_idx').loc[ids]
    assert np.allclose(chosen.lat,matched.lat) and np.allclose(chosen.lon,matched.lon)
    rows=np.floor((85-chosen.lat.to_numpy())*120).astype(int)
    cols=np.floor((chosen.lon.to_numpy()+180)*120).astype(int)
    indices=rows*nc+cols
    oldmask=np.memmap(G/f'src_D2sc_{name}_k.u8',dtype='uint8',mode='r')
    assert len(np.unique(indices))==int(np.count_nonzero(oldmask))
    assert oldmask[indices].all()
    travel_path=OUT/f'travel_{name}.f32'
    log_path=OUT/f'travel_{name}.log'
    existing_complete=travel_path.exists() and travel_path.stat().st_size==nr*nc*4 and log_path.exists() and 'done.' in log_path.read_text()
    if not existing_complete:
        with log_path.open('w') as log:
            subprocess.run([str(BASE/'src/ttdijkstra'),str(G/'friction.f32'),str(G/f'src_D2sc_{name}_k.u8'),
                str(travel_path),str(nr),str(nc),'120','0'],stdout=log,stderr=log,check=True)
    new=np.memmap(OUT/f'travel_{name}.f32',dtype='float32',mode='r',shape=(nr,nc))
    old=np.memmap(G/f'tt_D2sc_{name}_k.f32',dtype='float32',mode='r',shape=(nr,nc))
    sums={30:0.,120:0.};denominator=0.;maxdiff=0.;disagreements={30:0,120:0}
    for row in range(0,nr,300):
        valid=(ctry[row:row+300]>=0)&(pop[row:row+300]>0)
        p=pop[row:row+300][valid].astype('float64')
        a=new[row:row+300][valid];b=old[row:row+300][valid]
        denominator+=p.sum()
        paired_reached=(a<=120)&(b<=120)
        maxdiff=max(maxdiff,float(np.abs(a[paired_reached]-b[paired_reached]).max(initial=0)))
        for threshold in [30,120]:
            sums[threshold]+=p[a<=threshold].sum()
            disagreements[threshold]+=int(((a<=threshold)!=(b<=threshold)).sum())
    results[name]={'n_selected':len(ids),'unique_ids':len(np.unique(ids)),
        'facility_csv_matches_numpy':True,'mask_matches_facility_cells':True,
        'source_cells':len(np.unique(indices)),'max_recomputed_time_difference_within_120_minutes':maxdiff,
        'coverage_cell_disagreements':disagreements,'coverage':{str(t):sums[t]/denominator for t in sums},
        'q':float(matched.is_urban.astype(bool).mean()),
        'selection_csv_sha256':hashlib.sha256((BASE/f'work/selection_{name}.csv').read_bytes()).hexdigest()}
    (OUT/'results.json').write_text(json.dumps(results,indent=2))
    assert disagreements[30]==0 and disagreements[120]==0
    assert maxdiff<1e-4
    print('WITNESS_VERIFIED',name,results[name]['coverage'],flush=True)
results['constrained_width_pp']=(results['rand']['coverage']['30']-results['cluster']['coverage']['30'])*100
(OUT/'results.json').write_text(json.dumps(results,indent=2))
print('GLOBAL_WITNESS_COMPLETE',flush=True)
