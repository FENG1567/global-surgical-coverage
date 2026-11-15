"""Common-friction allocation experiment within observed SAEH participants."""
from pathlib import Path
import json
import subprocess
import hashlib
import argparse
import numpy as np
import pandas as pd

BASE=Path('__PROJECT_ROOT__')
ROOT=BASE/'upgrade_20261112'
parser=argparse.ArgumentParser()
parser.add_argument('--tag',default='mexico_state_constraints_margin4')
parser.add_argument('--margin-degrees',type=int,default=4)
parser.add_argument('--joint-search-repetitions',type=int,default=16)
parser.add_argument('--hospital-only',action='store_true')
args=parser.parse_args()
OUT=ROOT/'derived'/args.tag
OUT.mkdir(parents=True,exist_ok=True)
G=BASE/'work/grid'
shape=(17400,43200)
pixel=1/120
scope=pd.read_csv(ROOT/'derived/country_scope/population_and_bounds.csv')
countryrow=scope[scope.iso3.eq('MEX')].iloc[0]
margin=args.margin_degrees*120
r0=max(0,int(countryrow.r0)-margin);r1=min(shape[0],int(countryrow.r1)+margin)
c0=max(0,int(countryrow.c0)-margin);c1=min(shape[1],int(countryrow.c1)+margin)
H,W=r1-r0,c1-c0
top=85-r0*pixel
friction=np.memmap(G/'friction.f32',dtype='float32',mode='r',shape=shape)[r0:r1,c0:c1].copy()
population=np.memmap(G/'pop2025.f32',dtype='float32',mode='r',shape=shape)[r0:r1,c0:c1].copy()
country=np.memmap(G/'country.i16',dtype='int16',mode='r',shape=shape)[r0:r1,c0:c1]
urban=np.memmap(G/'urban.u8',dtype='uint8',mode='r',shape=shape)[r0:r1,c0:c1].astype(bool)
rural=np.memmap(G/'rural.u8',dtype='uint8',mode='r',shape=shape)[r0:r1,c0:c1].astype(bool)
target=country==int(countryrow.idx)
weights=np.where(target,population,0)
assert abs(weights.sum(dtype=np.float64)/countryrow.population-1)<1e-8
friction.tofile(OUT/'friction_crop.f32')
source=(BASE/'src/ttdijkstra.c').read_text()
source=source.replace('const double PIX = 1.0 / 120.0;',
    'const double PIX = 1.0 / 120.0;\n    const double LAT_TOP = (argc > 8) ? atof(argv[8]) : 85.0;')
source=source.replace('double lat = 85.0 - PIX','double lat = LAT_TOP - PIX')
(OUT/'ttdijkstra_crop.c').write_text(source)
subprocess.run(['gcc','-O3','-o',str(OUT/'ttdijkstra_crop'),str(OUT/'ttdijkstra_crop.c'),'-lm'],check=True)
h=pd.read_csv(ROOT/'inputs/mexico_hospital_points.csv',dtype={'CLUES':str})
if args.hospital_only:
    h=h[pd.to_numeric(h.tipo_clave,errors='coerce').eq(2)].copy()
h=h[h.lat.between(-90,90)&h.lon.between(-180,180)].reset_index(drop=True)
invalid_geography=~(h.lat.between(13,34)&h.lon.between(-120,-85))
h.loc[invalid_geography].to_csv(OUT/'invalid_mexico_coordinates.csv',index=False)
h=h.loc[~invalid_geography].reset_index(drop=True)
rr=np.floor((85-h.lat.to_numpy())/pixel).astype(int)-r0
cc=np.floor((h.lon.to_numpy()+180)/pixel).astype(int)-c0
assert ((rr>=1)&(rr<H-1)&(cc>=1)&(cc<W-1)).all()
density=np.array([population[r-1:r+2,c-1:c+2].sum(dtype=np.float64) for r,c in zip(rr,cc)])
u=density>np.median(density)
smod_urban=urban[rr,cc]
joint_stratum=u.astype(int)*2+smod_urban.astype(int)
h['local_pop']=density;h['above_median_local_pop']=u
h['smod_urban_facility']=smod_urban
h.to_csv(OUT/'candidate_facilities_with_local_pop.csv',index=False)
cells=np.floor(h.lat).astype(int).to_numpy()*1000+np.floor(h.lon).astype(int).to_numpy()
rng=np.random.default_rng(20261112)

def cluster(indices,k):
    order=rng.permutation(np.unique(cells[indices]))
    selected=[]
    for cell in order:
        inside=indices[cells[indices]==cell]
        selected.extend(inside[:max(0,k-len(selected))])
        if len(selected)==k:
            break
    return np.asarray(selected,dtype=int)

rows=[]
def evaluate(label,selected,service,k,urban_k=None):
    selected=np.asarray(selected,dtype=int)
    assert len(selected)==k and len(np.unique(selected))==k
    if urban_k is not None:
        assert int(u[selected].sum())==urban_k
    mask=np.zeros((H,W),dtype='uint8');mask[rr[selected],cc[selected]]=1
    mask.tofile(OUT/'sources.u8')
    with (OUT/f'{label}.log').open('w') as log:
        subprocess.run([str(OUT/'ttdijkstra_crop'),str(OUT/'friction_crop.f32'),str(OUT/'sources.u8'),
            str(OUT/'travel_time.f32'),str(H),str(W),'120','0',str(top)],stdout=log,stderr=log,check=True)
    travel=np.memmap(OUT/'travel_time.f32',dtype='float32',mode='r',shape=(H,W))
    row={'scenario':label,'service':service,'n_candidates':len(h),'k':k,
         'realized_q':float(u[selected].mean()),'urban_count':int(u[selected].sum()),
         'smod_urban_count':int(smod_urban[selected].sum()),
         'smod_urban_share':float(smod_urban[selected].mean())}
    for threshold in [15,30,60,120]:
        covered=travel<=threshold
        for population_group,group_mask in [('all',target),('urban',target&urban),('rural',target&rural)]:
            denominator=population[group_mask].sum(dtype=np.float64)
            numerator=population[group_mask&covered].sum(dtype=np.float64)
            row[f'{population_group}_{threshold}']=float(numerator/denominator)
    pd.DataFrame({'CLUES':h.CLUES.iloc[selected]}).to_csv(OUT/f'{label}_selection.csv',index=False)
    rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'coverage_results.csv',index=False)
    print(json.dumps(row),flush=True)


main=ROOT/'derived/mexico_state_constraints'

reference=pd.read_csv(main/'coverage_results.csv')
lookup=pd.Series(np.arange(len(h)),index=h.CLUES)
for service in sorted(set(reference.service)-{'all'}):
    pattern='observed|state_joint_cluster' if 'state_constraints' in main.name else 'observed|jointmatched|jointsearch'
    feasible=reference[reference.service.eq(service)&reference.scenario.str.contains(pattern)]
    chosen_labels=list(dict.fromkeys([service+'_observed',
        feasible.loc[feasible.all_30.idxmin(),'scenario'],feasible.loc[feasible.all_30.idxmax(),'scenario']]))
    for label in chosen_labels:
        identifiers=pd.read_csv(main/f'{label}_selection.csv',dtype={'CLUES':str})
        selected=lookup.loc[identifiers.CLUES].to_numpy()
        expected=reference[reference.scenario.eq(label)].iloc[0]
        evaluate(label,selected,service,int(expected.k),int(expected.urban_count))
comparison=pd.DataFrame(rows).merge(reference,on='scenario',suffixes=('_margin4','_margin2'),validate='one_to_one')
metrics=[c for c in reference if c.startswith(('all_','urban_','rural_')) and c.rsplit('_',1)[-1].isdigit()]
maximum=max(float((comparison[c+'_margin4']-comparison[c+'_margin2']).abs().max()) for c in metrics)
comparison.to_csv(OUT/'boundary_comparison.csv',index=False)
(OUT/'audit.json').write_text(json.dumps({'margin_degrees':args.margin_degrees,'n_comparisons':len(rows),
    'maximum_coverage_difference_pp':maximum*100,'reference':main.name,
    'selection_rule':'observed plus minimum and maximum 30-minute coverage among feasible constructions'},indent=2))
print('FOLLOWUP_BOUNDARY_CHECK_COMPLETE',main.name,maximum*100,flush=True)
