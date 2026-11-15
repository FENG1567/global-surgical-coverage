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
parser.add_argument('--tag',default='brazil_state_constraints')
parser.add_argument('--margin-degrees',type=int,default=2)
parser.add_argument('--joint-search-repetitions',type=int,default=16)
args=parser.parse_args()
OUT=ROOT/'derived'/args.tag
OUT.mkdir(parents=True,exist_ok=True)
G=BASE/'work/grid'
shape=(17400,43200)
pixel=1/120
scope=pd.read_csv(ROOT/'derived/country_scope/population_and_bounds.csv')
countryrow=scope[scope.iso3.eq('BRA')].iloc[0]
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
h=pd.read_csv(ROOT/'inputs/brazil_hospital_points.csv',dtype={'cnes':str})
h['CLUES']=h.cnes.str.replace(r'\.0$','',regex=True).str.zfill(7)
counts=pd.read_csv(ROOT/'inputs/brazil_hospital_annual.csv',dtype={'CNES':str})
counts['CNES']=counts.CNES.str.zfill(7)
matching=[]
for service in ['chole','cesarean','appendectomy']:
    for window,years in [('2024',[2024]),('pre',[2017,2018,2019]),('2023',[2023])]:
        totals=counts[counts.service.eq(service)&counts.year.isin(years)].groupby('CNES')['n'].sum()
        h[f'n_{service}_{window}']=h.CLUES.map(totals).fillna(0)
        matching.append({'service':service,'window':window,'records':int(totals.sum()),
            'records_linked_to_facility_table':int(h[f'n_{service}_{window}'].sum()),
            'active_hospitals':len(totals),'active_hospitals_linked':int(h[f'n_{service}_{window}'].gt(0).sum())})
pd.DataFrame(matching).to_csv(OUT/'linkage_audit.csv',index=False)
h=h[h.lat.between(-90,90)&h.lon.between(-180,180)].reset_index(drop=True)
invalid_geography=~(h.lat.between(-35,6)&h.lon.between(-75,-33))
h.loc[invalid_geography].to_csv(OUT/'invalid_brazil_coordinates.csv',index=False)
h=h.loc[~invalid_geography].reset_index(drop=True)
for item in matching:
    column=f"n_{item['service']}_{item['window']}"
    item['records_with_usable_coordinates']=int(h[column].sum())
    item['case_weighted_coordinate_coverage']=float(h[column].sum()/item['records'])
    item['active_hospitals_with_usable_coordinates']=int(h[column].gt(0).sum())
pd.DataFrame(matching).to_csv(OUT/'linkage_audit.csv',index=False)
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
import geopandas as gpd
import rasterio
from rasterio import features
polygons=gpd.read_file('__BRAZIL_SOURCE_ROOT__/river/data/official/ibge_malha/extracted/BR_Municipios_2022.shp')
codecol=next(c for c in polygons.columns if c.upper().startswith('CD_MUN'))
polygons['m6']=polygons[codecol].astype(str).str[:6].astype(int)
transform=rasterio.transform.from_origin(-180+c0*pixel,top,pixel,pixel)
municipality=features.rasterize(zip(polygons.geometry,polygons.m6),out_shape=(H,W),transform=transform,fill=0,dtype='int32')
valid_muni=(municipality>0)&target
muni_codes,muni_inverse=np.unique(municipality[valid_muni],return_inverse=True)
muni_pop=np.bincount(muni_inverse,weights=population[valid_muni])
muni_urban_pop=np.bincount(muni_inverse,weights=(population*urban)[valid_muni])
osm_time=np.memmap(G/'tt_D1.f32',dtype='float32',mode='r',shape=shape)[r0:r1,c0:c1]
osm_numerator=np.bincount(muni_inverse,weights=(population*(osm_time<=30))[valid_muni],minlength=len(muni_codes))
pd.DataFrame({'m6':muni_codes,'pop':muni_pop,'urban_population':muni_urban_pop,'D1_osm':np.divide(osm_numerator,muni_pop,out=np.zeros_like(osm_numerator),where=muni_pop>0)}).to_csv(OUT/'municipality_population.csv',index=False)
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
    if label.endswith('_prior') or label.endswith('_observed') or label.endswith('_2023'):
        numerator=np.bincount(muni_inverse,weights=(population*(travel<=30))[valid_muni],minlength=len(muni_codes))
        pd.DataFrame({'m6':muni_codes,'coverage30':np.divide(numerator,muni_pop,out=np.zeros_like(numerator),where=muni_pop>0)}).to_csv(OUT/f'{label}_municipality.csv',index=False)
    pd.DataFrame({'CLUES':h.CLUES.iloc[selected]}).to_csv(OUT/f'{label}_selection.csv',index=False)
    rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'coverage_results.csv',index=False)
    print(json.dumps(row),flush=True)


SERVICES=['chole', 'cesarean', 'appendectomy']
STATE_COLUMN='uf'
assert h[STATE_COLUMN].notna().all()
state_values=h[STATE_COLUMN].astype(str).to_numpy()
groups=pd.MultiIndex.from_arrays([state_values,u.astype(int),smod_urban.astype(int)]).factorize()[0]
all_groups=np.unique(groups)
for service in SERVICES:
    count_column='n_'+service+'_2024' if 'BRA'=='BRA' else 'n_'+service
    observed=np.flatnonzero(h[count_column].to_numpy()>0)
    k=len(observed);uk=int(u[observed].sum())
    quotas={int(group):int((groups[observed]==group).sum()) for group in all_groups}
    evaluate(service+'_observed',observed,service,k,uk)
    for repetition in range(8):
        selected=np.concatenate([cluster(np.flatnonzero(groups==group),quotas[int(group)]) for group in all_groups])
        assert all(int((groups[selected]==group).sum())==quotas[int(group)] for group in all_groups)
        evaluate(service+f'_state_joint_cluster_{repetition:02d}',selected,service,k,uk)
        selected_frame=h.iloc[selected]
        pd.DataFrame({'state':state_values[selected],'local_population_class':u[selected],
            'smod_urban':smod_urban[selected]}).value_counts().rename('n_selected').reset_index().to_csv(OUT/f'{service}_state_budget_{repetition:02d}.csv',index=False)
(OUT/'audit.json').write_text(json.dumps({'country':'BRA','state_column':STATE_COLUMN,
    'constraints':'exact observed facility counts within each state crossed with both local-population and SMOD-urban facility classes',
    'clustered_repetitions':8,'seed':20261112,'country_population_retained':float(weights.sum(dtype=np.float64)/countryrow.population),
    'interpretation':'feasible-construction lower bounds under stronger observed spatial aggregates; no optimization certificate or real-error estimate'},indent=2))
print('STATE_CONSTRAINT_EXPERIMENT_COMPLETE','BRA',flush=True)
