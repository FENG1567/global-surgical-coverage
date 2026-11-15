"""Derive Brazil's common-friction experiment from the checked Mexico kernel."""
from pathlib import Path

root=Path('__PROJECT_ROOT__/upgrade_20261112')
source=(root/'code/mexico_spatial_experiment.py').read_text(encoding='utf8')
source=source.replace('mexico_spatial_joint','brazil_spatial_v1').replace("scope.iso3.eq('MEX')","scope.iso3.eq('BRA')")
source=source.replace("h=pd.read_csv(ROOT/'inputs/mexico_hospital_points.csv',dtype={'CLUES':str})",'''h=pd.read_csv(ROOT/'inputs/brazil_hospital_points.csv',dtype={'cnes':str})
h['CLUES']=h.cnes.str.replace(r'\\.0$','',regex=True).str.zfill(7)
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
pd.DataFrame(matching).to_csv(OUT/'linkage_audit.csv',index=False)''')
source=source.replace("h.lat.between(13,34)&h.lon.between(-120,-85)","h.lat.between(-35,6)&h.lon.between(-75,-33)")
source=source.replace('invalid_mexico_coordinates.csv','invalid_brazil_coordinates.csv')
source=source.replace('h=h.loc[~invalid_geography].reset_index(drop=True)', '''h=h.loc[~invalid_geography].reset_index(drop=True)
for item in matching:
    column=f"n_{item['service']}_{item['window']}"
    item['records_with_usable_coordinates']=int(h[column].sum())
    item['case_weighted_coordinate_coverage']=float(h[column].sum()/item['records'])
    item['active_hospitals_with_usable_coordinates']=int(h[column].gt(0).sum())
pd.DataFrame(matching).to_csv(OUT/'linkage_audit.csv',index=False)''')
source=source.replace("h.to_csv(OUT/'candidate_facilities_with_local_pop.csv',index=False)",'''h.to_csv(OUT/'candidate_facilities_with_local_pop.csv',index=False)
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
pd.DataFrame({'m6':muni_codes,'pop':muni_pop,'urban_population':muni_urban_pop,'D1_osm':np.divide(osm_numerator,muni_pop,out=np.zeros_like(osm_numerator),where=muni_pop>0)}).to_csv(OUT/'municipality_population.csv',index=False)''')
source=source.replace("pd.DataFrame({'CLUES':h.CLUES.iloc[selected]}).to_csv",'''if label.endswith('_prior') or label.endswith('_observed') or label.endswith('_2023'):
        numerator=np.bincount(muni_inverse,weights=(population*(travel<=30))[valid_muni],minlength=len(muni_codes))
        pd.DataFrame({'m6':muni_codes,'coverage30':np.divide(numerator,muni_pop,out=np.zeros_like(numerator),where=muni_pop>0)}).to_csv(OUT/f'{label}_municipality.csv',index=False)
    pd.DataFrame({'CLUES':h.CLUES.iloc[selected]}).to_csv''')
source=source.replace("for service,countcol in [('chole','n_chole'),('cesarean','n_cesarea')]:", "for service,countcol in [('chole','n_chole_2024'),('cesarean','n_cesarean_2024'),('appendectomy','n_appendectomy_2024')]:")
source=source.replace("evaluate(service+'_cluster_qmatched',cluster_q,service,k,uk)",'''evaluate(service+'_cluster_qmatched',cluster_q,service,k,uk)
    for window in ['pre','2023']:
        active=np.flatnonzero(h[f'n_{service}_{window}'].to_numpy()>0)
        evaluate(service+('_prior' if window=='pre' else '_2023'),active,service,len(active))
    for minimum_count in [5,10]:
        active=np.flatnonzero(h[countcol].to_numpy()>=minimum_count)
        evaluate(service+f'_min{minimum_count}',active,service,len(active))''')
source=source.replace('coordinate-valid SAEH2023 observed participating facilities','coordinate-valid Brazilian CNES2025 hospital list from original analysis')
source=source.replace('population proximity to observed participating facilities, no all-provider Mexican coverage claim','population proximity to facilities with observed SUS-funded procedure activity, conditional on the fixed CNES list')
source=source.replace("'hospitals':hashlib.sha256((ROOT/'inputs/mexico_hospital_points.csv').read_bytes()).hexdigest()", "'hospitals':hashlib.sha256((ROOT/'inputs/brazil_hospital_points.csv').read_bytes()).hexdigest(),'activity':hashlib.sha256((ROOT/'inputs/brazil_hospital_annual.csv').read_bytes()).hexdigest()")
source=source.replace("'2019 coordinate registry for 2023 activity','52 observed facilities had no registry match'", "'2025 hospital coordinates applied to historical activity','all CNES hospitals include facilities without observed SUS target-procedure activity'")
source=source.replace('MEXICO_SPATIAL_COMPLETE','BRAZIL_SPATIAL_COMPLETE')
(root/'code/brazil_spatial_experiment.py').write_text(source,encoding='utf8')
