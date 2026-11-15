from pathlib import Path
import json
import pandas as pd
from validate_spatial_auc import run_comparison

BASE=Path('__PROJECT_ROOT__/upgrade_20261112')
OUT=BASE/'derived/multiservice_validation'
OUT.mkdir(parents=True,exist_ok=False)
spatial=BASE/'derived/brazil_spatial_v1'
population=pd.read_csv(spatial/'municipality_population.csv')
outcomes=pd.read_csv(BASE/'inputs/brazil_municipality_annual.csv')
coordinates=pd.read_csv('__BRAZIL_SOURCE_ROOT__/01_data/interim/municipios_master.csv',dtype={'codigo_ibge':str})
coordinates['m6']=pd.to_numeric(coordinates.codigo_ibge.str[:6])
coordinates=coordinates[['m6','latitude','longitude']].drop_duplicates('m6')
results=[];inclusion=[]
for service in ['chole','cesarean','appendectomy']:
    outcome=outcomes[outcomes.service.eq(service)&outcomes.year.eq(2024)].copy()
    outcome=outcome.rename(columns={'MUNIC_RES':'m6'})
    outcome['has_local']=outcome.local.gt(0).astype(int)
    before=len(outcome)
    outcome=outcome[outcome.n>=10].copy()
    threshold_rows=len(outcome)
    exposure=pd.read_csv(spatial/f'{service}_prior_municipality.csv').rename(columns={'coverage30':'D2_act_pre'})
    d=outcome.merge(population,on='m6',how='left',validate='one_to_one').merge(exposure,on='m6',how='left',validate='one_to_one').merge(coordinates,on='m6',how='left',validate='one_to_one')
    required=['pop','urban_population','D1_osm','D2_act_pre','latitude','longitude']
    missing=d[required].isna().any(axis=1)|(d['pop']<=0)
    d.loc[missing,['m6','n','local']].to_csv(OUT/f'{service}_excluded.csv',index=False)
    d=d.loc[~missing].copy()
    inclusion.append({'service':service,'municipalities_with_cases':before,'municipalities_n_at_least10':threshold_rows,
        'excluded_missing_spatial_inputs':int(missing.sum()),'analysis_municipalities':len(d),'positive_local':int(d.has_local.sum())})
    summary,predictions,folds=run_comparison(d,repetitions=50)
    destination=OUT/service;destination.mkdir()
    (destination/'summary.json').write_text(json.dumps(summary,indent=2))
    predictions.to_csv(destination/'oof.csv.gz',index=False)
    folds.to_csv(destination/'folds.csv',index=False)
    d.to_csv(destination/'municipality_analysis.csv',index=False)
    for comparison,values in summary['paired_increments'].items():
        results.append({'service':service,'comparison':comparison,**values})
    print('MULTISERVICE_VALIDATED',service,len(d),summary['paired_increments'],flush=True)
pd.DataFrame(results).to_csv(OUT/'paired_increment_summary.csv',index=False)
pd.DataFrame(inclusion).to_csv(OUT/'inclusion.csv',index=False)
print('MULTISERVICE_VALIDATION_COMPLETE',flush=True)
