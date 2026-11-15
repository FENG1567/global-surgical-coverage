from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,brier_score_loss,log_loss
from sklearn.exceptions import ConvergenceWarning

BASE=Path('__PROJECT_ROOT__/upgrade_20261112')
OUT=BASE/'derived/temporal_multiservice_validation'
OUT.mkdir(parents=True,exist_ok=False)
spatial=BASE/'derived/brazil_spatial_v1'
population=pd.read_csv(spatial/'municipality_population.csv')
outcomes=pd.read_csv(BASE/'inputs/brazil_municipality_annual.csv').rename(columns={'MUNIC_RES':'m6'})
results=[]
for service in ['chole','cesarean','appendectomy']:
    exposure=pd.read_csv(spatial/f'{service}_prior_municipality.csv').rename(columns={'coverage30':'D2_act_pre'})
    d=outcomes[outcomes.service.eq(service)&outcomes.year.isin([2023,2024])&outcomes.n.ge(10)].copy()
    d=d.merge(population,on='m6',how='left',validate='many_to_one').merge(exposure,on='m6',how='left',validate='many_to_one')
    valid=d[['pop','urban_population','D1_osm','D2_act_pre']].notna().all(axis=1)&d['pop'].gt(0)
    d.loc[~valid,['year','m6','n']].to_csv(OUT/f'{service}_excluded.csv',index=False)
    d=d.loc[valid].copy()
    d['urban_share']=d.urban_population/d['pop'];d['logpop']=np.log1p(d['pop']);d['has_local']=d.local.gt(0).astype(int)
    train=d[d.year.eq(2023)];test=d[d.year.eq(2024)]
    predictions=test[['m6','has_local','n','local']].copy()
    models={'baseline':['logpop','urban_share'],'baseline+OSM':['logpop','urban_share','D1_osm'],
            'baseline+activity':['logpop','urban_share','D2_act_pre']}
    for model,cols in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            fit=LogisticRegression(max_iter=2000).fit(train[cols],train.has_local)
        probability=fit.predict_proba(test[cols])[:,1]
        predictions[model]=probability
        results.append({'service':service,'model':model,'train_year':2023,'test_year':2024,
            'n_train':len(train),'n_test':len(test),'n_test_not_present_in_train':int((~test.m6.isin(train.m6)).sum()),
            'auc':float(roc_auc_score(test.has_local,probability)),
            'brier':float(brier_score_loss(test.has_local,probability)),
            'log_loss':float(log_loss(test.has_local,probability))})
    predictions.to_csv(OUT/f'{service}_predictions.csv.gz',index=False)
pd.DataFrame(results).to_csv(OUT/'model_scores.csv',index=False)
(OUT/'design.json').write_text(json.dumps({'training_outcome_year':2023,'test_outcome_year':2024,
    'activity_feature_years':[2017,2018,2019],'minimum_recorded_resident_cases_each_year':10,
    'geographic_scope':'same country, temporal holdout; municipalities may recur across years',
    'covariates':'fixed GHSL2025 projected population and original SMOD2020; stable spatial definitions',
    'limitations':['2025 coordinate register used for historical activity','not an independent national healthcare system test'],
    'specification_frozen_before_this_temporal_rerun':True,
    'analysis_is_posthoc_extension_of_existing_project':True},indent=2))
print(pd.DataFrame(results).to_json(orient='records'),flush=True)
print('TEMPORAL_VALIDATION_COMPLETE',flush=True)
