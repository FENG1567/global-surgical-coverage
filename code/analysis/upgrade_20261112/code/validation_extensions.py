"""Prespecified stricter validation and prior-service persistence comparisons."""
from pathlib import Path
import json
import warnings
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,brier_score_loss,log_loss
from sklearn.exceptions import ConvergenceWarning

BASE=Path('__PROJECT_ROOT__')
parser=argparse.ArgumentParser()
parser.add_argument('--input',type=Path,default=BASE/'upgrade_20261112/derived/urban_repair/municipalities_corrected.csv')
parser.add_argument('--output-tag',default='validation_extensions')
parser.add_argument('--strict-chole',action='store_true')
parser.add_argument('--registry-prior-service',choices=['chole','cesarean','appendectomy'])
args=parser.parse_args()
OUT=BASE/'upgrade_20261112/derived'/args.output_tag
OUT.mkdir(parents=True,exist_ok=False)
d=pd.read_csv(args.input)
if 'urban_population' in d:
    assert d['pop'].gt(0).all() and d.urban_population.between(0,d['pop']).all()
    d['urban_share']=d.urban_population/d['pop']
if 'region' not in d:
    # IBGE municipality codes begin with the two-digit state identifier.
    region_codes={**dict.fromkeys([11,12,13,14,15,16,17],'North'),
                  **dict.fromkeys([21,22,23,24,25,26,27,28,29],'Northeast'),
                  **dict.fromkeys([31,32,33,35],'Southeast'),
                  **dict.fromkeys([41,42,43],'South'),
                  **dict.fromkeys([50,51,52,53],'Midwest')}
    d['region']=(pd.to_numeric(d.m6)//10000).map(region_codes)
    assert d.region.notna().all()
prior={}
if args.registry_prior_service:
    h=pd.read_csv(BASE/'upgrade_20261112/derived/brazil_spatial_v1/candidate_facilities_with_local_pop.csv')
    column=f'n_{args.registry_prior_service}_pre'
    counts=h.groupby('municipio6')[column].sum()
    prior={int(code):float(count) for code,count in counts.items()}
else:
    for year in [2017,2018,2019]:
        p=Path(f'__SIH_PARQUET_ROOT__/sih_{year}_cholec.parquet')
        source=pd.read_parquet(p,columns=['MUNIC_MOV','PROC_REA'])
        if args.strict_chole:
            source=source[source.PROC_REA.astype(str).isin(['0407030026','0407030034'])]
        counts=pd.to_numeric(source.MUNIC_MOV,errors='coerce').dropna().astype('int64').value_counts()
        for code,count in counts.items():
            prior[int(code)]=prior.get(int(code),0)+int(count)
d['prior_treatment_count']=d.m6.map(prior).fillna(0)
d['prior_local_service']=d.prior_treatment_count.gt(0).astype(int)
d['log_prior_treatment_count']=np.log1p(d.prior_treatment_count)
d['logpop']=np.log1p(d['pop'])
models={
 'population_urban':['logpop','urban_share'],
 'population_urban_osm':['logpop','urban_share','D1_osm'],
 'population_urban_activity':['logpop','urban_share','D2_act_pre'],
 'prior_service_baseline':['logpop','urban_share','prior_local_service','log_prior_treatment_count'],
 'prior_service_plus_activity':['logpop','urban_share','prior_local_service','log_prior_treatment_count','D2_act_pre'],
 'prior_service_plus_osm':['logpop','urban_share','prior_local_service','log_prior_treatment_count','D1_osm'],
}
y=d.has_local.to_numpy(dtype=int)
blocklat=np.floor(d.latitude.to_numpy()/2).astype(int)
blocklon=np.floor(d.longitude.to_numpy()/2).astype(int)
blocks=blocklat*1000+blocklon
unique=np.unique(blocks)
rng=np.random.default_rng(20261111)
scores=[];fold_diagnostics=[];all_predictions=[]

def evaluate(scheme,rep,folds,buffered=False):
    oof={m:np.full(len(d),np.nan) for m in models}
    for fold in np.unique(folds):
        test=folds==fold;train=~test
        if buffered:
            test_coords=set(zip(blocklat[test],blocklon[test]))
            forbidden=set((r+dr,c+dc) for r,c in test_coords for dr in [-1,0,1] for dc in [-1,0,1])
            train &= np.array([(r,c) not in forbidden for r,c in zip(blocklat,blocklon)])
        fold_diagnostics.append({'scheme':scheme,'repartition':rep,'fold':str(fold),
            'n_train':int(train.sum()),'n_test':int(test.sum()),'buffered':buffered})
        if len(np.unique(y[train]))<2 or len(np.unique(y[test]))<2:
            raise ValueError((scheme,fold,'class absent'))
        for model,cols in models.items():
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                fit=LogisticRegression(max_iter=2000).fit(d.loc[train,cols],y[train])
            oof[model][test]=fit.predict_proba(d.loc[test,cols])[:,1]
    for model,probability in oof.items():
        assert np.isfinite(probability).all()
        scores.append({'scheme':scheme,'repartition':rep,'model':model,
            'auc':float(roc_auc_score(y,probability)),
            'brier':float(brier_score_loss(y,probability)),
            'log_loss':float(log_loss(y,probability))})
    if scheme.startswith('leave_region'):
        predictions=d[['m6','has_local','region']].copy()
        predictions['scheme']=scheme
        for model,p in oof.items():predictions[model]=p
        all_predictions.append(predictions)

for rep in range(50):
    permutation=rng.permutation(unique)
    mapping={b:i%5 for i,b in enumerate(permutation)}
    evaluate('spatial_blocks',rep,np.array([mapping[b] for b in blocks]))
regions=d.region.fillna('MISSING').astype(str).to_numpy()
assert 'MISSING' not in regions
evaluate('leave_region_out',0,regions)
evaluate('leave_region_out_buffered',0,regions,buffered=True)
pd.DataFrame(scores).to_csv(OUT/'model_scores.csv',index=False)
pd.DataFrame(fold_diagnostics).to_csv(OUT/'fold_diagnostics.csv',index=False)
pd.concat(all_predictions).to_csv(OUT/'region_oof_predictions.csv.gz',index=False)
d[['m6','prior_treatment_count','prior_local_service']].to_csv(OUT/'prior_service_aggregates.csv',index=False)
summary=[]
for scheme in pd.DataFrame(scores).scheme.unique():
    table=pd.DataFrame(scores).query('scheme == @scheme').pivot(index='repartition',columns='model',values='auc')
    for left,right in [('population_urban_activity','population_urban'),('population_urban_osm','population_urban'),
                       ('prior_service_plus_activity','prior_service_baseline'),('prior_service_plus_osm','prior_service_baseline')]:
        delta=table[left]-table[right]
        summary.append({'scheme':scheme,'comparison':left+' minus '+right,'mean_delta_auc':float(delta.mean()),
            'repartition_lo':float(delta.quantile(.025)),'repartition_hi':float(delta.quantile(.975)),
            'positive_repartitions':int((delta>0).sum()),'n_repartitions':len(delta)})
(OUT/'summary.json').write_text(json.dumps({'comparisons':summary,
 'prior_service_definition':'2017-2019 records grouped by treatment municipality, no 2024 information',
 'strict_two_code_cholecystectomy':args.strict_chole,
 'registry_prior_service':args.registry_prior_service,
 'registry_prior_definition':'2017-2019 procedure AIH counts grouped by linked hospital CNES2025 municipality' if args.registry_prior_service else None,
 'buffer_definition':'exclude training municipalities in same or immediately adjacent 2-degree block to any held-out-region block',
 'scope':'internal-external geographic validation within Brazil SUS cholecystectomy data',
 'range_definition':'percentiles across repartitions; region schemes each evaluated once'},indent=2))
print(json.dumps(summary),flush=True)
