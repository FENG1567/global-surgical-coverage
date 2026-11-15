"""Refit published denominator models from released municipal aggregates.
SPDX-License-Identifier: MIT
"""
from pathlib import Path
import argparse,importlib.util,json
import numpy as np
import pandas as pd
REPO=Path(__file__).resolve().parents[1]

def helper():
    path=REPO/'code/analysis/upgrade_20261112/derived/bmc_revision_20261113/prediction/code/reanalyse_prediction.py'
    spec=importlib.util.spec_from_file_location('prediction_release',path)
    model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
    return model

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--service',choices=['chole','cesarean','appendectomy','all'],default='chole')
    p.add_argument('--threshold',type=int,choices=[1,5,10,20],default=1)
    p.add_argument('--repartitions',type=int,default=1,help='10 for the full published denominator design')
    p.add_argument('--output-dir',type=Path,default=REPO/'artifacts/model_refit')
    a=p.parse_args()
    if not 1<=a.repartitions<=10:p.error('repartitions must be between 1 and 10')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    m=helper();outcomes=m.load_outcomes(REPO/'data/model_inputs/brazil_municipality_annual.csv')
    reference=pd.read_csv(REPO/'data/results/prediction/flexible_n1_fixed/denominator_spatial_scores_by_repartition_merged.csv')
    reports=[]
    for service in (m.SERVICES if a.service=='all'else[a.service]):
        features=pd.read_csv(REPO/f'data/model_inputs/{service}_municipality_features.csv',dtype={'m6':str})
        observed=m.observed_outcome(features,outcomes,service,2024)
        predictions,scores,diagnostics=m.denominator_cv(observed,service,a.threshold,a.repartitions,20261111)
        target=reference.loc[reference.service.eq(service)&reference.threshold.eq(a.threshold)&reference.repartition.lt(a.repartitions)]
        joined=scores.merge(target,on=['service','threshold','repartition','model'],suffixes=('_refit','_reference'),validate='one_to_one')
        assert len(joined)==len(scores)==len(target),'model/repartition keys mismatch'
        differences={field:float(np.abs(joined[field+'_refit']-joined[field+'_reference']).max())for field in ['auc','brier','log_loss']}
        assert max(differences.values())<1e-7,differences
        scores.to_csv(a.output_dir/f'{service}_scores.csv',index=False)
        diagnostics.to_csv(a.output_dir/f'{service}_fold_diagnostics.csv',index=False)
        predictions.to_csv(a.output_dir/f'{service}_oof_predictions.csv.gz',index=False)
        reports.append({'service':service,'threshold':a.threshold,'repartitions':a.repartitions,'models':scores.model.nunique(),'max_absolute_score_difference':differences})
    (a.output_dir/'verification.json').write_text(json.dumps({'status':'pass','analyses':reports,'scope':'refitting from released city-level analytical inputs; not raw SIH or raster regeneration'},indent=2))
    print(json.dumps(reports),flush=True)
if __name__=='__main__':main()
