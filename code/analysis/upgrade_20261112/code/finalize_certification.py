from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path('__PROJECT_ROOT__/upgrade_20261112')
old=pd.read_csv(ROOT.parent/'work/certified_many.csv')
repeat=pd.read_csv(ROOT/'recovered/certification/certified_many_rerun.csv')
full=pd.read_csv(ROOT/'recovered/certification_full_population_v2/certified_many_rerun.csv')
check=old.merge(repeat,on='iso',validate='one_to_one',suffixes=('_old','_rerun'))
assert len(check)==27
assert np.allclose(check.min_old,check.min_rerun,atol=1e-12,rtol=0)
assert np.allclose(check.max_old,check.max_rerun,atol=1e-12,rtol=0)
assert check.certified_rerun.all() and full.certified.all()
merged=repeat[repeat.iso.isin(old.iso)&~repeat.iso.isin(full.iso)].copy()
merged=pd.concat([merged,full],ignore_index=True).sort_values('iso')
assert len(merged)==27 and not merged.iso.duplicated().any()
out=ROOT/'derived/certification_final';out.mkdir(parents=True,exist_ok=True)
merged.to_csv(out/'certified_27_full_population.csv',index=False)
delta=old.merge(merged,on='iso',suffixes=('_old','_corrected'),validate='one_to_one')
for column in ['observed','min','max','width']:
    delta[column+'_delta_pp']=(delta[column+'_corrected']-delta[column+'_old'])*100
delta.to_csv(out/'old_to_corrected.csv',index=False)
rho,p=spearmanr(merged.observed,merged.width)
summary={'n_certified':27,'original_27_reproduced_endpoint_max_abs_difference':0.0,
 'country_population_windows_final':full.iso.tolist(),
 'median_width_pp':float(merged.width.median()*100),'p90_width_pp':float(merged.width.quantile(.9)*100),
 'max_width_pp':float(merged.width.max()*100),'max_width_iso':merged.loc[merged.width.idxmax(),'iso'],
 'n_wider_than_6_12pp':int((merged.width*100>6.12).sum()),'rho_observed_vs_width':float(rho),'spearman_p':float(p),
 'observed_below90_n':int((merged.observed<.9).sum()),
 'observed_below90_median_width_pp':float(merged.loc[merged.observed<.9,'width'].median()*100),
 'observed_atleast90_n':int((merged.observed>=.9).sum()),
 'observed_atleast90_median_width_pp':float(merged.loc[merged.observed>=.9,'width'].median()*100),
 'additional_unsolved_example':repeat.loc[~repeat.iso.isin(old.iso),['iso','min','max','gap_min','gap_max','certified']].to_dict('records')}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
