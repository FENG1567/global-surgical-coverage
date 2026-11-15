"""Reconcile observed SAEH facilities with the historical CLUES register."""
from pathlib import Path
import hashlib
import json
import pandas as pd

BASE=Path('__BRAZIL_SOURCE_ROOT__/mexico/data')
OUT=Path('__PROJECT_ROOT__/upgrade_20261112/derived/mexico')
OUT.mkdir(parents=True,exist_ok=True)
registry=pd.read_csv(BASE/'clues_clean.csv',dtype={'CLUES':str,'ENTIDAD':str,'MUNIC':str,'tipo_clave':str})
observed=pd.read_csv(BASE/'mexico_hospital_points.csv',dtype={'CLUES':str,'ENTIDAD':str,'MUNIC':str,'tipo_clave':str})
assert not observed.CLUES.duplicated().any()
assert not registry.CLUES.duplicated().any()
observed['coordinate_valid']=observed.lat.between(-90,90)&observed.lon.between(-180,180)
observed['institution_prefix']=observed.CLUES.str[2:5]
observed['chole_observed']=observed.n_chole.gt(0)
observed['cesarean_observed']=observed.n_cesarea.gt(0)
observed['candidate_basis']='at_least_one_SAEH_2023_inpatient_record_in_existing_extraction'
observed.to_csv(OUT/'observed_candidate_facilities.csv',index=False)
state=observed.groupby(['ENTIDAD','coordinate_valid']).agg(n_facilities=('CLUES','size'),
    n_admissions=('n_total','sum'),n_chole=('n_chole','sum'),n_cesarea=('n_cesarea','sum')).reset_index()
state.to_csv(OUT/'coordinates_by_state.csv',index=False)
prefixes=set(observed.institution_prefix)
eligible=registry[registry.tipo_clave.eq('2')&registry.estatus.eq('EN OPERACION')&registry.CLUES.str[2:5].isin(prefixes)].copy()
eligible['observed_in_2023_extract']=eligible.CLUES.isin(observed.CLUES)
eligible.to_csv(OUT/'historical_registry_eligible.csv',index=False)
report={'scope':'SAEH SSA 2023 observed participating facilities, not all Mexican hospitals',
 'observed_facilities':len(observed),'valid_coordinates':int(observed.coordinate_valid.sum()),
 'observed_institution_prefixes':observed.institution_prefix.value_counts().to_dict(),
 'observed_registry_type':observed.tipo.value_counts(dropna=False).to_dict(),
 'historical_registry_eligible_facilities':len(eligible),
 'eligible_unobserved_in_2023':int((~eligible.observed_in_2023_extract).sum()),
 'limitations':['2019 registry and 2023 activity differ in time','unobserved registry facilities have unknown 2023 participation and cannot automatically receive zero capability labels',
 'primary allocation experiment uses coordinate-valid observed participating facilities, conditioning on this selected facility universe',
 'existing hospital-level extraction has not been regenerated from individual records in this run'],
 'services':{},'input_sha256':{n:hashlib.sha256((BASE/n).read_bytes()).hexdigest() for n in ['clues_clean.csv','mexico_hospital_points.csv']}}
for service,col in [('chole','n_chole'),('cesarean','n_cesarea')]:
 good=observed.coordinate_valid
 report['services'][service]={'active_all':int(observed[col].gt(0).sum()),
   'active_with_coordinates':int((observed[col].gt(0)&good).sum()),
   'recorded_admissions':int(observed[col].sum()),
   'case_weighted_coordinate_coverage':float(observed.loc[good,col].sum()/observed[col].sum())}
(OUT/'candidate_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(report,ensure_ascii=False),flush=True)
