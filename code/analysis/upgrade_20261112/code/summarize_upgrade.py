from pathlib import Path
import json
import shutil
import pandas as pd

ROOT=Path('__PROJECT_ROOT__/upgrade_20261112')
DATA=ROOT/'server_results/derived'
OUT=ROOT/'final_tables';OUT.mkdir(parents=True,exist_ok=True)
width_rows=[]
for country,folder in [('BRA','brazil_spatial_v1'),('MEX','mexico_joint_search')]:
    table=pd.read_csv(DATA/folder/'coverage_results.csv')
    for service in table.service.unique():
        if service=='all':continue
        observed=table[table.scenario.eq(service+'_observed')].iloc[0]
        for condition in ['count_only','count_and_local_population_class','count_and_joint_composition']:
            selected=table[table.service.eq(service)&table.k.eq(observed.k)].copy()
            if condition=='count_and_local_population_class':
                selected=selected[selected.scenario.str.contains('observed|qmatched|jointmatched|jointsearch')]
                assert selected.urban_count.eq(observed.urban_count).all()
            if condition=='count_and_joint_composition':
                selected=selected[selected.scenario.str.contains('observed|jointmatched|jointsearch')]
                assert selected.urban_count.eq(observed.urban_count).all()
                assert selected.smod_urban_count.eq(observed.smod_urban_count).all()
            for threshold in [30,120]:
                values=selected[f'all_{threshold}']
                width_rows.append({'country':country,'service':service,'constraint':condition,
                    'threshold_minutes':threshold,'n_candidates':int(observed.n_candidates),'k':int(observed.k),
                    'n_feasible_constructions':len(selected),'observed_coverage_percent':observed[f'all_{threshold}']*100,
                    'low_percent':values.min()*100,'high_percent':values.max()*100,
                    'width_lower_bound_pp':(values.max()-values.min())*100,
                    'min_scenario':selected.loc[values.idxmin(),'scenario'],
                    'max_scenario':selected.loc[values.idxmax(),'scenario'],
                    'observed_local_population_q':observed.realized_q,
                    'observed_smod_urban_share':observed.smod_urban_share})
widths=pd.DataFrame(width_rows)
widths.to_csv(OUT/'constructed_widths.csv',index=False)

increments=pd.read_csv(DATA/'multiservice_validation/paired_increment_summary.csv')
inclusion=pd.read_csv(DATA/'multiservice_validation/inclusion.csv')
cv=increments[increments.comparison.eq('+activity - baseline')].merge(inclusion,on='service')
cv.to_csv(OUT/'multiservice_spatial_validation.csv',index=False)
temporal=pd.read_csv(DATA/'temporal_multiservice_validation/model_scores.csv')
temporal_rows=[];strong_rows=[]
for service in ['chole','cesarean','appendectomy']:
    t=temporal[temporal.service.eq(service)].set_index('model')
    temporal_rows.append({'service':service,'train_n':int(t.loc['baseline','n_train']),
        'test_n':int(t.loc['baseline','n_test']),'baseline_auc':t.loc['baseline','auc'],
        'activity_auc':t.loc['baseline+activity','auc'],
        'activity_delta_auc':t.loc['baseline+activity','auc']-t.loc['baseline','auc'],
        'osm_delta_auc':t.loc['baseline+OSM','auc']-t.loc['baseline','auc'],
        'brier_improvement':t.loc['baseline','brier']-t.loc['baseline+activity','brier']})
    strong=json.loads((DATA/f'{service}_registry_prior_extensions_v2/summary.json').read_text())
    for item in strong['comparisons']:
        if item['comparison']=='prior_service_plus_activity minus prior_service_baseline':
            strong_rows.append({'service':service,**item})
pd.DataFrame(temporal_rows).to_csv(OUT/'temporal_validation.csv',index=False)
pd.DataFrame(strong_rows).to_csv(OUT/'prior_service_adjusted_validation.csv',index=False)

audit=pd.read_csv(ROOT/'derived/brazil/file_audit_2017_2018_2019_2023_2024.csv')
hospital=pd.read_csv(ROOT/'derived/brazil/hospital_annual_2017_2018_2019_2023_2024.csv')
assert len(audit)==1620 and audit.groupby('year').size().eq(324).all()
assert audit.invalid_hospital.eq(0).all() and audit.invalid_municipality_pair.eq(0).all()
counts=hospital.groupby(['year','service']).agg(aihs=('n','sum'),active_facilities=('CNES','nunique')).reset_index()
counts.to_csv(OUT/'brazil_annual_activity.csv',index=False)
summary={'source_files_processed':len(audit),'raw_aih_records':int(audit.records.sum()),
 'selected_aih_records':int(audit.selected_records.sum()),'source_years':sorted(audit.year.unique().tolist()),
 'within_file_selected_aih_duplicate_count':int(audit.duplicate_aih_selected_within_file.sum()),
 'count_unit':'AIH records, not unique patients',
 'code_dictionary_months_checked':60,
 'certification':json.loads((ROOT/'derived/certification_final/summary.json').read_text()),
 'global_witness':json.loads((DATA/'global_witness/results.json').read_text()),
 'composition_curve':json.loads((DATA/'composition_curve/summary.json').read_text()),
 'brazil_boundary':json.loads((DATA/'brazil_boundary_sensitivity/audit.json').read_text()),
 'strict_chole_raw_code_agreement':json.loads((DATA/'activity_vintage_audit/summary.json').read_text()),
 'joint_quota_recount':json.loads((OUT/'followup_constraint_audit.json').read_text(encoding='utf8')),
 'additional_boundary_checks':{name:json.loads((DATA/name/'audit.json').read_text(encoding='utf8')) for name in
     ['brazil_state_constraints_margin4','mexico_state_constraints_margin4','mexico_hospital_only_margin4']},
 'source_reconciliation':'released aggregates reconciled across sources; independent algorithm cross-checks included'}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
shutil.copy2(ROOT/'derived/certification_final/certified_27_full_population.csv',OUT/'certified_27_full_population.csv')
print(widths[widths.constraint.eq('count_and_joint_composition')&widths.threshold_minutes.eq(30)].to_string(index=False),flush=True)
print(cv[['service','mean','lo','hi','analysis_municipalities','n_positive']].to_string(index=False),flush=True)
print(pd.DataFrame(strong_rows)[['service','scheme','mean_delta_auc']].to_string(index=False),flush=True)
print('UPGRADE_TABLES_COMPLETE',flush=True)
