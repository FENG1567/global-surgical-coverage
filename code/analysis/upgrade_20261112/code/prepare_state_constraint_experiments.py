from pathlib import Path

ROOT=Path('__PROJECT_ROOT__/upgrade_20261112')
for country,filename,default_tag,state_column,services in [
 ('BRA','brazil_spatial_experiment.py','brazil_state_constraints','uf',['chole','cesarean','appendectomy']),
 ('MEX','mexico_spatial_experiment.py','mexico_state_constraints','ENTIDAD',['chole','cesarean'])]:
    source=(ROOT/'code'/filename).read_text(encoding='utf8')
    source=source.split("evaluate('all_participants'")[0]
    prior_default='brazil_spatial_v1' if country=='BRA' else 'mexico_spatial_joint'
    source=source.replace(f"default='{prior_default}'",f"default='{default_tag}'")
    source+=f'''
SERVICES={services!r}
STATE_COLUMN={state_column!r}
assert h[STATE_COLUMN].notna().all()
state_values=h[STATE_COLUMN].astype(str).to_numpy()
groups=pd.MultiIndex.from_arrays([state_values,u.astype(int),smod_urban.astype(int)]).factorize()[0]
all_groups=np.unique(groups)
for service in SERVICES:
    count_column='n_'+service+'_2024' if {country!r}=='BRA' else ('n_chole' if service=='chole' else 'n_cesarea')
    observed=np.flatnonzero(h[count_column].to_numpy()>0)
    k=len(observed);uk=int(u[observed].sum())
    quotas={{int(group):int((groups[observed]==group).sum()) for group in all_groups}}
    evaluate(service+'_observed',observed,service,k,uk)
    for repetition in range(8):
        selected=np.concatenate([cluster(np.flatnonzero(groups==group),quotas[int(group)]) for group in all_groups])
        assert all(int((groups[selected]==group).sum())==quotas[int(group)] for group in all_groups)
        evaluate(service+f'_state_joint_cluster_{{repetition:02d}}',selected,service,k,uk)
        selected_frame=h.iloc[selected]
        pd.DataFrame({{'state':state_values[selected],'local_population_class':u[selected],
            'smod_urban':smod_urban[selected]}}).value_counts().rename('n_selected').reset_index().to_csv(OUT/f'{{service}}_state_budget_{{repetition:02d}}.csv',index=False)
(OUT/'audit.json').write_text(json.dumps({{'country':{country!r},'state_column':STATE_COLUMN,
    'constraints':'exact observed facility counts within each state crossed with both local-population and SMOD-urban facility classes',
    'clustered_repetitions':8,'seed':20261112,'country_population_retained':float(weights.sum(dtype=np.float64)/countryrow.population),
    'interpretation':'feasible-construction lower bounds under stronger observed spatial aggregates; no optimization certificate or real-error estimate'}},indent=2))
print('STATE_CONSTRAINT_EXPERIMENT_COMPLETE',{country!r},flush=True)
'''
    (ROOT/'code'/f'{country.lower()}_state_constraints.py').write_text(source,encoding='utf8')
