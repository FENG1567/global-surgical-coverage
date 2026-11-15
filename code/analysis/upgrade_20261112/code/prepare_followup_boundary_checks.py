from pathlib import Path

ROOT = Path('__PROJECT_ROOT__/upgrade_20261112')
for country, reference, filename in [
    ('BRA', 'brazil_state_constraints', 'bra_state_boundary.py'),
    ('MEX', 'mexico_state_constraints', 'mex_state_boundary.py'),
    ('MEX', 'mexico_hospital_only', 'mex_hospital_boundary.py'),
]:
    source_name = 'brazil_spatial_experiment.py' if country == 'BRA' else 'mexico_spatial_experiment.py'
    source = (ROOT / 'code' / source_name).read_text(encoding='utf8').split("evaluate('all_participants'")[0]
    old_tag = 'brazil_spatial_v1' if country == 'BRA' else 'mexico_spatial_joint'
    source = source.replace(f"default='{old_tag}'", f"default='{reference}_margin4'")
    source = source.replace("--margin-degrees',type=int,default=2", "--margin-degrees',type=int,default=4")
    source += f"\nmain=ROOT/'derived/{reference}'\n"
    source += '''
reference=pd.read_csv(main/'coverage_results.csv')
lookup=pd.Series(np.arange(len(h)),index=h.CLUES)
for service in sorted(set(reference.service)-{'all'}):
    pattern='observed|state_joint_cluster' if 'state_constraints' in main.name else 'observed|jointmatched|jointsearch'
    feasible=reference[reference.service.eq(service)&reference.scenario.str.contains(pattern)]
    chosen_labels=list(dict.fromkeys([service+'_observed',
        feasible.loc[feasible.all_30.idxmin(),'scenario'],feasible.loc[feasible.all_30.idxmax(),'scenario']]))
    for label in chosen_labels:
        identifiers=pd.read_csv(main/f'{label}_selection.csv',dtype={'CLUES':str})
        selected=lookup.loc[identifiers.CLUES].to_numpy()
        expected=reference[reference.scenario.eq(label)].iloc[0]
        evaluate(label,selected,service,int(expected.k),int(expected.urban_count))
comparison=pd.DataFrame(rows).merge(reference,on='scenario',suffixes=('_margin4','_margin2'),validate='one_to_one')
metrics=[c for c in reference if c.startswith(('all_','urban_','rural_')) and c.rsplit('_',1)[-1].isdigit()]
maximum=max(float((comparison[c+'_margin4']-comparison[c+'_margin2']).abs().max()) for c in metrics)
comparison.to_csv(OUT/'boundary_comparison.csv',index=False)
(OUT/'audit.json').write_text(json.dumps({'margin_degrees':args.margin_degrees,'n_comparisons':len(rows),
    'maximum_coverage_difference_pp':maximum*100,'reference':main.name,
    'selection_rule':'observed plus minimum and maximum 30-minute coverage among feasible constructions'},indent=2))
print('FOLLOWUP_BOUNDARY_CHECK_COMPLETE',main.name,maximum*100,flush=True)
'''
    (ROOT / 'code' / filename).write_text(source, encoding='utf8')
