"""Export all published source tables and verify main-table arithmetic.
SPDX-License-Identifier: MIT
"""
from pathlib import Path
import argparse,json,shutil,csv
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]
NAMES={'chole':'Cholecystectomy','cesarean':'Cesarean delivery','appendectomy':'Appendectomy'}
def close(a,b,tol=1e-5):assert abs(float(a)-float(b))<tol,(a,b)
def verify():
    table=pd.read_csv(R/'data/tables/Main_Table_2.csv')
    bounds=pd.read_csv(R/'data/results/clinical/m2_bounds.csv')
    for service,name in NAMES.items():
        part=bounds[bounds.service.eq(service)]
        row=table[table.Procedure.eq(name)&table.Candidates.eq('Structure')].iloc[0]
        close(row['30-min width ≥ (pp)'],100*(part.maximum_30.max()-part.minimum_30.min()),.006)
        close(row['120-min width ≥ (pp)'],100*(part.maximum_120.max()-part.minimum_120.min()),.006)
    table=pd.read_csv(R/'data/tables/Main_Table_3.csv')
    scores=pd.read_csv(R/'data/results/prediction/flexible_n1_fixed/denominator_spatial_absolute_performance_merged.csv')
    for service,name in NAMES.items():
        row=table[table.Procedure.eq(name)].iloc[0]
        subset=scores[scores.service.eq(service)&scores.eligibility_threshold_resident_AIHS.eq(1)].set_index('model')
        for model,column in [('population_urban','Baseline AUC'),('population_urban_activity','+Activity AUC'),('strong_baseline','Strong AUC'),('strong_baseline_plus_activity','Strong+activity AUC')]:
            close(row[column],subset.loc[model,'auc_mean'],.000006)
        close(row['Resident AIHs'],subset.loc['population_urban','resident_AIHS'])
    table=pd.read_csv(R/'data/tables/Main_Table_4.csv')
    metrics=pd.read_csv(R/'data/results/clinical/m1_scenario_metrics.csv')
    sets={'full_original':'Original universe','B_structure_positive_or_unknown':'B: positive or unknown'}
    rules={'random':'Random (12 assignments)','high_local_population_priority':'High local population','low_local_population_priority':'Low local population'}
    for (candidate,rule),part in metrics.groupby(['candidate_set','rule']):
        row=table[table['Candidate set'].eq(sets[candidate])&table.Rule.eq(rules[rule])].iloc[0]
        for original,column in [('national_abs_error_pp','National MAE (pp)'),('mean_absolute_state_error_pp','State MAE (pp)'),('state_rank_spearman','State rank correlation'),('lowest_five_state_overlap','Lowest-five overlap')]:close(row[column],part[original].mean(),.0006)
    return {'status':'pass','main_tables_2_to_4':'recomputed checks from underlying analysis aggregates','table_1':'seven-statistic summary derived from all 27 certified country rows; manuscript Table S4 (analytical source Table_S2.csv)','supplementary_tables':'source-table export; underlying analyses are separate'}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=R/'artifacts/tables');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    result=verify();display=json.loads((R/'metadata/table_display_spec.json').read_text())['tables']
    t1=pd.read_csv(R/'data/tables/Main_Table_1.csv',dtype=str).fillna('')
    assert t1.columns.tolist()==display['Main_Table_1']['columns'] and t1.values.tolist()==display['Main_Table_1']['rows']
    endpoints=pd.read_csv(R/'data/tables/Table_S2.csv');assert len(endpoints)==27
    summary=t1.set_index('Measure')['Value'];close(summary['Median width (pp)'],100*endpoints.width.median(),.006);close(summary['90th percentile width (pp)'],100*endpoints.width.quantile(.9),.006);close(summary['Maximum width (pp)'],100*endpoints.width.max(),.006)
    crosswalk=list(csv.DictReader((R/'metadata/publication_table_crosswalk.csv').open()))
    for row in crosswalk:
        current=row['manuscript_table'];src=R/row['analytical_source']
        shutil.copy2(src,a.output_dir/(current+'.csv'))
        spec=display[current];d=pd.DataFrame(spec['rows'],columns=spec['columns'])
        lines=['| '+' | '.join(map(str,d.columns))+' |','| '+' | '.join(['---']*len(d.columns))+' |']
        lines+=['| '+' | '.join(str(x).replace('|',chr(92)+'|').replace(chr(10),'<br>')for x in row)+' |'for row in d.itertuples(index=False,name=None)]
        (a.output_dir/(current+'.md')).write_text(chr(10).join(lines)+chr(10))
    shutil.copy2(R/'data/tables/Main_Table_1_full_fields.csv',a.output_dir/'Main_Table_1_full_fields.csv')
    result['tables']=len(display);result['source_csv_files']=len(crosswalk)+1
    assert len(display)==27 and len(crosswalk)==27
    result['numbering']='current manuscript numbering; analytical paths mapped by metadata/publication_table_crosswalk.csv'
    result['display']='Native publication table strings, including paired AUC columns; underlying analytical CSVs also exported'
    (a.output_dir/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
