"""Verify the release inventory, source version, budgets and data boundaries.
SPDX-License-Identifier: MIT
"""
from pathlib import Path
import argparse,ast,csv,hashlib,json,re
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb')as stream:
  for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def verify():
 files=[p for p in R.rglob('*')if p.is_file()and not any(x in p.relative_to(R).parts for x in ['.git','__pycache__','artifacts','workspace','.venv','external_data'])]
 assert not[p for p in files if p.stat().st_size>=25*1024*1024],'file exceeds GitHub browser limit'
 assert not[p for p in files if p.suffix.lower()in ['.dbc','.dbf','.parquet','.f32','.u8','.docx','.pem','.key','.log']]
 for p in files:
  if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
  assert 'completion_manifest.json'!=p.name
  assert not('flexible_n1'in p.parts),'invalid old model output included'
 for source in (R/'data/model_inputs').glob('*.csv'):
  fields=set(pd.read_csv(source,nrows=0).columns)
  assert not fields.intersection({'N_AIH','DT_INTER','DT_SAIDA','patient_id','patient_name'})
 for name in ['rand','cluster']:
  budget=pd.read_csv(R/f'data/results/global/selection_{name}_budget.csv')
  assert len(budget)==budget.iso3.nunique()==216
  assert budget.n_selected.eq(budget.k_c).all()and budget.n_selected.sum()==184367
  selected=pd.read_csv(R/f'data/results/global/selection_{name}_design.csv.gz')
  assert len(selected)==selected.facility_idx.nunique()==184367
  actual=selected.groupby('iso3').size().reindex(budget.iso3).to_numpy()
  assert np.array_equal(actual,budget.k_c.to_numpy())
 scenario=pd.read_csv(R/'data/results/global/scenario_budget_final.csv').set_index('scenario')
 spread=100*(scenario.loc['rand','c30']-scenario.loc['cluster','c30'])
 assert abs(spread-5.583633)<.00001,spread
 fixed=json.loads((R/'data/results/prediction/flexible_n1_fixed/result_manifest.json').read_text())
 expected=fixed['executables']['supplement_script_sha256']
 source=R/'code/analysis/upgrade_20261112/derived/bmc_revision_20261113/prediction/code/supplement_flexible_threshold1.py'
 assert sha(source)==expected,'fixed result/source hash mismatch'
 provenance=json.loads((R/'metadata/source_provenance.json').read_text())
 entry=next(x for x in provenance if x['file']==str(source.relative_to(R)))
 assert entry['source_sha256']==expected
 assert len(list((R/'data/tables').glob('*.csv')))==28
 assert len(json.loads((R/'metadata/table_display_spec.json').read_text())['tables'])==27
 for i in range(1,24):assert(R/f'data/tables/Table_S{i}.csv').is_file()
 for row in csv.DictReader((R/'docs/FIGURE_MAP.csv').open()):
  for item in row['source_csv_relative_to_data_figure_sources'].split(';'):assert(R/'data/figure_sources'/item).is_file(),item
 for row in csv.DictReader((R/'metadata/external_data_manifest.csv').open()):
  if row['file_name'].endswith('.shp')and'.zip'in row['url']:assert row['access']=='provider-manual'
 manifest=R/'metadata/release_manifest.json'
 if manifest.exists():
  for row in json.loads(manifest.read_text())['files']:
   assert sha(R/row['path'])==row['sha256'],'release checksum mismatch: '+row['path']
 return {'status':'pass','files_checked':len(files),'main_and_supplementary_tables':27,'original_country_budgets':'216/216 exact, 184367 facilities in each selection','original_constrained_width_lower_bound_pp':float(spread),'fixed_source_sha256':expected,'all_files_below_25_MiB':True,'scope':'release checks and explicit data boundaries; not fresh raw/geographic analysis'}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);a=p.parse_args();result=verify()
 if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
