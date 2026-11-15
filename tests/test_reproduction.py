"""Meaningful regression checks for the model source and spatial engine.
SPDX-License-Identifier: MIT
"""
from pathlib import Path
import importlib.util,json,subprocess,sys,tempfile,unittest
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
class Reproduction(unittest.TestCase):
 def test_final_supplement_keeps_different_repartitions(self):
  source=R/'code/analysis/upgrade_20261112/derived/bmc_revision_20261113/prediction/code/supplement_flexible_threshold1.py'
  module=load('final_flexible',source)
  model=module.MODEL
  features=pd.read_csv(R/'data/model_inputs/chole_municipality_features.csv',dtype={'m6':str})
  outcomes=model.load_outcomes(R/'data/model_inputs/brazil_municipality_annual.csv')
  predictions,scores,_=module.fit_service(features,outcomes,'chole',2,20261111)
  for name in module.FLEXIBLE_MODELS:
   a=predictions[predictions.repartition.eq(0)].sort_values('m6')['p_'+name].to_numpy()
   b=predictions[predictions.repartition.eq(1)].sort_values('m6')['p_'+name].to_numpy()
   self.assertFalse(np.array_equal(a,b),name+' repeats the final OOF vector')
  reference=pd.read_csv(R/'data/results/prediction/flexible_n1_fixed/flexible_threshold1_scores_by_repartition.csv')
  expected=reference[reference.service.eq('chole')&reference.repartition.lt(2)]
  joined=scores.merge(expected,on=['service','threshold','repartition','model'],suffixes=('_new','_reference'),validate='one_to_one')
  self.assertEqual(len(joined),8)
  for field in ['auc','brier','log_loss']:self.assertLess(np.abs(joined[field+'_new']-joined[field+'_reference']).max(),1e-7)
 def test_native_travel_and_exact_milp_against_independent_references(self):
  with tempfile.TemporaryDirectory(prefix='surgical-repro-test-')as directory:
   temp=Path(directory);config=temp/'paths.json';config.write_text(json.dumps({'workspace':str(temp/'work'),'overrides':{}}))
   subprocess.run([sys.executable,str(R/'scripts/materialize_analysis.py'),'--config',str(config)],check=True,capture_output=True,text=True)
   project=temp/'work/project'
   for name in ['test_travel_engine.py','test_exact_bounds.py']:
    result=subprocess.run([sys.executable,str(project/'upgrade_20261112/code'/name)],check=True,capture_output=True,text=True)
    self.assertTrue(result.stdout.strip())
if __name__=='__main__':unittest.main()
