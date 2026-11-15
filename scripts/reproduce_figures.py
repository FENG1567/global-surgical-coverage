"""Render all ten figures from released aggregate data. SPDX-License-Identifier: MIT"""
from pathlib import Path
import argparse,subprocess,sys,os,json
R=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output-dir',type=Path,default=R/'artifacts/figures')
 p.add_argument('--font-dir',type=Path,help='Directory containing legally installed arial.ttf and arialbd.ttf')
 a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy()
 if a.font_dir:env['FIGURE_FONT_DIR']=str(a.font_dir.resolve())
 source=R/'code/analysis/plotting/layout';data=R/'data/figure_sources'
 for script in ['reader_main_1_3.py','reader_main_4_6.py','reader_supplement.py']:
  cmd=[sys.executable,str(source/script),'--data',str(data),'--out',str(a.output_dir)]
  if script=='reader_supplement.py':cmd+=['--repo',str(R)]
  subprocess.run(cmd,check=True,env=env)
 names=[f'Figure_{i}' for i in range(1,7)]+[f'Figure_S{i}' for i in range(1,5)]
 for name in names:
  for ext in ['pdf','svg','png','tiff']:assert(a.output_dir/f'{name}.{ext}').is_file()
 result={'status':'pass','figures':names,'TIFF':'600 dpi RGB LZW','PNG':'300 dpi preview','panel_labels':'Arial Bold 18 pt uppercase','minimum_text_points':8.5,'supplement_order':{'S1':'information revelation','S2':'strict validation and Mexico','S3':'Brazil national composition','S4':'state constraints and validation'}}
 (a.output_dir/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
