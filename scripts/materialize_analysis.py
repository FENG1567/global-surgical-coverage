"""Relocate archived scientific code into a configurable external workspace.
SPDX-License-Identifier: MIT
"""
from pathlib import Path
import argparse,ast,json,re,subprocess,sys
R=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--config',type=Path,default=R/'config/paths.example.json')
 p.add_argument('--run',help='script path relative to code/analysis; requires existing external inputs')
 p.add_argument('arguments',nargs=argparse.REMAINDER)
 a=p.parse_args();c=json.loads(a.config.read_text())
 workspace=Path(c['workspace']).expanduser()
 if not workspace.is_absolute():workspace=(R/workspace).resolve()
 values={'__PROJECT_ROOT__':workspace/'project','__BRAZIL_SOURCE_ROOT__':workspace/'brazil_sources','__SIH_ROOT__':workspace/'sih','__SCRATCH_ROOT__':workspace/'scratch','__AUTHOR_OUTPUTS__':workspace/'author_outputs','__SIH_PARQUET_ROOT__':workspace/'sih_parquet','__SIGTAP_ROOT__':workspace/'sigtap'}
 for key,value in c.get('overrides',{}).items():values[key]=Path(value).expanduser().resolve()
 target=values['__PROJECT_ROOT__'];target.mkdir(parents=True,exist_ok=True)
 for src in (R/'code/analysis').rglob('*'):
  if not src.is_file()or src.suffix not in ['.py','.c','.R','.sh']:continue
  dest=target/src.relative_to(R/'code/analysis');dest.parent.mkdir(parents=True,exist_ok=True)
  text=src.read_text()
  text=re.sub(r'__(?:PROJECT_ROOT|BRAZIL_SOURCE_ROOT|SIH_ROOT|SCRATCH_ROOT|AUTHOR_OUTPUTS|SIH_PARQUET_ROOT|SIGTAP_ROOT)__[^\n\"\']*',lambda match:match.group().replace('\\','/'),text)
  for token,path in values.items():text=text.replace(token,path.as_posix())
  dest.write_text(text)
  if dest.suffix=='.py':ast.parse(text,filename=str(dest))
 native=target/'upgrade_20261112/derived/mexico_spatial_v2';native.mkdir(parents=True,exist_ok=True)
 subprocess.run(['gcc','-O3','-o',str(native/'ttdijkstra_crop'),str(target/'native/ttdijkstra_crop.c'),'-lm'],check=True)
 subprocess.run(['gcc','-O3','-o',str(target/'src/ttdijkstra'),str(target/'src/ttdijkstra.c'),'-lm'],check=True)
 print(json.dumps({'materialized_project':str(target),'inputs':'external; see docs/FULL_ANALYSIS.md'}),flush=True)
 if a.run:
  source=(target/a.run).resolve()
  if not source.is_relative_to(target.resolve())or not source.is_file():p.error('run must name a staged analysis script')
  extra=a.arguments[1:]if a.arguments[:1]==['--']else a.arguments
  subprocess.run([sys.executable,str(source),*extra],cwd=target,check=True)
if __name__=='__main__':main()
