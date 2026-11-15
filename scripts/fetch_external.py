"""List or retrieve explicitly named external files, validating known SHA-256.
No download occurs unless --id is supplied. SPDX-License-Identifier: MIT
"""
from pathlib import Path
import argparse,csv,hashlib,json,urllib.request
R=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--id');p.add_argument('--destination',type=Path,default=R/'external_data');a=p.parse_args()
 rows=list(csv.DictReader((R/'metadata/external_data_manifest.csv').open(encoding='utf-8-sig')))
 if not a.id:
  for row in rows:print(row['id'],row['access'],row['file_name'])
  return
 row=next((x for x in rows if x['id']==a.id),None)
 if row is None:p.error('unknown id')
 if row['access']!='direct-download':p.error('provider/manual retrieval required; see the source URL and docs/DATA_SOURCES.md')
 a.destination.mkdir(parents=True,exist_ok=True);dest=a.destination/row['file_name']
 if dest.exists():p.error('destination exists; verify it rather than overwrite')
 part=dest.with_suffix(dest.suffix+'.partial');h=hashlib.sha256()
 with urllib.request.urlopen(row['url'],timeout=120)as response,part.open('wb')as handle:
  while chunk:=response.read(1024*1024):h.update(chunk);handle.write(chunk)
 if row['sha256']and h.hexdigest()!=row['sha256']:raise ValueError('download checksum differs from study snapshot; partial file retained, not accepted')
 part.rename(dest);print(json.dumps({'file':str(dest),'sha256':h.hexdigest(),'source':row['url']}))
if __name__=='__main__':main()
