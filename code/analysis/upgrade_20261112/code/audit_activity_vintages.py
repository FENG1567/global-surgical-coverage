from pathlib import Path
import json
import hashlib
import pandas as pd
import pyarrow.parquet as pq

BASE=Path('__PROJECT_ROOT__/upgrade_20261112')
OUT=BASE/'derived/activity_vintage_audit';OUT.mkdir(parents=True,exist_ok=True)
path=Path('__SIH_PARQUET_ROOT__/sih_2024_cholec.parquet')
names=pq.read_schema(path).names
fields=[c for c in ['CNES','PROC_REA','ANO_CMPT','MES_CMPT','MUNIC_RES','MUNIC_MOV'] if c in names]
old=pd.read_parquet(path,columns=fields)
old['CNES']=pd.to_numeric(old.CNES,errors='coerce').astype('Int64').astype(str).str.zfill(7)
old_counts=old.groupby('CNES').size().rename('n_old').reset_index()
new=pd.read_csv(BASE/'inputs/brazil_hospital_annual.csv',dtype={'CNES':str})
new=new[new.year.eq(2024)&new.service.eq('chole')][['CNES','n']].rename(columns={'n':'n_new'})
comparison=old_counts.merge(new,on='CNES',how='outer',validate='one_to_one').fillna({'n_old':0,'n_new':0})
comparison['difference']=comparison.n_new-comparison.n_old
comparison.to_csv(OUT/'hospital_count_comparison.csv',index=False)
strict_old=old[old.PROC_REA.astype(str).isin(['0407030026','0407030034'])]
strict_counts=strict_old.groupby('CNES').size().rename('n_old_strict').reset_index()
strict_comparison=strict_counts.merge(new,on='CNES',how='outer',validate='one_to_one').fillna(0)
strict_comparison.to_csv(OUT/'strict_hospital_count_comparison.csv',index=False)
report={'old_parquet_rows':len(old),'old_parquet_schema':names,
 'old_hospitals':len(old_counts),'new_aihs':int(new.n_new.sum()),'new_hospitals':len(new),
 'old_strict_two_code_rows':len(strict_old),
 'old_strict_hospitals':len(strict_counts),
 'strict_hospital_counts_differ':int(strict_comparison.n_old_strict.ne(strict_comparison.n_new).sum()),
 'excess_old_rows_outside_primary_two_codes':len(old)-len(strict_old),
 'hospital_counts_differ':int(comparison.difference.ne(0).sum()),
 'old_source':str(path),'old_source_bytes':path.stat().st_size,
 'new_source_unit':'five-year local DATASUS RD processing-month archive, primary official procedure codes',
 'new_hospital_table_sha256':hashlib.sha256((BASE/'inputs/brazil_hospital_annual.csv').read_bytes()).hexdigest(),
 'interpretation':'the original cholec file includes three additional biliary procedure codes; strict count agreement is checked at hospital level'}
for col in ['PROC_REA','ANO_CMPT']:
    if col in old:report[col+'_counts']={str(k):int(v) for k,v in old[col].value_counts(dropna=False).items()}
(OUT/'summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
