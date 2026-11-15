"""Aggregate public SIH DBC files to hospital and residence-municipality counts."""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import re
import shutil
import time
import numpy as np
import pandas as pd
from dbfread import DBF
from pyreaddbc import dbc2dbf

CODES = {
    'chole': ['0407030026','0407030034'],
    'cesarean': ['0411010026','0411010034','0411010042'],
    'appendectomy': ['0407020039','0407020047'],
}
ROOT = Path('__PROJECT_ROOT__/upgrade_20261112')
STAGE = Path('__SCRATCH_ROOT__')


def one(source_string):
    source = Path(source_string)
    match = re.fullmatch(r'RD([A-Z]{2})(\d{2})(\d{2})\.dbc',source.name,re.I)
    uf, yy, mm = match.groups()
    year, month = 2000+int(yy), int(mm)
    destination = ROOT/'derived/brazil/parts'/source.stem
    if (destination/'audit.json').exists():
        return json.loads((destination/'audit.json').read_text())
    destination.mkdir(parents=True,exist_ok=True)
    stage = STAGE/source.stem
    stage.mkdir(parents=True,exist_ok=True)
    compressed, decoded = stage/'source.dbc', stage/'decoded.dbf'
    start = time.monotonic()
    try:
        shutil.copyfile(source,compressed)
        dbc2dbf(str(compressed),str(decoded))
        dbf = DBF(str(decoded),encoding='latin1',load=False)
        names, formats, offsets = ['deleted'],['S1'],[0]
        position=1
        for field in dbf.fields:
            names.append(field.name); formats.append(f'S{field.length}'); offsets.append(position)
            position += field.length
        dtype = np.dtype({'names':names,'formats':formats,'offsets':offsets,'itemsize':dbf.header.recordlen})
        data = np.memmap(decoded,dtype=dtype,mode='r',offset=dbf.header.headerlen,shape=(dbf.header.numrecords,))
        proc = np.char.strip(data['PROC_REA'])
        chosen = np.isin(proc,np.array([c.encode() for v in CODES.values() for c in v])) & (data['deleted'] != b'*')
        fields = ['CNES','MUNIC_RES','MUNIC_MOV','PROC_REA','IDENT','N_AIH','SEQ_AIH5']
        d = pd.DataFrame({f:np.char.decode(np.char.strip(data[f][chosen]),'latin1') for f in fields})
        d['service'] = d.PROC_REA.map({c:s for s,cs in CODES.items() for c in cs})
        audit = {'file':str(source),'year':year,'month':month,'uf':uf,'records':len(data),
                 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'selected_records':len(d),
                 'selected_by_code':d.PROC_REA.value_counts().to_dict(),
                 'selected_ident':d.IDENT.value_counts().to_dict(),
                 'duplicate_aih_selected_within_file':int(d.N_AIH.duplicated().sum()),
                 'coding':CODES,'count_unit':'selected paid AIH records in processing-month files; no person-level deduplication',
                 'seconds':0}
        valid_hospital=d.CNES.str.fullmatch(r'\d{7}') & ~d.CNES.eq('0000000')
        valid_muni=d.MUNIC_RES.str.fullmatch(r'\d{6}') & ~d.MUNIC_RES.eq('000000') & d.MUNIC_MOV.str.fullmatch(r'\d{6}') & ~d.MUNIC_MOV.eq('000000')
        audit['invalid_hospital']=int((~valid_hospital).sum())
        audit['invalid_municipality_pair']=int((~valid_muni).sum())
        hospital=d[valid_hospital].groupby(['service','CNES']).size().rename('n').reset_index()
        flow=d[valid_muni].copy()
        flow['local']=flow.MUNIC_RES.eq(flow.MUNIC_MOV).astype(int)
        municipality=flow.groupby(['service','MUNIC_RES']).agg(n=('local','size'),local=('local','sum')).reset_index()
        for table in [hospital,municipality]:
            table['year'],table['month'],table['uf']=year,month,uf
        hospital.to_parquet(destination/'hospital.parquet',index=False)
        municipality.to_parquet(destination/'municipality.parquet',index=False)
        audit['seconds']=round(time.monotonic()-start,3)
        (destination/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
        del data
        return audit
    finally:
        # Only this generated per-file staging directory is eligible for cleanup.
        resolved = stage.resolve()
        if resolved.parent == STAGE.resolve():
            shutil.rmtree(resolved)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--years',nargs='+',type=int,default=[2017,2018,2019,2023,2024])
    parser.add_argument('--uf')
    parser.add_argument('--month',type=int)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    files=[]
    for year in args.years:
        for source in sorted((Path('__SIH_ROOT__')/str(year)).rglob('*.dbc')):
            m=re.fullmatch(r'RD([A-Z]{2})(\d{2})(\d{2})\.dbc',source.name,re.I)
            if m and (args.uf is None or m[1].upper()==args.uf.upper()) and (args.month is None or int(m[3])==args.month):
                files.append(str(source))
    if not files:
        raise ValueError('No input files')
    out=ROOT/'derived/brazil'
    out.mkdir(parents=True,exist_ok=True)
    audits=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(one,f):f for f in files}
        for future in concurrent.futures.as_completed(jobs):
            audit=future.result()
            audits.append(audit)
            print(json.dumps({'done':len(audits),'total':len(files),'file':Path(audit['file']).name,'selected':audit['selected_records'],'seconds':audit['seconds']}),flush=True)
    hospitals=pd.concat([pd.read_parquet(out/'parts'/Path(f).stem/'hospital.parquet') for f in files],ignore_index=True)
    municipalities=pd.concat([pd.read_parquet(out/'parts'/Path(f).stem/'municipality.parquet') for f in files],ignore_index=True)
    suffix='_'.join(map(str,args.years))+(f'_{args.uf}' if args.uf else '')+(f'_m{args.month}' if args.month else '')
    hospitals.groupby(['year','service','CNES'],as_index=False)['n'].sum().to_csv(out/f'hospital_annual_{suffix}.csv',index=False)
    municipalities.groupby(['year','service','MUNIC_RES'],as_index=False)[['n','local']].sum().to_csv(out/f'municipality_annual_{suffix}.csv',index=False)
    pd.DataFrame([{k:v for k,v in a.items() if not isinstance(v,dict)} for a in audits]).to_csv(out/f'file_audit_{suffix}.csv',index=False)
    (out/f'run_{suffix}.json').write_text(json.dumps({'files':len(files),'years':args.years,'coding':CODES,'all_successful':True,'selected_records':sum(a['selected_records'] for a in audits)},indent=2))
    print('EXTRACTION_COMPLETE',suffix,flush=True)


if __name__=='__main__':
    main()
