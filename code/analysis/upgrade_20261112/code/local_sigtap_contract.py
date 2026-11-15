from pathlib import Path
import zipfile
import re
import json
import hashlib

root = Path('__SIGTAP_ROOT__')
out = Path('__PROJECT_ROOT__/upgrade_20261112/contracts')
out.mkdir(parents=True, exist_ok=True)
rows, sources = [], []
for year in [2017,2018,2019,2023,2024]:
    for month in range(1,13):
        matches = sorted((root/str(year)).glob(f'*{year}{month:02d}*.zip'))
        if len(matches) != 1:
            raise ValueError((year,month,'expected unique SIGTAP archive',len(matches)))
        path = matches[0]
        with zipfile.ZipFile(path) as archive:
            name = next(n for n in archive.namelist() if n.lower().endswith('tb_procedimento.txt'))
            lines = archive.read(name).decode('latin1').splitlines()
            for line in lines:
                procedure, label = line[:10], line[10:260].strip()
                if re.search('COLECISTECTOMIA|APENDICECTOMIA|PARTO CESARIANO',label,re.I):
                    rows.append({'year':year,'month':month,'code':procedure,'name':label})
        sources.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
result={'status':'official_local_SIGTAP_monthly_dictionary','months':len(sources),'procedure_entries':rows,'sources':sources}
(out/'sigtap_monthly_evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
unique = sorted({(r['code'],r['name']) for r in rows})
print(json.dumps({'months':len(sources),'unique_procedures':unique},ensure_ascii=False))
