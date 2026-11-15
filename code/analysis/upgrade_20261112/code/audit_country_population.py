from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.ndimage import find_objects

BASE=Path('__PROJECT_ROOT__')
G=BASE/'work/grid'
OUT=BASE/'upgrade_20261112/derived/country_scope'
OUT.mkdir(parents=True,exist_ok=True)
nr,nc=17400,43200
country=np.memmap(G/'country.i16',dtype='int16',mode='r',shape=(nr,nc))
pop=np.memmap(G/'pop2025.f32',dtype='float32',mode='r',shape=(nr,nc))
codes=pd.read_csv(G/'countries.csv')
total=np.zeros(int(codes.idx.max())+1)
for start in range(0,nr,300):
    c=country[start:start+300].ravel()
    p=pop[start:start+300].ravel()
    valid=c>=0
    total += np.bincount(c[valid],weights=p[valid],minlength=len(total))
labels=np.array(country,dtype='int16')
labels[labels<0]=0
boxes=find_objects(labels,max_label=int(codes.idx.max()))
del labels
rows=[]
for item in codes.itertuples():
    box=boxes[item.idx-1]
    rows.append({'iso3':item.iso3,'idx':int(item.idx),'population':float(total[item.idx]),
      'r0':box[0].start if box else None,'r1':box[0].stop if box else None,
      'c0':box[1].start if box else None,'c1':box[1].stop if box else None})
pd.DataFrame(rows).to_csv(OUT/'population_and_bounds.csv',index=False)
details_path=BASE/'upgrade_20261112/derived/certification/endpoint_details.json'
if details_path.exists():
    details=json.loads(details_path.read_text())
    merged=pd.DataFrame(details)[['iso','country_population_in_window']].merge(pd.DataFrame(rows),left_on='iso',right_on='iso3')
    merged['population_fraction_retained']=merged.country_population_in_window/merged.population
    merged.to_csv(OUT/'certification_population_retention.csv',index=False)
print('COUNTRY_SCOPE_COMPLETE',flush=True)
