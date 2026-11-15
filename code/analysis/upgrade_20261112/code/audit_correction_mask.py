from pathlib import Path
import json
import numpy as np

BASE=Path('__PROJECT_ROOT__')
G=BASE/'work/grid'
OUT=BASE/'upgrade_20261112/derived/correction_mask'
OUT.mkdir(parents=True,exist_ok=True)
target_path=G/'src_D2meas_corr.u8'
target=np.memmap(target_path,dtype='uint8',mode='r')
paths=[p for p in G.glob('src_*.u8') if p.name in ['src_D2meas.u8','src_D1_p75.u8','src_D1_p85.u8','src_D1_p65.u8','src_D1.u8']]
report={'target':str(target_path),'target_nonzero':int(np.count_nonzero(target)),'direct_comparisons':[],'intersection_comparisons':[]}
for path in paths:
    source=np.memmap(path,dtype='uint8',mode='r')
    report['direct_comparisons'].append({'mask':path.name,'nonzero':int(np.count_nonzero(source)),
        'different_cells':int(np.count_nonzero(source!=target)),'target_outside_mask':int(np.count_nonzero((target>0)&(source==0)))})
basepath=G/'src_D2meas.u8'
if basepath.exists():
    base=np.memmap(basepath,dtype='uint8',mode='r')
    for path in paths:
        source=np.memmap(path,dtype='uint8',mode='r')
        different=0
        for start in range(0,len(target),10000000):
            intersection=(base[start:start+10000000]>0)&(source[start:start+10000000]>0)
            different+=int(np.count_nonzero(intersection!=(target[start:start+10000000]>0)))
        report['intersection_comparisons'].append({'base':'src_D2meas.u8','second':path.name,'different_cells':different})
(OUT/'mask_diagnostics.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
