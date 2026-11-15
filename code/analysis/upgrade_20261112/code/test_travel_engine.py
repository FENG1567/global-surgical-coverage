"""Compare the regional C engine against a heap shortest-path reference."""
from pathlib import Path
import heapq
import json
import subprocess
import numpy as np

ROOT=Path('__PROJECT_ROOT__/upgrade_20261112')
OUT=ROOT/'derived/travel_engine_test'
OUT.mkdir(parents=True,exist_ok=True)
exe=ROOT/'derived/mexico_spatial_v2/ttdijkstra_crop'
rng=np.random.default_rng(17)
reports=[]
for seed in range(3):
    H,W=35,39;top=35.;maximum=120.;pixel=1/120
    f=rng.uniform(.00005,.002,(H,W)).astype('float32')
    if seed==1:f[:,20]=-9999;f[17,20]=.001
    if seed==2:f[:,20]=-9999
    source=np.zeros((H,W),dtype='uint8');source[4,5]=1;source[28,31]=1
    f.tofile(OUT/'f.f32');source.tofile(OUT/'source.u8')
    subprocess.run([str(exe),str(OUT/'f.f32'),str(OUT/'source.u8'),str(OUT/'d.f32'),str(H),str(W),'120','0',str(top)],
                   check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    expected=np.full((H,W),np.float32(1e9));queue=[]
    for r,c in np.argwhere(source):
        expected[r,c]=0;heapq.heappush(queue,(0.,int(r),int(c)))
    while queue:
        value,r,c=heapq.heappop(queue)
        if value!=float(expected[r,c]):continue
        dx=111320*pixel*np.cos(np.deg2rad(top-pixel*(r+.5)))
        dy=110574*pixel
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                if dr==dc==0:continue
                rr,cc=r+dr,c+dc
                if not (0<=rr<H and 0<=cc<W) or f[rr,cc]<0:continue
                step=np.hypot(dx,dy) if dr and dc else (dx if dr==0 else dy)
                cost=np.float32(.5*(float(f[r,c])+float(f[rr,cc]))*step)
                new=np.float32(np.float32(value)+cost)
                if new<expected[rr,cc] and new<=maximum:
                    expected[rr,cc]=new;heapq.heappush(queue,(float(new),rr,cc))
    actual=np.fromfile(OUT/'d.f32',dtype='float32').reshape(H,W)
    error=float(np.abs(actual-expected).max())
    assert error<1e-4,(seed,error)
    for threshold in [15,30,60,120]:assert np.array_equal(actual<=threshold,expected<=threshold)
    reports.append({'case':seed,'max_abs_error_minutes':error,'all_threshold_masks_identical':True})
(OUT/'results.json').write_text(json.dumps(reports,indent=2))
print(json.dumps(reports),flush=True)
