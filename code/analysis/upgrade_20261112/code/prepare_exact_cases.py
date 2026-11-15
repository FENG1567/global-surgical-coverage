from pathlib import Path

root=Path('__PROJECT_ROOT__/upgrade_20261112')
source=(root/'recovered/43_certify_many_server.py').read_text(encoding='utf8')
source=source.replace('import json, os, sys, time','import json, os, sys, time\nfrom pathlib import Path')
source=source.replace('TAU = 30', '''TAU = 30
OUT=Path(BASE)/'upgrade_20261112/derived/exact_cases'
OUT.mkdir(parents=True,exist_ok=True)
SCOPE=pd.read_csv(Path(BASE)/'upgrade_20261112/derived/country_scope/population_and_bounds.csv').set_index('iso3')''')
source=source.replace('todo = [iso for iso, n in counts.items() if 8 <= n <= NMAX and iso in idxmap]',"todo = ['DJI','BLZ','GNB']")
source=source.replace('fr = FR[r0:r1, c0:c1].astype(np.float64)', '''scope=SCOPE.loc[iso]
        r0=min(r0,max(0,int(scope.r0)-240));r1=max(r1,min(NR,int(scope.r1)+240))
        c0=min(c0,max(0,int(scope.c0)-240));c1=max(c1,min(NC,int(scope.c1)+240))
        fr = FR[r0:r1, c0:c1].astype(np.float64)''')
marker='        w = pc / pc.sum()'
assert marker in source
source=source.split(marker)[0]+'''        density=[]
        for latitude,longitude in zip(h.lat,h.lon):
            row=int((85-latitude)/PIX);col=int((longitude+180)/PIX)
            density.append(float(pop_glob[max(0,row-1):row+2,max(0,col-1):col+2].sum(dtype=np.float64)))
        urban=np.asarray(density)>np.median(density)
        assert abs(pc.sum()/scope.population-1)<1e-6
        np.savez_compressed(OUT/f'{iso}.npz',A=A,weights=pc,urban=urban,k=k,
            country_population=scope.population,country_population_retained=pc.sum()/scope.population)
        h.to_csv(OUT/f'{iso}_facilities.csv',index=False)
        print('EXACT_CASE_READY',iso,n,k,A.shape,flush=True)
    except Exception as e:
        raise
print('EXACT_CASES_COMPLETE',flush=True)
'''
(root/'code/extract_exact_cases.py').write_text(source,encoding='utf8')
