"""Exhaustive fixed-count sets and nested audit simulations on real geometries."""
from pathlib import Path
import itertools
import json
import math
import numpy as np
import pandas as pd

BASE=Path('__PROJECT_ROOT__/upgrade_20261112')
OUT=BASE/'derived/information_value_v2'
OUT.mkdir(parents=True,exist_ok=False)
rng=np.random.default_rng(20261112)
results=[];endpoint_rows=[]
for path in sorted((BASE/'derived/exact_cases').glob('*.npz')):
    z=np.load(path)
    A=z['A'];weights=z['weights'];urban=z['urban'];k=int(z['k']);n=A.shape[0]
    assert n<63
    count=math.comb(n,k)
    assert count<=200000
    powers=np.left_shift(np.uint64(1),np.arange(n,dtype=np.uint64))
    masks=np.fromiter((sum(1<<i for i in c) for c in itertools.combinations(range(n),k)),dtype=np.uint64,count=count)
    pattern=np.sum(A.astype(np.uint64)*powers[:,None],axis=0,dtype=np.uint64)
    unique,inverse=np.unique(pattern,return_inverse=True)
    pattern_weights=np.bincount(inverse,weights=weights,minlength=len(unique))/weights.sum()
    coverage=np.zeros(count)
    for bits,weight in zip(unique,pattern_weights):
        if bits:coverage+=(masks&bits!=0)*weight
    urban_count=np.zeros(count,dtype=np.int16)
    for i in np.flatnonzero(urban):urban_count+=((masks&powers[i])!=0)
    catchment=A@weights/weights.sum()
    base_width=float(coverage.max()-coverage.min())
    endpoint_rows.append({'iso':path.stem,'n':n,'k':k,'enumerated_subsets':count,
        'min':float(coverage.min()),'max':float(coverage.max()),'width':base_width,
        'country_population_retained':float(z['country_population_retained'])})
    # Simulated complete labels are sampled uniformly over the fixed-k set.
    for truth_rep in range(100):
        truth_index=int(rng.integers(count));truth=masks[truth_index]
        truth_q=int(urban_count[truth_index])
        feasible_q=urban_count==truth_q
        orders={'random':rng.permutation(n),'largest_catchment':np.argsort(-catchment,kind='stable')}
        for prior_label,initial in [('count_only',np.ones(count,dtype=bool)),('count_and_exact_q',feasible_q)]:
            prior_width=float(np.ptp(coverage[initial]))
            for strategy in ['random','largest_catchment']:
                order=orders[strategy]
                previous=prior_width
                for fraction in [0,.1,.2,.4,.6]:
                    m=int(round(fraction*n))
                    revealed=np.bitwise_or.reduce(powers[order[:m]],initial=np.uint64(0))
                    feasible=initial&((masks&revealed)==(truth&revealed))
                    assert feasible[truth_index]
                    width=float(np.ptp(coverage[feasible]))
                    assert width<=previous+1e-12
                    previous=width
                    results.append({'iso':path.stem,'truth_rep':truth_rep,'prior':prior_label,'strategy':strategy,
                        'fraction':fraction,'n_labels_revealed':m,'remaining_subsets':int(feasible.sum()),
                        'baseline_width':base_width,'prior_width':prior_width,'remaining_width':width,
                        'width_reduction_from_prior':prior_width-width,'truth_q_count':truth_q})
    print('ENUMERATED',path.stem,count,'width',base_width,flush=True)
pd.DataFrame(endpoint_rows).to_csv(OUT/'exhaustive_endpoints.csv',index=False)
data=pd.DataFrame(results)
data.to_csv(OUT/'audit_simulations.csv',index=False)
summary=data.groupby(['iso','prior','strategy','fraction'],as_index=False).agg(
    mean_remaining_width=('remaining_width','mean'),median_remaining_width=('remaining_width','median'),
    mean_reduction=('width_reduction_from_prior','mean'),minimum_remaining=('remaining_width','min'),
    maximum_remaining=('remaining_width','max'),n_simulated_truths=('truth_rep','count'))
summary.to_csv(OUT/'summary.csv',index=False)
(OUT/'design.json').write_text(json.dumps({'truths_per_country':100,'seed':20261112,
 'paired_random_orders_across_information_priors':True,
 'truth_generation':'uniform random fixed-count subsets, simulated labels on real facility and population geometry',
 'fractions':[0,.1,.2,.4,.6],'nested_revelation':True,
 'targeted_strategy':'descending catchment population using geometry only, no hidden label information',
 'exact_endpoints':'all fixed-k subsets enumerated; coverage pattern aggregation retains total country population',
 'meaning':'model-conditional simulation of information acquisition, no measured registry truth or monetary utility'},indent=2))
print('INFORMATION_VALUE_COMPLETE',flush=True)
