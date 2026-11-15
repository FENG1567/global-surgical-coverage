from pathlib import Path
import itertools
import json
import numpy as np
from scipy.optimize import milp,LinearConstraint,Bounds
from scipy import sparse

OUT=Path('__PROJECT_ROOT__/upgrade_20261112/derived/exact_bound_tests')
OUT.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(19)
reports=[]
for case in range(5):
    n,m,k=8,13,3
    A=rng.random((n,m))<.2
    A[:,0]=False
    weights=rng.random(m);weights/=weights.sum()
    selections=list(itertools.combinations(range(n),k))
    covers=np.array([weights[A[list(s)].any(axis=0)].sum() for s in selections])
    upper=sparse.hstack([-sparse.csr_matrix(A.T.astype(float)),sparse.eye(m,format='csr')])
    jj,ii=np.nonzero(A)
    row=np.repeat(np.arange(len(jj)),2)
    col=np.empty(2*len(jj),dtype=int);col[::2]=jj;col[1::2]=n+ii
    lower=sparse.coo_matrix((np.tile([1.,-1.],len(jj)),(row,col)),shape=(len(jj),n+m)).tocsr()
    budget=sparse.csr_matrix(np.r_[np.ones(n),np.zeros(m)].reshape(1,-1))
    constraints=[LinearConstraint(upper,-np.inf,0),LinearConstraint(lower,-np.inf,0),LinearConstraint(budget,k,k)]
    result={}
    for sign,label in [(1,'min'),(-1,'max')]:
        fit=milp(c=np.r_[np.zeros(n),sign*weights],integrality=np.r_[np.ones(n),np.zeros(m)],
                 bounds=Bounds(np.zeros(n+m),np.ones(n+m)),constraints=constraints,
                 options={'mip_rel_gap':1e-8})
        assert fit.success
        result[label]=sign*fit.fun
    assert abs(result['min']-covers.min())<1e-8
    assert abs(result['max']-covers.max())<1e-8
    truth=np.array(selections[0]);remaining=np.ones(len(selections),dtype=bool);previous=np.ptp(covers)
    order=rng.permutation(n)
    for reveal in range(1,n+1):
        revealed=order[:reveal]
        remaining=np.array([all((i in s)==(i in truth) for i in revealed) for s in selections])
        width=np.ptp(covers[remaining])
        assert width<=previous+1e-10
        previous=width
    assert previous==0
    reports.append({'case':case,'n_subsets':len(selections),'milp_matches_enumeration':True,
                    'nested_labels_monotone':True,'all_labels_width_zero':True})
(OUT/'results.json').write_text(json.dumps(reports,indent=2))
print(json.dumps(reports),flush=True)
