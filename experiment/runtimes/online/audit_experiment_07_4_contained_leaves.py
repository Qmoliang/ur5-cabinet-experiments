"""Exploratory bound: how many directional leaves are wholly contained in peers?"""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'formal_results/experiment_07/development_07_4'

def audit(path):
    a=np.load(path);U=a['leaf_shapes'];owners=a['leaf_owners'];codes=a['leaf_codes'];n=len(U);keep=np.ones(n,dtype=bool);oi=[];ii=[]
    order=np.argsort(owners,kind='stable');first=np.r_[0,np.flatnonzero(np.diff(owners[order]))+1];ends=np.r_[first[1:],n]
    for begin,end in zip(first,ends):
        group=order[begin:end]
        for inner in group:
            for outer in group:
                if outer!=inner:oi.append(outer);ii.append(inner)
    oi=np.array(oi,dtype=int);ii=np.array(ii,dtype=int);delta=U[oi]-U[ii];slack=np.linalg.eigvalsh(delta)[:,0];scale=np.max(np.abs(U[oi]),axis=(1,2));equal=np.all(U[oi]==U[ii],axis=(1,2));contains=(slack>1e-14*scale+1e-20)|(equal&(codes[oi]<codes[ii]))
    keep[ii[contains]]=False
    witnesses={int(i):int(o) for o,i in zip(oi[contains&keep[oi]],ii[contains&keep[oi]]) if not keep[i]}
    assert len(witnesses)==int(np.sum(~keep))
    result=dict(source=str(path),total_leaves=n,retained_leaves=int(keep.sum()),removed_leaves=int((~keep).sum()),removed_fraction=float(np.mean(~keep)),all_removed_have_retained_psd_witness=True,does_not_refit_proxies=True,does_not_measure_runtime_speedup=True)
    np.savez_compressed(path.with_name(path.stem+'_contained_leaf_witnesses.npz'),kept=np.flatnonzero(keep),removed=np.array(list(witnesses)),containing=np.array(list(witnesses.values())))
    return result
if __name__=='__main__':
    rows=[]
    for seed in range(5):
        record=dict(seed=seed,**audit(OUT/f'same_input_seed{seed}.npz'));rows.append(record);print(json.dumps(record),flush=True)
    (OUT/'contained_leaf_audit.json').write_text(json.dumps(dict(exploratory_only=True,runs=rows),indent=2),encoding='utf-8')
