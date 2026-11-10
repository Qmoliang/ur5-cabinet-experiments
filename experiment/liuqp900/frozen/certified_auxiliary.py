"""Verify the auxiliary velocity against the original six-variable QP KKT system."""
from types import SimpleNamespace
import numpy as np

def verify_solution(H,g,A,l,u,payload):
    x=np.asarray(payload['velocity']);y=np.asarray(payload['original_dual'])
    if x.shape!=(6,) or y.shape!=(len(A),) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):raise RuntimeError('invalid auxiliary certificate shape')
    values=A@x;primal=float(max(np.max(l-values),np.max(values-u),0.))
    positive=np.maximum(y,0);negative=np.minimum(y,0);up=np.isfinite(u);lo=np.isfinite(l)
    if np.max(positive[~up],initial=0)>1e-8 or np.max(-negative[~lo],initial=0)>1e-8:raise RuntimeError('invalid infinite-bound dual')
    stationarity=H@x+g+A.T@y;relative=float(np.linalg.norm(stationarity,np.inf)/(1+np.linalg.norm(H@x,np.inf)+np.linalg.norm(g,np.inf)+np.linalg.norm(A.T@y,np.inf)))
    comp=max(float(np.max(np.abs(positive[up]*(u[up]-values[up])),initial=0)),float(np.max(np.abs(negative[lo]*(values[lo]-l[lo])),initial=0)))
    force=g+A.T@y;dual=-.5*force@np.linalg.solve(H,force)-u[up]@positive[up]-l[lo]@negative[lo]
    obj=.5*x@H@x+g@x;gap=float(obj-dual);eigen=float(np.linalg.eigvalsh(H).min())
    gap_certificate=(primal<=1e-12 and comp<=.5*max(eigen,0)*1e-10 and abs(gap)<=.5*max(eigen,0)*1e-10)
    stationarity_ok=relative<=1e-7 or gap_certificate
    if primal>1e-7 or not stationarity_ok or comp>1e-7*(1+abs(obj)) or gap>1e-8*(1+abs(obj)) or eigen<=0:
        raise RuntimeError(f'auxiliary KKT certificate failed: primal={primal}, relative={relative}, complementarity={comp}, gap={gap}')
    info=SimpleNamespace(status='solved',prim_res=primal,dual_res=float(np.linalg.norm(stationarity,np.inf)),iter=int(payload.get('iterations',0)),obj_val=float(obj),pilot_solver='CERTIFIED_AUXILIARY',relative_stationarity=relative,complementarity=comp,primal_dual_gap=gap)
    return SimpleNamespace(x=x.copy(),y=y.copy(),info=info),dict(primal=primal,relative_stationarity=relative,complementarity=comp,primal_dual_gap=gap,min_hessian_eigenvalue=eigen)
