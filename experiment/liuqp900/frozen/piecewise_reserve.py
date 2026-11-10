"""Exact elimination of one-sided reserve epigraph variables.

Each Newton subproblem is a six-variable QP with the SAME physical rows.
Convex Armijo descent handles sign changes; no configuration path is searched.
"""
from types import SimpleNamespace
import numpy as np
from reserve_objective_contract import reserves
from verified_qp import solve_verified

def solve_reserve(H,g,A,l,u,q,previous,low,high,omega,error,dt,acceleration,release=None,joint_metric=True,**unused):
    rp,rm=reserves(q,previous,low,high,dt,acceleration);scale=omega*max(error,1e-3)**2
    positive=scale/rp**2 if joint_metric else np.zeros(6);negative=scale/rm**2 if joint_metric else np.zeros(6)
    AA=A;ll=l;uu=u
    if release is not None:
        b,rate=release;AA=np.vstack([A,-b]);ll=np.r_[l,-np.inf];uu=np.r_[u,-rate]
    seed=solve_verified(H,g,AA,ll,uu)
    if seed is None:return None
    x=seed.x.copy();iterations=0
    def weights(v):return np.where(v>=0,positive,negative)
    def objective(v):return float(.5*v@H@v+g@v+.5*np.dot(weights(v),v*v))
    for count in range(32):
        weight=weights(x);result=solve_verified(H+np.diag(weight),g,AA,ll,uu,hint=x)
        if result is None:return None
        iterations+=result.info.iter;v=result.x;gradientR=weights(v)*v
        residual=H@v+g+gradientR+AA.T@result.y
        relative=float(np.linalg.norm(residual,np.inf)/(1+np.linalg.norm(g,np.inf)+np.linalg.norm(H@v+gradientR,np.inf)+np.linalg.norm(AA.T@result.y,np.inf)))
        if relative<=1e-8:
            alpha=0. if release is None else float(result.y[-1]);delta=gradientR.copy()
            if release is not None:delta-=alpha*np.asarray(release[0])
            return dict(velocity=v,delta=delta,positive_reserve=rp,negative_reserve=rm,release_dual=alpha,consumption_cost=float(.5*np.dot(weights(v),v*v)),original_cost=float(.5*v@H@v+g@v),raw=result,original_dual=result.y[:len(A)].copy(),iterations=iterations,region_iterations=count+1,convex_stationarity=relative)
        direction=v-x;gradient=H@x+g+weights(x)*x;slope=float(gradient@direction);value=objective(x);step=1.
        for _ in range(32):
            trial=x+step*direction
            if objective(trial)<=value+1e-4*step*slope+1e-14:break
            step*=.5
        else:return None
        x=trial
    return None
