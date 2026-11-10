"""Numerical fallback for the SAME convex QP, with independent KKT checks.

Used identically for original and modified objectives. Never relaxes rows.
"""
from types import SimpleNamespace
import time
import numpy as np
NUMERICAL_REJECTIONS=0
from scipy.optimize import minimize,linprog,nnls
from scipy.linalg import null_space,qr
from scipy import sparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
import clarabel

def _solve_slsqp(H,g,A,l,u,hint=None):
    started=time.perf_counter();H=np.asarray(H);g=np.asarray(g);A=np.asarray(A);l=np.asarray(l);u=np.asarray(u)
    eqmask=np.isfinite(l)&np.isfinite(u)&(np.abs(u-l)<1e-13)
    up=np.flatnonzero(np.isfinite(u)&~eqmask);lo=np.flatnonzero(np.isfinite(l)&~eqmask)
    B=np.vstack([A[up],-A[lo]]);b=np.r_[u[up],-l[lo]];norm=np.maximum(np.linalg.norm(B,axis=1),1e-12);B=B/norm[:,None];b=b/norm
    eqidx=np.flatnonzero(eqmask);E=A[eqidx];eb=l[eqidx]
    if len(E):
        Q,R,piv=qr(E.T,pivoting=True,mode='economic');rank=np.linalg.matrix_rank(E,tol=1e-10);take=piv[:rank];Ec=E[take];ecb=eb[take]
    else:take=np.empty(0,dtype=int);Ec=E;ecb=eb
    x0=np.zeros(len(g)) if hint is None or not np.all(np.isfinite(hint)) or np.max(np.abs(hint))>10 else np.asarray(hint).copy()
    if max(float(np.max(B@x0-b)) if len(B) else 0.,float(np.max(np.abs(E@x0-eb))) if len(E) else 0.)>1e-3:
        feasible=linprog(np.zeros(len(g)),A_ub=B,b_ub=b,A_eq=Ec if len(Ec) else None,b_eq=ecb if len(Ec) else None,bounds=[(None,None)]*len(g),method='highs')
        if not feasible.success:return None
        x0=feasible.x
    scale=max(1.,np.linalg.norm(H,2));constraints=[dict(type='ineq',fun=lambda x:b-B@x,jac=lambda x:-B)]
    if len(Ec):constraints.append(dict(type='eq',fun=lambda x:Ec@x-ecb,jac=lambda x:Ec))
    result=minimize(lambda x:(.5*x@H@x+g@x)/scale,x0,jac=lambda x:(H@x+g)/scale,constraints=constraints,method='SLSQP',options=dict(ftol=1e-14,maxiter=500,disp=False))
    x=result.x;slack=b-B@x;primal=max(0.,float(np.max(-slack)),float(np.max(np.abs(E@x-eb))) if len(E) else 0.)
    if not np.all(np.isfinite(x)) or primal>1e-7:return None
    active=np.flatnonzero(slack<1e-7);gradient=H@x+g;N=null_space(Ec,rcond=1e-10) if len(Ec) else np.eye(len(g))
    try:
        if len(active) and N.shape[1]:lam,_=nnls(N.T@B[active].T,-N.T@gradient,maxiter=10000,atol=1e-12)
        else:lam=np.zeros(len(active))
    except (RuntimeError,np.linalg.LinAlgError):
        global NUMERICAL_REJECTIONS
        NUMERICAL_REJECTIONS+=1
        return None
    residual=gradient+(B[active].T@lam if len(active) else 0)
    eta=np.linalg.lstsq(Ec.T,-residual,rcond=1e-10)[0] if len(Ec) else np.zeros(0)
    stationarity=residual+(Ec.T@eta if len(Ec) else 0);dual=float(np.max(np.abs(stationarity)))
    relative=dual/(1+np.linalg.norm(g)+np.linalg.norm(H@x));comp=float(np.max(np.abs(lam*slack[active]))) if len(active) else 0.
    if relative>1e-6 or comp>1e-6*(1+abs(.5*x@H@x+g@x)):return None
    all_lam=np.zeros(len(B));all_lam[active]=lam;all_lam=all_lam/norm;y=np.zeros(len(A));y[up]+=all_lam[:len(up)];y[lo]-=all_lam[len(up):]
    if len(Ec):y[eqidx[take]]=eta
    return SimpleNamespace(x=x,y=y,info=SimpleNamespace(status='solved',prim_res=primal,dual_res=dual,iter=int(result.nit),obj_val=float(.5*x@H@x+g@x),run_time=time.perf_counter()-started,pilot_solver='SLSQP_KKT',relative_stationarity=relative,complementarity=comp))

def _solve_clarabel(H,g,A,l,u):
    started=time.perf_counter();H=np.asarray(H);g=np.asarray(g);A=np.asarray(A);l=np.asarray(l);u=np.asarray(u)
    eq=np.isfinite(l)&np.isfinite(u)&(np.abs(u-l)<1e-13)
    ei=np.flatnonzero(eq);ui=np.flatnonzero(np.isfinite(u)&~eq);li=np.flatnonzero(np.isfinite(l)&~eq)
    B=np.vstack([A[ei],A[ui],-A[li]]);b=np.r_[l[ei],u[ui],-l[li]]
    scales=np.maximum.reduce([np.linalg.norm(B,axis=1),np.abs(b)/1000.,np.full(len(b),1e-8)])
    Bs=B/scales[:,None];bs=b/scales
    cones=[]
    if len(ei):cones.append(clarabel.ZeroConeT(len(ei)))
    if len(ui)+len(li):cones.append(clarabel.NonnegativeConeT(len(ui)+len(li)))
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=300
    settings.tol_gap_abs=1e-12;settings.tol_gap_rel=1e-12;settings.tol_feas=1e-12
    if hasattr(settings,'max_threads'):settings.max_threads=1
    solver=clarabel.DefaultSolver(sparse.triu(sparse.csc_matrix(H),format='csc'),g,sparse.csc_matrix(Bs),bs,cones,settings)
    result=solver.solve()
    if str(result.status) not in ('Solved','AlmostSolved'):return None
    x=np.asarray(result.x);dual=np.asarray(result.z)/scales;y=np.zeros(len(A))
    y[ei]=dual[:len(ei)];start=len(ei);du=np.maximum(0,dual[start:start+len(ui)]);dl=np.maximum(0,dual[start+len(ui):])
    y[ui]+=du;y[li]-=dl
    values=A@x;primal=max(0.,float(np.max(l-values)),float(np.max(values-u)))
    stationarity=H@x+g+A.T@y;residual=float(np.max(np.abs(stationarity)));relative=residual/(1+np.linalg.norm(g)+np.linalg.norm(H@x))
    comp=max(float(np.max(np.abs(du*(u[ui]-values[ui])))) if len(ui) else 0.,float(np.max(np.abs(dl*(values[li]-l[li])))) if len(li) else 0.)
    if not np.all(np.isfinite(x)) or primal>1e-7 or relative>1e-6 or comp>1e-6*(1+abs(.5*x@H@x+g@x)):return None
    return SimpleNamespace(x=x,y=y,info=SimpleNamespace(status='solved',prim_res=primal,dual_res=residual,iter=int(result.iterations),obj_val=float(.5*x@H@x+g@x),run_time=time.perf_counter()-started,pilot_solver='CLARABEL_KKT',relative_stationarity=relative,complementarity=comp))

def solve_verified(H,g,A,l,u,hint=None):
    answer=_solve_clarabel(H,g,A,l,u)
    return answer if answer is not None else _solve_slsqp(H,g,A,l,u,hint)
