"""Convex one-sided joint-reserve consumption objective and scalar release.

No successful trajectory or reference configuration is accepted by this API.
"""
import numpy as np
from verified_qp import solve_verified
from unified_joint_braking import stopping_distance

def reserves(q,previous,low,high,dt,acceleration):
    resolution=acceleration*dt*dt
    positive=np.maximum(high-q-stopping_distance(np.maximum(previous,0),dt,acceleration),resolution)
    negative=np.maximum(q-low-stopping_distance(np.maximum(-previous,0),dt,acceleration),resolution)
    return positive,negative

def solve_reserve(H,g,A,l,u,q,previous,low,high,omega,error,dt,acceleration,release=None,hold=None):
    n=len(g);rp,rm=reserves(q,previous,low,high,dt,acceleration)
    task_scale=omega*max(error,1e-3)**2
    P=np.zeros((3*n,3*n));P[:n,:n]=H;P[n:,n:]=task_scale*np.eye(2*n)
    linear=np.r_[g,np.zeros(2*n)];original=np.c_[A,np.zeros((len(A),2*n))]
    plus=np.c_[np.eye(n),-np.diag(rp),np.zeros((n,n))]
    minus=np.c_[-np.eye(n),np.zeros((n,n)),-np.diag(rm)]
    nonnegative=np.c_[np.zeros((2*n,n)),-np.eye(2*n)]
    AA=np.vstack([original,plus,minus,nonnegative]);ll=np.r_[l,np.full(4*n,-np.inf)];uu=np.r_[u,np.zeros(4*n)]
    if hold is not None:
        J,value=hold;AA=np.vstack([AA,np.c_[J,np.zeros((len(J),2*n))]]);ll=np.r_[ll,value];uu=np.r_[uu,value]
    release_index=None
    if release is not None:
        b,rate=release;release_index=len(AA);AA=np.vstack([AA,np.r_[-np.asarray(b),np.zeros(2*n)]]);ll=np.r_[ll,-np.inf];uu=np.r_[uu,-rate]
    solved=solve_verified(P,linear,AA,ll,uu)
    if solved is None:return None
    v=solved.x[:n];m=len(A);delta=solved.y[m:m+n]-solved.y[m+n:m+2*n]
    if hold is not None:
        J,value=hold;eq_index=m+4*n;delta=delta+J.T@solved.y[eq_index:eq_index+len(J)]
    release_dual=0.
    if release_index is not None:release_dual=float(solved.y[release_index]);delta=delta-release_dual*np.asarray(release[0])
    return dict(velocity=v,delta=delta,positive_reserve=rp,negative_reserve=rm,release_dual=release_dual,
                consumption_cost=float(.5*task_scale*np.sum(solved.x[n:]**2)),
                original_cost=float(.5*v@H@v+g@v),solver=solved.info.pilot_solver,raw=solved)
