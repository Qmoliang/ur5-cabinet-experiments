"""Current-velocity resource consumption cost, with no future-state decisions."""
import numpy as np
from verified_qp import solve_verified
from unified_joint_braking import stopping_distance

def reserves(q,previous,low,high,dt,acceleration):
    floor=acceleration*dt*dt
    return (np.maximum(high-q-stopping_distance(np.maximum(previous,0),dt,acceleration),floor),
            np.maximum(q-low-stopping_distance(np.maximum(-previous,0),dt,acceleration),floor))

def solve_reserve(H,g,A,l,u,q,previous,low,high,omega,error,dt,acceleration,release=None,hold=None,spatial=None,cuts=(),joint_metric=True):
    n=len(g);rp,rm=reserves(q,previous,low,high,dt,acceleration)
    nj=2*n if joint_metric else 0;ns=int(spatial is not None);size=n+nj+ns;scale=omega*max(error,1e-3)**2
    P=np.zeros((size,size));P[:n,:n]=H;P[n:,n:]=scale*np.eye(nj+ns)
    lin=np.r_[g,np.zeros(nj+ns)];rows=[np.pad(A,((0,0),(0,nj+ns)))];lbs=[l];ubs=[u];terms=[];cursor=len(A)
    def add(B,upper,gradient):
        nonlocal cursor
        B=np.atleast_2d(B);rows.append(B);lbs.append(np.full(len(B),-np.inf));ubs.append(np.atleast_1d(upper))
        if gradient is not None:terms.append((cursor,len(B),gradient))
        cursor+=len(B)
    if joint_metric:
        plus=np.zeros((n,size));plus[:,:n]=np.eye(n);plus[:,n:n+n]=-np.diag(rp)
        minus=np.zeros((n,size));minus[:,:n]=-np.eye(n);minus[:,n+n:n+2*n]=-np.diag(rm)
        add(plus,np.zeros(n),np.eye(n));add(minus,np.zeros(n),-np.eye(n))
    if spatial is not None:
        normals,available=spatial;B=np.zeros((len(normals),size));B[:,:n]=normals;B[:,-1]=-available
        add(B,np.zeros(len(B)),normals.T)
    if nj+ns:
        B=np.zeros((nj+ns,size));B[:,n:]=-np.eye(nj+ns);add(B,np.zeros(nj+ns),None)
    for row,bound in cuts:
        B=np.r_[row,np.zeros(nj+ns)];add(B,bound,np.asarray(row)[:,None])
    if hold is not None:raise ValueError('Not used by this experiment')
    release_index=None
    if release is not None:
        b,rate=release;release_index=cursor;add(np.r_[-np.asarray(b),np.zeros(nj+ns)],-rate,-np.asarray(b)[:,None])
    solved=solve_verified(P,lin,np.vstack(rows),np.concatenate(lbs),np.concatenate(ubs))
    if solved is None:return None
    delta=np.zeros(n)
    for start,count,G in terms:delta+=G@solved.y[start:start+count]
    v=solved.x[:n]
    return dict(velocity=v,delta=delta,positive_reserve=rp,negative_reserve=rm,
        release_dual=0. if release_index is None else float(solved.y[release_index]),
        consumption_cost=float(.5*scale*np.sum(solved.x[n:]**2)),original_cost=float(.5*v@H@v+g@v),
        raw=solved,original_dual=solved.y[:len(A)].copy(),iterations=solved.info.iter)
