"""Exact negative-witness selection using analytic distance lower bounds.
No obstacle is removed from the published map or safety oracle. Only exact
point-distance evaluations unable to attain the negative minimum are skipped.
"""
import time,numpy as np
from diagnostic_uncached_policy import LowDimPolicy as Previous
class LowDimPolicy(Previous):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);pg=self.ctx.proxy_guard
  outer=pg.sphere.copy() if pg.representation=='sphere' else np.sqrt(np.max(pg.values,axis=1))
  outer=np.where(pg.isbox,np.linalg.norm(pg.box,axis=1),outer)
  self.distance_outer_radius=outer+pg.offset;self.bound_candidates=0;self.exact_candidates=0
 def _bounded_geometry(self,points,ri,oi):
  pg=self.ctx.proxy_guard;distance=np.full(len(oi),np.inf);normals=np.zeros((len(oi),3));self.bound_candidates+=len(oi)
  order=np.arange(len(ri)) if np.all(ri[1:]>=ri[:-1]) else np.argsort(ri,kind='stable');ordered=ri[order];cuts=np.r_[0,np.flatnonzero(ordered[1:]!=ordered[:-1])+1,len(ri)]
  for start,end in zip(cuts[:-1],cuts[1:]):
   indices=order[start:end];robot=ri[indices[0]];ids=oi[indices];point=points[int(robot)];lower=np.linalg.norm(pg.centers[ids]-point,axis=1)-self.distance_outer_radius[ids]
   first=int(np.argmin(lower));probe,_=pg.point_geometry(point[None,:],ids[first:first+1],optimizer=True);self.exact_candidates+=1
   threshold=min(float(probe[0]),float(self.future_guard.radii[int(robot)]+self.ctx.controller.safety_margin))
   keep=indices[lower<=threshold+1e-12]
   if len(keep):
    dd,nn=pg.point_geometry(np.repeat(point[None,:],len(keep),axis=0),oi[keep],optimizer=True);distance[keep]=dd;normals[keep]=nn;self.exact_candidates+=len(keep)
  return distance,normals
 def continuation_witnesses(self,q):
  started=time.perf_counter();g=self.future_guard;pg=self.ctx.proxy_guard;rows=[];worst=0.
  for a in self.sample_fractions:
   qs=(1-a)*q+a*self.qg;p=g.positions(qs).copy();ri,oi=pg.pairs(p,p,g.radii+self.ctx.controller.safety_margin+2e-7)
   observations=[]
   if len(oi):
    dist,n=self._bounded_geometry(p,ri,oi);gap=dist-g.radii[ri];slack=gap-self.ctx.controller.safety_margin
    # Keep worst obstacle for each robot point, not a hand-selected body.
    for i in np.unique(ri):
     ix=np.flatnonzero(ri==i);j=ix[np.argmin(slack[ix])]
     if slack[j]<0:observations.append((float(slack[j]),int(i),-n[j],('environment',int(i),int(oi[j]))))
   low=p-g.radii[:,None]-g.workspace_lower;high=g.workspace_upper-p-g.radii[:,None]
   for i in range(len(p)):
    vals=np.r_[low[i],high[i]];k=int(np.argmin(vals))
    if vals[k]<0:
     normal=np.eye(3)[k%3]*(1 if k<3 else -1);observations.append((float(vals[k]),i,normal,('workspace',i,k)))
   pairs=g.self_pairs;ii,jj=pairs.T;vec=p[ii]-p[jj];length=np.linalg.norm(vec,axis=1);slack=length-g.radii[ii]-g.radii[jj]-g.self_margin
   sel=np.flatnonzero(slack<0);needed=[o[1] for o in observations]+ii[sel].tolist()+jj[sel].tolist();jac=self._jac(p,needed)
   for value,i,n,key in observations:
    b=(1-a)*(n@jac[i]);rows.append(dict(slack=value,b=b,kind=key[0],id=list(key[1:]),fraction=float(a)));worst=max(worst,-value)
   # One worst self constraint per sample avoids pair multiplicity dominating.
   if len(sel):
    t=int(sel[np.argmin(slack[sel])]);b=(1-a)*vec[t]/max(length[t],1e-12)@(jac[int(ii[t])]-jac[int(jj[t])]);value=float(slack[t]);rows.append(dict(slack=value,b=b,kind='self',id=[int(ii[t]),int(jj[t])],fraction=float(a)));worst=max(worst,-value)
  self.witness_queries+=1
  # Group by sample: mean square deficit is independent of map cell count.
  value=float(sum(x['slack']**2 for x in rows)/max(1,len(rows)))
  return dict(rows=rows,deficit=worst,potential=value,seconds=time.perf_counter()-started)
 def summary(self):
  out=super().summary();out.update(exact_bound_candidates=self.bound_candidates,exact_distance_evaluations=self.exact_candidates,witness_optimization='analytic enclosing-sphere signed-distance lower bounds; original exact distance and normal retained for all possible negative minimizers; original full map and execution safety unchanged')
  return out
