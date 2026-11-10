"""Same fixed continuation, refreshing observed witnesses without rescanning the map each tick."""
import numpy as np,time
from continuation_policy import LowDimPolicy as Previous
class LowDimPolicy(Previous):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.witness_anchor=None;self.witness_cycle=-1000;self.cached_rows=[];self.refreshes=0;self.fast_updates=0;self.admission_detail={};self.no_descent_since=None
 def continuation_witnesses(self,q):
  if self.witness_anchor is None or self.k-self.witness_cycle>=100 or np.max(np.abs(q-self.witness_anchor))>=.04:
   state=super().continuation_witnesses(q);self.witness_anchor=q.copy();self.witness_cycle=self.k;self.cached_rows=state['rows'];self.refreshes+=1;return state
  started=time.perf_counter();guard=self.future_guard;pg=self.ctx.proxy_guard;rows=[]
  for a in self.sample_fractions:
   subset=[r for r in self.cached_rows if r['fraction']==a]
   if not subset:continue
   p=guard.positions((1-a)*q+a*self.qg).copy();needed=[]
   for r in subset:needed.extend(r['id'] if r['kind']=='self' else r['id'][:1])
   jac=self._jac(p,needed);env=[r for r in subset if r['kind']=='environment']
   if env:
    ii=np.array([r['id'][0] for r in env]);oi=np.array([r['id'][1] for r in env]);dist,normal=pg.point_geometry(p[ii],oi,optimizer=True)
    for j,r in enumerate(env):
     i=int(ii[j]);rows.append(dict(r,slack=float(dist[j]-guard.radii[i]-self.ctx.controller.safety_margin),b=(1-a)*(-normal[j]@jac[i])))
   for r in subset:
    if r['kind']=='environment':continue
    i,j=r['id']
    if r['kind']=='workspace':
     axis=j%3;n=np.eye(3)[axis]*(1 if j<3 else -1);value=float(p[i,axis]-guard.radii[i]-guard.workspace_lower[axis] if j<3 else guard.workspace_upper[axis]-p[i,axis]-guard.radii[i]);b=(1-a)*(n@jac[i])
    else:
     d=p[i]-p[j];norm=np.linalg.norm(d);value=float(norm-guard.radii[i]-guard.radii[j]-guard.self_margin);b=(1-a)*(d/max(norm,1e-12))@(jac[i]-jac[j])
    rows.append(dict(r,slack=value,b=b))
  rows=[r for r in rows if r['slack']<0]
  if not rows:
   # A cache cannot establish entry readiness: full geometry is searched again.
   self.witness_anchor=None;return self.continuation_witnesses(q)
  self.fast_updates+=1;self.witness_queries+=1
  return dict(rows=rows,deficit=max(-r['slack'] for r in rows),potential=sum(r['slack']**2 for r in rows)/len(rows),seconds=time.perf_counter()-started)
 def _admit(self,c,arrays,answer,coordinate=None):
  raw=answer['velocity'];v=np.clip(raw,*c.joint_velocity_bounds());H,g,A,l,u=arrays
  def reject(name):self.admission_detail[name]=self.admission_detail.get(name,0)+1
  if max(np.max(l-A@v),np.max(A@v-u),0)>1e-7:reject('postclip_original_rows');return None
  brake=self.previous-np.clip(self.previous,-self.acceleration*self.dt,self.acceleration*self.dt)
  for scale in [1.,.75,.5,.25,.125,.0625,0.]:
   candidate=brake+scale*(v-brake)
   if coordinate is not None and coordinate@candidate<=2e-6:reject('requested_progress');continue
   if not self.memory_safe(self.q,candidate):reject('no_return_memory');continue
   if not self.ctx.guard.braking_safe(self.ctx.proxy_guard,self.q,candidate):reject('physical_continuous_or_braking');continue
   if np.linalg.norm(candidate)<1e-8:reject('zero_only_admission')
   return candidate,scale
  reject('all_scales_rejected');return None
 def _solve(self,c,s,arrays,release=None):
  answer=super()._solve(c,s,arrays,release)
  if self.mode!='CERT_PREP' or release is not None or not self.future_state['rows']:return answer
  # When the least-squares compromise makes no witness-potential progress,
  # request a realizable fraction of the maximum current descent support.
  rows=self.future_state['rows'];b=sum(-r['slack']*r['b'] for r in rows);norm=np.linalg.norm(b)
  if norm<1e-12:return answer
  b/=norm
  ad=None if answer is None else self._admit(c,arrays,answer)
  if ad is not None and b@ad[0]>4e-6:return answer
  chosen=self._try_direction(c,s,arrays,b)
  return answer if chosen is None else chosen[0]
 def summary(self):
  out=super().summary();out.update(continuation_cache_full_refreshes=self.refreshes,continuation_cache_fast_updates=self.fast_updates,admission_rejection_detail=self.admission_detail,continuation_cache_rule='same observed witnesses, fresh kinematics and exact signed distances every call; complete refresh after 100 cycles or 0.04rad displacement; mandatory fresh scan and whole-curve certificate before entry')
  return out
