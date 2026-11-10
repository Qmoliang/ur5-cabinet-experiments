"""Current-velocity preparation for one fixed terminal contraction certificate.
No tree, waypoints, optimized future configurations, or stored successful route.
The virtual points are fixed evaluations of q(s)=(1-s)q+s*qg, never decisions.
"""
import json,time
from pathlib import Path
import numpy as np,mujoco
from goal_entry_policy_activated import LowDimPolicy as Previous
from goal_entry_policy import clipped
from static_safety import ExecutionGuard

class LowDimPolicy(Previous):
 def __init__(self,ctx,job,event):
  super().__init__(ctx,job,event)
  self.future_guard=ExecutionGuard(ctx.model,ctx.robot,self.dt,.006,self.acceleration);self.future_guard.set_self_pairs(ctx.selfrows.pairs);self.future_guard.margin=ctx.controller.safety_margin
  self.future_rows=[];self.future_state=None;self.entry_certified=False;self.entry_calls=0;self.entry_passes=0;self.witness_queries=0;self.prepare_steps=0;self.entry_steps=0;self.prepare_failures=0;self.best_deficit=float('inf');self.last_deficit_progress=0;self.entry_events=[];self.gate_rejections={};self.entry_certificate_ms=[]
  self.restore_memories=job.get('initial_memories')
  if self.restore_memories:
   self.memories=json.loads(Path(self.restore_memories).read_text());self.emit('inherited_no_return_memories',count=len(self.memories))
  self.terminal_basis=None;self.sample_fractions=np.linspace(0,1,9)[1:-1]
 def _jac(self,points,indices):
  guard=self.future_guard;mujoco.mj_comPos(guard.model,guard.data);out={}
  for i in set(map(int,indices)):
   J=np.zeros((3,guard.model.nv));mujoco.mj_jac(guard.model,guard.data,J,None,points[i],int(guard.bodies[i]));out[i]=J[:,:6]
  return out
 def continuation_witnesses(self,q):
  started=time.perf_counter();g=self.future_guard;pg=self.ctx.proxy_guard;rows=[];worst=0.
  for a in self.sample_fractions:
   qs=(1-a)*q+a*self.qg;p=g.positions(qs).copy();ri,oi=pg.pairs(p,p,g.radii+self.ctx.controller.near_distance)
   observations=[]
   if len(oi):
    dist,n=pg.point_geometry(p[ri],oi,optimizer=True);gap=dist-g.radii[ri];slack=gap-self.ctx.controller.safety_margin
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
 def continuation_certificate(self):
  self.entry_calls+=1;t=time.perf_counter()
  physical=self.future_guard.segment(self.ctx.proxy_guard,self.q,self.qg)
  memory=physical and self._memory_segment(self.q,self.qg,self.memories)
  self.entry_certificate_ms.append((time.perf_counter()-t)*1000)
  if physical and memory:self.entry_passes+=1;return True
  key='continuous_physical' if not physical else 'no_return_memory';self.gate_rejections[key]=self.gate_rejections.get(key,0)+1
  return False
 def _terms(self,c,s):
  if self.mode=='CERT_ENTER':
   # Original Cartesian objective plus a scalar terminal contraction direction.
   return np.zeros((6,6)),np.zeros(6)
  rows=self.future_state['rows'];K=np.asarray([r['b'] for r in rows]).reshape(-1,6);v=np.asarray([-c.task_gain*r['slack'] for r in rows]);v=np.minimum(v,c.max_task_speed)
  if len(rows):K=K/np.sqrt(len(rows));v=v/np.sqrt(len(rows))
  else:K=s['J'];v=s['desired']
  N=np.eye(6)-np.linalg.pinv(K,rcond=1e-8)@K
  posture=self.posture_lever*N;desired=clipped(-c.task_gain*posture@(self.q-self.qg),c.max_task_speed)
  HH=c.task_weight*(K.T@K+posture.T@posture-s['J'].T@s['J']);gg=c.task_weight*(s['J'].T@s['desired']-K.T@v-posture.T@desired)
  return HH,gg
 def _extras(self,c,s):
  rows,bounds=super()._extras(c,s)
  if self.mode=='CERT_ENTER':
   d=self.qg-self.q;norm=np.linalg.norm(d)
   if norm>1e-10:
    e=d/norm;B=np.linalg.svd(e[None,:],full_matrices=True)[2][1:]
    rows=np.vstack([rows,B,-B,-e,e]);bounds=np.r_[bounds,np.zeros(10),0,min(.5,norm/self.dt)]
  return rows,bounds
 def __call__(self,c,s,arrays):
  self.expected=None;self.last_delta=np.zeros(6);self.current_s=s
  if not self.escape_active:
   history=[x['error_m'] for x in self.history[-100:]]
   if not (len(history)==100 and history[0]-min(history)<.0002 and self.error>.002):return np.zeros(6)
   self.escape_active=True;self.command_epoch+=1;self.brake_pending=np.max(np.abs(self.previous))>1e-8;self.emit('actual_stall_activates_continuation_certificate',error_m=self.error)
  if self.terminal or self.brake_pending:return np.zeros(6)
  self.future_state=self.continuation_witnesses(self.q);D=self.future_state['deficit'];self.future_rows=self.future_state['rows']
  if D<self.best_deficit-.0002:self.best_deficit=D;self.last_deficit_progress=self.k
  certified=False
  if not self.future_rows:certified=self.continuation_certificate()
  newmode='CERT_ENTER' if certified else 'CERT_PREP'
  if newmode!=self.mode:
   old=self.mode;self.mode=newmode;self.command_epoch+=1;self.solve_cache={};self.brake_pending=np.max(np.abs(self.previous))>1e-8
   item=dict(cycle=self.k,previous_mode=old,mode=newmode,deficit_m=D,q=self.q.tolist(),memory_count=len(self.memories));self.entry_events.append(item);self.emit('continuation_mode_changed',**{k:v for k,v in item.items() if k not in ('cycle','mode')})
   if self.brake_pending:return np.zeros(6)
  if certified:
   answer=self._solve(c,s,arrays)
   if answer is not None and self._admit(c,arrays,answer) is not None:self.entry_steps+=1;return self._publish(c,answer)
  else:
   answer=self._solve(c,s,arrays)
   if answer is not None:
    chosen=self._admit(c,arrays,answer)
    if chosen is not None:
     self.prepare_steps+=1
     if self.k%50==0:self.emit('continuation_preparation',deficit_m=D,potential=self.future_state['potential'],witnesses=len(self.future_rows),worst=sorted([{k:v for k,v in r.items() if k!='b'} for r in self.future_rows],key=lambda r:r['slack'])[:3],actual_goal_rate=float((self.ee-self.ctx.target)@s['J']@chosen[0]),safe_scale=chosen[1])
     return self._publish(c,answer)
  self.prepare_failures+=1;self.brake_pending=True;return np.zeros(6)
 def summary(self):
  out=super().summary();out.update(continuation_kind='single fixed terminal contraction feedback, never a searched configuration route',continuation_queries=self.witness_queries,continuation_certificates=self.entry_calls,continuation_passes=self.entry_passes,continuation_certificate_ms_p99=None if not self.entry_certificate_ms else float(np.percentile(self.entry_certificate_ms,99)),continuation_gate_rejections=self.gate_rejections,continuation_prepare_steps=self.prepare_steps,continuation_entry_steps=self.entry_steps,continuation_prepare_failures=self.prepare_failures,continuation_best_deficit_m=self.best_deficit,continuation_events=self.entry_events,no_return_constraints_discarded=0,optimized_future_configurations=0,waypoint_tracking=False,posture_preparation='terminal posture only in nullspace of pulled-back continuation witness task')
  return out
