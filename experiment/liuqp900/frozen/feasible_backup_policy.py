"""Execution certificates carry a feasible deterministic braking continuation."""
import time,json,hashlib
from pathlib import Path
import numpy as np
from frame_transfer_policy import LowDimPolicy as Previous
from original_row_certificate import build_assembler,check_rows
from verified_qp import solve_verified
from certified_auxiliary import verify_solution
from goal_entry_policy_v4 import SELF_CERTIFICATE_ROUNDOFF_M,ROW_ACCEPTANCE_TOLERANCE
class LowDimPolicy(Previous):
 def __init__(self,ctx,job,event):
  super().__init__(ctx,job,event);self.assembler=build_assembler(job['task'],job.get('representation','ellipsoid'));self.cert_cache={};self.cache_state=None;self.admitted={};self.backup=None;self.backup_actions=0;self.proposal_actions=0;self.stale_actions=0;self.certified_calls=0;self.certified_passes=0;self.cert_rejects={};self.cert_ms=[];self.row_maximum=0.;self.execution_ledger=[];self.certificate_serial=0;self.last_certificate=None;self.feasibility_violations=0
  q=ctx.q0.copy();previous=np.asarray(ctx.controller.applied_velocity).copy();arrays,ids,J=self.assembler.rows(q,previous,ctx.target);brake=previous-np.clip(previous,-self.acceleration*self.dt,self.acceleration*self.dt);solution=solve_verified(*arrays)
  options=[brake]+([] if solution is None else [solution.x])
  for v in options:
   cert=self._certify(q,previous,v)
   if cert is not None:self.backup=cert;self.emit('feasible_backup_initialized',steps=len(cert['steps']),first_velocity=v.tolist());break
  if self.backup is None:raise RuntimeError('No original-row-feasible initial continuation; not silently assuming zero velocity is feasible')
 def _memory_stamp(self):return tuple((tuple(m['normal']),float(m['offset'])) for m in self.memories)
 def memory_safe(self,q,v,additional=None):
  if additional is None or not hasattr(self,'assembler'):return super().memory_safe(q,v,additional)
  old=self.memories;self.memories=old+[additional]
  try:return self._certify(q,self.previous,v) is not None
  finally:self.memories=old
 def _key(self,q,previous):return q.tobytes()+previous.tobytes()
 def _extra_violation(self,q,v,arrays,ids,J):
  A,u=arrays[2],arrays[4];values=[]
  inward=(SELF_CERTIFICATE_ROUNDOFF_M+self.dt*ROW_ACCEPTANCE_TOLERANCE)/self.dt
  indices=[i for i,k in enumerate(ids) if k[0]=='self' and u[i]*self.dt<self.ctx.controller.near_distance and np.linalg.norm(A[i])>1e-12]
  if indices:values.append(float(np.max(A[indices]@v-u[indices]+inward)))
  f=self.feature(q)
  for m in self.memories:
   n=np.asarray(m['normal']);gradient=n[:3]@J+n[3:]@self.P;values.append(float(-gradient@v-self.ctx.controller.task_gain*(n@f-m['offset'])))
  return max([0.]+values)
 def _certify(self,q,previous,v):
  state=self._key(q,previous)
  if self.cache_state!=state:self.cache_state=state;self.cert_cache={};self.admitted={}
  key=(v.tobytes(),self._memory_stamp())
  if key in self.cert_cache:return self.cert_cache[key]
  start=time.perf_counter();self.certified_calls+=1;reason=None;steps=[];maximum=0.
  if not self.ctx.guard.braking_safe(self.ctx.proxy_guard,q,v):reason='continuous_physical_or_braking'
  elif not self.memory_safe(q,v):reason='continuous_memory_or_braking'
  else:
   current=q.copy();last=previous.copy();speed=v.copy()
   for k in range(11):
    arrays,ids,J=self.assembler.rows(current,last,self.ctx.target);raw=check_rows(arrays,speed);extra=self._extra_violation(current,speed,arrays,ids,J);violation=max(raw,extra);maximum=max(maximum,violation)
    if violation>ROW_ACCEPTANCE_TOLERANCE:
     reason='original_current_rows' if k==0 else 'original_backup_rows'
     break
    # Reusable rest certificates require the exact next (q, previous) state.
    # A tiny previous velocity still needs its explicit zero-speed transition.
    terminal=bool(np.all(speed == 0.) and np.all(last == 0.))
    steps.append(dict(q=current.copy(),previous=last.copy(),velocity=speed.copy(),row_violation=raw,extra_violation=extra,terminal=terminal))
    if terminal:break
    current=current+self.dt*speed;last=speed.copy();speed=speed-np.clip(speed,-self.acceleration*self.dt,self.acceleration*self.dt)
   if not steps or not steps[-1]['terminal']:reason=reason or 'braking_horizon_exhausted'
  elapsed=time.perf_counter()-start;self.cert_ms.append(elapsed*1000)
  if reason is not None:
   self.cert_rejects[reason]=self.cert_rejects.get(reason,0)+1;self.cert_cache[key]=None;return None
  self.certified_passes+=1;self.certificate_serial+=1;cert=dict(id=self.certificate_serial,steps=steps,memory_count=len(self.memories),memory_signature=self._memory_stamp(),maximum_violation=maximum);self.cert_cache[key]=cert;return cert
 def _admit(self,c,arrays,answer,coordinate=None):
  raw=np.clip(answer['velocity'],*c.joint_velocity_bounds())
  if check_rows(arrays,raw)>ROW_ACCEPTANCE_TOLERANCE:return None
  brake=self.previous-np.clip(self.previous,-self.acceleration*self.dt,self.acceleration*self.dt)
  for scale in [1.,.75,.5,.25,.125,.0625,0.]:
   v=raw.copy() if scale==1. else brake+scale*(raw-brake)
   if coordinate is not None and coordinate@v<=2e-6:continue
   cert=self._certify(self.q,self.previous,v)
   if cert is not None:self.admitted[id(answer)]=(answer['velocity'].copy(),v.copy(),cert);return v,scale
  return None
 def _publish(self,c,answer):
  selected=self.admitted.get(id(answer))
  if selected is None or not np.array_equal(selected[0],answer['velocity']):
   if self._admit(c,c.last_qp_arrays,answer) is None:return np.zeros(6)
   selected=self.admitted[id(answer)]
  _,v,cert=selected
  if not np.array_equal(v,answer['velocity']):
   # A certified scaled velocity is the exact minimizer of the same original
   # Hessian with this linear perturbation. Original constraints stay intact.
   H,g,A,l,u=c.last_qp_arrays;delta=-H@v-g;answer=dict(answer,velocity=v,delta=delta,original_dual=np.zeros(len(A)),iterations=0);verify_solution(H,g+delta,A,l,u,answer)
  return super()._publish(c,answer)
 def __call__(self,c,s,arrays):
  delta=super().__call__(c,s,arrays)
  if self.expected is None and not self.brake_pending and s['x'] is not None:self._certify(self.q,self.previous,np.asarray(s['x']))
  return delta
 def filter_viability(self,q,previous,command):
  selected=None;state=self._key(q,previous)
  if self.cache_state==state:
   for cert in self.cert_cache.values():
    if cert is None:continue
    first=cert['steps'][0]
    if np.max(abs(first['velocity']-command))<=1e-10 and np.array_equal(first['q'],q) and np.array_equal(first['previous'],previous):selected=cert;break
  from_proposal=selected is not None
  if selected is None:selected=self.backup
  if selected is None:raise RuntimeError('Missing feasible backup')
  first=selected['steps'][0]
  if not np.array_equal(first['q'],q) or not np.array_equal(first['previous'],previous):
   self.feasibility_violations+=1;raise RuntimeError('Actual state diverged from the certified backup; no uncertified velocity is issued')
  if selected['memory_signature']!=self._memory_stamp():
   refreshed=self.cert_cache.get((first['velocity'].tobytes(),self._memory_stamp())) if self.cache_state==state else None
   if refreshed is None:raise RuntimeError('New memory has no original-row-certified backup')
   selected=refreshed;first=selected['steps'][0]
  v=first['velocity'].copy();self.row_maximum=max(self.row_maximum,first['row_violation']);self.proposal_actions+=int(from_proposal);self.backup_actions+=int(not from_proposal);changed=bool(np.max(abs(v-command))>1e-10);self.stale_actions+=int(changed)
  self.execution_ledger.append(dict(cycle=self.k,certificate=selected['id'],proposal=from_proposal,substituted=changed,original_row_violation=first['row_violation'],extra_violation=first['extra_violation'],backup_steps=len(selected['steps'])))
  remaining=selected['steps'][1:] if len(selected['steps'])>1 else selected['steps'];self.backup=dict(selected,steps=remaining);self.last_certificate=selected
  if changed and (self.stale_actions<=15 or self.k%100==0):self.emit('certified_backup_replaces_uncertified_or_stale_command',certificate=selected['id'],remaining_steps=len(remaining))
  return v
 def summary(self):
  out=super().summary();folder=Path(__import__('os').environ['UNIFIED_CASE_DIR']);(folder/'feasibility_ledger.json').write_text(json.dumps(self.execution_ledger));
  if self.backup is not None:(folder/'final_backup.json').write_text(json.dumps(self.backup,default=lambda x:x.tolist() if hasattr(x,'tolist') else x,indent=2))
  out.update(revision='binding_frame_and_original_row_backup',feasible_backup_proofs=self.certified_passes,backup_certificate_calls=self.certified_calls,backup_certificate_rejections=self.cert_rejects,backup_certificate_ms_p99=None if not self.cert_ms else float(np.percentile(self.cert_ms,99)),certified_proposal_steps=self.proposal_actions,certified_backup_steps=self.backup_actions,stale_commands_replaced=self.stale_actions,maximum_executed_original_row_violation=self.row_maximum,backup_state_mismatches=self.feasibility_violations,certificate_assembler_calls=self.assembler.calls,original_row_source_hash=self.assembler.source_hash,backup_kind='current proposed velocity followed by fixed original-acceleration braking; all original rows, physical and memory constraints checked; no optimized future configurations');return out
