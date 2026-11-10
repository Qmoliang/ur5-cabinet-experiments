"""Original LiuQP objective with the exact same certified execution utilities.
No terminal IK, posture target, escape state, or no-return memory is constructed.
"""
from pathlib import Path
from collections import OrderedDict
import json,os,numpy as np
from latency_backup_policy import LowDimPolicy as CertificateUtilities
from original_row_certificate import build_assembler
from verified_qp import solve_verified
from goal_entry_policy_v4 import SELF_CERTIFICATE_ROUNDOFF_M,ROW_ACCEPTANCE_TOLERANCE
class LowDimPolicy(CertificateUtilities):
 def __init__(self,ctx,job,event):
  self.ctx=ctx;self.job=job;self.event_callback=event;self.k=0;self.q=ctx.q0.copy();self.previous=np.asarray(ctx.controller.applied_velocity).copy();self.dt=ctx.controller.dt if hasattr(ctx.controller,'dt') else .02;self.acceleration=ctx.controller.acceleration_limit;self.memories=[];self.mode='ORIGINAL_LIUQP';self.command_epoch=0;self.brake_pending=False;self.terminal=None;self.expected=None;self.last_delta=np.zeros(6);self.last_delay_ticks=1;self.baseline_calls=0
  self.assembler=build_assembler(job['task'],job.get('representation','ellipsoid'));self.cert_cache={};self.cache_state=None;self.admitted={};self.backup=None;self.backup_actions=0;self.proposal_actions=0;self.stale_actions=0;self.certified_calls=0;self.certified_passes=0;self.cert_rejects={};self.cert_ms=[];self.row_maximum=0.;self.execution_ledger=[];self.certificate_serial=0;self.last_certificate=None;self.feasibility_violations=0;self.published_certificates={};self.hold_registry=OrderedDict();self.extended={};self.hold_proofs=0;self.hold_requests=[];self.row_reuses=0;self.row_cache=OrderedDict();original_rows=self.assembler.rows
  def cached_rows(q,previous,target):
   key=q.tobytes()+previous.tobytes()+np.asarray(target).tobytes()
   if key in self.row_cache:self.row_reuses+=1;self.row_cache.move_to_end(key);return self.row_cache[key]
   out=original_rows(q,previous,target);self.row_cache[key]=out
   while len(self.row_cache)>2*10*11:self.row_cache.popitem(last=False)
   return out
  self.assembler.rows=cached_rows
  arrays,ids,J=self.assembler.rows(self.q,self.previous,ctx.target);brake=self.previous-np.clip(self.previous,-self.acceleration*self.dt,self.acceleration*self.dt);solution=solve_verified(*arrays)
  for v in [brake]+([] if solution is None else [solution.x]):
   cert=self._certify(self.q,self.previous,v)
   if cert is not None:self.backup=cert;self.emit('feasible_backup_initialized',steps=len(cert['steps']),first_velocity=v.tolist());break
  if self.backup is None:raise RuntimeError('No original-row-feasible initial continuation')
 def emit(self,event,**data):self.event_callback(event=event,cycle=self.k,mode=self.mode,**data)
 def observe(self,k,q,previous):self.k=k;self.q=q.copy();self.previous=previous.copy()
 def feedback(self,k,q,v,modified):
  if self.brake_pending and np.max(abs(v))<1e-8:self.brake_pending=False
 def __call__(self,c,s,arrays):
  assert c.extra_joint_goal is None and not c.extra_task_frames
  self.baseline_calls+=1;self.expected=None;self.last_delta=np.zeros(6);return np.zeros(6)
 def memory_safe(self,q,v,additional=None):
  assert not self.memories and additional is None
  return True
 def _extra_violation(self,q,v,arrays,ids,J):
  A,u=arrays[2],arrays[4];inward=(SELF_CERTIFICATE_ROUNDOFF_M+self.dt*ROW_ACCEPTANCE_TOLERANCE)/self.dt;indices=[i for i,k in enumerate(ids) if k[0]=='self' and u[i]*self.dt<self.ctx.controller.near_distance and np.linalg.norm(A[i])>1e-12]
  return max(0.,float(np.max(A[indices]@v-u[indices]+inward))) if indices else 0.
 def summary(self):
  folder=Path(os.environ['UNIFIED_CASE_DIR']);(folder/'feasibility_ledger.json').write_text(json.dumps(self.execution_ledger));(folder/'goalward_memories.json').write_text('[]')
  if self.backup is not None:(folder/'final_backup.json').write_text(json.dumps(self.backup,default=lambda x:x.tolist() if hasattr(x,'tolist') else x))
  return dict(baseline_original_objective=True,baseline_hook_calls=self.baseline_calls,goal_posture_used=False,escape_enabled=False,memory_count=0,nonzero_task_gradient_deltas=0,feasible_backup_proofs=self.certified_passes,backup_certificate_calls=self.certified_calls,backup_certificate_rejections=self.cert_rejects,certified_proposal_steps=self.proposal_actions,certified_backup_steps=self.backup_actions,maximum_executed_original_row_violation=self.row_maximum,original_row_source_hash=self.assembler.source_hash,backup_state_mismatches=self.feasibility_violations,common_certificate_method='unchanged inherited _certify, selected-command filtering, original-row reconstruction, self numerical reserve and braking tail')
