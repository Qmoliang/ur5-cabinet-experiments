"""Rejoin the original Cartesian task after certified actual task progress."""
import numpy as np
from executed_frame_policy import LowDimPolicy as Previous
from goal_entry_policy_v6 import LowDimPolicy as GoalSolver
class LowDimPolicy(Previous):
 def __init__(self,*a,**kw):
  self.goal_rejoin=False;self.rejoin_testing=False;self.rejoin_events=[];self.rejoin_stalls=[];self.goal_best=np.inf;self.goal_progress_cycle=0;self.goal_first_execution=False;self.ee_escape_hit=None
  super().__init__(*a,**kw)
 def _terms(self,c,s):
  if self.goal_rejoin or self.rejoin_testing:return np.zeros((6,6)),np.zeros(6)
  return super()._terms(c,s)
 def start_surface(self,*args,**kwargs):
  accepted=super().start_surface(*args,**kwargs)
  if accepted:
   hit=self.feature(self.q);distance=self.error;direction=np.r_[(self.ctx.target-self.ee)/max(distance,1e-12),np.zeros(2)];delta=min(self.radius/2,distance/4);self.ee_escape_hit=distance
   self.surface_pending=dict(normal=direction.tolist(),offset=float(direction@hit+delta),hit_feature=hit.tolist(),progress_buffer_m=delta,goal_value=distance-delta,hit_distance=distance,selected_row=list(self.blocker_id),hit_cycle=self.k,memory_kind='original_task_goal_fiber')
  return accepted
 def _register(self,cert):
  if self.goal_rejoin:
   first=cert['steps'][0];q=first['q'];v=first['velocity'];g=self.ctx.guard;site=self.ctx.model.site('attachment_site').id;g.positions(q);before=np.linalg.norm(g.data.site_xpos[site]-self.ctx.target);g.positions(q+self.dt*v);after=np.linalg.norm(g.data.site_xpos[site]-self.ctx.target)
   if after>before+1e-12:return
  return super()._register(cert)
 def _try_rejoin(self,c,s,arrays):
  m=self.surface_pending
  if m is None or self.ee_escape_hit is None:return None
  delta=m['progress_buffer_m']
  if self.error>self.ee_escape_hit-delta or self._value(m,self.q)<delta:return None
  if np.max(abs(self.previous))>1e-8:
   self.command_epoch+=1;self.brake_pending=True;self.expected=None;c.pilot_supplied_solution=None;return np.zeros(6)
  old_mode=self.mode;old_memories=self.memories;old_cache=self.solve_cache;self.mode='GOAL_REJOIN';self.memories=old_memories+[m];self.rejoin_testing=True;self.solve_cache={};answer=None;chosen=None;rate=0.
  try:
   stop=self._certify(self.q,self.previous,np.zeros(6))
   if stop is not None:
    answer=GoalSolver._solve(self,c,s,arrays)
    chosen=None if answer is None else self._admit(c,arrays,answer)
    if chosen is not None:rate=float(-(self.ee-self.ctx.target)@s['J']@chosen[0]/max(self.error,1e-12))
  finally:self.mode=old_mode;self.memories=old_memories;self.rejoin_testing=False;self.solve_cache=old_cache
  if chosen is None or rate<=1e-4:return None
  committed=dict(m,activation_cycle=self.k);self.memories.append(committed);self.surface_pending=None;self.surface_visited.extend(self.surface_nodes);self.surface_escape=False;self.goal_rejoin=True;self.mode='GOAL_REJOIN';self.goal_best=self.error;self.goal_progress_cycle=self.k;self.goal_first_execution=False;self.actual_checkpoint=None;self.command_epoch+=1;self.solve_cache={}
  record=dict(cycle=self.k,hit_error_m=self.ee_escape_hit,error_m=self.error,certified_goal_rate_m_s=rate,memory_index=len(self.memories)-1,goal_fiber_preserved=True);self.rejoin_events.append(record);self.emit('certified_original_goal_rejoin',**{k:v for k,v in record.items() if k!='cycle'});return self._publish(c,answer)
 def __call__(self,c,s,arrays):
  self.current_s=s
  if self.goal_rejoin:
   self.expected=None;self.last_delta=np.zeros(6)
   if self.brake_pending:return np.zeros(6)
   if self.error<self.goal_best-.0002:self.goal_best=self.error;self.goal_progress_cycle=self.k
   if self.k-self.goal_progress_cycle>=100 and self.error>.002:
    record=dict(cycle=self.k,error_m=self.error,best_error_m=self.goal_best,memories=len(self.memories),previous_terminal_options=sorted(self.option_tried));self.rejoin_stalls.append(record);self.emit('goal_rejoin_stalled_at_new_certified_progress_region',**{k:v for k,v in record.items() if k!='cycle'});self.goal_rejoin=False;self.mode='CERT_PREP';self.option_tried={self.option_rank};self.option_last=None;self.option_best=np.inf;self.best_deficit=np.inf;self.actual_checkpoint=None;self.surface_feature_goal=None;self.update_surface_feature_progress();self.command_epoch+=1;self.brake_pending=np.max(abs(self.previous))>1e-8;self.solve_cache={};return np.zeros(6)
   answer=GoalSolver._solve(self,c,s,arrays)
   if answer is not None and self._admit(c,arrays,answer) is not None:return self._publish(c,answer)
   self.brake_pending=True;return np.zeros(6)
  if self.surface_escape and not self.brake_pending:
   entered=self._try_rejoin(c,s,arrays)
   if entered is not None:return entered
  if self.surface_escape:
   # The old feature-distance/terminal-ray handoff is replaced by the
   # actual original-task gate above. Keep the same pending memory intact.
   pending=self.surface_pending;self.surface_pending=None
   try:return super().__call__(c,s,arrays)
   finally:self.surface_pending=pending
  return super().__call__(c,s,arrays)
 def filter_viability(self,q,previous,command):
  v=super().filter_viability(q,previous,command)
  if self.goal_rejoin and not self.goal_first_execution and self.execution_ledger[-1]['proposal'] and np.max(abs(v))>1e-8:
   self.goal_first_execution=True;self.goal_best=self.error;self.goal_progress_cycle=self.k
  return v
 def summary(self):
  out=super().summary();out.update(revision='goal_fiber_rejoin_with_binding_and_feasible_backup',original_goal_rejoins=self.rejoin_events,original_goal_rejoin_stalls=self.rejoin_stalls,new_memory_kind='pure Cartesian task progress, retaining all terminal postures at the original target; inherited memories unchanged',original_goal_objective_in_rejoin=True);return out
