"""Evaluate a new physical task only after its command actually executes."""
import numpy as np
from task_cone_policy import LowDimPolicy as Previous
class LowDimPolicy(Previous):
 def __init__(self,*a,**kw):
  self.frame_generation=0;self.frame_waiting=False;self.frame_has_selected_action=False;self.frame_attempts=0;self.frame_activation=0;self.frame_execution_events=[];self.frame_transfer_deferrals=0
  super().__init__(*a,**kw)
 def _begin_frame(self):
  self.frame_generation+=1;self.frame_waiting=True;self.frame_has_selected_action=False;self.frame_attempts=0;self.frame_activation=self.k;self.surface_feature_goal=None;self.update_surface_feature_progress()
 def emit(self,event,**data):
  if event=='physical_blocker_frame_transferred':self._begin_frame()
  return super().emit(event,**data)
 def start_surface(self,*args,**kwargs):
  accepted=super().start_surface(*args,**kwargs)
  if accepted:self._begin_frame()
  return accepted
 def _current_binding_rows(self,s,arrays):
  candidates=super()._current_binding_rows(s,arrays)
  if self.blocker_id is None:return candidates
  geometry=self._geometry()
  if geometry is None or geometry[3]>self.ctx.controller.near_distance:return candidates
  waiting=self.frame_waiting and (self.frame_has_selected_action or self.frame_attempts==0 or self.k-self.frame_activation<100)
  if waiting:
   same=[x for x in candidates if self._frame(x[2])==self._frame(self.blocker_id)];self.frame_transfer_deferrals+=int(len(same)!=len(candidates));return same
  return candidates
 def surface_call(self,*args,**kwargs):
  generation=self.frame_generation;delta=super().surface_call(*args,**kwargs)
  if generation==self.frame_generation:self.frame_attempts+=1
  return delta
 def _publish(self,*args,**kwargs):
  delta=super()._publish(*args,**kwargs)
  if self.surface_escape and self.expected is not None and np.max(abs(self.expected))>1e-8:self.frame_has_selected_action=True
  return delta
 def filter_viability(self,q,previous,command):
  v=super().filter_viability(q,previous,command)
  if self.surface_escape and self.frame_waiting and self.execution_ledger[-1]['proposal'] and np.max(abs(v))>1e-8:
   self.frame_waiting=False;self.surface_feature_goal=None;self.update_surface_feature_progress();event=dict(cycle=self.k,activation_cycle=self.frame_activation,frame_generation=self.frame_generation,row=list(self.blocker_id));self.frame_execution_events.append(event);self.emit('new_blocking_frame_first_action_executed',**{k:v for k,v in event.items() if k!='cycle'})
  return v
 def summary(self):
  out=super().summary();out.update(revision='execution_based_frame_transfer_with_feasible_holds',frame_execution_events=self.frame_execution_events,frame_transfer_deferrals=self.frame_transfer_deferrals,frame_progress_reference='reset at real frame-task execution, not inherited from an earlier task; same original 100-cycle and 0.2mm actual feature progress test');return out
