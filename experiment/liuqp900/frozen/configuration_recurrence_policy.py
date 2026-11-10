"""Detect actual configuration recurrence even inside one coarse surface node."""
import numpy as np
from scipy.spatial import cKDTree
from compatibility_clipped_policy import LowDimPolicy as Previous
class LowDimPolicy(Previous):
 def __init__(self,*a,**kw):
  self.recurrence_records=[];self.recurrence_events=[];self.recurrence_tree=None;self.recurrence_tree_size=0;self.configuration_travel=0.;self.recurrence_last_q=None;self.executed_direction=None
  super().__init__(*a,**kw)
 def start_surface(self,*a,**kw):
  ok=super().start_surface(*a,**kw)
  if ok:self.recurrence_records=[];self.recurrence_tree=None;self.recurrence_tree_size=0;self.configuration_travel=0.;self.recurrence_last_q=self.q.copy();self.executed_direction=None
  return ok
 def filter_viability(self,q,previous,command):
  v=super().filter_viability(q,previous,command)
  if self.surface_escape and self.execution_ledger[-1]['proposal'] and np.linalg.norm(v)>1e-8:self.executed_direction=v/np.linalg.norm(v)
  return v
 def surface_call(self,c,s,arrays):
  point=self.posture_lever*self.q
  if self.recurrence_last_q is not None:self.configuration_travel+=self.posture_lever*float(np.linalg.norm(self.q-self.recurrence_last_q))
  self.recurrence_last_q=self.q.copy();frame=None if self.blocker_id is None else self._frame(self.blocker_id);direction=self.executed_direction;tol=.001
  if len(self.recurrence_records)-self.recurrence_tree_size>=64:
   self.recurrence_tree=cKDTree(np.asarray([x['point'] for x in self.recurrence_records]));self.recurrence_tree_size=len(self.recurrence_records)
  candidates=[] if self.recurrence_tree is None else self.recurrence_tree.query_ball_point(point,tol)
  candidates.extend(i for i in range(self.recurrence_tree_size,len(self.recurrence_records)) if np.linalg.norm(self.recurrence_records[i]['point']-point)<=tol)
  found=None
  if not self.brake_pending and not self.surface_reversed and not self.frame_waiting and direction is not None:
   for i in candidates:
    old=self.recurrence_records[i]
    if self.k-old['cycle']>=100 and self.configuration_travel-old['travel']>=2*c.near_distance and old['frame']==frame and old['direction'] is not None and old['direction']@direction>0:
     found=old;break
  if not self.recurrence_records or np.linalg.norm(point-self.recurrence_records[-1]['point'])>=tol/2 or frame!=self.recurrence_records[-1]['frame']:
   self.recurrence_records.append(dict(point=point.copy(),cycle=self.k,travel=self.configuration_travel,frame=frame,direction=None if direction is None else direction.copy()))
  if found is not None:
   geometry=self._geometry()
   if geometry is not None:
    tangent=self._tangent(geometry);self.circulation_tangent=-tangent;self.surface_reversed=True;self.reverse_used=True;self.command_epoch+=1;self.brake_pending=True;self.expected=None;self.last_delta=np.zeros(6);self.solve_cache={};c.pilot_supplied_solution=None
    rec=dict(cycle=self.k,previous_policy_cycle=found['cycle'],configuration_distance_m=float(np.linalg.norm(point-found['point'])),travel_since_previous_m=self.configuration_travel-found['travel'],frame=list(frame),direction_alignment=float(found['direction']@direction));self.recurrence_events.append(rec);self.emit('actual_configuration_recurrence_reselects_winding',**{k:v for k,v in rec.items() if k!='cycle'});return np.zeros(6)
  return super().surface_call(c,s,arrays)
 def summary(self):
  out=super().summary();out.update(revision='world_winding_with_configuration_recurrence',configuration_recurrence_events=self.recurrence_events,recurrence_resolution_m=.001,recurrence_minimum_travel_m=2*self.ctx.controller.near_distance,recurrence_minimum_feedback_cycles=100,recurrence_role='visited configurations are only recurrence evidence; they are never targets, waypoints or search nodes',original_exit_gate_unchanged=True);return out
