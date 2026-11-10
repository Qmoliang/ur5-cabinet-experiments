"""Use constraints opposing the actual escape task, and rejoin goal tracking when clear."""
import numpy as np
from latency_backup_policy import LowDimPolicy as Previous
from goal_entry_policy_v6 import LowDimPolicy as CurrentSolver
from frame_transfer_policy import LowDimPolicy as FrameSelection
from goal_entry_policy import clipped
class LowDimPolicy(Previous):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);self.escape_shadow_calls=0;self.escape_shadow_failures=0;self.free_frame_tasks=0
 def _terms(self,c,s):
  if not self.surface_escape:return super()._terms(c,s)
  geom=self._geometry()
  if geom is None:return super()._terms(c,s)
  point,n,J,gap=geom
  if gap<=c.near_distance:return super()._terms(c,s)
  # Once this contact is clear, a perpetual constant-speed tangent no longer
  # expresses the task. Track the same body's frozen terminal position, while
  # retaining original EE tracking and two posture features in its nullspace.
  self.free_frame_tasks+=1;K=J;desired=clipped(c.task_gain*(self._goal_point_for_blocker()-point),c.max_task_speed);inverse=np.linalg.pinv(K,rcond=1e-8);N=np.eye(6)-inverse@K
  posture=self.P@N;ee=s['J']@N;posture_desired=clipped(-c.task_gain*self.P@(self.q-self.qg)-self.P@inverse@desired,c.max_task_speed);ee_desired=clipped(s['desired']-s['J']@inverse@desired,c.max_task_speed)
  return c.task_weight*(K.T@K+posture.T@posture+ee.T@ee-s['J'].T@s['J']),c.task_weight*(s['J'].T@s['desired']-K.T@desired-posture.T@posture_desired-ee.T@ee_desired)
 def _current_binding_rows(self,s,arrays):
  # The original goal-tracking duals can oppose a needed retreat. Diagnose
  # the CURRENT escape objective, with the same original and memory rows.
  self.escape_shadow_calls+=1;answer=CurrentSolver._solve(self,self.ctx.controller,s,arrays)
  if answer is None:self.escape_shadow_failures+=1;return []
  actual=dict(s,y=answer['original_dual'],status='solved');return FrameSelection._current_binding_rows(self,actual,arrays)
 def _try_direction(self,c,s,arrays,b):
  if self.surface_escape:
   geom=self._geometry()
   if geom is not None and geom[3]>c.near_distance:
    d=self._goal_point_for_blocker()-geom[0];norm=np.linalg.norm(d)
    if norm>1e-8:b=(d/norm)@geom[2]
    elif np.linalg.norm(s['desired'])>1e-8:b=(s['desired']/np.linalg.norm(s['desired']))@s['J']
  return super()._try_direction(c,s,arrays,b)
 def summary(self):
  out=super().summary();out.update(revision='current_escape_task_binding_and_feasible_holds',binding_duals='current low-dimensional escape QP, not the nominal Cartesian attraction',escape_shadow_qps=self.escape_shadow_calls,escape_shadow_failures=self.escape_shadow_failures,cleared_frame_goal_tasks=self.free_frame_tasks,near_surface_task_dimension=2,clear_frame_task_dimension=3,posture_feature_dimension=2,optimized_future_configurations=0);return out
