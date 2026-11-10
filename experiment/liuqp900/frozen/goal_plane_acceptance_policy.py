"""Goal-side continuation objective with actual stage acceptance and stale-command reset."""
import numpy as np
from goal_side_plane_policy import LowDimPolicy as Previous
class LowDimPolicy(Previous):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.actual_checkpoint=None;self.goal_plane_acceptance=[];self.goal_plane_rejected_stages=0
 def continuation_witnesses(self,q):
  state=super().continuation_witnesses(q)
  if self.mode not in ('CERT_PREP','RESERVE_GOAL'):return state
  m=-state['deficit'];a=self.actual_checkpoint
  if a is None:self.actual_checkpoint=dict(cycle=self.k,q=q.copy(),margin=m,rows=[dict(r,b=r['b'].copy()) for r in state['rows']]);return state
  if self.k-a['cycle']<100:return state
  predicted=min([r['slack']+r['b']@(q-a['q']) for r in a['rows']],default=0.)-a['margin'];actual=m-a['margin'];rejected=actual<-.0002 or (predicted>.0002 and actual<.25*predicted)
  rec=dict(cycle=self.k,start_cycle=a['cycle'],predicted_gain_m=float(predicted),actual_gain_m=float(actual),before_m=a['margin'],after_m=m,accepted=actual>=.0002,rejected=bool(rejected),physical_sampled_deficit_m=state['physical_sampled_deficit'])
  self.goal_plane_acceptance.append(rec);self.emit('goal_plane_actual_progress_assessed',**{k:v for k,v in rec.items() if k!='cycle'})
  if rejected:self.goal_plane_rejected_stages+=1;self.command_epoch+=1;self.brake_pending=np.max(np.abs(self.previous))>1e-8
  self.actual_checkpoint=dict(cycle=self.k,q=q.copy(),margin=m,rows=[dict(r,b=r['b'].copy()) for r in state['rows']]);return state
 def __call__(self,c,s,arrays):
  delta=super().__call__(c,s,arrays)
  if self.brake_pending:self.expected=None;self.last_delta=np.zeros(6);c.pilot_supplied_solution=None;return np.zeros(6)
  return delta
 def summary(self):
  out=super().summary();out.update(diagnostic_intervention='new terminal-side support-plane objective with full witness refresh and actual stage acceptance; no witness cache or changed safety',goal_plane_actual_progress=self.goal_plane_acceptance,goal_plane_rejected_stages=self.goal_plane_rejected_stages,actual_progress_rule='every original 100-cycle window compare newly rebuilt plane margin with the old model over actual executed joint displacement; reject old queued commands and brake/recompute on regression or insufficient realization')
  return out
