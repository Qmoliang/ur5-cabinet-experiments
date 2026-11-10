"""Preserve the established preparation law until its actual progress stalls."""
import numpy as np
from goal_fiber_policy import LowDimPolicy as Previous
from legacy_goal_plane_memory_policy import LowDimPolicy as Legacy
class LowDimPolicy(Previous):
 def __init__(self,*args,**kwargs):
  self.extension_active=False;self.extension_events=[];self.legacy_preparation_calls=0;self.legacy_margin_best=np.inf;self.legacy_margin_progress=None
  super().__init__(*args,**kwargs)
 def _terms(self,c,s):
  if not self.extension_active:return Legacy._terms(self,c,s)
  return super()._terms(c,s)
 def __call__(self,c,s,arrays):
  if self.extension_active:return super().__call__(c,s,arrays)
  self.legacy_preparation_calls+=1;delta=Legacy.__call__(self,c,s,arrays)
  if not self.escape_active and not self.brake_pending and s['x'] is not None:
   nominal=np.clip(np.asarray(s['x']),*c.joint_velocity_bounds())
   cert=self._certify(self.q,self.previous,nominal)
   if cert is not None:
    self.published_certificates[cert['id']]=(self.command_epoch,cert);self._extend_published(cert)
  if self.escape_active and self.future_state is not None and not self.brake_pending:
   margin=self.future_state['deficit']
   if self.legacy_margin_progress is None or margin<self.legacy_margin_best-.0002:
    self.legacy_margin_best=margin;self.legacy_margin_progress=self.k
   if self.error>.001 and self.k-self.legacy_margin_progress>=100 and self.k-self.surface_feature_last>=100:
    event=dict(cycle=self.k,mode_before=self.mode,error_m=self.error,continuation_deficit_m=margin,cycles_since_continuation_progress=self.k-self.legacy_margin_progress,cycles_since_actual_feature_progress=self.k-self.surface_feature_last)
    self.extension_active=True;self.extension_events.append(event);self.emit('legacy_preparation_stall_enables_extension',**{k:v for k,v in event.items() if k!='cycle'})
    self.option_last=None;self.option_best=np.inf;self.actual_checkpoint=None;self.mode='CERT_PREP';self.command_epoch+=1;self.brake_pending=np.max(abs(self.previous))>1e-8;self.expected=None;self.last_delta=np.zeros(6);self.solve_cache={};c.pilot_supplied_solution=None;return np.zeros(6)
  return delta
 def summary(self):
  out=super().summary();out.update(revision='legacy_preparation_preserved_until_actual_stall',legacy_preparation_calls=self.legacy_preparation_calls,extension_events=self.extension_events,legacy_goal_and_posture_objective_unchanged_before_extension=True,extension_activation='same 0.2mm/100-cycle measured continuation and actual feature stagnation; all physical/original-row guards remain active');return out
