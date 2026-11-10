"""Require actual feature stagnation before changing the physical task."""
import numpy as np
from surface_escape_policy import LowDimPolicy as Previous
class LowDimPolicy(Previous):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);self.surface_feature_goal=None;self.surface_feature_best=np.inf;self.surface_feature_last=0;self.surface_activation_deferred=0
 def update_surface_feature_progress(self):
  value=float(np.sqrt(self.error**2+np.sum((self.P@(self.q-self.qg))**2)))
  if self.surface_feature_goal is None or not np.array_equal(self.surface_feature_goal,self.qg):
   self.surface_feature_goal=self.qg.copy();self.surface_feature_best=value;self.surface_feature_last=self.k
  elif value<self.surface_feature_best-.0002:self.surface_feature_best=value;self.surface_feature_last=self.k
  return value
 def observe(self,*a,**kw):
  super().observe(*a,**kw);self.update_surface_feature_progress()
 def start_surface(self,c,s,arrays,raw):
  value=self.update_surface_feature_progress()
  if self.k-self.surface_feature_last<100:
   self.surface_activation_deferred+=1;return False
  self.emit('surface_activation_has_actual_feature_stall',feature_distance=value,best_feature_distance=self.surface_feature_best,cycles_since_actual_feature_record=self.k-self.surface_feature_last)
  return super().start_surface(c,s,arrays,raw)
 def summary(self):
  out=super().summary();out.update(t0_revision='surface_escape_after_dual_actual_stagnation',surface_activation_deferred_while_actual_progress=self.surface_activation_deferred,surface_activation_rule='both RMS terminal penetration and actual Cartesian-plus-two-feature error have no original 0.2mm record improvement for 100 cycles; reset actual feature record on terminal change');return out
