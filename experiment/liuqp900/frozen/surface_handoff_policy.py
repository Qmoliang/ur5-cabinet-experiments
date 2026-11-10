"""Reset actual-progress evidence at the controller handoff, not before it."""
from surface_stall_policy import LowDimPolicy as Previous
class LowDimPolicy(Previous):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);self.surface_feature_active=False;self.surface_handoff_resets=0
 def update_surface_feature_progress(self):
  active=bool(self.escape_active)
  if active!=self.surface_feature_active:
   self.surface_feature_goal=None;self.surface_feature_active=active;self.surface_handoff_resets+=1
   self.emit('surface_feature_record_reset_at_actual_handoff',escape_active=active)
  return super().update_surface_feature_progress()
 def summary(self):
  out=super().summary();out.update(t0_revision='surface_escape_with_actual_handoff_progress',surface_handoff_progress_resets=self.surface_handoff_resets,pre_escape_feature_progress_used_for_surface_decisions=False);return out
