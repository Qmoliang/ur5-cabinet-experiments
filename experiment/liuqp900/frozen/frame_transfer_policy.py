"""Transfer the physical escape task when the binding constraint moves."""
import numpy as np
from surface_handoff_policy import LowDimPolicy as Previous
class LowDimPolicy(Previous):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);self.blocker_transfers=[];self.invalid_dual_updates=0
 def _current_binding_rows(self,s,arrays):
  if not str(s['status']).lower().startswith('solved') or s['y'] is None:
   self.invalid_dual_updates+=1;return []
  A,u=arrays[2],arrays[4];rows=[]
  for i,key in enumerate(s['row_ids']):
   if key[0] not in ('collision','frontier','self','workspace'):continue
   norm=float(np.linalg.norm(A[i]))
   if norm<=1e-10 or not np.isfinite(u[i]) or self.dt*u[i]>.0005:continue
   force=max(0.,float(s['y'][i]))*norm
   if force>1e-6:rows.append((force,i,tuple(key)))
  return sorted(rows,key=lambda x:(-x[0],x[1]))
 def surface_call(self,c,s,arrays):
  if not self.brake_pending and self.blocker_id is not None:
   geometry=self._geometry();candidates=self._current_binding_rows(s,arrays)
   if geometry is not None and candidates:
    old=tuple(self.blocker_id);force,index,key=candidates[0];old_force=max((v for v,i,k in candidates if k[1]==old[1] and k[0]==old[0]),default=0.)
    clear=geometry[3]>c.near_distance;stale=self.k-self.surface_feature_last>=100 and force>old_force+1e-6
    if key!=old and (clear or (stale and key[1:]!=old[1:])):
     self.blocker_id=key;self.surface_cache=None;new_geometry=self._geometry()
     if new_geometry is not None:
      # Keep pending/committed global feature memories and visit history.
      # Only replace the local physical frame and its directional reference.
      self.circulation_tangent=None;self.tangent_override=None;self.surface_reversed=False;self.reverse_used=False;self.surface_last_point=new_geometry[0].copy();self.surface_move_cycle=self.k;self.surface_last_travel=self.surface_travel;self.solve_cache={};self.command_epoch+=1;self.brake_pending=np.max(np.abs(self.previous))>1e-8
      event=dict(cycle=self.k,old_row=list(old),new_row=list(self.blocker_id),old_gap_m=float(geometry[3]),new_gap_m=float(new_geometry[3]),force=force,old_frame_force=old_force,reason='old_frame_clear' if clear else 'actual_stagnation_with_new_dominant_constraint',memory_count=len(self.memories));self.blocker_transfers.append(event);self.emit('physical_blocker_frame_transferred',**{k:v for k,v in event.items() if k!='cycle'});self.expected=None;self.last_delta=np.zeros(6);c.pilot_supplied_solution=None;return np.zeros(6)
     self.blocker_id=old;self.surface_cache=None
  return super().surface_call(c,s,arrays)
 def summary(self):
  out=super().summary();out.update(revision='current_binding_frame_transfer',actual_blocker_transfers=self.blocker_transfers,invalid_duals_ignored=self.invalid_dual_updates,memories_discarded_by_frame_transfer=0);return out
