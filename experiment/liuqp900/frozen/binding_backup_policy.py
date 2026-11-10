"""Keep the selected binding row; distinguish proxy transport from frame transfer."""
import numpy as np
from selected_backup_policy import LowDimPolicy as Previous
from goal_entry_policy_v2 import LowDimPolicy as ExactGeometry
from surface_escape_policy import LowDimPolicy as Surface
class LowDimPolicy(Previous):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);self.binding_row_updates=[];self.world_frame_transports=[]
 def _geometry(self):
  if self.blocker_id is None:return None
  return ExactGeometry._geometry(self)
 def _frame(self,key):
  kind,i,j=key;b=self.ctx.guard.bodies
  return ('relative',int(b[i]),int(b[j])) if kind=='self' else ('body',int(b[i]))
 def surface_call(self,c,s,arrays):
  if not self.brake_pending and self.blocker_id is not None:
   geometry=self._geometry();candidates=self._current_binding_rows(s,arrays)
   if geometry is not None and candidates:
    old=tuple(self.blocker_id);force,index,key=candidates[0];old_force=max((v for v,i,k in candidates if k==old),default=0.)
    clear=geometry[3]>c.near_distance;stale=self.k-self.surface_feature_last>=100 and force>old_force+1e-6
    if key!=old and (clear or stale):
     self.blocker_id=key;self.surface_cache=None;new_geometry=self._geometry();assert tuple(self.blocker_id)==key,'geometry evaluation changed the chosen binding row'
     if new_geometry is not None:
      frame_changed=self._frame(old)!=self._frame(key)
      self.surface_last_point=new_geometry[0].copy();self.solve_cache={}
      event=dict(cycle=self.k,old_row=list(old),new_row=list(key),old_gap_m=float(geometry[3]),new_gap_m=float(new_geometry[3]),force=force,old_row_force=old_force,frame_changed=frame_changed,memory_count=len(self.memories))
      if frame_changed:
       # Direction is an episode property; retry exhaustion belongs to the old task.
       reference=None if self.circulation_tangent is None else self.circulation_tangent.copy()
       normal=new_geometry[1];projected=None if reference is None else reference-normal*(normal@reference)
       norm=0. if projected is None else float(np.linalg.norm(projected))
       self.circulation_tangent=None if norm<=1e-8 else projected/norm
       if self.circulation_tangent is not None:
        record=dict(cycle=self.k,old_row=list(old),new_row=list(key),old_tangent=reference.tolist(),new_tangent=self.circulation_tangent.tolist(),new_normal=normal.tolist(),projection_norm=norm,retry_allowance_reset_for_new_task=True);self.world_frame_transports.append(record)
        self.emit('world_winding_kept_new_task_retry_reset',**{k:v for k,v in record.items() if k!='cycle'})
       self.tangent_override=None;self.surface_reversed=False;self.reverse_used=False;self.surface_move_cycle=self.k;self.surface_last_travel=self.surface_travel;self.command_epoch+=1;self.brake_pending=np.max(np.abs(self.previous))>1e-8;self.blocker_transfers.append(event);self.emit('physical_blocker_frame_transferred',**{k:v for k,v in event.items() if k!='cycle'});self.expected=None;self.last_delta=np.zeros(6);c.pilot_supplied_solution=None;return np.zeros(6)
      # Same rigid body, new contact row: parallel-transport the tangent via
      # the existing projection. Do not reset/brake the same physical task.
      self.binding_row_updates.append(event)
      if len(self.binding_row_updates)<=30 or self.k%100==0:self.emit('binding_surface_row_updated',**{k:v for k,v in event.items() if k!='cycle'})
     else:self.blocker_id=old;self.surface_cache=None
  return Surface.surface_call(self,c,s,arrays)
 def summary(self):
  out=super().summary();out.update(revision='binding_identity_and_feasible_backup',binding_surface_row_updates=self.binding_row_updates,binding_geometry_keeps_selected_row=True,same_frame_updates_preserve_tangent=True,world_frame_transports=self.world_frame_transports,world_tangent_transport=True,reversal_attempt_scope='per controlled robot task, unchanged from original',exit_gate_unchanged=True);return out
