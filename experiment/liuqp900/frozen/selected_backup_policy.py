"""A feasible counterfactual is not an authorized task command."""
import numpy as np
from feasible_backup_policy import LowDimPolicy as Previous
from frame_transfer_policy import LowDimPolicy as Frame
class LowDimPolicy(Previous):
 def __init__(self,*a,**kw):
  self.published_certificates={};super().__init__(*a,**kw)
 def _certify(self,q,previous,v):
  if self.cache_state!=self._key(q,previous):self.published_certificates={}
  return super()._certify(q,previous,v)
 def _publish(self,c,answer):
  delta=super()._publish(c,answer)
  if self.expected is not None:
   cert=self.cert_cache.get((self.expected.tobytes(),self._memory_stamp()))
   if cert is not None:self.published_certificates[cert['id']]=(self.command_epoch,cert)
  return delta
 def __call__(self,c,s,arrays):
  delta=Frame.__call__(self,c,s,arrays)
  if not self.escape_active and not self.brake_pending and s['x'] is not None:
   nominal=np.clip(np.asarray(s['x']),*c.joint_velocity_bounds())
   cert=self._certify(self.q,self.previous,nominal)
   if cert is not None:self.published_certificates[cert['id']]=(self.command_epoch,cert)
  return delta
 def filter_viability(self,q,previous,command):
  all_certificates=self.cert_cache
  if self.backup is not None and self.backup['memory_signature']!=self._memory_stamp() and self.cache_state==self._key(q,previous):
   v=self.backup['steps'][0]['velocity'];refreshed=all_certificates.get((v.tobytes(),self._memory_stamp()))
   if refreshed is not None:self.backup=refreshed
  self.cert_cache={(cert['steps'][0]['velocity'].tobytes(),cert['memory_signature']):cert for epoch,cert in self.published_certificates.values() if epoch==self.command_epoch}
  try:return super().filter_viability(q,previous,command)
  finally:self.cert_cache=all_certificates
 def summary(self):
  out=super().summary();out.update(revision='selected_command_and_feasible_backup',execution_authorization='only actually published current-epoch commands; unselected feasible counterfactuals never authorize execution');return out
