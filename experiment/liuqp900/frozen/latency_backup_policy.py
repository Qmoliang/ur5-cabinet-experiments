"""Certify holding a published velocity through measured computation delay."""
from collections import OrderedDict
import numpy as np
from binding_backup_policy import LowDimPolicy as Previous
class LowDimPolicy(Previous):
 def __init__(self,*a,**kw):
  super().__init__(*a,**kw);self.hold_registry=OrderedDict();self.extended={};self.hold_proofs=0;self.hold_requests=[];self.row_reuses=0;self.row_cache=OrderedDict();original_rows=self.assembler.rows
  def cached_rows(q,previous,target):
   key=q.tobytes()+previous.tobytes()+np.asarray(target).tobytes()
   if key in self.row_cache:self.row_reuses+=1;self.row_cache.move_to_end(key);return self.row_cache[key]
   out=original_rows(q,previous,target);self.row_cache[key]=out
   # Two outstanding commands, at most the existing ten-step brake budget,
   # each with its deterministic braking states. This is only an exact cache.
   while len(self.row_cache)>2*10*11:self.row_cache.popitem(last=False)
   return out
  self.assembler.rows=cached_rows
 def _register(self,cert):
  step=cert['steps'][0];state=self._key(step['q'],step['previous']);items=self.hold_registry.setdefault(state,{})
  items[cert['id']]=(self.command_epoch,cert);self.hold_registry.move_to_end(state)
  while len(self.hold_registry)>2*10:self.hold_registry.popitem(last=False)
 def _extend_published(self,cert):
  horizon=min(10,max(1,int(getattr(self,'last_delay_ticks',1))));key=(cert['id'],self.command_epoch)
  self._register(cert)
  if self.extended.get(key,0)>=horizon:return
  saved=(self.cache_state,self.cert_cache,self.admitted,self.published_certificates);v=cert['steps'][0]['velocity'].copy();current=cert['steps'][0]['q'].copy();count=1
  try:
   for k in range(1,horizon):
    current=current+self.dt*v;future=self._certify(current,v,v)
    if future is None:break
    self._register(future);self.hold_proofs+=1;count+=1
  finally:self.cache_state,self.cert_cache,self.admitted,self.published_certificates=saved
  self.extended[key]=horizon
  if len(self.hold_requests)<100:self.hold_requests.append(dict(cycle=self.k,measured_previous_delay_ticks=int(getattr(self,'last_delay_ticks',1)),requested_ticks=horizon,certified_ticks=count))
 def _publish(self,c,answer):
  delta=super()._publish(c,answer)
  if self.expected is not None:
   cert=self.cert_cache.get((self.expected.tobytes(),self._memory_stamp()))
   if cert is not None:self._extend_published(cert)
  return delta
 def __call__(self,c,s,arrays):
  delta=super().__call__(c,s,arrays)
  if not self.escape_active:
   for epoch,cert in list(self.published_certificates.values()):
    if epoch==self.command_epoch:self._extend_published(cert)
  return delta
 def filter_viability(self,q,previous,command):
  state=self._key(q,previous);saved=(self.cache_state,self.cert_cache,self.published_certificates);all_certificates=dict(self.cert_cache) if self.cache_state==state else {};approved={}
  for ident,(epoch,cert) in self.hold_registry.get(state,{}).items():
   if epoch==self.command_epoch and cert['memory_signature']==self._memory_stamp():
    approved[ident]=(epoch,cert);all_certificates[(cert['steps'][0]['velocity'].tobytes(),cert['memory_signature'])]=cert
  self.cache_state=state;self.cert_cache=all_certificates;self.published_certificates=approved
  try:return super().filter_viability(q,previous,command)
  finally:self.cache_state,self.cert_cache,self.published_certificates=saved
 def summary(self):
  out=super().summary();out.update(revision='binding_transfer_and_latency_covered_backup',holding_certificates=self.hold_proofs,holding_proof_windows=self.hold_requests,identical_original_row_reuses=self.row_reuses,hold_rule='previous measured controller delay, capped by the original 10-step certification budget; every held prefix has its own original-row-feasible braking tail; unexpected additional delay falls back to certified braking',future_optimized_states=0);return out
