"""Bounded continuous certificate for the fixed terminal feedback; counterexamples refine witnesses."""
import time,numpy as np
from continuation_maximin_policy import LowDimPolicy as Previous
from goal_entry_policy import clipped
class LowDimPolicy(Previous):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.continuation_intervals=0;self.continuation_refinements=0;self.certificate_timeouts=0
 def _terms(self,c,s):
  if self.mode!='CERT_ENTER':return super()._terms(c,s)
  # The certified feedback must actually progress along its certified ray,
  # even when its initial Cartesian distance to the target increases.
  weight=self.posture_lever**2;velocity=clipped(c.task_gain*(self.qg-self.q),.5)
  return c.task_weight*(weight*np.eye(6)-s['J'].T@s['J']),c.task_weight*(s['J'].T@s['desired']-weight*velocity)
 def continuation_certificate(self):
  self.entry_calls+=1;started=time.perf_counter();g=self.future_guard;direction=self.qg-self.q;count=max(1,int(np.ceil(np.max(np.abs(direction))/.04)));seen=[];original=g.positions
  def positions(q):
   a=float((q-self.q)@direction/max(direction@direction,1e-30))
   if 1e-9<a<1-1e-9:seen.append(a)
   return original(q)
  def deadline():
   if time.perf_counter()-started>14.:raise TimeoutError('terminal certificate budget')
  g.positions=positions;g.deadline_check=deadline;physical=True;memory=True;timed_out=False
  try:
   for i in range(count):
    a=self.q+direction*(i/count);b=self.q+direction*((i+1)/count);seen=[];self.continuation_intervals+=1
    if not g.segment(self.ctx.proxy_guard,a,b):physical=False;break
    if not self._memory_segment(a,b,self.memories):memory=False;break
  except TimeoutError:physical=False;timed_out=True;self.certificate_timeouts+=1
  finally:g.positions=original;g.deadline_check=None;self.entry_certificate_ms.append((time.perf_counter()-started)*1000)
  if physical and memory:self.entry_passes+=1;return True
  key='certificate_timeout' if timed_out else 'continuous_physical' if not physical else 'no_return_memory';self.gate_rejections[key]=self.gate_rejections.get(key,0)+1
  if not physical and not timed_out:
   before=len(self.sample_fractions);self.sample_fractions=np.unique(np.r_[self.sample_fractions,seen]);self.continuation_refinements+=len(self.sample_fractions)-before;self.witness_anchor=None
  if sum(self.gate_rejections.values())<=30:self.emit('terminal_certificate_rejected',reason=key,fractions=len(self.sample_fractions),intervals=count,elapsed_s=time.perf_counter()-started)
  return False
 def summary(self):
  out=super().summary();out.update(continuation_validation_intervals=self.continuation_intervals,continuation_counterexample_fractions_added=self.continuation_refinements,continuation_certificate_timeouts=self.certificate_timeouts,entry_execution='positive scalar contraction under original current-velocity QP constraints; terminal objective matches the certified feedback',certificate_partition='original 0.04rad resolution, each interval retains original curved continuous guard; failing-interval samples only refine this same scalar feedback family')
  return out
