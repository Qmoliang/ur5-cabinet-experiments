"""Sharpen the existing no-return certificate without changing its tolerance."""
import numpy as np
from goal_plane_acceptance_policy import LowDimPolicy as Previous
class LowDimPolicy(Previous):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.memory_bound_refinements=0;self.memory_bound_examples=[]
 def _memory_segment(self,a,b,memories,depth=0):
  fa=self.feature(a);fb=self.feature(b)
  for m in memories:
   n=np.asarray(m['normal']);va=float(n@fa-m['offset']);vb=float(n@fb-m['offset']);curve=np.linalg.norm(n[:3])*self.site_curvature*np.sum(np.abs(b-a))**2/8
   if min(va,vb)-curve>=-1e-10:continue
   # Same second-derivative bound, but retain the endpoint-dependent chord.
   B=np.longdouble(4)*np.longdouble(curve);slope=np.longdouble(vb)-np.longdouble(va)
   tau=np.clip((B-slope)/max(2*B,np.longdouble('1e-60')),0,1)
   lower=np.longdouble(va)+tau*(slope-B)+B*tau*tau
   if np.nextafter(lower,np.longdouble('-inf'))>=np.longdouble('-1e-10'):
    self.memory_bound_refinements+=1
    if len(self.memory_bound_examples)<20:self.memory_bound_examples.append(dict(start_value=va,end_value=vb,old_lower=min(va,vb)-curve,new_lower=float(lower),tau=float(tau),depth=depth))
    continue
   if min(va,vb)<-1e-9 or depth>=8:return False
   mid=(a+b)/2
   return self._memory_segment(a,mid,memories,depth+1) and self._memory_segment(mid,b,memories,depth+1)
  return True
 def summary(self):
  out=super().summary();out.update(memory_certificate_kind='same tolerance, same model curvature bound, endpoint-consistent parabolic lower bound; original accepted bounds retained',memory_certificate_tolerance=-1e-10,memory_certificate_refinements=self.memory_bound_refinements,memory_certificate_examples=self.memory_bound_examples,no_return_constraints_discarded=0)
  return out
