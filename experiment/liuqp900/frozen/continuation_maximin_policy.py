"""Maximize the weakest predicted continuation clearance, not average deficit."""
import numpy as np
from scipy.optimize import linprog
from continuation_cached_policy import LowDimPolicy as Previous
from goal_entry_policy_activated import LowDimPolicy as OriginalSolver
class LowDimPolicy(Previous):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.continuation_rows=np.empty((0,6));self.continuation_bounds=np.empty(0);self.maximin_calls=0;self.maximin_no_gain=0;self.maximin_failures=0;self.maximin_diagnostics=[]
 def _extras(self,c,s):
  rows,bounds=super()._extras(c,s)
  if self.mode=='CERT_PREP' and len(self.continuation_rows):rows=np.vstack([rows,self.continuation_rows]);bounds=np.r_[bounds,self.continuation_bounds]
  return rows,bounds
 def _solve(self,c,s,arrays,release=None):
  if self.mode!='CERT_PREP' or release is not None or not self.future_state['rows']:return OriginalSolver._solve(self,c,s,arrays,release)
  self.continuation_rows=np.empty((0,6));self.continuation_bounds=np.empty(0)
  H,g,A,l,u=arrays;extra,eb=self._extras(c,s);up=np.isfinite(u);lo=np.isfinite(l)
  physical=np.vstack([A[up],-A[lo],extra]);physical_u=np.r_[u[up],-l[lo],eb]
  B=np.asarray([r['b'] for r in self.future_state['rows']]);h=np.array([r['slack'] for r in self.future_state['rows']]);minimum=float(h.min())
  AA=np.vstack([np.c_[physical,np.zeros(len(physical))],np.c_[-self.dt*B,np.ones(len(B))]])
  ub=np.r_[physical_u,h];objective=np.r_[np.zeros(6),-1.]
  solution=linprog(objective,A_ub=AA,b_ub=ub,bounds=[(None,None)]*6+[(None,.0002)],method='highs');self.maximin_calls+=1
  if not solution.success:self.maximin_failures+=1;return OriginalSolver._solve(self,c,s,arrays)
  eta=float(solution.x[-1]);improvement=eta-minimum
  if improvement<=2e-9:
   self.maximin_no_gain+=1
   if len(self.maximin_diagnostics)<30:self.maximin_diagnostics.append(dict(cycle=self.k,minimum=minimum,eta=eta,reason='no_positive_weakest_clearance_support'))
   return OriginalSolver._solve(self,c,s,arrays)
  # Half of feasible weakest-clearance improvement; original support fraction.
  for fraction in [.5,.25,.125,.0625]:
   wanted=minimum+fraction*improvement
   self.continuation_rows=-B;self.continuation_bounds=(h-wanted)/self.dt;self.solve_cache={}
   answer=OriginalSolver._solve(self,c,s,arrays)
   if answer is not None:
    ad=self._admit(c,arrays,answer)
    if ad is not None and min(h+self.dt*B@ad[0])>minimum+2e-10:
     if self.k%100==0:self.emit('weakest_continuation_clearance_improved',minimum_m=minimum,optimal_predicted_m=eta,admitted_predicted_m=float(min(h+self.dt*B@ad[0])),safe_scale=ad[1])
     return answer
  self.continuation_rows=np.empty((0,6));self.continuation_bounds=np.empty(0);self.solve_cache={};self.maximin_failures+=1
  return OriginalSolver._solve(self,c,s,arrays)
 def summary(self):
  out=super().summary();out.update(maximin_calls=self.maximin_calls,maximin_no_gain=self.maximin_no_gain,maximin_failures=self.maximin_failures,maximin_diagnostics=self.maximin_diagnostics,continuation_objective='maximize weakest one-step predicted clearance along the fixed terminal feedback, then realize half feasible gain through original LiuQP')
  return out
