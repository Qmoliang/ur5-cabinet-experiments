"""No improvement of executed best distance detects oscillation as well as rest."""
import numpy as np
from blocker_policy_v4 import LowDimPolicy as Previous

class LowDimPolicy(Previous):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.progress_mode=None;self.best_executed_error=float('inf');self.last_improvement_cycle=0
    def observe(self,k,q,previous):
        super().observe(k,q,previous)
        if self.mode!=self.progress_mode:
            self.progress_mode=self.mode;self.best_executed_error=self.error;self.last_improvement_cycle=k
        elif self.error<self.best_executed_error-.0002:
            self.best_executed_error=self.error;self.last_improvement_cycle=k
    def _select_blocker(self,s,arrays,nominal):
        if self.k-self.last_improvement_cycle<100 or self.error<.002:return None
        H,g,A,l,u=arrays;candidates=[]
        for i,row in enumerate(s['row_ids']):
            if row[0] not in ('collision','frontier','self','workspace'):continue
            norm=np.linalg.norm(A[i])
            if norm<1e-10 or not np.isfinite(u[i]) or self.dt*u[i]>.0005:continue
            force=max(0.,float(s['y'][i]))*norm;key=tuple(row);prior=self.blocker_best.get(key)
            if prior is not None and .5*self.error**2>=prior-1e-8:continue
            if force>1e-6:candidates.append((force,i,key))
        return max(candidates,key=lambda x:x[0]) if candidates else None
    def summary(self):
        out=super().summary();out.update(stall_evidence='no 0.2 mm improvement in actual best goal error for 100 cycles; oscillation is not progress',best_executed_error_m=self.best_executed_error,last_improvement_cycle=self.last_improvement_cycle)
        return out
