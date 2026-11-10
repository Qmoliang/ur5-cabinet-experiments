"""Select only genuine travel/stopping-reserve limits, not infinite goal time."""
import numpy as np
from release_policy import LowDimPolicy as Previous
from unified_joint_braking import stopping_distance

class LowDimPolicy(Previous):
    def _select_release(self,s,nominal):
        v=nominal['velocity'];lower=self.q-self.low;upper=self.high-self.q
        resolution=self.acceleration*self.dt*self.dt;candidates=[]
        for j in range(6):
            closing_lower=max(-float(v[j]),0.);closing_upper=max(float(v[j]),0.)
            lower_budget=float(stopping_distance(closing_lower,self.dt,self.acceleration))+resolution
            upper_budget=float(stopping_distance(closing_upper,self.dt,self.acceleration))+resolution
            if lower[j]<=lower_budget and v[j]<=1e-6:candidates.append((float(lower[j]/lower_budget),j,1.,'actual_lower_stopping_reserve'))
            if upper[j]<=upper_budget and v[j]>=-1e-6:candidates.append((float(upper[j]/upper_budget),j,-1.,'actual_upper_stopping_reserve'))
        return min(candidates,key=lambda x:(x[0],x[1])) if candidates else None

    def emit(self,name,**kwargs):
        if name=='release_selected' and 'time_ratio' in kwargs:kwargs['reserve_to_stopping_budget_ratio']=kwargs.pop('time_ratio')
        return super().emit(name,**kwargs)

    def summary(self):
        out=super().summary();out['selection_revision']='actual geometric joint-travel/stopping budget only; no division by near-zero goal progress'
        return out
