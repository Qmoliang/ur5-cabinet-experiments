"""Solve the effective original six-variable objective before admission.

An augmented solver's approximate minimizer is never compared after execution.
Admission previews the actual original-QP solution that will be executed.
"""
import numpy as np
from viability_policy import LowDimPolicy as Previous
from verified_qp import solve_verified

class LowDimPolicy(Previous):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.polish_count=0;self.polish_max_difference=0.;self.polish_release_rejections=0
    def _solve(self,c,s,arrays,release=None):
        answer=super()._solve(c,s,arrays,release=release)
        if answer is None:return None
        H,g,A,l,u=arrays;result=solve_verified(H,g+answer['delta'],A,l,u,hint=answer['velocity'])
        if result is None:return None
        self.polish_count+=1;self.polish_max_difference=max(self.polish_max_difference,float(np.max(np.abs(result.x-answer['velocity']))))
        if release is not None and release[0]@result.x<release[1]-1e-7:
            self.polish_release_rejections+=1;return None
        answer.update(velocity=result.x.copy(),original_dual=result.y.copy(),iterations=result.info.iter)
        return answer
    def summary(self):
        out=super().summary();out.update(polish_count=self.polish_count,polish_max_velocity_difference=self.polish_max_difference,polish_release_rejections=self.polish_release_rejections,execution_solution='independently solved original 6-variable QP with proposed gradient; this same velocity is admitted and executed')
        return out
