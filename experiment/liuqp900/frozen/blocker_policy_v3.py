"""Verify numerical optimality before publishing any physical candidate."""
from blocker_policy_v2 import LowDimPolicy as Previous
from certified_auxiliary import verify_solution
from verified_qp import _solve_slsqp

class LowDimPolicy(Previous):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.certificate_refinements=0;self.certificate_rejections=0;self.augmented_failures=0
    def _solve(self,c,s,arrays,release=None):
        answer=super()._solve(c,s,arrays,release=release)
        if answer is None:self.augmented_failures+=1;return None
        H,g,A,l,u=arrays;gradient=g+answer['delta']
        try:verify_solution(H,gradient,A,l,u,answer);return answer
        except RuntimeError:pass
        self.certificate_refinements+=1
        result=_solve_slsqp(H,gradient,A,l,u,hint=answer['velocity'])
        if result is None:self.certificate_rejections+=1;return None
        answer.update(velocity=result.x.copy(),original_dual=result.y.copy(),iterations=result.info.iter)
        if release is not None and release[0]@result.x<release[1]-1e-7:self.certificate_rejections+=1;return None
        try:verify_solution(H,gradient,A,l,u,answer);return answer
        except RuntimeError:self.certificate_rejections+=1;return None
    def summary(self):
        out=super().summary();out.update(certificate_refinements=self.certificate_refinements,certificate_rejections=self.certificate_rejections,augmented_failures=self.augmented_failures)
        return out
