from release_policy_v2 import LowDimPolicy as Previous

class LowDimPolicy(Previous):
    def _solve(self,*args,**kwargs):
        answer=super()._solve(*args,**kwargs)
        if answer is not None:
            arrays=args[2];m=len(arrays[2])
            if 'raw' in answer:
                answer['original_dual']=answer['raw'].y[:m].copy();answer['iterations']=answer['raw'].info.iter
            elif self.metric_enabled is False and kwargs.get('release') is None:
                answer['original_dual']=args[1]['y'].copy();answer['iterations']=0
        return answer

    def _preview(self,c,arrays,answer,coordinate=None):
        accepted=super()._preview(c,arrays,answer,coordinate)
        if accepted is not None and 'original_dual' in answer:
            c.pilot_supplied_solution=dict(velocity=answer['velocity'].copy(),original_dual=answer['original_dual'].copy(),iterations=answer.get('iterations',0))
        return accepted

    def summary(self):
        out=super().summary();out['execution_solution']='auxiliary minimizer accepted by original six-variable primal/dual/KKT checks, avoiding a redundant approximate re-solve'
        return out
