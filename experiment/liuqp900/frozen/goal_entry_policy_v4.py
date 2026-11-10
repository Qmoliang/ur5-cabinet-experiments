"""Align self-distance proposal rows with the unchanged execution certificate."""
import numpy as np
from goal_entry_policy_v3 import LowDimPolicy as Previous
from piecewise_reserve import solve_reserve
from certified_auxiliary import verify_solution

# Existing ExecutionGuard.self_segment_bound subtracts 2e-9 metres.
# Existing proposal row acceptance permits 1e-7 in velocity-row units.
SELF_CERTIFICATE_ROUNDOFF_M=2e-9
ROW_ACCEPTANCE_TOLERANCE=1e-7

class LowDimPolicy(Previous):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.self_certificate_rows=0;self.direct_goal_certificates=0
    def _extras(self,c,s):
        rows,bounds=super()._extras(c,s);H,g,A,l,u=c.last_qp_arrays
        indices=[i for i,row in enumerate(s['row_ids']) if row[0]=='self' and u[i]*self.dt<c.near_distance and np.linalg.norm(A[i])>1e-12]
        if indices:
            inward=(SELF_CERTIFICATE_ROUNDOFF_M+self.dt*ROW_ACCEPTANCE_TOLERANCE)/self.dt
            rows=np.vstack([rows,A[indices]]);bounds=np.r_[bounds,u[indices]-inward];self.self_certificate_rows+=len(indices)
        return rows,bounds
    def _solve(self,c,s,arrays,release=None):
        cache_key=(id(s),self.mode,len(self.memories))
        if release is None and cache_key in self.solve_cache:return self.solve_cache[cache_key]
        H,g,A,l,u=arrays;extraH,extrag=self._terms(c,s);rows,bounds=self._extras(c,s)
        AA=np.vstack([A,rows]);ll=np.r_[l,np.full(len(rows),-np.inf)];uu=np.r_[u,bounds];self.mode_qp_calls+=1
        answer=solve_reserve(H+extraH,g+extrag,AA,ll,uu,self.q,self.previous,self.low,self.high,c.task_weight,self.error,self.dt,self.acceleration,release=release,joint_metric=True)
        if answer is None:self.augmented_failures+=1;return None
        raw=answer['raw'];v=answer['velocity'];delta=answer['delta']+extraH@v+extrag
        if len(rows):delta+=rows.T@raw.y[len(A):len(AA)]
        answer.update(delta=delta,original_dual=raw.y[:len(A)].copy())
        try:verify_solution(H,g+delta,A,l,u,answer)
        except RuntimeError:self.certificate_rejections+=1;return None
        if (len(rows) and np.max(rows@v-bounds)>ROW_ACCEPTANCE_TOLERANCE) or (release is not None and release[0]@v<release[1]-ROW_ACCEPTANCE_TOLERANCE):self.extra_infeasible+=1;return None
        self.direct_goal_certificates+=1
        if release is None:self.solve_cache[cache_key]=answer
        return answer
    def summary(self):
        out=super().summary();out.update(self_certificate_roundoff_m=SELF_CERTIFICATE_ROUNDOFF_M,self_qp_inward_reserve_m=SELF_CERTIFICATE_ROUNDOFF_M+self.dt*ROW_ACCEPTANCE_TOLERANCE,self_certificate_rows=self.self_certificate_rows,direct_goal_certificates=self.direct_goal_certificates,execution_solution='same six-variable piecewise minimizer, independently verified against effective original LiuQP before admission')
        return out
