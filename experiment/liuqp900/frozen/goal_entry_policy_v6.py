"""Terminal posture in the current-task nullspace; no soft reserve barrier."""
import numpy as np
from goal_entry_policy_v5 import LowDimPolicy as Previous
from goal_entry_policy import clipped
from verified_qp import solve_verified
from certified_auxiliary import verify_solution

class LowDimPolicy(Previous):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.posture_lever=float(np.mean(np.linalg.norm(self.P,axis=1)));self.full_posture_best=float('inf');self.full_posture_last_progress=0;self.full_posture_mode=None
    def observe(self,k,q,previous):
        super().observe(k,q,previous)
        distance=float(np.sqrt(self.error**2+self.posture_lever**2*np.sum((q-self.qg)**2)))
        if self.mode!=self.full_posture_mode:
            self.full_posture_mode=self.mode;self.full_posture_best=distance;self.full_posture_last_progress=k
        elif distance<self.full_posture_best-.0002:
            self.full_posture_best=distance;self.full_posture_last_progress=k
        self.feature_last_progress=self.full_posture_last_progress
    def _terms(self,c,s):
        if self.error<=.001:return np.zeros((6,6)),np.zeros(6)
        J=s['J'];K=J;desired=self._entry_velocity(c,s)
        if self.memory_enabled and self.mode=='BYPASS' and self.blocker_id is not None:
            geometry=self._geometry()
            if geometry is not None:
                point,n,Jpoint,gap=geometry
                if gap>c.near_distance:
                    K=Jpoint;desired=clipped(c.task_gain*(self._goal_point_for_blocker()-point),c.max_task_speed)
                else:
                    tangent=self._tangent(geometry);K=np.vstack([n@Jpoint,tangent@Jpoint])
                    kind=self.blocker_id[0];margin=c.safety_margin if kind in ('collision','frontier') else self.ctx.guard.self_margin+4e-9 if kind=='self' else 0.
                    desired=clipped(np.array([max(0.,c.task_gain*(margin-gap)),c.max_task_speed]),c.max_task_speed)
        inverse=np.linalg.pinv(K,rcond=1e-8);N=np.eye(6)-inverse@K
        Hextra=c.task_weight*(K.T@K-J.T@J);gextra=c.task_weight*(J.T@s['desired']-K.T@desired)
        if self.entry_enabled:
            posture=self.posture_lever*N;preparation=clipped(-c.task_gain*posture@(self.q-self.qg),c.max_task_speed)
            Hextra+=c.task_weight*posture.T@posture;gextra-=c.task_weight*posture.T@preparation
        return Hextra,gextra
    def _solve(self,c,s,arrays,release=None):
        key=(id(s),self.mode,len(self.memories))
        if release is None and key in self.solve_cache:return self.solve_cache[key]
        H,g,A,l,u=arrays;HH,gg=self._terms(c,s);rows,bounds=self._extras(c,s)
        AA=np.vstack([A,rows]);ll=np.r_[l,np.full(len(rows),-np.inf)];uu=np.r_[u,bounds]
        if release is not None:
            b,rate=release;AA=np.vstack([AA,-b]);ll=np.r_[ll,-np.inf];uu=np.r_[uu,-rate]
        result=solve_verified(H+HH,g+gg,AA,ll,uu);self.mode_qp_calls+=1
        if result is None:self.augmented_failures+=1;return None
        v=result.x;delta=HH@v+gg
        if len(rows):delta+=rows.T@result.y[len(A):len(A)+len(rows)]
        alpha=0. if release is None else float(result.y[-1])
        if release is not None:delta-=alpha*np.asarray(release[0])
        answer=dict(velocity=v,delta=delta,original_dual=result.y[:len(A)].copy(),iterations=result.info.iter,release_dual=alpha,consumption_cost=0.,original_cost=float(.5*v@H@v+g@v),raw=result)
        try:verify_solution(H,g+delta,A,l,u,answer)
        except RuntimeError:self.certificate_rejections+=1;return None
        if (len(rows) and np.max(rows@v-bounds)>1e-7) or (release is not None and release[0]@v<release[1]-1e-7):self.extra_infeasible+=1;return None
        self.direct_goal_certificates+=1
        if release is None:self.solve_cache[key]=answer
        return answer
    def summary(self):
        out=super().summary();out.update(soft_joint_reserve_penalty=False,original_joint_braking_constraints_retained=True,posture_preparation='full terminal configuration projected into current-task nullspace, never full joint-waypoint tracking',posture_feature_dimension=None,memory_posture_feature_dimension=2,posture_task_nullity='at most 3 for Cartesian tracking and 4 for two-component surface tracking',posture_scale_m=self.posture_lever,optimized_future_configurations=0)
        return out
