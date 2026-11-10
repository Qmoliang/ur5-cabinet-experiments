"""Reserve-aware LiuQP plus targeted release of a currently depleted joint.

Only current q/velocity, final goal, original QP and declared limits are used.
No RRT trajectory, goal posture, waypoint or path graph enters this policy.
"""
from collections import Counter
from pathlib import Path
import os,json
import numpy as np
from scipy.optimize import linprog
from reserve_objective import solve_reserve,reserves

PARAMETERS=dict(version='reserve_release_v1',support_fraction=.5,release_backoffs=12,
    numerical_progress_factor=20,goal_rate_threshold=1e-4,release_credit_fraction=.05,
    risk_horizon_rule='time to joint boundary below estimated time to goal',
    denominator_floor='acceleration * dt^2',reserve_cost_scale='original task_weight * max(goal_error, goal_tolerance)^2',
    admission='original joint clipping, original row check, original guard scales',
    virtual_task_speed_cap=False,scene_specific_joint_or_direction=False)

class LowDimPolicy:
    def __init__(self,ctx,job,event):
        self.ctx=ctx;self.job=job;self.event=event;self.mode='RESERVE_GOAL';self.terminal=None
        self.brake_pending=False;self.command_epoch=0;self.expected=None;self.last_delta=np.zeros(6)
        self.low=ctx.model.jnt_range[:6,0]+ctx.controller.joint_padding
        self.high=ctx.model.jnt_range[:6,1]-ctx.controller.joint_padding
        self.dt=ctx.controller.dt;self.acceleration=ctx.controller.acceleration_limit
        self.metric_enabled=job.get('method')!='release_only';self.release_enabled=job.get('method')!='metric'
        self.release_joint=None;self.release_sign=None;self.hitV=None;self.hit_reserve=None
        self.history=[];self.events=[];self.mode_counts=Counter();self.rejections=Counter();self.selector_counts=Counter()
        self.auxiliary_calls=0;self.lp_calls=0;self.preview_calls=0;self.scaled_admissions=0;self.clipped_admissions=0
        self.peak_error=0.;self.release_start_error=None;self.actual_release_progress=0.;self.last_resource=None

    def emit(self,name,**kwargs):
        item=dict(event=name,cycle=int(getattr(self.ctx.controller,'pilot_cycle',0)),mode=self.mode,**kwargs)
        self.events.append(item);self.event(**item)

    def switch(self,mode,reason):
        if mode!=self.mode:
            self.mode=mode;self.command_epoch+=1
            if np.max(np.abs(self.previous))>1e-8:self.brake_pending=True
            self.emit('mode_changed',reason=reason,epoch=self.command_epoch)

    def observe(self,k,q,previous):
        self.k=k;self.q=np.asarray(q).copy();self.previous=np.asarray(previous).copy()
        self.ee=self.ctx.data.site_xpos[self.ctx.model.site('attachment_site').id].copy()
        self.error=float(np.linalg.norm(self.ee-self.ctx.target));self.peak_error=max(self.peak_error,self.error)
        self.mode_counts[self.mode]+=1
        if self.brake_pending and np.max(np.abs(previous))<1e-8:self.brake_pending=False
        if self.release_joint is not None:
            j=self.release_joint;resource=self.q[j]-self.low[j] if self.release_sign>0 else self.high[j]-self.q[j]
            if self.last_resource is not None:self.actual_release_progress+=resource-self.last_resource
            self.last_resource=float(resource)
        self.history.append(dict(cycle=k,error_m=self.error,q=self.q.copy(),previous=self.previous.copy(),mode=self.mode,release_joint=-1 if self.release_joint is None else self.release_joint,release_sign=0 if self.release_sign is None else self.release_sign))
        if self.error<=1e-3 and self.mode=='RELEASE':self.switch('RESERVE_GOAL','goal_tolerance_reached');self.release_joint=None

    def feedback(self,k,q,velocity,modified):
        if modified:self.rejections['runtime_guard_modified']+=1

    def _solve(self,c,s,arrays,release=None):
        H,g,A,l,u=arrays
        if not self.metric_enabled and release is None:
            return dict(velocity=s['x'].copy(),delta=np.zeros(6),release_dual=0.,consumption_cost=0.,original_cost=float(.5*s['x']@H@s['x']+g@s['x']))
        if self.metric_enabled:
            self.auxiliary_calls+=1
            return solve_reserve(H,g,A,l,u,self.q,self.previous,self.low,self.high,c.task_weight,self.error,self.dt,self.acceleration,release=release)
        from verified_qp import solve_verified
        b,rate=release;AA=np.vstack([A,-b]);ll=np.r_[l,-np.inf];uu=np.r_[u,-rate];self.auxiliary_calls+=1
        result=solve_verified(H,g,AA,ll,uu)
        if result is None:return None
        alpha=float(result.y[-1]);return dict(velocity=result.x,delta=-alpha*b,release_dual=alpha,consumption_cost=0.,original_cost=float(.5*result.x@H@result.x+g@result.x),raw=result)

    def _preview(self,c,arrays,answer,coordinate=None):
        raw=np.asarray(answer['velocity']);lo,hi=c.joint_velocity_bounds();bounded=np.clip(raw,lo,hi)
        H,g,A,l,u=arrays
        if max(np.max(l-A@bounded),np.max(A@bounded-u),0)>1e-7:
            self.rejections['postclip_original_rows']+=1;return None
        brake=np.clip(self.previous-np.clip(self.previous,-self.acceleration*self.dt,self.acceleration*self.dt),lo,hi)
        for alpha in [1.,.75,.5,.25,.125,.0625,0.]:
            trial=bounded if alpha==1. else brake+alpha*(bounded-brake);self.preview_calls+=1
            if self.ctx.guard.braking_safe(self.ctx.proxy_guard,self.q,trial):
                if coordinate is not None and coordinate@trial<=20e-7:
                    self.rejections['safe_but_no_release_progress']+=1;return None
                self.clipped_admissions+=int(np.any(raw!=bounded));self.scaled_admissions+=int(alpha<1)
                self.expected=raw.copy();self.last_delta=answer['delta'].copy()
                return trial,alpha
        self.rejections['no_certified_command']+=1;return None

    def _select_release(self,s,nominal):
        v=nominal['velocity'];J=s['J'];direction=(self.ctx.target-self.ee)/max(self.error,1e-30)
        rate=float(direction@J@v);goal_time=self.error/max(rate,1e-7)
        resolution=self.acceleration*self.dt*self.dt
        lower=self.q-self.low;upper=self.high-self.q
        # A geometric joint-bound diagnosis, not a velocity/acceleration row
        # interpreted as an obstacle normal.
        candidates=[]
        for j in range(6):
            if lower[j]<=resolution and v[j]<=1e-6:candidates.append((0.,j,1.,'lower_travel_exhausted'))
            if upper[j]<=resolution and v[j]>=-1e-6:candidates.append((0.,j,-1.,'upper_travel_exhausted'))
            if v[j]<-1e-6:
                ratio=(lower[j]/(-v[j]))/goal_time
                if ratio<1:candidates.append((ratio,j,1.,'lower_time_budget'))
            if v[j]>1e-6:
                ratio=(upper[j]/v[j])/goal_time
                if ratio<1:candidates.append((ratio,j,-1.,'upper_time_budget'))
        if not candidates:return None
        return min(candidates,key=lambda x:(x[0],x[1]))

    def __call__(self,c,s,arrays):
        self.expected=None;self.last_delta=np.zeros(6)
        if self.terminal or self.brake_pending:return np.zeros(6)
        nominal=self._solve(c,s,arrays)
        if nominal is None:self.rejections['reserve_qp_failed']+=1;return np.zeros(6)
        direction=(self.ctx.target-self.ee)/max(self.error,1e-30);goal_rate=float(direction@s['J']@nominal['velocity'])
        if self.mode=='RESERVE_GOAL' and self.release_enabled and self.error>.001:
            selected=self._select_release(s,nominal)
            if selected is not None:
                ratio,j,sign,reason=selected
                self.release_joint=j;self.release_sign=sign;self.hitV=.5*self.error**2
                self.hit_reserve=float(self.q[j]-self.low[j] if sign>0 else self.high[j]-self.q[j]);self.last_resource=self.hit_reserve
                self.release_start_error=self.error;self.selector_counts[reason]+=1
                self.switch('RELEASE',reason);self.emit('release_selected',joint=j,sign=sign,reason=reason,time_ratio=ratio,reserve_rad=self.hit_reserve,error_m=self.error)
                if self.brake_pending:return np.zeros(6)
        if self.mode=='RELEASE':
            j=self.release_joint;sign=self.release_sign;b=np.eye(6)[j]*sign
            credit=.05*max(self.hitV-.5e-6,0.)
            if .5*self.error**2<self.hitV-credit and b@nominal['velocity']>=-1e-6 and goal_rate>1e-5:
                self.emit('release_recovered',joint=j,reserve_rad=float(self.q[j]-self.low[j] if sign>0 else self.high[j]-self.q[j]),error_m=self.error)
                self.switch('RESERVE_GOAL','task_progress_without_reconsuming_reserve');self.release_joint=None
                if self.brake_pending:return np.zeros(6)
            else:
                H,g,A,l,u=arrays;up=np.isfinite(u);lo=np.isfinite(l);self.lp_calls+=1
                support=linprog(-b,A_ub=np.vstack([A[up],-A[lo]]),b_ub=np.r_[u[up],-l[lo]],bounds=[(None,None)]*6,method='highs')
                if support.success and -support.fun>40e-7:
                    requested=.5*float(-support.fun)
                    for _ in range(PARAMETERS['release_backoffs']):
                        if requested<=20e-7:break
                        answer=self._solve(c,s,arrays,release=(b,requested))
                        if answer is not None:
                            accepted=self._preview(c,arrays,answer,b)
                            if accepted is not None:
                                if self.k%50==0:self.emit('release_progress_proposal',joint=j,requested=requested,certified_rate=float(b@accepted[0]),safe_scale=accepted[1],dual=answer['release_dual'])
                                return answer['delta']
                        requested*=.5
                self.rejections['release_unavailable']+=1
                # Never turn an unavailable release into a path/IK fallback.
                self.brake_pending=np.max(np.abs(self.previous))>1e-8
                return np.zeros(6)
        accepted=self._preview(c,arrays,nominal)
        if accepted is not None:return nominal['delta']
        self.brake_pending=np.max(np.abs(self.previous))>1e-8
        return np.zeros(6)

    def summary(self):
        folder=Path(os.environ['UNIFIED_CASE_DIR'])
        np.savez_compressed(folder/'resource_history.npz',cycle=np.asarray([x['cycle'] for x in self.history]),q=np.asarray([x['q'] for x in self.history]),previous=np.asarray([x['previous'] for x in self.history]),goal_error=np.asarray([x['error_m'] for x in self.history]),release_joint=np.asarray([x['release_joint'] for x in self.history]),release_sign=np.asarray([x['release_sign'] for x in self.history]))
        return dict(parameters=PARAMETERS,metric_enabled=self.metric_enabled,release_enabled=self.release_enabled,events=self.events,mode_counts=dict(self.mode_counts),rejections=dict(self.rejections),selector_counts=dict(self.selector_counts),auxiliary_qp_calls=self.auxiliary_calls,support_lp_calls=self.lp_calls,certificate_previews=self.preview_calls,clipped_admissions=self.clipped_admissions,scaled_admissions=self.scaled_admissions,peak_error_m=self.peak_error,escape_start_error_m=self.release_start_error,actual_release_progress_rad=self.actual_release_progress,terminal=self.terminal,path_planning_calls=0,history_replay=False,success_trajectory_read=False)
