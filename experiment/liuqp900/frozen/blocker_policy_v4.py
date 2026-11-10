"""Release a measured blocking geometric row, not an arbitrary pose joint."""
import numpy as np
from scipy.optimize import linprog
from blocker_policy_v3 import LowDimPolicy as Previous
from polished_policy import LowDimPolicy as OriginalControl

class LowDimPolicy(Previous):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.blocker_id=None;self.blocker_hitV=None;self.blocker_events=[];self.blocker_best={}
    def _select_blocker(self,s,arrays,nominal):
        if len(self.history)<100 or self.error<.002:return None
        errors=np.array([x['error_m'] for x in self.history[-100:]])
        if errors.max()-errors.min()>.0002:return None
        direction=(self.ctx.target-self.ee)/self.error
        H,g,A,l,u=arrays;candidates=[]
        for i,row in enumerate(s['row_ids']):
            if row[0] not in ('collision','frontier','self','workspace'):continue
            norm=np.linalg.norm(A[i])
            if norm<1e-10 or not np.isfinite(u[i]) or self.dt*u[i]>.0005:continue
            force=max(0.,float(s['y'][i]))*norm
            key=tuple(row);prior=self.blocker_best.get(key)
            if prior is not None and .5*self.error**2>=prior-1e-8:continue
            if force>1e-6:candidates.append((force,i,key))
        return max(candidates,key=lambda x:x[0]) if candidates else None
    def __call__(self,c,s,arrays):
        self.expected=None;self.last_delta=np.zeros(6)
        if self.error<=1e-3 and self.mode=='OBSTACLE_RELEASE':
            self.switch('RESERVE_GOAL','goal_tolerance_reached');self.blocker_id=None
        if self.terminal or self.brake_pending:return np.zeros(6)
        if self.mode=='RELEASE':return OriginalControl.__call__(self,c,s,arrays)
        nominal=self._solve(c,s,arrays)
        if nominal is None:nominal=dict(velocity=s['x'].copy(),delta=np.zeros(6))  # diagnosis only; never publish this fallback
        if self.mode!='OBSTACLE_RELEASE':
            selected=self._select_blocker(s,arrays,nominal)
            if selected is None:return OriginalControl.__call__(self,c,s,arrays)
            force,i,key=selected;self.blocker_id=key;self.blocker_hitV=.5*self.error**2;self.blocker_best[key]=self.blocker_hitV
            self.switch('OBSTACLE_RELEASE','actual_executed_stall_with_geometric_positive_dual')
            self.emit('blocker_selected',row_id=list(key),force=force,error_m=self.error,gap_above_nominal_m=float(arrays[4][i]*self.dt))
            if self.brake_pending:return np.zeros(6)
        lookup={tuple(row):i for i,row in enumerate(s['row_ids'])}
        i=lookup.get(self.blocker_id)
        if i is None:
            self.emit('blocker_no_longer_active',row_id=list(self.blocker_id));self.switch('RESERVE_GOAL','selected_geometry_left_active_set');self.blocker_id=None
            return np.zeros(6) if self.brake_pending else OriginalControl.__call__(self,c,s,arrays)
        H,g,A,l,u=arrays;norm=np.linalg.norm(A[i]);b=-A[i]/max(norm,1e-12)
        direction=(self.ctx.target-self.ee)/max(self.error,1e-30);goal_rate=float(direction@s['J']@nominal['velocity'])
        if .5*self.error**2<self.blocker_hitV-.05*max(self.blocker_hitV-.5e-6,0) and b@nominal['velocity']>=-1e-6 and goal_rate>1e-5:
            self.emit('blocker_released_with_task_progress',row_id=list(self.blocker_id),error_m=self.error,gap_above_nominal_m=float(self.dt*u[i]))
            self.switch('RESERVE_GOAL','task_progress_without_reclosing_selected_geometry');self.blocker_id=None
            return np.zeros(6) if self.brake_pending else OriginalControl.__call__(self,c,s,arrays)
        up=np.isfinite(u);lo=np.isfinite(l);self.lp_calls+=1
        support=linprog(-b,A_ub=np.vstack([A[up],-A[lo]]),b_ub=np.r_[u[up],-l[lo]],bounds=[(None,None)]*6,method='highs')
        self.expected=None;self.last_delta=np.zeros(6)
        if support.success and -support.fun>40e-7:
            rate=-.5*float(support.fun)
            for _ in range(12):
                if rate<=20e-7:break
                answer=self._solve(c,s,arrays,release=(b,rate))
                if answer is not None:
                    accepted=self._preview(c,arrays,answer,b)
                    if accepted is not None:
                        if self.k%50==0:self.emit('blocker_release_proposal',row_id=list(self.blocker_id),gap_above_nominal_m=float(self.dt*u[i]),certified_gap_rate_m_s=float(norm*(b@accepted[0])),error_m=self.error,dual=answer['release_dual'])
                        return answer['delta']
                rate*=.5
        self.rejections['blocker_release_unavailable']+=1;self.brake_pending=True
        return np.zeros(6)
    def summary(self):
        out=super().summary();out.update(geometric_blocker_release=True,blocker_id=self.blocker_id,blocker_memory=[dict(row_id=list(key),hitV=value) for key,value in self.blocker_best.items()],blocking_direction='negative current geometric row; no reference configuration or fixed joint',release_scope='single scalar geometric clearance; task and null-space effects remain coordinated by original QP')
        return out
