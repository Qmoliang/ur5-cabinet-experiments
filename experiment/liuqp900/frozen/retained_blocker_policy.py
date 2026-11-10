"""Current observed geometry survives pruning of its instantaneous QP row."""
import numpy as np
import mujoco
from scipy.optimize import linprog
from blocker_policy_v5 import LowDimPolicy as Previous
from polished_policy import LowDimPolicy as OriginalControl

class LowDimPolicy(Previous):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Read only already-observed proxy identifiers; never read archive path.
        with np.load(self.ctx.entry['input_file']) as z:ids=z['ids'].copy()
        self.proxy_index={int(key):i for i,key in enumerate(ids)}
        self.retained_geometry_queries=0;self.pruned_row_continuations=0
    def _retained_geometry(self,c,s,arrays):
        kind,robot,identifier=self.blocker_id
        lookup={tuple(row):i for i,row in enumerate(s['row_ids'])}
        row=lookup.get(self.blocker_id)
        if kind not in ('collision','frontier'):
            if row is None:return None
            a=arrays[2][row];norm=np.linalg.norm(a)
            return -a/max(norm,1e-12),norm,float(self.dt*arrays[4][row])
        index=self.proxy_index.get(int(identifier))
        if index is None:return None
        guard=self.ctx.guard;pg=self.ctx.proxy_guard;points=guard.positions(self.q);point=points[int(robot)].copy()
        distance,normal=pg.point_geometry(point[None,:],np.array([index]))
        mujoco.mj_comPos(guard.model,guard.data)
        jp=np.zeros((3,guard.model.nv));jr=np.zeros_like(jp)
        mujoco.mj_jac(guard.model,guard.data,jp,jr,point,int(guard.bodies[int(robot)]))
        a=normal[0]@jp[:,:6];norm=float(np.linalg.norm(a))
        if norm<1e-12:return None
        self.retained_geometry_queries+=1;self.pruned_row_continuations+=int(row is None)
        return -a/norm,norm,float(distance[0]-guard.radii[int(robot)]-c.safety_margin)
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
        H,g,A,l,u=arrays
        geometry=self._retained_geometry(c,s,arrays)
        if geometry is None:
            self.rejections['retained_geometry_unavailable']+=1;self.brake_pending=True;return np.zeros(6)
        b,norm,gap=geometry
        if gap+c.safety_margin>=c.near_distance:
            self.emit('retained_geometry_cleared',row_id=list(self.blocker_id),surface_gap_m=gap+c.safety_margin,error_m=self.error)
            self.switch('RESERVE_GOAL','measured_clearance_reached_original_near_distance');self.blocker_id=None
            return np.zeros(6) if self.brake_pending else OriginalControl.__call__(self,c,s,arrays)
        direction=(self.ctx.target-self.ee)/max(self.error,1e-30);goal_rate=float(direction@s['J']@nominal['velocity'])
        if .5*self.error**2<self.blocker_hitV-.05*max(self.blocker_hitV-.5e-6,0) and b@nominal['velocity']>=-1e-6 and goal_rate>1e-5:
            self.emit('blocker_released_with_task_progress',row_id=list(self.blocker_id),error_m=self.error,gap_above_nominal_m=float(gap))
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
                        if self.k%50==0:self.emit('blocker_release_proposal',row_id=list(self.blocker_id),gap_above_nominal_m=float(gap),certified_gap_rate_m_s=float(norm*(b@accepted[0])),error_m=self.error,dual=answer['release_dual'])
                        return answer['delta']
                rate*=.5
        self.rejections['blocker_release_unavailable']+=1;self.brake_pending=True
        return np.zeros(6)
    def summary(self):
        out=super().summary();out.update(retained_geometry_queries=self.retained_geometry_queries,pruned_row_continuations=self.pruned_row_continuations,geometric_exit='measured original near-distance clearance, or task progress without reclosure; row disappearance is not an exit')
        return out
