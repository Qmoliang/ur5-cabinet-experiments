"""Target-side surface following with task-consistent null-space preparation."""
import numpy as np
from goal_entry_policy_v4 import LowDimPolicy as Previous
from goal_entry_policy import JointControl,clipped

class LowDimPolicy(Previous):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.identifiers=np.empty(len(self.proxy_index),dtype=np.int64)
        for identifier,index in self.proxy_index.items():self.identifiers[index]=identifier
        self.surface_cache=None;self.circulation_tangent=None;self.tangent_override=None;self.reverse_used=False
        self.surface_transfers=0;self.frame_transfers=0;self.null_preparations=0;self.local_failures=0;self.no_action_start=None
    def observe(self,k,q,previous):
        self.surface_cache=None
        super().observe(k,q,previous)
    def _geometry(self):
        if not self.memory_enabled or self.blocker_id is None or self.blocker_id[0] not in ('collision','frontier'):return super()._geometry()
        robot=int(self.blocker_id[1])
        if self.surface_cache is not None and self.surface_cache[0]==robot:return self.surface_cache[1]
        guard=self.ctx.guard;pg=self.ctx.proxy_guard;point=guard.positions(self.q)[robot].copy();rad=guard.radii[robot]
        ri,oi=pg.pairs(point[None,:],point[None,:],np.array([rad+self.ctx.controller.near_distance]))
        if len(oi):
            distance,normal=pg.point_geometry(np.repeat(point[None,:],len(oi),axis=0),oi);index=int(oi[np.argmin(distance)]);new=('frontier' if pg.isbox[index] else 'collision',robot,int(self.identifiers[index]))
            if tuple(self.blocker_id)!=new:
                old=self.blocker_id;self.blocker_id=new;self.surface_transfers+=1
                if self.surface_transfers<=200:self.emit('surface_proxy_transported',previous=list(old),current=list(new))
        geometry=super()._geometry();self.surface_cache=(robot,geometry);return geometry
    def _goal_normal(self,geometry):
        point,n,J,gap=geometry;kind,i,j=self.blocker_id
        if kind in ('collision','frontier'):
            index=self.proxy_index[int(j)];d,normal=self.ctx.proxy_guard.point_geometry(self.goal_points[i][None,:],np.array([index]));return -normal[0]
        if kind=='self':
            vector=self.goal_points[j]-self.goal_points[i];return vector/max(np.linalg.norm(vector),1e-12)
        return n
    def _tangent(self,geometry):
        point,n,J,gap=geometry
        reference=self.tangent_override if self.tangent_override is not None else self.circulation_tangent
        if reference is None:reference=self._goal_point_for_blocker()-point
        tangent=reference-n*(n@reference)
        if np.linalg.norm(tangent)<1e-8:
            reference=self._goal_point_for_blocker()-point;tangent=reference-n*(n@reference)
        if np.linalg.norm(tangent)<1e-8:tangent=self.outward-n*(n@self.outward)
        if np.linalg.norm(tangent)<1e-8:
            reference=np.eye(3)[np.argmin(np.abs(n))];tangent=reference-n*(n@reference)
        return tangent/np.linalg.norm(tangent)
    def _entry_velocity(self,c,s):
        if not self.entry_enabled:return s['desired']
        x=self.ee-self.ctx.target;d=self.outward;axial=float(d@x);lateral=x-axial*d;z=self.P@(self.q-self.qg)
        compatibility=1/(1+float(z@z)/self.radius**2+float(lateral@lateral)/self.radius**2)
        axial_speed=-c.task_gain*compatibility*axial
        if axial<0:axial_speed=c.task_gain*min(self.radius,max(-axial,.001))
        entry=clipped(-c.task_gain*lateral+axial_speed*d,c.max_task_speed);weight=float(np.exp(-(self.error/(2*self.radius))**2))
        return (1-weight)*s['desired']+weight*entry
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
            projected=self.P@N;z=self.P@(self.q-self.qg)
            preparation=clipped(-c.task_gain*z-self.P@inverse@desired,c.max_task_speed)
            Hextra+=c.task_weight*(projected.T@projected);gextra-=c.task_weight*projected.T@preparation
        return Hextra,gextra
    def _extras(self,c,s):
        # Keep original physical rows, no-return rows, and the derived self
        # certificate reserve. Do NOT hold the 40 mm near-obstacle threshold.
        mode=self.mode
        if mode=='BYPASS':self.mode='SURFACE_EXTRAS'
        try:return super()._extras(c,s)
        finally:self.mode=mode
    def _tracking_solution(self,c,s,arrays):
        mode=self.mode;self.mode='RESERVE_GOAL'
        try:return self._solve(c,s,arrays)
        finally:self.mode=mode
    def _surface_action(self,c,s,arrays,geometry):
        tangent=self._tangent(geometry);Jpoint=geometry[2]
        allow_reverse=self.circulation_tangent is None or (not self.reverse_used and self.k-self.feature_last_progress>=100)
        for sign in ([1.,-1.] if allow_reverse else [1.]):
            self.tangent_override=sign*tangent
            try:chosen=self._try_direction(c,s,arrays,sign*tangent@Jpoint)
            finally:self.tangent_override=None
            if chosen is not None:return chosen,sign*tangent,sign
        return None
    def _use_tangent(self,tangent,sign):
        changed=self.circulation_tangent is None
        if not changed and sign<0:
            self.reverse_used=True;self.command_epoch+=1;self.brake_pending=np.max(np.abs(self.previous))>1e-8;self.emit('surface_direction_reversed_after_exhaustion')
        self.circulation_tangent=tangent.copy()
        if changed:self.emit('target_side_direction_committed',sign=sign,tangent=tangent.tolist())
    def __call__(self,c,s,arrays):
        self.expected=None;self.last_delta=np.zeros(6);self.current_s=s
        if self.error<=.001 and self.mode in ('OPEN','BYPASS'):
            self.switch('RESERVE_GOAL','original_goal_tolerance');self.blocker_id=None;self.pending_memory=None
        if self.terminal or self.brake_pending:return np.zeros(6)
        if self.mode=='RELEASE':return JointControl.__call__(self,c,s,arrays)
        nominal=self._tracking_solution(c,s,arrays);diagnosis=nominal if nominal is not None else dict(velocity=s['x'])
        if self.mode=='RESERVE_GOAL':
            selected=self._select_blocker(s,arrays,diagnosis) if self.geometric_enabled else None
            if selected is None:
                if nominal is None:self.brake_pending=True;return np.zeros(6)
                return JointControl.__call__(self,c,s,arrays)
            force,i,key=selected;self.blocker_id=key;hit=self.feature(self.q);direction=self.goal_feature-hit;distance=float(np.linalg.norm(direction))
            if distance<1e-6:return JointControl.__call__(self,c,s,arrays)
            direction/=distance;delta=min(self.radius/2,distance/4)
            self.pending_memory=dict(normal=direction.tolist(),offset=float(direction@hit+delta),hit_feature=hit.tolist(),progress_buffer_m=delta,goal_value=distance-delta,hit_distance=distance,selected_row=list(key),hit_cycle=self.k)
            self.blocker_hitV=.5*self.error**2;self.blocker_best[key]=self.blocker_hitV;self.circulation_tangent=None;self.reverse_used=False;self.no_action_start=None
            self.switch('OPEN','actual_goal_feature_stall');self.emit('target_directed_release',row_id=list(key),hit_distance=distance)
            if self.brake_pending:return np.zeros(6)
        geometry=self._geometry()
        if geometry is None:self.brake_pending=True;self.rejections['surface_geometry_unavailable']+=1;return np.zeros(6)
        point,n,Jpoint,gap=geometry;side=float(n@self._goal_normal(geometry));current_distance=float(np.linalg.norm(self.feature(self.q)-self.goal_feature))
        admitted_nominal=None if nominal is None else self._admit(c,arrays,nominal)
        normal_progress=None if admitted_nominal is None else float((self.feature(self.q)-self.goal_feature)@np.vstack([s['J'],self.P])@admitted_nominal[0])
        if self.memory_enabled and self.pending_memory is not None:
            memory=self.pending_memory;delta=memory['progress_buffer_m'];brake=self.previous-np.clip(self.previous,-self.acceleration*self.dt,self.acceleration*self.dt)
            ready=(self._value(memory,self.q)>=delta and current_distance<=memory['hit_distance']-delta and side>=0 and normal_progress is not None and normal_progress<0)
            if ready and self.memory_safe(self.q,brake,memory):
                committed=dict(memory,activation_cycle=self.k);self.memories.append(committed);self.emit('goalward_memory_committed',memory_index=len(self.memories)-1,actual_value=self._value(committed,self.q),distance=current_distance,hit_distance=memory['hit_distance'],goal_side=side)
                self.pending_memory=None;self.blocker_id=None;self.circulation_tangent=None;self.switch('RESERVE_GOAL','target_side_and_certified_net_feature_progress');return np.zeros(6) if self.brake_pending else JointControl.__call__(self,c,s,arrays)
        if not self.memory_enabled:
            if gap>=c.near_distance:
                self.switch('RESERVE_GOAL','clearance_only_ablation');self.blocker_id=None;self.pending_memory=None;return np.zeros(6) if self.brake_pending else JointControl.__call__(self,c,s,arrays)
            b=n@Jpoint;b/=max(np.linalg.norm(b),1e-12);chosen=self._try_direction(c,s,arrays,b)
            if chosen is not None:return self._publish(c,chosen[0])
            self.brake_pending=True;self.rejections['clearance_release_unavailable']+=1;return np.zeros(6)
        if self.mode=='BYPASS' and gap>c.near_distance and side>=0 and normal_progress is not None and normal_progress<0:
            self.no_action_start=None;return self._publish(c,nominal)
        previous_mode=self.mode;self.mode='BYPASS'
        action=self._surface_action(c,s,arrays,geometry)
        self.mode=previous_mode
        if action is not None:
            chosen,tangent,sign=action
            if self.mode!='BYPASS':
                self.switch('BYPASS','safe_goal_oriented_tangent_exists')
                if self.brake_pending:return np.zeros(6)
            self._use_tangent(tangent,sign)
            if self.brake_pending:return np.zeros(6)
            self.no_action_start=None
            if self.k%50==0:self.emit('surface_following',surface_gap=gap,goal_side=side,feature_distance=current_distance,safe_scale=chosen[1][1])
            return self._publish(c,chosen[0])
        # A target-oriented null-space preparation can change the feasible
        # tangent cone without pretending that arbitrary posture motion helps.
        mode=self.mode;self.mode='BYPASS';self.solve_cache.pop((id(s),'BYPASS',len(self.memories)),None)
        preparation=self._solve(c,s,arrays);self.mode=mode
        if preparation is not None:
            admitted=self._admit(c,arrays,preparation)
            if admitted is not None:
                z=self.P@(self.q-self.qg);rate=float(z@self.P@admitted[0]);tangent=self._tangent(geometry)
                if rate<-1e-8 and tangent@Jpoint@admitted[0]>=-1e-8:
                    self.null_preparations+=1;self.no_action_start=None;return self._publish(c,preparation)
        if gap>c.near_distance:
            replacement=self._select_blocker(s,arrays,diagnosis)
            if replacement is not None and tuple(replacement[2])!=tuple(self.blocker_id):
                self.blocker_id=replacement[2];self.surface_cache=None;self.circulation_tangent=None;self.reverse_used=False;self.frame_transfers+=1;self.switch('OPEN','blocking_body_changed_keep_same_escape_memory');self.emit('blocking_frame_transferred',row_id=list(self.blocker_id));return np.zeros(6)
        self.mode='OPEN';b=n@Jpoint;b/=max(np.linalg.norm(b),1e-12);chosen=self._try_direction(c,s,arrays,b);self.mode=previous_mode
        if chosen is not None:
            if self.mode!='OPEN':self.switch('OPEN','tangent_unavailable_reopen_surface')
            if self.brake_pending:return np.zeros(6)
            self.no_action_start=None;return self._publish(c,chosen[0])
        self.rejections['local_direction_family_unavailable']+=1;self.brake_pending=True
        if self.no_action_start is None:self.no_action_start=self.k
        elif self.k-self.no_action_start>=100 and np.max(np.abs(self.previous))<1e-8:
            self.local_failures+=1;self.terminal='local_direction_family_exhausted';self.emit('policy_failure_current_direction_family_exhausted')
        return np.zeros(6)
    def summary(self):
        out=super().summary();out.update(posture_preparation='projected into the null space of the currently controlled frame; task drift compensated',escape_primary_task='two components of blocked-point surface velocity; final goal remains unchanged',surface_proxy_transfers=self.surface_transfers,blocking_frame_transfers=self.frame_transfers,null_preparation_commands=self.null_preparations,local_direction_family_failures=self.local_failures,voluntary_40mm_bypass_constraint=False,exit_requires_goal_side_and_feature_distance_decrease=True)
        return out
