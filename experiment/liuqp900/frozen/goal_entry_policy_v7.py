import numpy as np
from goal_entry_policy_v6 import LowDimPolicy as Previous
from goal_entry_policy import JointControl

class LowDimPolicy(Previous):
    def __init__(self,ctx,job,event):
        super().__init__(ctx,job,event);self.radial_subactions=0
        self.escape_active=bool(job.get('start_snapshot'))
    def __call__(self,c,s,arrays):
        self.expected=None;self.last_delta=np.zeros(6);self.current_s=s
        if not self.escape_active:
            history=[item['error_m'] for item in self.history[-100:]]
            stalled=(len(history)==100 and history[0]-min(history)<.0002 and self.error>.002 and np.max(np.abs(s['x']))<.002)
            if not stalled:return np.zeros(6)
            self.escape_active=True;self.command_epoch+=1;self.brake_pending=np.max(np.abs(self.previous))>1e-8
            self.emit('original_liuqp_stall_activates_target_end_module',error_m=self.error,epoch=self.command_epoch)
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
        # Radial correction is a subaction of the SAME boundary-following
        # episode; it must not re-enable Cartesian attraction or flush the
        # command queue on every temporary tangent obstruction.
        b=n@Jpoint;b/=max(np.linalg.norm(b),1e-12);chosen=self._try_direction(c,s,arrays,b)
        if chosen is not None:
            self.radial_subactions+=int(self.mode=='BYPASS')
            self.no_action_start=None;return self._publish(c,chosen[0])
        self.rejections['local_direction_family_unavailable']+=1;self.brake_pending=True
        if self.no_action_start is None:self.no_action_start=self.k
        elif self.k-self.no_action_start>=100 and np.max(np.abs(self.previous))<1e-8:
            self.local_failures+=1;self.terminal='local_direction_family_exhausted';self.emit('policy_failure_current_direction_family_exhausted')
        return np.zeros(6)
    def summary(self):
        out=super().summary();out.update(persistent_radial_subactions=self.radial_subactions,escape_active=self.escape_active,activation='immediate for declared real-stall replay; otherwise original LiuQP until measured original-task stagnation',normal_correction_preserves_bypass_mode=True)
        return out
