import numpy as np,mujoco
from goal_entry_policy import LowDimPolicy as Previous,JointControl
from piecewise_reserve import solve_reserve
from verified_qp import solve_verified,_solve_slsqp
from certified_auxiliary import verify_solution

class LowDimPolicy(Previous):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.feature_best=float('inf');self.feature_last_progress=0;self.feature_mode=None
    def observe(self,k,q,previous):
        super().observe(k,q,previous);distance=float(np.linalg.norm(self.feature(q)-self.goal_feature))
        if self.mode!=self.feature_mode:
            self.feature_mode=self.mode;self.feature_best=distance;self.feature_last_progress=k
        elif distance<self.feature_best-.0002:
            self.feature_best=distance;self.feature_last_progress=k
    def _select_blocker(self,s,arrays,nominal):
        if self.k-self.feature_last_progress<100 or self.error<.002:return None
        H,g,A,l,u=arrays;y=nominal.get('original_dual')
        if y is None:
            HH,gg=self._terms(self.ctx.controller,s);shadow=solve_verified(H+HH,g+gg,A,l,u)
            y=s['y'] if shadow is None else shadow.y
        options=[]
        for i,row in enumerate(s['row_ids']):
            if row[0] not in ('collision','frontier','self','workspace'):continue
            norm=np.linalg.norm(A[i]);force=max(0.,float(y[i]))*norm
            if norm<1e-10 or not np.isfinite(u[i]) or self.dt*u[i]>.0005 or force<=1e-6:continue
            key=tuple(row);prior=self.blocker_best.get(key)
            # The full method's no-return certificates carry the persistent
            # memory; a Cartesian-only tabu must not veto posture progress.
            if not self.memory_enabled and prior is not None and .5*self.error**2>=prior-1e-8:continue
            options.append((force,i,key))
        return max(options,key=lambda item:item[0]) if options else None
    def _geometry(self):
        kind,i,j=self.blocker_id
        if kind in ('collision','frontier'):return super()._geometry()
        guard=self.ctx.guard;points=guard.positions(self.q).copy();mujoco.mj_comPos(guard.model,guard.data)
        def jac(index):
            J=np.zeros((3,guard.model.nv));mujoco.mj_jac(guard.model,guard.data,J,None,points[index],int(guard.bodies[index]));return J[:,:6]
        if kind=='self':
            point=points[j]-points[i];length=float(np.linalg.norm(point));return point,point/max(length,1e-12),jac(j)-jac(i),length-guard.radii[i]-guard.radii[j]
        if kind=='workspace':
            axis=j%3;n=np.eye(3)[axis]*(1 if j<3 else -1);gap=points[i,axis]-guard.workspace_lower[axis]-guard.radii[i] if j<3 else guard.workspace_upper[axis]-points[i,axis]-guard.radii[i]
            return points[i],n,jac(i),float(gap)
        return None
    def _goal_point_for_blocker(self):
        kind,i,j=self.blocker_id
        return self.goal_points[j]-self.goal_points[i] if kind=='self' else self.goal_points[i]
    def _solve(self,c,s,arrays,release=None):
        cache_key=(id(s),self.mode,len(self.memories))
        if release is None and cache_key in self.solve_cache:return self.solve_cache[cache_key]
        H,g,A,l,u=arrays;extraH,extrag=self._terms(c,s);rows,bounds=self._extras(c,s)
        AA=np.vstack([A,rows]);ll=np.r_[l,np.full(len(rows),-np.inf)];uu=np.r_[u,bounds]
        self.mode_qp_calls+=1
        answer=solve_reserve(H+extraH,g+extrag,AA,ll,uu,self.q,self.previous,self.low,self.high,c.task_weight,self.error,self.dt,self.acceleration,release=release,joint_metric=True)
        if answer is None:self.augmented_failures+=1;return None
        raw=answer['raw'];v=answer['velocity'];delta=answer['delta']+extraH@v+extrag
        if len(rows):delta+=rows.T@raw.y[len(A):len(AA)]
        effective=solve_verified(H,g+delta,A,l,u,hint=v)
        if effective is None:self.certificate_rejections+=1;return None
        answer.update(velocity=effective.x.copy(),delta=delta,original_dual=effective.y.copy(),iterations=effective.info.iter)
        try:verify_solution(H,g+delta,A,l,u,answer)
        except RuntimeError:
            effective=_solve_slsqp(H,g+delta,A,l,u,hint=v);self.certificate_refinements+=1
            if effective is None:self.certificate_rejections+=1;return None
            answer.update(velocity=effective.x.copy(),original_dual=effective.y.copy(),iterations=effective.info.iter)
            try:verify_solution(H,g+delta,A,l,u,answer)
            except RuntimeError:self.certificate_rejections+=1;return None
        vv=answer['velocity']
        if (len(rows) and np.max(rows@vv-bounds)>1e-7) or (release is not None and release[0]@vv<release[1]-1e-7):self.extra_infeasible+=1;return None
        if release is None:self.solve_cache[cache_key]=answer
        return answer
    def __call__(self,c,s,arrays):
        self.expected=None;self.last_delta=np.zeros(6);self.current_s=s
        if self.error<=.001 and self.mode in ('OPEN','BYPASS'):
            self.switch('RESERVE_GOAL','original_goal_tolerance');self.blocker_id=None;self.pending_memory=None
        if self.terminal or self.brake_pending:return np.zeros(6)
        nominal=self._solve(c,s,arrays)
        if self.mode=='RELEASE':
            if nominal is None:self.brake_pending=True;return np.zeros(6)
            return JointControl.__call__(self,c,s,arrays)
        diagnosis=nominal if nominal is not None else dict(velocity=s['x'])
        if self.mode=='RESERVE_GOAL':
            selected=self._select_blocker(s,arrays,diagnosis) if self.geometric_enabled else None
            if selected is None:
                if nominal is None:self.brake_pending=True;self.rejections['entry_objective_numerical_failure']+=1;return np.zeros(6)
                return JointControl.__call__(self,c,s,arrays)
            force,i,key=selected
            if key[0] not in ('collision','frontier','self','workspace'):return JointControl.__call__(self,c,s,arrays)
            self.blocker_id=key;hit=self.feature(self.q);direction=self.goal_feature-hit;distance=float(np.linalg.norm(direction))
            if distance<1e-6:return JointControl.__call__(self,c,s,arrays)
            direction/=distance;delta=min(self.radius/2,distance/4)
            self.pending_memory=dict(normal=direction.tolist(),offset=float(direction@hit+delta),hit_feature=hit.tolist(),progress_buffer_m=delta,goal_value=float(distance-delta),selected_row=list(key),hit_cycle=self.k)
            self.blocker_hitV=.5*self.error**2;self.blocker_best[key]=self.blocker_hitV;self.axis=None;self.tangent_flips=0;self.side_best=-delta;self.side_progress_cycle=self.k
            self.switch('OPEN','actual_stall_target_entry_incompatible');self.emit('target_directed_release',row_id=list(key),goal_feature_distance=distance,progress_buffer=delta)
            if self.brake_pending:return np.zeros(6)
        geometry=self._geometry()
        if geometry is None:self.brake_pending=True;self.rejections['retained_geometry_unavailable']+=1;return np.zeros(6)
        point,n,Jpoint,gap=geometry
        if self.memory_enabled and self.pending_memory is not None and self._value(self.pending_memory,self.q)>=self.pending_memory['progress_buffer_m']:
            brake=self.previous-np.clip(self.previous,-self.acceleration*self.dt,self.acceleration*self.dt)
            if self.memory_safe(self.q,brake,self.pending_memory):
                memory=dict(self.pending_memory,activation_cycle=self.k);self.memories.append(memory);self.emit('goalward_memory_committed',memory_index=len(self.memories)-1,actual_value=self._value(memory,self.q),goal_value=memory['goal_value'])
                self.pending_memory=None;self.blocker_id=None;self.switch('RESERVE_GOAL','progress_toward_terminal_position_and_posture');return np.zeros(6) if self.brake_pending else JointControl.__call__(self,c,s,arrays)
        if self.mode=='OPEN' and gap>=c.near_distance:
            if not self.memory_enabled:
                self.switch('RESERVE_GOAL','clearance_only_ablation');self.blocker_id=None;self.pending_memory=None;return np.zeros(6) if self.brake_pending else JointControl.__call__(self,c,s,arrays)
            self.switch('BYPASS','clearance_opened_but_terminal_feature_progress_pending')
            if self.brake_pending:return np.zeros(6)
        if self.mode=='OPEN':
            b=n@Jpoint;b=b/max(np.linalg.norm(b),1e-12);chosen=self._try_direction(c,s,arrays,b)
        else:
            if self.axis is None:
                toward=self._goal_point_for_blocker()-point;tangent=toward-n*(n@toward)
                if np.linalg.norm(tangent)<1e-8:tangent=self.outward-n*(n@self.outward)
                if np.linalg.norm(tangent)<1e-8:
                    basis=np.eye(3)[np.argmin(np.abs(n))];tangent=basis-n*(n@basis)
                tangent/=np.linalg.norm(tangent);axis=np.cross(n,tangent);options=[]
                for sign in [1.,-1.]:
                    answer=self._try_direction(c,s,arrays,sign*tangent@Jpoint)
                    if answer is not None:
                        progress=float(self._gradient(self.pending_memory,s)@answer[1][0]);options.append((progress,sign,answer))
                if options:
                    progress,sign,chosen=max(options,key=lambda x:x[0]);self.axis=axis*sign;self.emit('circulation_direction_selected',sign=sign,predicted_feature_progress=progress,axis=self.axis.tolist())
                else:chosen=None
            else:
                tangent=np.cross(self.axis,n);tangent/=max(np.linalg.norm(tangent),1e-12);chosen=self._try_direction(c,s,arrays,tangent@Jpoint)
                if chosen is None and self.k-self.side_progress_cycle>=100 and self.tangent_flips==0:
                    self.axis*=-1;self.tangent_flips=1;self.command_epoch+=1;self.brake_pending=True;self.emit('circulation_branch_reversed_after_actual_no_progress');return np.zeros(6)
        if chosen is not None:
            if self.k%50==0:self.emit('target_directed_motion',surface_gap=gap,feature_progress=None if self.pending_memory is None else self._value(self.pending_memory,self.q),safe_scale=chosen[1][1])
            return self._publish(c,chosen[0])
        self.rejections['target_directed_action_unavailable']+=1;self.brake_pending=True;return np.zeros(6)
    def summary(self):
        out=super().summary();out.update(reserve_solver='six-variable convex piecewise quadratic with certified stationarity; same one-sided objective',stall_metric='actual 3+2 goal feature progress',blocker_dual='effective new goal-entry objective, not the original Cartesian-only objective')
        return out
