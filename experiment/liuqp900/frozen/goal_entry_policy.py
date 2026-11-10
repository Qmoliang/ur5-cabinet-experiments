"""Target-end position/posture field and persistent low-dimensional bypass.

Only the present six joint velocities are optimized. A static terminal IK
hypothesis supplies two posture coordinates and one entry direction. No route
from the current state to that hypothesis is computed or read.
"""
from pathlib import Path
import json,os
import numpy as np,mujoco
from scipy.optimize import linprog
from retained_blocker_policy import LowDimPolicy as Previous
from polished_policy import LowDimPolicy as JointControl
from reserve_objective_contract import solve_reserve
from verified_qp import solve_verified,_solve_slsqp
from certified_auxiliary import verify_solution

ROOT=Path(__file__).resolve().parent
def clipped(x,limit):
    return x*min(1.,limit/max(float(np.linalg.norm(x)),1e-30))

class LowDimPolicy(Previous):
    def __init__(self,ctx,job,event):
        super().__init__(ctx,job,event)
        info=json.loads((ROOT/'GOAL_PORTS.json').read_text());self.port=info['ports'][job.get('port_rank',0)]
        self.qg=np.array(self.port['goal_q']);self.P=np.array(self.port['posture_matrix']);self.outward=np.array(self.port['outward']);self.radius=self.port['linear_radius_m']
        self.goal_feature=np.r_[ctx.target,np.zeros(2)];self.goal_points=ctx.guard.positions(self.qg).copy()
        self.entry_enabled=job.get('entry_objective',True);self.geometric_enabled=job.get('geometric_release',True);self.memory_enabled=job.get('persistent_memory',True)
        self.memories=[];self.pending_memory=None;self.axis=None;self.tangent_sign=1.;self.tangent_flips=0
        self.side_best=-float('inf');self.side_progress_cycle=0;self.mode_qp_calls=0;self.memory_scaled=0;self.memory_failures=0;self.extra_infeasible=0
        self.feature_history=[];self.current_s=None;self.solve_cache={}
        # A model-only bound on Cartesian attachment-site arc curvature.
        body=int(ctx.model.site_bodyid[ctx.model.site('attachment_site').id]);length=np.linalg.norm(ctx.model.site_pos[ctx.model.site('attachment_site').id])
        while body:
            length+=np.linalg.norm(ctx.model.body_pos[body]);start=ctx.model.body_jntadr[body]
            for j in range(start,start+ctx.model.body_jntnum[body]):length+=2*np.linalg.norm(ctx.model.jnt_pos[j])
            body=int(ctx.model.body_parentid[body])
        self.site_curvature=2*length
    def feature(self,q):
        self.ctx.guard.positions(q);ee=self.ctx.guard.data.site_xpos[self.ctx.model.site('attachment_site').id].copy()
        return np.r_[ee,self.P@(q-self.qg)]
    def observe(self,k,q,previous):
        super().observe(k,q,previous);self.solve_cache={}
        z=self.P@(self.q-self.qg);self.feature_history.append(np.r_[k,self.error,z])
        if self.pending_memory is not None:
            value=self._value(self.pending_memory,self.q)
            if value>self.side_best+.0002:self.side_best=value;self.side_progress_cycle=k
    def _value(self,m,q):return float(np.asarray(m['normal'])@self.feature(q)-m['offset'])
    def _gradient(self,m,s):
        w=np.asarray(m['normal']);return w[:3]@s['J']+w[3:]@self.P
    def _terms(self,c,s):
        Hextra=np.zeros((6,6));gextra=np.zeros(6)
        if not self.entry_enabled or self.error<=.001:return Hextra,gextra
        x=self.ee-self.ctx.target;d=self.outward;axial=float(d@x);lateral=x-axial*d;z=self.P@(self.q-self.qg)
        compatibility=1./(1.+float(z@z)/self.radius**2+float(lateral@lateral)/self.radius**2)
        inward=-c.task_gain*compatibility*axial
        if axial<0:inward=c.task_gain*min(self.radius,max(-axial,.001))
        entry=clipped(-c.task_gain*lateral+inward*d,c.max_task_speed)
        local_weight=float(np.exp(-(self.error/(2*self.radius))**2))
        desired=(1-local_weight)*s['desired']+local_weight*entry
        posture_velocity=clipped(-c.task_gain*z,c.max_task_speed)
        Hextra=c.task_weight*(self.P.T@self.P)
        gextra=-c.task_weight*s['J'].T@(desired-s['desired'])-c.task_weight*self.P.T@posture_velocity
        return Hextra,gextra
    def _geometry(self):
        kind,robot,identifier=self.blocker_id
        if kind not in ('collision','frontier'):return None
        idx=self.proxy_index.get(int(identifier))
        if idx is None:return None
        guard=self.ctx.guard;points=guard.positions(self.q);point=points[int(robot)].copy()
        distance,normal=self.ctx.proxy_guard.point_geometry(point[None,:],np.array([idx]));n=-normal[0]
        mujoco.mj_comPos(guard.model,guard.data);J=np.zeros((3,guard.model.nv));mujoco.mj_jac(guard.model,guard.data,J,None,point,int(guard.bodies[int(robot)]));J=J[:,:6]
        return point,n,J,float(distance[0]-guard.radii[int(robot)])
    def _extras(self,c,s):
        rows=[];bounds=[]
        for m in self.memories:
            rows.append(-self._gradient(m,s));bounds.append(c.task_gain*self._value(m,self.q))
        if self.mode=='BYPASS':
            geom=self._geometry()
            if geom is not None:
                point,n,J,gap=geom;rows.append(-n@J);bounds.append(c.task_gain*(gap-c.near_distance))
        return np.array(rows).reshape(-1,6),np.asarray(bounds)
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
    def _memory_segment(self,a,b,memories,depth=0):
        fa=self.feature(a);fb=self.feature(b)
        for m in memories:
            n=np.asarray(m['normal']);va=float(n@fa-m['offset']);vb=float(n@fb-m['offset'])
            curve=np.linalg.norm(n[:3])*self.site_curvature*np.sum(np.abs(b-a))**2/8
            if min(va,vb)-curve>=-1e-10:continue
            if min(va,vb)<-1e-9 or depth>=8:return False
            mid=(a+b)/2
            return self._memory_segment(a,mid,memories,depth+1) and self._memory_segment(mid,b,memories,depth+1)
        return True
    def memory_safe(self,q,v,additional=None):
        memories=self.memories+([] if additional is None else [additional])
        if not memories:return True
        current=q.copy();speed=v.copy()
        for _ in range(10):
            nxt=current+self.dt*speed
            if not self._memory_segment(current,nxt,memories):return False
            current=nxt;speed-=np.clip(speed,-self.acceleration*self.dt,self.acceleration*self.dt)
            if np.max(np.abs(speed))<1e-12:return True
        return False
    def filter_memory(self,q,previous,command,lo,hi):
        if not self.memories:return command
        clipped_command=np.clip(command,lo,hi);brake=np.clip(previous-np.clip(previous,-self.acceleration*self.dt,self.acceleration*self.dt),lo,hi)
        for scale in [1.,.75,.5,.25,.125,.0625,0.]:
            candidate=brake+scale*(clipped_command-brake)
            if self.memory_safe(q,candidate) and self.ctx.guard.braking_safe(self.ctx.proxy_guard,q,candidate):
                self.memory_scaled+=int(scale<1.);return candidate
        self.memory_failures+=1;return brake
    def _admit(self,c,arrays,answer,coordinate=None):
        raw=answer['velocity'];v=np.clip(raw,*c.joint_velocity_bounds());H,g,A,l,u=arrays
        if max(np.max(l-A@v),np.max(A@v-u),0)>1e-7:self.rejections['postclip_original_rows']+=1;return None
        brake=self.previous-np.clip(self.previous,-self.acceleration*self.dt,self.acceleration*self.dt)
        for scale in [1.,.75,.5,.25,.125,.0625,0.]:
            candidate=brake+scale*(v-brake)
            if coordinate is not None and coordinate@candidate<=2e-6:continue
            if self.memory_safe(self.q,candidate) and self.ctx.guard.braking_safe(self.ctx.proxy_guard,self.q,candidate):return candidate,scale
        self.rejections['no_certified_command']+=1;return None
    def _publish(self,c,answer):
        self.expected=answer['velocity'].copy();self.last_delta=answer['delta'].copy()
        c.pilot_supplied_solution=dict(velocity=self.expected.copy(),original_dual=answer['original_dual'].copy(),iterations=answer.get('iterations',0))
        return answer['delta']
    def _preview(self,c,arrays,answer,coordinate=None):
        accepted=self._admit(c,arrays,answer,coordinate)
        if accepted is not None:self._publish(c,answer)
        return accepted
    def _support(self,c,s,arrays,b):
        A,l,u=arrays[2:];rows,bounds=self._extras(c,s);up=np.isfinite(u);lo=np.isfinite(l);self.lp_calls+=1
        lp=linprog(-b,A_ub=np.vstack([A[up],-A[lo],rows]),b_ub=np.r_[u[up],-l[lo],bounds],bounds=[(None,None)]*6,method='highs')
        return None if not lp.success else -float(lp.fun)
    def _try_direction(self,c,s,arrays,b):
        support=self._support(c,s,arrays,b)
        if support is None or support<4e-6:return None
        rate=.5*support
        for _ in range(12):
            if rate<2e-6:return None
            answer=self._solve(c,s,arrays,release=(b,rate))
            if answer is not None:
                admitted=self._admit(c,arrays,answer,b)
                if admitted is not None:return answer,admitted
            rate*=.5
        return None
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
            if key[0] not in ('collision','frontier'):return JointControl.__call__(self,c,s,arrays)
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
                toward=self.goal_points[self.blocker_id[1]]-point;tangent=toward-n*(n@toward)
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
        out=super().summary();folder=Path(os.environ['UNIFIED_CASE_DIR']);(folder/'goalward_memories.json').write_text(json.dumps(self.memories,indent=2));np.savez_compressed(folder/'goal_features.npz',history=np.asarray(self.feature_history))
        out.update(goal_port=self.port,entry_objective=self.entry_enabled,geometric_release=self.geometric_enabled,persistent_memory=self.memory_enabled,memory_count=len(self.memories),memory_scaled_commands=self.memory_scaled,memory_certificate_failures=self.memory_failures,extra_row_rejections=self.extra_infeasible,current_goal_q_tracking=False,posture_feature_dimension=2,entry_direction_dimension=1,optimized_future_configurations=0)
        return out
