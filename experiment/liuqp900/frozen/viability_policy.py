"""Original LiuQP plus local reserve costs and current-command brake repair.

The only optimized physical variable is current joint velocity. Braking states
are the original deterministic stopping certificate, not trajectory decisions.
"""
import numpy as np
import mujoco
from release_policy_v3 import LowDimPolicy as Previous
from reserve_objective_contract import solve_reserve

class LowDimPolicy(Previous):
    def __init__(self,ctx,job,event):
        super().__init__(ctx,job,event)
        self.metric_enabled=job.get('joint_metric',True);self.spatial_enabled=job.get('spatial_metric',True)
        self.release_enabled=job.get('targeted_release',True);self.operating_gate=job.get('operating_gate',True)
        self.hard_margin=ctx.guard.margin;self.op_margin=ctx.controller.safety_margin-2e-7
        self.gate_active=False;self.repair_attempts=0;self.repair_successes=0;self.repair_records=[]
    def observe(self,k,q,previous):
        super().observe(k,q,previous)
        if self.operating_gate and not self.gate_active:
            guard=self.ctx.guard;guard.margin=self.op_margin
            brake=previous-np.clip(previous,-self.acceleration*self.dt,self.acceleration*self.dt)
            if guard.braking_safe(self.ctx.proxy_guard,q,brake):
                self.gate_active=True;self.command_epoch+=1;self.brake_pending=np.max(np.abs(previous))>1e-8
                self.emit('operating_stop_contract_activated',margin_m=self.op_margin+2e-7)
            else:guard.margin=self.hard_margin
    def _spatial(self,s,arrays):
        if not self.spatial_enabled:return None
        A=arrays[2];u=arrays[4];indices=[i for i,r in enumerate(s['row_ids']) if r[0] in ('collision','frontier','self','workspace') and np.linalg.norm(A[i])>1e-10 and np.isfinite(u[i]) and u[i]*self.dt<self.ctx.controller.near_distance]
        if not indices:return None
        normals=A[indices];gap=self.dt*u[indices]
        # All denominators are relative to each row's own nominal margin.
        speed=self.previous.copy();disp=np.zeros(6);closing=np.zeros(len(indices))
        for _ in range(10):
            disp+=self.dt*speed;closing=np.maximum(closing,normals@disp)
            speed-=np.clip(speed,-self.acceleration*self.dt,self.acceleration*self.dt)
            if np.max(np.abs(speed))<1e-12:break
        floor=np.maximum(2e-7,np.sum(np.abs(normals),axis=1)*self.acceleration*self.dt**2)
        return normals,np.maximum(gap-closing,floor)
    def _repair_cut(self,v):
        guard=self.ctx.guard;proxy=self.ctx.proxy_guard;n=np.maximum(1,np.ceil(np.abs(v)/(self.acceleration*self.dt)-1e-12)).astype(int)
        current=self.q.copy();speed=v.copy();worst=None
        for k in range(int(n.max())):
            for fraction in (.5,1.):
                qs=current+fraction*self.dt*speed;positions=guard.positions(qs)
                ri,oi=proxy.pairs(positions,positions,guard.radii+self.op_margin+2e-6)
                if not len(oi):continue
                d,normal=proxy.point_geometry(positions[ri],oi);gaps=d-guard.radii[ri];j=int(np.argmin(gaps))
                if worst is None or gaps[j]<worst[0]:
                    worst=(float(gaps[j]),qs.copy(),int(ri[j]),normal[j].copy(),self.dt*(np.minimum(k,n)+fraction*(k<n)))
            current+=self.dt*speed;speed-=np.clip(speed,-self.acceleration*self.dt,self.acceleration*self.dt)
        if worst is None:return None
        gap,qs,robot,normal,time=worst;deficit=self.op_margin+2e-7-gap
        if deficit<=0:return None
        guard.positions(qs);mujoco.mj_comPos(guard.model,guard.data)
        jp=np.zeros((3,guard.model.nv));jr=np.zeros_like(jp)
        point=guard.positions(qs)[robot].copy();mujoco.mj_comPos(guard.model,guard.data)
        mujoco.mj_jac(guard.model,guard.data,jp,jr,point,int(guard.bodies[robot]))
        c=(-normal@jp[:,:6])*time
        if np.linalg.norm(c)<1e-12:return None
        if len(self.repair_records)<200:self.repair_records.append(dict(cycle=self.k,gap_m=gap,deficit_m=deficit,normal=c.tolist()))
        return -c,-float(c@v+deficit)
    def _solve(self,c,s,arrays,release=None):
        H,g,A,l,u=arrays;cuts=[];answer=None
        for attempt in range(4):
            self.auxiliary_calls+=1
            answer=solve_reserve(H,g,A,l,u,self.q,self.previous,self.low,self.high,c.task_weight,self.error,self.dt,self.acceleration,release=release,spatial=self._spatial(s,arrays),cuts=cuts,joint_metric=self.metric_enabled)
            if answer is None:return None
            if not self.operating_gate:return answer
            v=np.clip(answer['velocity'],*c.joint_velocity_bounds());guard=self.ctx.guard;saved=guard.margin;guard.margin=self.op_margin
            try:safe=guard.braking_safe(self.ctx.proxy_guard,self.q,v)
            finally:guard.margin=saved
            if safe:
                self.repair_successes+=int(attempt>0);return answer
            cut=self._repair_cut(v)
            if cut is None:return answer
            cuts.append(cut);self.repair_attempts+=1
        return answer
    def summary(self):
        out=super().summary();out.update(spatial_metric=self.spatial_enabled,operating_gate=self.operating_gate,operating_gate_active=self.gate_active,repair_attempts=self.repair_attempts,repair_successes=self.repair_successes,repair_records=self.repair_records)
        return out
