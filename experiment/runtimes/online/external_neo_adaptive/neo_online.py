"""NEO QP with v4.3/v4.4 uncertainty-bearing online proxy geometry."""
from pathlib import Path
import sys,time,hashlib
from dataclasses import fields
import numpy as np
import mujoco
from scipy.optimize import minimize,LinearConstraint

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE))
sys.path.insert(0,str(BASE.parent/'known/external_neo/vendor'))
import quadprog
from protocol_liuqp_controller import ProtocolLiuQPController,ProtocolStepMetrics
from model import certificate_world_state,set_configuration
from native_mvt import NativeMultilevelMVT

class OnlineNEO(ProtocolLiuQPController):
    influence=.30
    def __init__(self,*args,**kwargs):
        self._own_index=None; self.index_audits=[]; self.diagnostics=[]; self.trace_q=[]
        kwargs.update(redundant_plane_pruning=False,persistent_qp_workspace=False)
        super().__init__(*args,**kwargs)
        self.probe=mujoco.MjData(self.model)
        self._warm=np.zeros((len(self.robot_spheres),len(self.obstacle_centers),3))

    def update_obstacles(self,centers,**kwargs):
        started=time.perf_counter()
        # Supplied map index remains owned by PerceptionPacket / original sweep guard.
        super().update_obstacles(centers,**kwargs)
        if self._own_index is not None: self._own_index.close()
        if self.representation=='sphere':
            half=np.repeat((self.obstacle_radii+self.obstacle_offsets)[:,None],3,axis=1)
        else:
            half=np.sqrt(np.maximum(np.diagonal(self.obstacle_shapes,axis1=1,axis2=2),0))
            if self.obstacle_uncertainty_shapes is not None:
                half+=np.sqrt(np.maximum(np.diagonal(self.obstacle_uncertainty_shapes,axis1=1,axis2=2),0))
            half+=self.obstacle_offsets[:,None]
        self.proxy_half=half
        rr=np.array([s.radius for s in self.robot_spheres]); padding=self.influence+5e-6
        self._own_index=NativeMultilevelMVT(self.obstacle_centers,half+5e-6,.015,
                                          float(rr.max()+padding),query_padding=padding,simd=True)
        self.obstacle_index=self._own_index
        self._warm=np.zeros((len(rr),len(centers),3))
        pos,_,r=certificate_world_state(self.model,self.data,self.robot_spheres)
        actual=self._own_index.query_spheres(pos,r); missing=0
        for p,rad,got in zip(pos,r,actual):
            delta=np.maximum(np.abs(self.obstacle_centers-p)-half,0)
            expected=np.flatnonzero(np.linalg.norm(delta,axis=1)<=rad+self.influence)
            missing+=len(np.setdiff1d(expected,got))
        assert missing==0, 'NEO influence MVT omitted candidates'
        self.index_audits.append(dict(proxy_count=len(centers),missing=missing,
                                     influence_m=self.influence,update_wall_ms=(time.perf_counter()-started)*1000))

    def geometry(self,positions,radii,full=False):
        batches=([np.arange(len(self.obstacle_centers)) for _ in radii] if full
                 else self.obstacle_index.query_spheres(positions,radii))
        ri=np.repeat(np.arange(len(radii)),[len(x) for x in batches])
        oi=np.concatenate(batches).astype(int) if len(ri) else np.empty(0,dtype=int)
        if not len(ri): return ri,oi,np.empty(0),np.empty((0,3)),0,0.
        delta=self.obstacle_centers[oi]-positions[ri]
        if self.representation=='sphere':
            length=np.linalg.norm(delta,axis=1)
            if np.any(length<1e-14): raise RuntimeError('coincident sphere centers')
            return ri,oi,length-radii[ri]-self.obstacle_radii[oi]-self.obstacle_offsets[oi],delta/length[:,None],0,0.
        Q=self.obstacle_shapes[oi]
        U=(self.obstacle_uncertainty_shapes[oi] if self.obstacle_uncertainty_shapes is not None else np.zeros_like(Q))
        R=np.eye(3)[None,:,:]*radii[ri,None,None]**2
        normal,it,res=self._support_kernel.normals_sum_pairs_newton_warm(
            positions[ri],R,self.obstacle_centers[oi],Q,U,self._warm[ri,oi],max_iterations=16)
        retry=res>1e-7
        if np.any(retry):
            n2,i2,r2=self._support_kernel.normals_sum_pairs_newton_warm(
                positions[ri[retry]],R[retry],self.obstacle_centers[oi[retry]],Q[retry],U[retry],
                np.zeros((int(retry.sum()),3)),max_iterations=64)
            normal[retry]=n2; it[retry]+=i2; res[retry]=r2
        if res.max()>1e-7: raise RuntimeError(f'support KKT residual {res.max()}')
        self._warm[ri,oi]=normal
        distance=np.einsum('ij,ij->i',delta,normal)-radii[ri]-self.obstacle_offsets[oi]
        for shape in [Q,U]: distance-=np.sqrt(np.maximum(np.einsum('ni,nij,nj->n',normal,shape,normal),0))
        return ri,oi,distance,normal,int(it.sum()),float(res.max())

    def jac_at(self,q):
        set_configuration(self.model,self.probe,q)
        J=np.zeros((3,self.model.nv)); Jr=np.zeros_like(J)
        mujoco.mj_jacSite(self.model,self.probe,J,Jr,self.ee_site_id)
        return J[:,:self.nv]

    def assemble(self,target):
        q=self.data.qpos[:self.nv].copy(); ee,J,v=self.task_feedback(target)
        e=max(float(np.abs(target-ee).sum()),1e-6); jm=np.zeros(self.nv)
        for k in range(self.nv):
            a=q.copy(); b=q.copy(); a[k]+=1e-5; b[k]-=1e-5
            jm[k]=(np.prod(np.linalg.svd(self.jac_at(a),compute_uv=False))-np.prod(np.linalg.svd(self.jac_at(b),compute_uv=False)))/2e-5
        H=J.T@J+.01*e*np.eye(self.nv); g=-J.T@v-e*jm
        lb,ub=self.joint_velocity_bounds(); dl=q-self.joint_min; du=self.joint_max-q
        lb=np.maximum(lb,np.where(dl<.9,-(dl-.015)/(.9-.015),-np.inf))
        ub=np.minimum(ub,np.where(du<.9,(du-.015)/(.9-.015),np.inf))
        pos,jac,r=certificate_world_state(self.model,self.data,self.robot_spheres)
        started=time.perf_counter(); ri,oi,d,n,iterations,residual=self.geometry(pos,r)
        geometry_ms=(time.perf_counter()-started)*1000; active=d<self.influence
        rows=[np.eye(self.nv),J,np.einsum('wi,rij->rwj',self.workspace_A,jac).reshape(-1,self.nv),
              np.einsum('pi,pij->pj',n[active],jac[ri[active]])]
        lows=[lb,v-10,np.full(len(pos)*6,-np.inf),np.full(active.sum(),-np.inf)]
        highs=[ub,v+10,((self.workspace_b[None,:]-r[:,None]-pos@self.workspace_A.T)/self.dt).ravel(),
               (d[active]-self.safety_margin)/(self.influence-self.safety_margin)]
        info=dict(ee=ee,J=J,v=v,e=e,jm=jm,geometry_ms=geometry_ms,ri=ri,oi=oi,d=d,
                  iterations=iterations,residual=residual,active=int(active.sum()),workspace_rows=len(pos)*6)
        return H,g,np.vstack(rows),np.concatenate(lows),np.concatenate(highs),info

    def solve(self,target):
        started=time.perf_counter(); self.trace_q.append(self.data.qpos[:self.nv].copy())
        H,g,A,lo,hi,info=self.assemble(np.asarray(target)); self.failed_qp_candidate=(H,g,A,lo,hi,info); assembled=time.perf_counter()
        nonzero=np.linalg.norm(A,axis=1)>1e-14
        if np.any(lo[~nonzero]>1e-12) or np.any(hi[~nonzero]<-1e-12): raise RuntimeError('infeasible fixed workspace row')
        aa=A[nonzero]; ll=lo[nonzero]; hh=hi[nonzero]; upper=np.isfinite(hh); lower=np.isfinite(ll)
        C=np.vstack([-aa[upper],aa[lower]]).T; b=np.r_[-hh[upper],ll[lower]]
        def violation(x): return float(max(0,np.max(lo-A@x),np.max(A@x-hi)))
        fallback=False; raw_error=None
        try:
            solution=quadprog.solve_qp(H,-g,C,b); u=solution[0]; nit=int(solution[3][0])
            if violation(u)>2e-6: raise RuntimeError('quadprog residual too large')
        except (ValueError,RuntimeError) as ex:
            raw_error=str(ex); fallback=True
            sol=minimize(lambda x:.5*x@H@x+g@x,self.previous_velocity,jac=lambda x:H@x+g,
                         method='SLSQP',constraints=[LinearConstraint(A,lo,hi)],options=dict(ftol=1e-12,maxiter=500))
            u=sol.x; nit=sol.nit
            if not sol.success or violation(u)>2e-6: raise RuntimeError(f'NEO QP failed: {raw_error}; {sol.message}; violation={violation(u)}')
        self.previous_velocity=u.copy(); self.last_qp_arrays=(H,g,A,lo,hi)
        # Pair decisions can be reconstructed from causal snapshots; avoid storing
        # millions of Python tuples in the old Liu-specific active-pair recorder.
        self.last_pair_records=()
        d=info['d']; idx=int(np.argmin(d)) if len(d) else None
        values={f.name:0 for f in fields(ProtocolStepMetrics)}
        values.update(status='solved' if not fallback else 'solved_slsqp',representation=self.representation,
                      solve_ms=(time.perf_counter()-assembled)*1000,setup_ms=(assembled-started)*1000-info['geometry_ms'],
                      geometry_ms=info['geometry_ms'],total_controller_ms=(time.perf_counter()-started)*1000,
                      ee_error=float(np.linalg.norm(target-info['ee'])),task_speed=float(np.linalg.norm(info['v'])),
                      qdot_norm=float(np.linalg.norm(u)),min_clearance=float(d.min()-.006) if len(d) else float('inf'),
                      raw_pairs=len(d),broadphase_candidate_pairs=len(d),active_obstacle_rows=info['active'],
                      workspace_rows=info['workspace_rows'],normal_pairs=info['active'],support_normal_iterations=info['iterations'],
                      support_normal_max_residual=info['residual'],qp_iterations=nit,qp_primal_residual=violation(u),
                      qp_dual_residual=float('nan'),qp_rho_mode='not_applicable_quadprog',
                      qp_row_sha256=hashlib.sha256(A.tobytes()+lo.tobytes()+hi.tobytes()).hexdigest(),
                      limiting_robot_index=int(info['ri'][idx]) if idx is not None else -1,
                      limiting_obstacle_index=int(info['oi'][idx]) if idx is not None else -1,
                      limiting_proxy_id=int(self.proxy_ids[info['oi'][idx]]) if idx is not None else -1)
        self.diagnostics.append(dict(cycle=len(self.diagnostics),fallback=fallback,primary_error=raw_error,
                                     violation=violation(u),slack_norm=float(np.linalg.norm(info['v']-info['J']@u)),
                                     influence_m=self.influence,active_rows=info['active']))
        if len(self.diagnostics)%250==0:
            print(f'NEO {self.representation} di={self.influence} step={len(self.diagnostics)} error={values["ee_error"]*1000:.3f}mm proxies={len(self.obstacle_centers)}',flush=True)
        return u,ProtocolStepMetrics(**values)

    def close(self):
        if self._own_index is not None: self._own_index.close(); self._own_index=None
