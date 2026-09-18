"""NEO core QP, position-only MuJoCo adaptation of Haviland & Corke (2021).
Independent controller; shared geometry kernels do not call Liu's solve/pruning.
"""
from pathlib import Path
import sys
import time
import numpy as np
import mujoco
import osqp
from scipy import sparse
from scipy.optimize import minimize, LinearConstraint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
import quadprog
sys.path.insert(0,str(ROOT/'src'/'core'))
from robot import JOINT_NAMES, DT, attachment_position, certificate_world_state, set_configuration
from native_support import NativeEllipsoidSupport

VELOCITY_LIMITS=np.array([1.,1.,1.,1.4,1.4,1.4])
WA=np.array([[-1.,0,0],[1.,0,0],[0,-1.,0],[0,1.,0],[0,0,-1.],[0,0,1.]])
WB=np.array([.95,.95,.95,.95,0.,1.35])

def translational_jacobian(model,data):
    jp=np.zeros((3,model.nv)); jr=np.zeros_like(jp)
    mujoco.mj_jacSite(model,data,jp,jr,model.site('attachment_site').id)
    return jp

def manipulability(jac):
    return float(np.prod(np.linalg.svd(jac,compute_uv=False)))

class EllipsoidDistances:
    def __init__(self,cover):
        self.cover=cover
        self.values,self.rotations=np.linalg.eigh(cover.ellipsoid_shapes)
        self.kernel=NativeEllipsoidSupport()
        self.kernel.set_pair_threads(8)
        self.warm=None
    def query(self,positions,radii,influence=np.inf):
        n=len(positions); k=len(self.cover.centers)
        if self.warm is None: self.warm=np.full((n,k),np.nan)
        delta=positions[:,None,:]-self.cover.centers[None,:,:]
        outside=np.maximum(np.abs(delta)-self.cover.aabb_half_extents[None,:,:],0.)
        lower=np.linalg.norm(outside,axis=2)-radii[:,None]
        ri,oi=np.nonzero(lower<=influence+1e-10)
        if len(ri)==0:
            return ri,oi,np.empty(0),np.empty((0,3)),0.
        normals,surfaces,mult,_,_,residual=self.kernel.closest_points_pairs_warm(
            positions[ri],self.cover.centers[oi],self.values[oi],self.rotations[oi],self.warm[ri,oi])
        self.warm[ri,oi]=mult
        local=np.einsum('nji,nj->ni',self.rotations[oi],delta[ri,oi])
        inside=np.sum(local**2/self.values[oi],axis=1)<1.-1e-9
        # The external-distance kernel is only certified outside. Fail explicitly.
        if inside.any(): raise RuntimeError('Robot certificate center entered obstacle ellipsoid')
        distances=np.linalg.norm(surfaces-positions[ri],axis=1)-radii[ri]
        return ri,oi,distances,normals,float(np.max(residual))

class NEOController:
    def __init__(self,model,data,robot,cover,influence):
        self.model=model; self.data=data; self.robot=robot
        self.influence=influence; self.safety=.006; self.geometry=EllipsoidDistances(cover)
        self.probe=mujoco.MjData(model); self.last=np.zeros(6)
        self.qmin=np.array([model.jnt_range[model.joint(n).id,0] for n in JOINT_NAMES])
        self.qmax=np.array([model.jnt_range[model.joint(n).id,1] for n in JOINT_NAMES])
        self.last_qp=None
    def gradient(self,q,step=1e-5):
        result=np.empty(6)
        for k in range(6):
            p=q.copy(); p[k]+=step; set_configuration(self.model,self.probe,p)
            a=manipulability(translational_jacobian(self.model,self.probe))
            p[k]-=2*step; set_configuration(self.model,self.probe,p)
            b=manipulability(translational_jacobian(self.model,self.probe))
            result[k]=(a-b)/(2*step)
        return result
    def assemble(self,target):
        q=self.data.qpos[:6].copy()
        J=translational_jacobian(self.model,self.data)
        error=target-attachment_position(self.model,self.data)
        v=2.*error
        if np.linalg.norm(v)>.18: v*=.18/np.linalg.norm(v)
        e=max(float(np.sum(np.abs(error))),1e-6)
        jm=self.gradient(q)
        # Original objective: .5*.01*||u||² + .5/e*||delta||² - jm.T*u,
        # J*u+delta=v. Multiply the eliminated objective by e for conditioning.
        H=J.T@J+.01*e*np.eye(6); g=-J.T@v-e*jm
        lb=np.maximum(-VELOCITY_LIMITS,(self.qmin+.015-q)/DT)
        ub=np.minimum(VELOCITY_LIMITS,(self.qmax-.015-q)/DT)
        # NEO joint velocity dampers, author-example influence=.9rad, eta=1.
        dl=q-self.qmin; du=self.qmax-q
        lb=np.maximum(lb,np.where(dl<.9,-(dl-.015)/(.9-.015),-np.inf))
        ub=np.minimum(ub,np.where(du<.9,(du-.015)/(.9-.015),np.inf))
        positions,jac,radii=certificate_world_state(self.model,self.data,self.robot)
        rows=[np.eye(6),J,np.einsum('wi,rij->rwj',WA,jac).reshape(-1,6)]
        lows=[lb,v-10.,np.full(len(positions)*6,-np.inf)]
        ups=[ub,v+10.,((WB[None,:]-radii[:,None]-positions@WA.T)/DT).ravel()]
        ri,oi,dist,normals,residual=self.geometry.query(positions,radii,self.influence)
        active=dist<self.influence
        A=np.einsum('pi,pij->pj',normals[active],jac[ri[active]])
        # normals point robot -> obstacle, so approaching speed is n.T*J*u.
        b=(dist[active]-self.safety)/(self.influence-self.safety)
        rows.append(A); lows.append(np.full(len(A),-np.inf)); ups.append(b)
        mat=np.vstack(rows); low=np.concatenate(lows); high=np.concatenate(ups)
        info=dict(error_before_m=float(np.linalg.norm(error)),manipulability=manipulability(J),
                  distance_candidate_pairs=len(dist),obstacle_rows=int(active.sum()),
                  min_candidate_clearance_m=float(np.min(dist)) if len(dist) else None,
                  closest_residual=residual,slack_weight=1/e)
        return H,g,mat,low,high,J,v,e,jm,info
    def solve(self,target):
        started=time.perf_counter()
        H,g,A,lo,hi,J,v,e,jm,info=self.assemble(target)
        # Goldfarb-Idnani dense active-set QP, as used by the NEO ecosystem.
        # Zero rows from fixed base spheres are tautologies; verify before removal.
        nonzero=np.linalg.norm(A,axis=1)>1e-14
        if np.any(lo[~nonzero]>1e-12) or np.any(hi[~nonzero]<-1e-12):
            raise RuntimeError('Infeasible zero constraint row')
        aa=A[nonzero];ll=lo[nonzero];hh=hi[nonzero]
        upper=np.isfinite(hh);lower=np.isfinite(ll)
        C=np.vstack([-aa[upper],aa[lower]]).T
        bvec=np.r_[-hh[upper],ll[lower]]
        raw_status='quadprog solved';status=raw_status;fallback=False;iterations=0
        def violation(x):
            if x is None or not np.all(np.isfinite(x)): return np.inf
            ax=A@x
            return float(max(0.,np.max(lo-ax),np.max(ax-hi)))
        try:
            solution=quadprog.solve_qp(H,-g,C,bvec)
            x=solution[0];iterations=int(solution[3][0])
            if violation(x)>2e-6: raise RuntimeError('quadprog residual exceeds tolerance')
        except (ValueError,RuntimeError) as ex:
            raw_status=str(ex);fallback=True
            sol=minimize(lambda z: .5*z@H@z+g@z,self.last,
                         jac=lambda z:H@z+g,method='SLSQP',
                         constraints=[LinearConstraint(A,lo,hi)],
                         options=dict(ftol=1e-12,maxiter=500))
            x=sol.x;status='slsqp:'+sol.message
            if not sol.success or violation(x)>2e-6:
                raise RuntimeError(f'QP failed: {raw_status}; {sol.message}; violation={violation(x)}')
        self.last=x.copy(); self.last_qp=(H,g,A,lo,hi)
        info.update(status=status,primary_solver_status=raw_status,fallback=fallback,
                    qp_iterations=iterations,qp_violation=violation(x),
                    qdot_norm=float(np.linalg.norm(x)),slack_norm=float(np.linalg.norm(v-J@x)),
                    total_controller_ms=1000*(time.perf_counter()-started))
        return x,info
