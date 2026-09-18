"""Numerical checks independent of closed-loop success."""
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy import sparse
import osqp
from run_comparison import original, ROOT, HERE
from known_volume import build_volume_cover
from robot import build_model,build_robot_certificate,set_configuration,attachment_position,certificate_world_state
from neo_controller import NEOController,translational_jacobian,quadprog

scene=original.load_scene();model=build_model(scene);data=mujoco.MjData(model)
robot=build_robot_certificate(model);cover=build_volume_cover(scene.boxes,'ellipsoid')
ctrl=NEOController(model,data,robot,cover,.046);target=np.array(scene.waypoints[-1])
traj=np.load(ROOT/'results'/'ellipsoid'/'q_history.npy')
checks=[]
for frame in [0,300,750,3000]:
 q=traj[frame];set_configuration(model,data,q)
 j=translational_jacobian(model,data); fd=np.zeros_like(j)
 for k in range(6):
  qp=q.copy();qp[k]+=1e-6;set_configuration(model,ctrl.probe,qp);a=attachment_position(model,ctrl.probe)
  qp[k]-=2e-6;set_configuration(model,ctrl.probe,qp);b=attachment_position(model,ctrl.probe)
  fd[:,k]=(a-b)/2e-6
 je=float(np.max(np.abs(j-fd)));assert je<1e-8
 ge=float(np.max(np.abs(ctrl.gradient(q,1e-5)-ctrl.gradient(q,2e-5))));assert ge<1e-7
 H,g,A,lo,hi,J,v,e,jm,info=ctrl.assemble(target)
 u,metrics=ctrl.solve(target)
 # Explicit 9-dimensional QP with task slack; scale whole objective by e.
 P=np.diag(np.r_[np.full(6,.01*e),np.ones(3)])
 c=np.r_[-e*jm,np.zeros(3)]
 A9=np.vstack([np.c_[A,np.zeros((len(A),3))],np.c_[J,np.eye(3)],np.c_[np.zeros((3,6)),np.eye(3)]])
 l9=np.r_[lo,v,np.full(3,-10.)];h9=np.r_[hi,v,np.full(3,10.)]
 nz=np.linalg.norm(A,axis=1)>1e-14
 aa=A[nz];ll=lo[nz];hh=hi[nz];upper=np.isfinite(hh);lower=np.isfinite(ll)
 C=np.column_stack([np.c_[J,np.eye(3)].T,
       np.c_[-aa[upper],np.zeros((sum(upper),3))].T,
       np.c_[aa[lower],np.zeros((sum(lower),3))].T])
 bb=np.r_[v,-hh[upper],ll[lower]]
 z=quadprog.solve_qp(P,-c,C,bb,meq=3)[0]
 qe=float(np.max(np.abs(z[:6]-u)));assert qe<2e-5,qe
 pos,jacs,radii=certificate_world_state(model,data,robot)
 ri,oi,d,n,res=ctrl.geometry.query(pos,radii,np.inf)
 full={(int(r),int(o)) for r,o,dist in zip(ri,oi,d) if dist<.046}
 rb,ob,db,nb,_=ctrl.geometry.query(pos,radii,.046)
 broad={(int(r),int(o)) for r,o,dist in zip(rb,ob,db) if dist<.046}
 assert full==broad
 # Finite-difference closest distance for most critical pair, holding geometry fixed.
 k=int(np.argmin(d));r=int(ri[k]);o=int(oi[k]);analytic=-n[k]@jacs[r]
 fd_dist=[]
 for axis in range(6):
  vals=[]
  for sign in [1,-1]:
   qp=q.copy();qp[axis]+=sign*1e-6;set_configuration(model,ctrl.probe,qp)
   ps,_,rs=certificate_world_state(model,ctrl.probe,robot)
   nr,s,*_=ctrl.geometry.kernel.closest_points_pairs_warm(ps[r:r+1],cover.centers[o:o+1],ctrl.geometry.values[o:o+1],ctrl.geometry.rotations[o:o+1],np.array([np.nan]))
   vals.append(np.linalg.norm(s[0]-ps[r])-rs[r])
  fd_dist.append((vals[0]-vals[1])/2e-6)
 de=float(np.max(np.abs(analytic-fd_dist)));assert de<1e-6,de
 checks.append(dict(frame=frame,jacobian_error=je,manipulability_gradient_error=ge,explicit_slack_velocity_error=qe,distance_gradient_error=de,all_pairs_candidate_oracle=True))
result=dict(passed=True,robot_spheres=len(robot),proxies=len(cover.centers),checks=checks)
(HERE/'numerical_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
