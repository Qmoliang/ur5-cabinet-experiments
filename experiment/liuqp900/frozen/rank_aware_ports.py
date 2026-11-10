"""Rank-aware static entry certificates; no path search or changed posture coordinates."""
import numpy as np
RANK_TOL=1e-8
RESIDUAL_TOL=1e-7
MIN_DISPLACEMENT=2e-7
class PortRejected(Exception):
 def __init__(self,reason,**details):super().__init__(reason);self.reason=reason;self.details=details

def reachable_lift(J,P):
 J=np.asarray(J);P=np.asarray(P);_,sp,Vp=np.linalg.svd(P,full_matrices=True);rp=int(np.sum(sp>RANK_TOL));Z=Vp[rp:].T
 U,s,Vt=np.linalg.svd(J@Z,full_matrices=False);r=int(np.sum(s>RANK_TOL));projector=U[:,:r]@U[:,:r].T
 lift=np.zeros((J.shape[1],J.shape[0])) if not r else (Z@Vt[:r].T/s[:r])@U[:,:r].T
 return lift,projector,dict(posture_rank=rp,reachable_translation_rank=r,combined_rank=int(np.linalg.matrix_rank(np.vstack([J,P]),tol=RANK_TOL)),projected_singular_values=s.tolist(),rank_tolerance=RANK_TOL)

def build_port(ctx,oracle,previous,reference):
 port=dict(previous);qg=np.asarray(port['goal_q'],dtype=float)
 if qg.shape!=(6,) or not np.all(np.isfinite(qg)):raise PortRejected('invalid_goal_configuration')
 oracle.margin=ctx.controller.safety_margin
 ee,J,_,_,_,_=oracle.evaluate(qg)
 if np.linalg.norm(ee-ctx.target)>=.0007 or not oracle.free(qg):raise PortRejected('static_goal_no_longer_feasible',position_error_m=float(np.linalg.norm(ee-ctx.target)))
 P=np.zeros((2,6));P[0,0]=np.linalg.norm(J[:,0]);P[1,2]=np.linalg.norm(J[:,2]);lift,projector,diag=reachable_lift(J,P)
 if diag['reachable_translation_rank']==0:raise PortRejected('no_resolved_translation_in_posture_nullspace',**diag)
 legacy=diag['combined_rank']==5 and diag['reachable_translation_rank']==3
 legacy_lift=np.linalg.pinv(np.vstack([J,P]),rcond=1e-8) if legacy else None
 rays=[];attempts=[]
 directions=[reference]+[sign*np.eye(3)[axis] for axis in range(3) for sign in [1.,-1.]]
 for index,requested in enumerate(directions):
  projected=requested.copy() if legacy else projector@requested;pn=float(np.linalg.norm(projected))
  if pn<=RANK_TOL:attempts.append(dict(direction_index=index,status='unresolved_in_local_linear_subspace',projection_norm=pn));continue
  d=projected/pn;v=legacy_lift@np.r_[d,np.zeros(2)] if legacy else lift@d
  residual=max(float(np.max(np.abs(J@v-d))),float(np.max(np.abs(P@v))))
  if residual>RESIDUAL_TOL:
   projected=projector@requested;pn=float(np.linalg.norm(projected))
   if pn<=RANK_TOL:attempts.append(dict(direction_index=index,status='unresolved_in_local_linear_subspace'));continue
   d=projected/pn;v=lift@d;residual=max(float(np.max(np.abs(J@v-d))),float(np.max(np.abs(P@v))))
  if residual>RESIDUAL_TOL:attempts.append(dict(direction_index=index,status='linear_residual_too_large',residual=residual));continue
  radius=ctx.controller.near_distance;accepted=False
  for _ in range(7):
   qb=qg+radius*v
   if np.all(qb>=oracle.lower) and np.all(qb<=oracle.upper) and oracle.segment_certified(qg,qb):
    displacement=oracle.gripper_position(qb)-ee;length=float(np.linalg.norm(displacement))
    if length>MIN_DISPLACEMENT:
     outward=displacement/length;rays.append(dict(outward=outward.tolist(),radius_m=length,linear_radius_m=radius,alignment=float(outward@reference),backward_configuration_for_certificate_only=qb.tolist(),requested_direction=requested.tolist(),reachable_direction=d.tolist(),projection_norm=pn,linear_residual=residual,posture_linear_residual=float(np.max(np.abs(P@(qb-qg))))));accepted=True;break
   radius*=.5
  attempts.append(dict(direction_index=index,status='certified' if accepted else 'no_certified_ray_in_original_backoff_budget'))
 if not rays:raise PortRejected('no_certified_local_entry_direction',**diag,ray_attempts=attempts)
 chosen=max(rays,key=lambda r:(r['linear_radius_m'],r['alignment']))
 port.update(posture_matrix=P.tolist(),posture_coordinates=['base_angle_unwrapped','elbow_flexion'],coordinate_selection='unchanged fixed UR5 branch coordinates; rank deficiency is handled in the reachable translation subspace',all_entry_rays=rays,entry_rank_diagnostics=diag,entry_ray_attempts=attempts,**chosen)
 return port

def initialize(ctx,oracle,source):
 p0=oracle.gripper_position(ctx.q0);d=p0-ctx.target;norm=float(np.linalg.norm(d));reference=d/norm if norm>1e-12 else np.array([1.,0.,0.]);records=[];selected=None
 for index,old in enumerate(source['ports']):
  if selected is not None:records.append(dict(source_rank=index,status='not_evaluated_unselected'));continue
  try:port=build_port(ctx,oracle,old,reference)
  except PortRejected as e:records.append(dict(source_rank=index,status='rejected',reason=e.reason,details=e.details));continue
  selected=dict(port,source_rank=index,rank=0);records.append(dict(source_rank=index,status='selected',rank_diagnostics=port['entry_rank_diagnostics']))
 return dict(task=source['task'],ports=[] if selected is None else [selected],initialization_status='ready' if selected is not None else 'no_usable_static_port',selection_rule='first certifiable terminal port in the original frozen order, before control; unselected backups do not veto a valid selection',port_diagnostics=records,rank_tolerance=RANK_TOL,posture_features_changed=False,optimized_route=False,goal_pose_values_modified=False,goal_pose_changed=bool(selected is not None and selected['source_rank']!=0))
