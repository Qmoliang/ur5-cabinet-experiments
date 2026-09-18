"""Diagnose the final millimetres and map change in the completed ablation."""
import json,csv
import numpy as np
import mujoco
from scipy.optimize import linprog
from run_ablation import HERE,NoManipNEO
from model import build_robot_certificate,set_configuration,certificate_world_state
from protocol_drawer_scene import formal_drawer_camera_quarter_scene
from run_study import dump

def main():
    outer=HERE/'results/neo_ellipsoid_46';s=json.loads((outer/'summary.json').read_text(encoding='utf-8'));folder=outer/s['run_name']
    scene=formal_drawer_camera_quarter_scene();target=np.array(scene.waypoints[-1]);model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));data=mujoco.MjData(model)
    robot=build_robot_certificate(model);q=np.load(folder/'q_history.npy');z=np.load(folder/'causal_proxy_snapshots.npz')
    with (folder/'cycles.csv').open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
    errors=np.array([float(r['error_m']) for r in rows]);lastgood=int(np.flatnonzero(errors<.001)[-1]);bad=lastgood+1
    pub=z['publish_cycles'];offsets=z['offsets'];lastsnapshot=int(np.searchsorted(pub,bad,side='right')-1)
    NoManipNEO.influence=.046
    def make(snapshot,configuration):
        set_configuration(model,data,configuration);sl=slice(offsets[snapshot],offsets[snapshot+1])
        return NoManipNEO(model,data,scene,robot,z['centers'][sl],representation='ellipsoid',
                         obstacle_shapes=z['ellipsoid_shapes'][sl],obstacle_uncertainty_shapes=z['proxy_uncertainty_shapes'][sl],
                         obstacle_offsets=z['uncertainty_offsets'][sl],proxy_ids=z['proxy_ids'][sl])
    ctrl=make(len(pub)-1,q[-1]);H,g,A,lo,hi,info=ctrl.assemble(target)
    direction=(target-info['ee'])/np.linalg.norm(target-info['ee']);up=np.isfinite(hi);lower=np.isfinite(lo)
    lp=linprog(-direction@info['J'],A_ub=np.vstack([A[up],-A[lower]]),b_ub=np.r_[hi[up],-lo[lower]],bounds=[(None,None)]*6,method='highs')
    u,_=ctrl.solve(target);pos,jac,r=certificate_world_state(model,data,robot);ri,oi,d,n,_,_=ctrl.geometry(pos,r)
    boxcenters=np.array([b.center for b in scene.boxes]);boxhalf=np.array([b.half_size for b in scene.boxes]);nearest=[]
    for k in np.argsort(d)[:8]:
        j=oi[k];delta=np.abs(boxcenters-ctrl.obstacle_centers[j])-boxhalf
        boxdist=np.linalg.norm(np.maximum(delta,0),axis=1)+np.minimum(delta.max(axis=1),0);owner=int(np.argmin(np.abs(boxdist)))
        normal=n[k]
        nearest.append(dict(robot_body=robot[ri[k]].body_name,robot_sphere=int(ri[k]),proxy_id=int(ctrl.proxy_ids[j]),
                            gap_mm=float(d[k]*1000),nearest_physical_box_to_proxy_center=scene.boxes[owner].name,
                            core_directional_support_mm=float(np.sqrt(normal@ctrl.obstacle_shapes[j]@normal)*1000),
                            U_directional_support_mm=float(np.sqrt(normal@ctrl.obstacle_uncertainty_shapes[j]@normal)*1000),
                            residual_mm=float(ctrl.obstacle_offsets[j]*1000)))
    terminal=dict(error_mm=float(errors[-1]*1000),qdot_norm=float(np.linalg.norm(u)),
                  qp_goal_progress_m_s=float(direction@info['J']@u),
                  max_feasible_goal_progress_m_s=float(-lp.fun) if lp.success else None,lp_success=bool(lp.success),nearest_pairs=nearest)
    ctrl.close()
    # At the first post-update configuration, hold q fixed and compare two maps.
    witness=[];publish_cycle=int(pub[lastsnapshot]);at_q=np.array(scene.q0) if publish_cycle==0 else q[publish_cycle-1]
    for snapshot in [max(0,lastsnapshot-1),lastsnapshot]:
        ctrl=make(snapshot,at_q);pos,_,r=certificate_world_state(model,data,robot);_,_,d,_,_,_=ctrl.geometry(pos,r)
        u,_=ctrl.solve(target);_,J,v=ctrl.task_feedback(target)
        witness.append(dict(snapshot_index=snapshot,publish_cycle=int(pub[snapshot]),
                            fixed_q_cycle_before_publication=publish_cycle,minimum_gap_mm=float(d.min()*1000),
                            qdot_norm=float(np.linalg.norm(u)),cartesian_velocity_m_s=(J@u).tolist()))
        ctrl.close()
    result=dict(terminal=terminal,last_good_post_step_s=(lastgood+1)*.02,first_final_bad_post_step_s=(bad+1)*.02,
                active_snapshot_publish_cycle=publish_cycle,active_snapshot_publish_time_s=publish_cycle*.02,
                map_change_at_identical_configuration=witness,
                limitation='Local velocity diagnosis; no global alternate-posture/path impossibility proof.')
    dump(HERE/'results/terminal_diagnosis.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
