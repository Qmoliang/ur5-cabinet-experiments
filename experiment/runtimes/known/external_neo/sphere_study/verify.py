"""Independent geometry and log reconstruction for the sphere study."""
import json
import numpy as np
import mujoco
from experiment import HERE, PARENT, ROOT, CASES, original, dump, sha, build_volume_cover
from experiment import build_model, build_robot_certificate, set_configuration, attachment_position, DT
from robot import certificate_world_positions, certificate_world_state
from neo_controller import VELOCITY_LIMITS, WA, WB
from audit import lever_bounds

def main():
    output=HERE/'results'
    for name in ['protected_hashes.json','source_hashes.json']:
        hashes=json.loads((output/name).read_text(encoding='utf-8'))
        assert all(sha(ROOT/p)==h for p,h in hashes.items()), name
    scene=original.load_scene(); model=build_model(scene); data=mujoco.MjData(model)
    robot=build_robot_certificate(model); radii=np.array([s.radius for s in robot])
    cover=build_volume_cover(scene.boxes,'sphere',.075); L=lever_bounds(model,robot)
    centers=np.array([b.center for b in scene.boxes]); halves=np.array([b.half_size for b in scene.boxes])
    target=np.array(scene.waypoints[-1]); results={}
    def exact_box_gaps(pos):
        v=np.abs(pos[:,None,:]-centers[None,:,:])-halves[None,:,:]
        return np.linalg.norm(np.maximum(v,0),axis=2)+np.minimum(np.max(v,axis=2),0)-radii[:,None]
    def proxy_gaps(pos):
        # Independent broadcasting oracle, does not call experiment.SphereDistances.
        return np.sqrt(np.sum((pos[:,None,:]-cover.centers[None,:,:])**2,axis=2))-radii[:,None]-cover.sphere_radii[None,:]
    qlo=np.array([model.jnt_range[model.joint(n).id,0] for n in original.JOINT_NAMES])+.015
    qhi=np.array([model.jnt_range[model.joint(n).id,1] for n in original.JOINT_NAMES])-.015
    for case in CASES:
        folder=output/case; summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        logs=json.loads((folder/'cycles.json').read_text(encoding='utf-8')); q=np.load(folder/'q_history.npy'); ee=np.load(folder/'ee_history.npy')
        assert q.shape==(3001,6) and ee.shape==(3001,3) and len(logs)==3000
        assert np.isfinite(q).all() and np.array_equal(q[0],scene.q0)
        velocity=float(np.max(np.abs(np.diff(q,axis=0)/DT)-VELOCITY_LIMITS))
        joint_margin=min(float(np.min(q-qlo)),float(np.min(qhi-q)))
        min_proxy=np.inf; min_box=np.inf; min_ws=np.inf; pen=0; hold=0; maxhold=0; confirmed=None
        error_diff=0.
        for k,x in enumerate(q):
            set_configuration(model,data,x); pos=certificate_world_positions(data,robot)
            min_proxy=min(min_proxy,float(proxy_gaps(pos).min()))
            min_box=min(min_box,float(exact_box_gaps(pos).min()))
            min_ws=min(min_ws,float((WB[None,:]-radii[:,None]-pos@WA.T).min()))
            error_diff=max(error_diff,float(np.max(np.abs(attachment_position(model,data)-ee[k]))))
            contacts=sum(data.contact[t].dist < -1e-8 for t in range(data.ncon)); pen+=int(contacts>0)
            if k:
                err=float(np.linalg.norm(ee[k]-target))
                assert abs(err-logs[k-1]['error_m'])<1e-12
                hold=hold+1 if err<.001 and contacts==0 else 0; maxhold=max(maxhold,hold)
                if hold==50 and confirmed is None: confirmed=k*DT
        lower_min=np.inf; unresolved=0; interval_checks=0
        def interval(a,b,depth=0):
            nonlocal lower_min,unresolved,interval_checks
            set_configuration(model,data,(a+b)/2)
            pos=certificate_world_positions(data,robot)
            lower=float((exact_box_gaps(pos)-(.5*L@np.abs(b-a))[:,None]).min())
            interval_checks+=1
            if lower>=.006:
                lower_min=min(lower_min,lower); return
            if depth>=8:
                unresolved+=1; lower_min=min(lower_min,lower); return
            mid=(a+b)/2; interval(a,mid,depth+1); interval(mid,b,depth+1)
        for a,b in zip(q[:-1],q[1:]): interval(a,b)
        assert velocity<1e-6 and joint_margin>=-1e-7 and min_ws>=-1e-7
        assert min_proxy>=0 and pen==0 and error_diff<1e-12
        assert maxhold==summary['maximum_success_hold_cycles']
        assert summary['success']==(maxhold>=50) and confirmed==summary['confirmed_success_time_s']
        assert summary['failure'] is None and summary['fallback_cycles']==0
        assert all(r['status']=='quadprog solved' and r['qp_violation']<2e-6 for r in logs)
        set_configuration(model,data,q[-1]); pos=certificate_world_positions(data,robot)
        gaps=proxy_gaps(pos); boxgaps=exact_box_gaps(pos)
        active=[]
        for ri,oi in np.argwhere(gaps<.006+1e-6):
            owner=int(cover.box_indices[oi]); delta=cover.centers[oi]-pos[ri]
            active.append(dict(robot_sphere=int(ri),robot_body=robot[ri].body_name,
                               obstacle_proxy=int(oi),cabinet_box=scene.boxes[owner].name,
                               proxy_gap_mm=float(gaps[ri,oi]*1000),
                               same_sphere_to_physical_box_gap_mm=float(boxgaps[ri,owner]*1000),
                               normal_robot_to_obstacle=(delta/np.linalg.norm(delta)).tolist()))
        result=dict(passed=True,saved_states=len(q),all_pair_endpoint_checks=len(q)*len(robot)*len(cover.centers),
                    minimum_proxy_surface_gap_mm=min_proxy*1000,minimum_proxy_safety_slack_mm=(min_proxy-.006)*1000,
                    endpoint_proxy_6mm_strictly_satisfied=bool(min_proxy>=.006),minimum_exact_box_endpoint_gap_mm=min_box*1000,
                    minimum_workspace_slack_mm=min_ws*1000,maximum_velocity_limit_excess_rad_s=velocity,
                    minimum_joint_padding_slack_rad=joint_margin,penetrating_states=pen,
                    maximum_reconstructed_ee_error_m=error_diff,success_hold_recomputed=maxhold,
                    continuous_cabinet_6mm_certified=unresolved==0,continuous_box_gap_lower_bound_mm=lower_min*1000,
                    interval_checks=interval_checks,unresolved_for_6mm=unresolved,
                    continuous_self_collision_certified=False,continuous_floor_certified=False,
                    final_ee_m=ee[-1].tolist(),final_qdot_norm=logs[-1]['qdot_norm'],
                    final_task_slack_norm=logs[-1]['slack_norm'],final_active_sphere_pairs=active)
        dump(folder/'audit.json',result); results[case]=result
        print(case,json.dumps(result),flush=True)
    # A counterfactual geometry check, not a collision-free sphere trajectory.
    witness={}
    for oldcase in ['neo_matched','neo_paper_influence']:
        q=np.load(PARENT/'results'/oldcase/'q_history.npy')[-1]
        set_configuration(model,data,q); pos=certificate_world_positions(data,robot); g=proxy_gaps(pos)
        ri,oi=np.unravel_index(np.argmin(g),g.shape)
        witness[oldcase]=dict(sphere_proxy_minimum_gap_at_ellipsoid_success_mm=float(g[ri,oi]*1000),
                             robot_body=robot[ri].body_name,cabinet_box=scene.boxes[int(cover.box_indices[oi])].name,
                             note='This final posture violates spherical envelopes if gap<6mm; does not prove all sphere-feasible target postures impossible.')
    dump(output/'verification.json',dict(passed=True,old_sources_and_results_unchanged=True,
         frozen_run_sources_unchanged=True,cases=results,ellipsoid_success_posture_counterfactual=witness))
    print('ALL CHECKS PASSED; counterfactual:',json.dumps(witness),flush=True)

if __name__=='__main__': main()
