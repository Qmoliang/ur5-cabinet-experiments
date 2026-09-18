"""Offline terminal-state diagnosis; never modifies frozen experiment runs."""
from pathlib import Path
from types import SimpleNamespace
import csv
import json
from collections import Counter
import numpy as np
import mujoco
import osqp
from scipy import sparse
from model import build_robot_certificate, set_configuration, certificate_world_state, attachment_position
from formal_protocol_v3_viewer import RaggedSnapshots
from run_protocol_v3_async_online import _protocol_scene
from run_protocol_v3_online_ablation import _build_mvt_only, _new_controller
from ellipsoid_model import optimal_support_normal_with_uncertainty

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'formal_results/experiment_07/terminal_diagnosis'

def read_csv(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def box_sdf(p, box):
    d = np.abs(np.asarray(p) - box.center) - box.half_size
    return float(np.linalg.norm(np.maximum(d, 0)) + min(np.max(d), 0))

def proxy_object(a):
    return SimpleNamespace(proxy_ids=a['proxy_ids'], centers=a['centers'],
        sphere_radii=a['sphere_radii'], base_sphere_radii=a['sphere_radii'],
        base_ellipsoid_shapes=a['ellipsoid_shapes'], ellipsoid_outer_shapes=a['ellipsoid_outer_shapes'],
        proxy_uncertainty_shapes=a['proxy_uncertainty_shapes'], proxy_offset_radii=a['uncertainty_offsets'])

def make_controller(model, data, scene, robot, proxies, summary):
    index = _build_mvt_only(proxies, 'ellipsoid', np.array([r.radius for r in robot]), simd=True)
    ctrl = _new_controller('ellipsoid', model, data, scene, robot, proxies, index,
        ellipsoid_pair_threads=2, ellipsoid_pair_affinity_mask=0)
    for name, source in [('task_gain', 'task_gain'), ('max_task_speed', 'max_task_speed_m_per_s'),
        ('osqp_adaptive_row_threshold', 'configured_osqp_adaptive_row_threshold'),
        ('osqp_absolute_tolerance', 'osqp_absolute_tolerance'),
        ('osqp_relative_tolerance', 'osqp_relative_tolerance'), ('osqp_max_iterations', 'osqp_max_iterations')]:
        setattr(ctrl, name, summary[source])
    return ctrl, index

def independent_qp(H, g, A, l, u, keep):
    solver = osqp.OSQP()
    solver.setup(P=sparse.triu(sparse.csc_matrix(H), format='csc'), q=g,
        A=sparse.csc_matrix(A[keep]), l=l[keep], u=u[keep], verbose=False,
        eps_abs=1e-9, eps_rel=1e-9, max_iter=100000, polishing=True)
    result = solver.solve(raise_error=False)
    if not result.info.status.lower().startswith('solved'):
        raise RuntimeError(result.info.status)
    return result

def diagnose(row, *, include_counterfactuals=True, geometry_only=False):
    run = ROOT / row['run']
    summary = json.loads((run/'summary.json').read_text(encoding='utf-8'))
    cycles = read_csv(run/'cycles.csv')
    last = cycles[-1]
    k = int(last['cycle'])
    history = np.load(run/'q_history.npy')
    model = mujoco.MjModel.from_xml_path(str(run/'scene.xml'))
    data = mujoco.MjData(model)
    scene = _protocol_scene(summary.get('camera_scene_version', summary['scene_version']))
    target = np.array(scene.waypoints[-1])
    robot = build_robot_certificate(model)
    # q_history stores AFTER each cycle; pair_states and QP are BEFORE that cycle.
    q = history[-2].copy()
    set_configuration(model, data, history[-1])
    ee_check = np.linalg.norm(attachment_position(model, data) - [float(last['ee_'+c]) for c in 'xyz'])
    set_configuration(model, data, q)
    generation, arrays = RaggedSnapshots(run/'causal_proxy_snapshots.npz').at_cycle(k)
    assert generation == int(last['active_generation'])
    proxies = proxy_object(arrays)
    ctrl, index = make_controller(model, data, scene, robot, proxies, summary)
    qdot, metrics = ctrl.solve(target)
    ee, J, desired = ctrl.task_feedback(target)
    direction = (target-ee)/np.linalg.norm(target-ee)
    pos, jac, radii = certificate_world_state(model, data, robot)
    names = [b.name for b in scene.boxes]
    labels = [names[int(np.argmin([abs(box_sdf(p,b)) for b in scene.boxes]))] for p in proxies.centers]
    id_to_index = {int(pid): i for i,pid in enumerate(proxies.proxy_ids)}
    H,g,A,l,u = ctrl.last_qp_arrays
    identities = ctrl._previous_qp_full_row_identity
    row_labels = [labels[id_to_index[tag[2]]] if tag[0]=='collision' else tag[0] for tag in identities]
    if geometry_only:
        from types import SimpleNamespace
        if include_counterfactuals:raise ValueError("geometry-only diagnosis cannot supply QP ablations")
        baseline = SimpleNamespace(x=qdot, y=np.zeros(len(A)))
    else:
        baseline = independent_qp(H,g,A,l,u,np.ones(len(A),dtype=bool))
    recorded = {}
    with (run/'pair_states.csv').open(encoding='utf-8', newline='') as f:
        for pair in csv.DictReader(f):
            if int(pair['cycle']) == k:
                recorded[(int(pair['robot_index']),int(pair['proxy_id']))] = pair
    reconstruction_errors=[]
    pair_details=[]
    pairs_by_key={(p.robot_index,p.proxy_id):p for p in ctrl.last_pair_records}
    for j,tag in enumerate(identities):
        if tag[0]!='collision':
            continue
        _,ri,pid=tag
        oi=id_to_index[pid]
        p = pairs_by_key[(ri,pid)]
        if (ri,pid) in recorded:
            reconstruction_errors.append(abs(p.clearance-float(recorded[(ri,pid)]['clearance'])))
        if p.clearance>0.00005 and baseline.y[j]<1e-6:
            continue
        b = scene.boxes[names.index(labels[oi])]
        support=optimal_support_normal_with_uncertainty(pos[ri],proxies.centers[oi],
            proxies.base_ellipsoid_shapes[oi],proxies.proxy_uncertainty_shapes[oi])
        segment=np.empty(6)
        distance=mujoco.mj_geomDistance(model,data,robot[ri].source_geom_id,model.geom(b.name).id,10,segment)
        pair_details.append(dict(robot_index=ri,body=robot[ri].body_name,source_geom=robot[ri].source_geom_id,
            proxy_id=pid,box=b.name,robot_center_m=pos[ri].tolist(),robot_radius_m=float(radii[ri]),
            proxy_center_m=proxies.centers[oi].tolist(),core_semiaxes_mm=(1000*np.sqrt(np.linalg.eigvalsh(proxies.base_ellipsoid_shapes[oi]))).tolist(),
            uncertainty_semiaxes_mm=(1000*np.sqrt(np.maximum(np.linalg.eigvalsh(proxies.proxy_uncertainty_shapes[oi]),0))).tolist(),
            core_support_along_normal_mm=float(1000*np.sqrt(support.normal@proxies.base_ellipsoid_shapes[oi]@support.normal)),
            uncertainty_support_along_normal_mm=float(1000*np.sqrt(max(support.normal@proxies.proxy_uncertainty_shapes[oi]@support.normal,0))),
            surface_gap_mm=1000*p.surface_clearance,barrier_h_mm=1000*p.clearance,
            certificate_to_box_gap_mm=1000*(box_sdf(pos[ri],b)-radii[ri]),
            source_geom_to_box_gap_mm=1000*distance,normal=support.normal.tolist(),
            dual=float(baseline.y[j]),velocity_slack_m_s=float(u[j]-A[j]@baseline.x)))
    ablations={}
    if include_counterfactuals:
        for label in ['none','drawer_back','drawer_bottom','drawer_ceiling','bottom_and_ceiling','all_collision']:
            drop = {'none':[], 'bottom_and_ceiling':['drawer_bottom','drawer_ceiling'],
                'all_collision':names}.get(label,[label])
            keep=np.array([label2 not in drop for label2 in row_labels])
            answer=independent_qp(H,g,A,l,u,keep)
            ablations[label]=dict(removed_rows=int(np.sum(~keep)),ee_velocity_mm_s=(1000*J@answer.x).tolist(),
                error_decrease_mm_s=float(1000*direction@J@answer.x))
        near_rows=np.array([j for j,tag in enumerate(identities) if tag[0]=='collision' and pairs_by_key[(tag[1],tag[2])].near_penalty])
        H_no_near=H-ctrl.near_weight*(A[near_rows].T@A[near_rows])
        answer=independent_qp(H_no_near,g,A,l,u,np.ones(len(A),dtype=bool))
        ablations['near_penalty_off_hard_safety_unchanged']=dict(error_decrease_mm_s=float(1000*direction@J@answer.x),ee_velocity_mm_s=(1000*J@answer.x).tolist())
    back=scene.boxes[names.index('drawer_back')]
    back_cert=min((box_sdf(p,back)-r, i) for i,(p,r) in enumerate(zip(pos,radii)))
    back_physical=min((mujoco.mj_geomDistance(model,data,gid,model.geom('drawer_back').id,10,np.empty(6)),gid)
        for gid in set(r.source_geom_id for r in robot))
    tail=cycles[-250:]
    result=dict(seed=int(row['seed']),source_run=str(run),target_m=target.tolist(),final_ee_m=history[-1].tolist(),
        final_error_mm=1000*float(last['error_m']),error_vector_before_last_mm=(1000*(target-ee)).tolist(),
        ee_reconstruction_error_m=float(ee_check),pair_clearance_reconstruction_max_error_m=float(max(reconstruction_errors,default=0)),
        rebuilt_pairs=len(pairs_by_key),recorded_pairs=len(recorded),matched_pairs=len(reconstruction_errors),
        tail_error_start_mm=1000*float(tail[0]['error_m']),tail_error_end_mm=1000*float(tail[-1]['error_m']),
        tail_status_counts=dict(Counter(c['status'] for c in tail)),
        tail_sweep_scale_range=[min(float(c['causal_sweep_scale']) for c in tail),max(float(c['causal_sweep_scale']) for c in tail)],
        back_min_certificate_gap_mm=1000*back_cert[0],back_closest_robot_index=back_cert[1],
        back_raw_mujoco_geom_gap_mm=1000*back_physical[0],
        back_mujoco_gap_consistent_with_certificate=bool(back_physical[0]>=back_cert[0]-1e-7),
        back_min_source_geom_gap_mm=1000*back_physical[0] if back_physical[0]>=back_cert[0]-1e-7 else None,
        tight_pairs=sorted(pair_details,key=lambda p:-p['dual']),
        collision_rows_by_box=dict(Counter(label for label in row_labels if label in names)),
        instantaneous_ablations=ablations,task_gain=ctrl.task_gain,near_weight=ctrl.near_weight,
        recorded_last_progress_mm_s=1000*(float(cycles[-2]['error_m'])-float(last['error_m']))/0.02)
    result['final_ee_m']=[float(last['ee_'+c]) for c in 'xyz']
    assert ee_check<1e-8 and max(reconstruction_errors,default=0)<1e-7
    if geometry_only:
        for pair in result['tight_pairs']:pair['dual']=None
        result['diagnostic_duals_available']=False
    index.close()
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=read_csv(ROOT/'tables/experiment_07/T27_drawer_formal_runs_07_2.csv')
    result=[]
    for row in rows:
        if row['group']!='ET30': continue
        r=diagnose(row)
        result.append(r)
        print(json.dumps({k:r[k] for k in ['seed','final_error_mm','back_min_certificate_gap_mm','back_min_source_geom_gap_mm','instantaneous_ablations']},ensure_ascii=False),flush=True)
    (OUT/'terminal_constraints.json').write_text(json.dumps(dict(diagnostic_only=True,
        note='Nearest box assigns proxy origin; ablations are instantaneous, no safety claim for dropped rows. Frozen data unchanged.',runs=result),indent=2),encoding='utf-8')

if __name__=='__main__': main()

