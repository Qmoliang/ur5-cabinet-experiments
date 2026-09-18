"""Replay frozen source camera poses to audit directional uncertainty inflation."""
import json
import numpy as np
import mujoco
from diagnose_et30_terminal import ROOT, OUT, read_csv
from model import set_configuration
from depth_camera_perception import UR5MountedDepthCamera, environment_endpoint_mask

report=json.loads((OUT/'terminal_constraints.json').read_text(encoding='utf-8'))
output=[]
for item in report['runs']:
    run=__import__('pathlib').Path(item['source_run'])
    arc=np.load(run/'final_causal_proxies.npz')
    points=arc['filtered_points']; cells=arc['filtered_uncertainty_shapes']; owners=arc['filtered_cluster_indices']
    keys=np.floor((points-np.array([-1.2,-1.2,0]))/.0075).astype(int)
    cell_index={tuple(k):i for i,k in enumerate(keys)}
    selected=[]
    for pair in item['tight_pairs']:
        if pair['dual']<.01: continue
        oi=int(np.flatnonzero(arc['proxy_ids']==pair['proxy_id'])[0])
        assigned=np.flatnonzero(owners==oi)
        d=-np.array(pair['normal'])
        selected.append(dict(pair=pair,d=d,assigned=set(assigned.tolist()), count=0,measurement_support_max=0.,
            raw_relative_cell_bound=0.,raw_proxy_directional_support=-float('inf'),seen=set()))
    model=mujoco.MjModel.from_xml_path(str(run/'scene.xml')); data=mujoco.MjData(model)
    saved_sources=np.load(run/'causal_source_configurations.npz')
    poses=saved_sources['q']
    frame_by_cycle={int(x['source_cycle']):x for x in read_csv(run/'perception_frames.csv')}
    raw_frame_counts_validated=0
    with UR5MountedDepthCamera(model,camera_names=('ur5_depth_wrist','ur5_depth_forearm'),width=160,height=90,
        pixel_stride=1,minimum_range=.07,optical_depth_error_bound=.003,occluding_self_filter=True,native_raycast=True) as camera:
        for pose_index,q in enumerate(poses):
            set_configuration(model,data,q)
            captured=camera.capture(data)
            logged=frame_by_cycle[int(saved_sources['source_cycles'][pose_index])]
            assert sum(len(o.points) for o in captured)==int(logged['raw_depth_points']), 'Raw camera count mismatch'
            raw_frame_counts_validated+=1
            for obs in captured:
                mask=environment_endpoint_mask(obs)
                p=obs.points[mask]; U=obs.sample_uncertainty_shapes[mask]
                rawkeys=np.floor((p-np.array([-1.2,-1.2,0]))/.0075).astype(int)
                indices=np.array([cell_index.get(tuple(k),-1) for k in rawkeys])
                for s in selected:
                    hit=np.isin(indices,list(s['assigned']))
                    if not np.any(hit): continue
                    selected_indices=indices[hit]; ph=p[hit]; Uh=U[hit]; d=s['d']
                    support=np.sqrt(np.maximum(np.einsum('i,nij,j->n',d,Uh,d),0))
                    displacement=(ph-points[selected_indices])@d
                    s['count']+=len(ph); s['seen'].update(selected_indices.tolist())
                    s['measurement_support_max']=max(s['measurement_support_max'],float(max(support)))
                    s['raw_relative_cell_bound']=max(s['raw_relative_cell_bound'],float(max(np.abs(displacement)+support)))
                    s['raw_proxy_directional_support']=max(s['raw_proxy_directional_support'],float(max((ph-np.array(s['pair']['proxy_center_m']))@d+support)))
    result=[]
    for s in selected:
        pair=s['pair']; d=s['d']; assigned=np.array(sorted(s['assigned']),dtype=int)
        Umax=float(max(np.sqrt(np.maximum(np.einsum('i,nij,j->n',d,cells[assigned],d),0)))) if len(assigned) else None
        result.append(dict(robot_index=pair['robot_index'],proxy_id=pair['proxy_id'],box=pair['box'],
            assigned_cells=len(assigned),raw_seen_cells=len(s['seen']),matched_raw_samples=s['count'],
            raw_measurement_support_max_mm=1000*s['measurement_support_max'],
            raw_measurement_plus_cell_relocation_bound_mm=1000*s['raw_relative_cell_bound'],
            stored_cell_U_max_support_mm=1000*Umax if Umax is not None else None,
            proxy_U_support_mm=pair['uncertainty_support_along_normal_mm'],
            core_support_mm=pair['core_support_along_normal_mm'],
            final_Q_plus_U_support_mm=pair['core_support_along_normal_mm']+pair['uncertainty_support_along_normal_mm'],
            replayed_assigned_raw_union_support_mm=1000*s['raw_proxy_directional_support']))
    record=dict(seed=item['seed'],source_pose_count=len(poses),raw_frame_counts_validated=raw_frame_counts_validated,pairs=result)
    output.append(record)
    print(json.dumps(record),flush=True)
    (OUT/'uncertainty_decomposition.json').write_text(json.dumps(dict(diagnostic_only=True,
        note='Replayed published native-ray camera poses. Directional raw union of assigned cells is diagnostic, not a replacement full-direction certificate.',runs=output),indent=2),encoding='utf-8')

