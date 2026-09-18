"""Replay every published source frame through the full 07.4 proxy manager."""
from pathlib import Path
import json,time
from dataclasses import asdict
import numpy as np,mujoco
from model import set_configuration
from depth_camera_perception import UR5MountedDepthCamera,environment_endpoint_mask
from incremental_proxy_manager import IncrementalMatchedProxyManager
from run_protocol_v3_async_online import PROXY_WORKSPACE_LOWER,PROXY_WORKSPACE_UPPER,PROXY_CLUSTER_SIZE,MAXIMUM_AABB_OVERSHOOT
from diagnose_et30_terminal import read_csv
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'formal_results/experiment_07/development_07_4'

def main():
    source=next(x for x in json.loads((ROOT/'formal_results/experiment_07/development_07_3/results.json').read_text())['online'] if x['group']=='ET30' and x['seed']==0);src=Path(source['run']);poses=np.load(src/'causal_source_configurations.npz');logs={int(x['source_cycle']):int(x['raw_depth_points']) for x in read_csv(src/'perception_frames.csv')}
    m=mujoco.MjModel.from_xml_path(str(src/'scene.xml'));d=mujoco.MjData(m)
    manager=IncrementalMatchedProxyManager(filter_size=.0075,cluster_size=PROXY_CLUSTER_SIZE,maximum_aabb_overshoot=MAXIMUM_AABB_OVERSHOOT,maximum_uncertainty_union_inflation=1.05,certificate_radius_limit=.07,radius_limit_representation='ellipsoid',uncertainty_fusion_mode='raw_directional_uncertainty',ellipsoid_cover_mode='adaptive_irredundant',minimum_core_semi_axis=.003)
    frames=[]
    with UR5MountedDepthCamera(m,camera_names=('ur5_depth_wrist','ur5_depth_forearm'),width=160,height=90,pixel_stride=1,minimum_range=.07,optical_depth_error_bound=.003,occluding_self_filter=True,native_raycast=True) as camera:
        for i,q in enumerate(poses['q']):
            set_configuration(m,d,q);observations=camera.capture(d);assert sum(len(o.points) for o in observations)==logs[int(poses['source_cycles'][i])]
            ps=[];us=[];rs=[]
            for o in observations:
                mask=environment_endpoint_mask(o)&np.all(o.points>=PROXY_WORKSPACE_LOWER,axis=1)&np.all(o.points<=PROXY_WORKSPACE_UPPER,axis=1);ps.append(o.points[mask]);us.append(o.sample_uncertainty_shapes[mask]);rs.append(np.zeros(int(mask.sum())))
            st=time.perf_counter();stats=manager.update(np.concatenate(ps),np.concatenate(rs),np.concatenate(us));audit=manager.coverage_audit();elapsed=(time.perf_counter()-st)*1000;assert audit.all_raw_sample_balls_certified
            row=dict(frame=i,source_cycle=int(poses['source_cycles'][i]),manager_and_audit_ms=elapsed,proxies=len(manager.snapshot.centers),spatial_cells=len(manager._spatial_cells),directional_leaves=len(manager._cells),raw_samples=manager.raw_directional_join.total_samples,coverage=asdict(audit),update_stats=asdict(stats));frames.append(row)
            (OUT/'full_manager_same_input_seed0.json').write_text(json.dumps(dict(source_run=str(src),mode='raw_directional_uncertainty',frames=frames),indent=2),encoding='utf-8')
            if i%5==0 or i==len(poses['q'])-1:print(json.dumps({k:row[k] for k in ['frame','manager_and_audit_ms','proxies','spatial_cells','directional_leaves']}),flush=True)
    s=manager.snapshot
    np.savez_compressed(OUT/'full_manager_same_input_seed0.npz',centers=s.centers,Q=s.base_ellipsoid_shapes,U=s.proxy_uncertainty_shapes,points=s.filtered_points,cell_U=s.filtered_uncertainty_shapes,owners=s.filtered_cluster_indices)
    print('Complete same-input manager replay passed',flush=True)
if __name__=='__main__':main()
