"""Same recorded source frames: legacy final cell U versus single-outer intervals."""
import json
from pathlib import Path
import numpy as np
import mujoco
from raw_support_intervals import RawSupportIntervals,RawLoewnerJoin
from diagnose_et30_terminal import ROOT,read_csv
from model import set_configuration
from depth_camera_perception import UR5MountedDepthCamera,environment_endpoint_mask
OUT=ROOT/'formal_results/experiment_07/development_07_3'

def run(mode="intervals"):
    OUT.mkdir(parents=True,exist_ok=True);results=[]
    diagnoses=json.loads((ROOT/'formal_results/experiment_07/terminal_diagnosis/terminal_constraints.json').read_text())['runs']
    for item in diagnoses:
        src=Path(item['source_run']);a=np.load(src/'final_causal_proxies.npz');points=a['filtered_points'];old=a['filtered_uncertainty_shapes'];keys=np.floor((points-[-1.2,-1.2,0])/.0075).astype(int);lookup={tuple(k):i for i,k in enumerate(keys)};acc=RawLoewnerJoin() if mode=="join" else RawSupportIntervals()
        m=mujoco.MjModel.from_xml_path(str(src/'scene.xml'));d=mujoco.MjData(m);poses=np.load(src/'causal_source_configurations.npz');logged={int(r['source_cycle']):int(r['raw_depth_points']) for r in read_csv(src/'perception_frames.csv')};frames=0;seen=0
        with UR5MountedDepthCamera(m,camera_names=('ur5_depth_wrist','ur5_depth_forearm'),width=160,height=90,pixel_stride=1,minimum_range=.07,optical_depth_error_bound=.003,occluding_self_filter=True,native_raycast=True) as camera:
            for qi,q in enumerate(poses['q']):
                set_configuration(m,d,q);obs=camera.capture(d);assert sum(len(o.points) for o in obs)==logged[int(poses['source_cycles'][qi])];frames+=1
                ps=[];us=[]
                for o in obs:
                    mask=environment_endpoint_mask(o);ps.append(o.points[mask]);us.append(o.sample_uncertainty_shapes[mask])
                p=np.concatenate(ps);u=np.concatenate(us);rawkeys=np.floor((p-[-1.2,-1.2,0])/.0075).astype(int);keep=np.array([tuple(k) in lookup for k in rawkeys]);p=p[keep];u=u[keep];rawkeys=rawkeys[keep]
                uniq,inv=np.unique(rawkeys,axis=0,return_inverse=True);kt=[tuple(k) for k in uniq];rep=points[[lookup[k] for k in kt]];order=np.argsort(inv,kind='stable');first=np.r_[0,np.flatnonzero(np.diff(inv[order]))+1]
                acc.update(p,u,kt,inv,rep,order,first);seen+=len(p)
        assert len(acc.cells)==len(points)
        new=[]
        for k in keys:
            _,R,h,_=acc.cells[tuple(k)];new.append((R*(3*h*h))@R.T)
        new=np.array(new);pairs=[]
        for pair in item['tight_pairs']:
            if pair['dual']<.01:continue
            oi=int(np.flatnonzero(a['proxy_ids']==pair['proxy_id'])[0]);assigned=a['filtered_cluster_indices']==oi;direction=np.array(pair['normal'])
            old_support=np.sqrt(np.einsum('i,nij,j->n',direction,old[assigned],direction));new_support=np.sqrt(np.einsum('i,nij,j->n',direction,new[assigned],direction))
            pairs.append(dict(robot_index=pair['robot_index'],box=pair['box'],proxy_id=pair['proxy_id'],old_cell_max_mm=float(old_support.max()*1000),new_cell_max_mm=float(new_support.max()*1000),median_cell_change_mm=float(np.median(new_support-old_support)*1000)))
        axes=np.eye(3);oldvol=np.sqrt(np.maximum(np.linalg.det(old),0));newvol=np.sqrt(np.maximum(np.linalg.det(new),0));record=dict(mode=mode,minimum_psd_slack=getattr(acc,"minimum_psd_slack",None),seed=item['seed'],frames=frames,matched_samples=seen,cells=len(points),minimum_raw_interval_slack=acc.minimum_interval_slack,max_box_corner_value=acc.max_box_corner_value,median_volume_ratio=float(np.median(newvol/oldvol)),fraction_cells_smaller_volume=float(np.mean(newvol<oldvol)),pairs=pairs)
        results.append(record);np.savez_compressed(OUT/f'same_input_{mode}_cells_seed{item["seed"]}.npz',points=points,old_U=old,new_U=new)
        (OUT/f'same_input_{mode}_fusion.json').write_text(json.dumps(dict(diagnostic_only=True,first_frame_axes=True,uses_same_legacy_stable_representatives=True,runs=results),indent=2),encoding='utf-8');print(json.dumps(record),flush=True)
if __name__=='__main__':
    import sys
    run(sys.argv[1] if len(sys.argv)>1 else 'intervals')
