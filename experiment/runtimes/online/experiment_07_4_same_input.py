"""07.4 identical recorded observations: single cell U versus raw directional leaves."""
from pathlib import Path
import json,time
import numpy as np,mujoco
from raw_directional_uncertainty import RawDirectionalJoin
from depth_camera_perception import UR5MountedDepthCamera,environment_endpoint_mask
from model import set_configuration
from diagnose_et30_terminal import read_csv
from run_protocol_v3_async_online import PROXY_WORKSPACE_LOWER,PROXY_WORKSPACE_UPPER
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'formal_results/experiment_07/development_07_4'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    baseline=ROOT/'formal_results/experiment_07/development_07_3';meta=json.loads((baseline/'results.json').read_text());terms=json.loads((baseline/'terminal_constraints.json').read_text())['runs'];output=[]
    for row in sorted([r for r in meta['online'] if r['group']=='ET30'],key=lambda r:r['seed']):
        src=Path(row['run']);arc=np.load(src/'final_causal_proxies.npz');points=arc['filtered_points'];old=arc['filtered_uncertainty_shapes'];keys=np.floor((points-[-1.2,-1.2,0])/.0075).astype(int);lookup={tuple(k):i for i,k in enumerate(keys)};acc=RawDirectionalJoin()
        m=mujoco.MjModel.from_xml_path(str(src/'scene.xml'));d=mujoco.MjData(m);sources=np.load(src/'causal_source_configurations.npz');logs={int(x['source_cycle']):int(x['raw_depth_points']) for x in read_csv(src/'perception_frames.csv')};frames=0;discarded=0;timings=[]
        with UR5MountedDepthCamera(m,camera_names=('ur5_depth_wrist','ur5_depth_forearm'),width=160,height=90,pixel_stride=1,minimum_range=.07,optical_depth_error_bound=.003,occluding_self_filter=True,native_raycast=True) as camera:
            for qi,q in enumerate(sources['q']):
                set_configuration(m,d,q);obs=camera.capture(d);assert sum(len(o.points) for o in obs)==logs[int(sources['source_cycles'][qi])];frames+=1
                ps=[];us=[]
                for o in obs:
                    mask=environment_endpoint_mask(o)&np.all(o.points>=PROXY_WORKSPACE_LOWER,axis=1)&np.all(o.points<=PROXY_WORKSPACE_UPPER,axis=1);ps.append(o.points[mask]);us.append(o.sample_uncertainty_shapes[mask])
                p=np.concatenate(ps);U=np.concatenate(us);rawkeys=np.floor((p-[-1.2,-1.2,0])/.0075).astype(int);keep=np.array([tuple(k) in lookup for k in rawkeys]);discarded+=int(np.sum(~keep));p=p[keep];U=U[keep];rawkeys=rawkeys[keep]
                uniq,inverse=np.unique(rawkeys,axis=0,return_inverse=True);kt=[tuple(k) for k in uniq];reps=points[[lookup[k] for k in kt]];start=time.perf_counter();acc.update(p,U,kt,inverse,reps);timings.append((time.perf_counter()-start)*1000)
        assert len(acc.cells)==len(points) and discarded==0
        target=next(x for x in terms if x['group']=='ET30' and x['seed']==row['seed']);pairs=[]
        for pair in target['tight_pairs']:
            if pair['barrier_h_mm']>.05:continue
            oi=int(np.flatnonzero(arc['proxy_ids']==pair['proxy_id'])[0]);assigned=np.flatnonzero(arc['filtered_cluster_indices']==oi);n=np.array(pair['normal']);before=[];after=[]
            for i in assigned:
                before.append(np.sqrt(n@old[i]@n));after.append(max(np.sqrt(n@leaf['shape']@n) for leaf in acc.cells[tuple(keys[i])].values()))
            pairs.append(dict(box=pair['box'],robot_index=pair['robot_index'],proxy_id=pair['proxy_id'],cells=len(assigned),old_max_support_mm=float(max(before)*1000),leaf_union_max_support_mm=float(max(after)*1000),median_change_mm=float(np.median(np.array(after)-before)*1000)))
        counts=[len(x) for x in acc.cells.values()];record=dict(seed=row['seed'],source_run=str(src),frames=frames,matched_raw_samples=acc.total_samples,unmatched_samples=discarded,spatial_cells=len(points),leaves=sum(counts),leaves_per_cell_median=float(np.median(counts)),leaves_per_cell_p95=float(np.percentile(counts,95)),leaves_per_cell_max=max(counts),minimum_psd_slack=acc.minimum_psd_slack,accumulator_update_ms_median=float(np.median(timings)),pairs=pairs)
        flattened=[(lookup[key],code,l) for key,leaves in acc.cells.items() for code,l in leaves.items()]
        np.savez_compressed(OUT/f'same_input_seed{row["seed"]}.npz',cell_points=points,old_U=old,leaf_owners=np.array([x[0] for x in flattened]),leaf_codes=np.array([x[1] for x in flattened]),leaf_shapes=np.array([x[2]['shape'] for x in flattened]),leaf_counts=np.array([x[2]['count'] for x in flattened]))
        output.append(record);(OUT/'same_input.json').write_text(json.dumps(dict(development_data=True,runs=output),indent=2),encoding='utf-8');print(json.dumps(record),flush=True)
if __name__=='__main__':main()
