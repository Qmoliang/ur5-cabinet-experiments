"""Exploratory F0: same recorded inputs, fixed maps, common terminal start."""
from pathlib import Path
from types import SimpleNamespace
import json
from collections import Counter
import numpy as np,mujoco
from model import set_configuration,attachment_position,build_robot_certificate,certificate_world_positions
from diagnose_et30_terminal import make_controller,proxy_object
from formal_protocol_v3_viewer import RaggedSnapshots
from pointcloud_proxy import minkowski_outer_shapes
from official_drawer_07_3 import scene
from experiment_07_3_geometry import ROOT,distances
OUT=ROOT/'formal_results/experiment_07/development_07_4'

def main():
    row=next(r for r in json.loads((ROOT/'formal_results/experiment_07/development_07_3/results.json').read_text())['online'] if r['group']=='ET30' and r['seed']==0);src=Path(row['run']);summary=json.loads((src/'summary.json').read_text());m=mujoco.MjModel.from_xml_path(str(src/'scene.xml'));d=mujoco.MjData(m);robot=build_robot_certificate(m);q0=np.load(src/'q_history.npy')[-1];s=scene();target=np.array(s.waypoints[-1]);radii=np.array([r.radius for r in robot])
    _,a=RaggedSnapshots(src/'causal_proxy_snapshots.npz').at_cycle(2249);legacy=proxy_object(a)
    a=np.load(OUT/'full_manager_same_input_seed0.npz');Q=a['Q'];U=a['U'];base=np.sqrt(np.linalg.eigvalsh(Q)[:,-1]);sphere=base+np.sqrt(np.linalg.eigvalsh(U)[:,-1]);new=SimpleNamespace(proxy_ids=np.arange(len(Q)),centers=a['centers'],sphere_radii=sphere,base_sphere_radii=base,base_ellipsoid_shapes=Q,ellipsoid_outer_shapes=minkowski_outer_shapes(Q,U),proxy_uncertainty_shapes=U,proxy_offset_radii=np.zeros(len(Q)))
    result=[]
    for group,p in [('ET30_frozen',legacy),('D30_same_input_frozen',new)]:
        q=q0.copy();set_configuration(m,d,q);ctrl,index=make_controller(m,d,s,robot,p,summary);hist=[q.copy()];trace=[];hold=0;first=None;minbox=np.inf;penetration=0;states=Counter();initial=float(np.linalg.norm(target-attachment_position(m,d)))
        for k in range(1000):
            velocity,metrics=ctrl.solve(target);states[metrics.status]+=1;q=q+.02*velocity;set_configuration(m,d,q);hist.append(q.copy());error=float(np.linalg.norm(target-attachment_position(m,d)))
            contact=any(d.contact[j].dist < -1e-8 for j in range(d.ncon));penetration+=int(contact);minbox=min(minbox,float(distances(certificate_world_positions(d,robot),s.boxes,radii).min()));hold=hold+1 if error<.001 and not contact else 0
            if hold>=50 and first is None:first=(k+1)*.02
            if k%50==0 or k==999:trace.append(dict(extra_s=(k+1)*.02,error_mm=1000*error,min_proxy_h_mm=1000*metrics.min_clearance))
        index.close();record=dict(group=group,source_run=str(src),diagnostic_only=True,new_observations=False,formal_success=False,initial_error_mm=initial*1000,final_error_mm=error*1000,hold50_time_s=first,penetrating_cycles=penetration,minimum_discrete_box_gap_mm=minbox*1000,statuses=dict(states),trace=trace);result.append(record)
        folder=OUT/group;folder.mkdir(exist_ok=True);np.save(folder/'q_history.npy',np.array(hist));(folder/'scene.xml').write_bytes((src/'scene.xml').read_bytes());(folder/'summary.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        (OUT/'frozen_map_diagnostic.json').write_text(json.dumps(dict(exploratory=True,same_recorded_inputs=True,runs=result),indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in record.items() if k!='trace'}),flush=True)
if __name__=='__main__':main()
