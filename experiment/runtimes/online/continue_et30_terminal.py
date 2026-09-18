"""Frozen-map continuation diagnostics; no formal-run mutation or safety claim."""
import json
import numpy as np
import mujoco
from diagnose_et30_terminal import ROOT, OUT, read_csv, proxy_object, make_controller
from formal_protocol_v3_viewer import RaggedSnapshots
from model import build_robot_certificate, set_configuration, attachment_position
from run_protocol_v3_async_online import _protocol_scene

def continuation(row, disable_near, max_cycles):
    run=ROOT/row['run']
    summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    cycles=read_csv(run/'cycles.csv')
    _,arrays=RaggedSnapshots(run/'causal_proxy_snapshots.npz').at_cycle(int(cycles[-1]['cycle']))
    model=mujoco.MjModel.from_xml_path(str(run/'scene.xml'))
    data=mujoco.MjData(model)
    q=np.load(run/'q_history.npy')[-1].copy()
    set_configuration(model,data,q)
    scene=_protocol_scene(summary.get('camera_scene_version',summary['scene_version']))
    target=np.array(scene.waypoints[-1])
    robot=build_robot_certificate(model)
    ctrl,index=make_controller(model,data,scene,robot,proxy_object(arrays),summary)
    if disable_near: ctrl.near_weight=0
    initial=float(np.linalg.norm(target-attachment_position(model,data)))
    hold=0
    trace=[]
    first=None
    minimum_h=float('inf')
    penetrating_steps=0
    statuses={}
    for k in range(max_cycles):
        qdot,metrics=ctrl.solve(target)
        statuses[metrics.status]=statuses.get(metrics.status,0)+1
        minimum_h=min(minimum_h,metrics.min_clearance)
        q=q+0.02*qdot
        set_configuration(model,data,q)
        err=float(np.linalg.norm(target-attachment_position(model,data)))
        penetration=any(data.contact[c].dist < -1e-8 for c in range(data.ncon))
        penetrating_steps+=int(penetration)
        hold=hold+1 if err<0.001 and not penetration else 0
        if k%250==0 or k==max_cycles-1 or hold>=50:
            trace.append(dict(extra_time_s=(k+1)*0.02,error_mm=1000*err))
        if hold>=50:
            first=(k+1)*0.02
            break
    index.close()
    return dict(seed=int(row['seed']),variant='near_penalty_off' if disable_near else 'unchanged_controller',
        frozen_map=True,new_observations=False,continuous_sweep_audited=False,extra_time_s=(k+1)*0.02,
        initial_error_mm=1000*initial,final_error_mm=1000*err,first_hold50_s=first,
        minimum_start_of_step_h_mm=1000*minimum_h,discrete_penetrating_steps=penetrating_steps,
        statuses=statuses,trace=trace)

def main():
    results=[]
    rows=[r for r in read_csv(ROOT/'tables/experiment_07/T27_drawer_formal_runs_07_2.csv') if r['group']=='ET30']
    for disable_near in [False,True]:
        for row in rows:
            result=continuation(row,disable_near,1000 if disable_near else 3000)
            results.append(result)
            (OUT/'frozen_continuation.json').write_text(json.dumps(dict(diagnostic_only=True,runs=results),indent=2),encoding='utf-8')
            print(json.dumps({k:v for k,v in result.items() if k!='trace'}),flush=True)

if __name__=='__main__': main()
