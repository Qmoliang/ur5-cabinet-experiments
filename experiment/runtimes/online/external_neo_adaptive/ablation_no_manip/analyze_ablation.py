"""Audit complete trajectories and independently test captured failed QPs."""
import json,sys
import numpy as np
import mujoco
from scipy.optimize import linprog
from run_ablation import HERE,PARENT,CASES
sys.path.insert(0,str(PARENT))
import audit_study
from run_study import CASES as ORIGINAL_CASES,dump
from model import build_robot_certificate,set_configuration,attachment_position,certificate_world_state
from protocol_drawer_scene import formal_drawer_camera_quarter_scene

def main():
    audit_study.HERE=HERE;audit_study.CASES={case:ORIGINAL_CASES[case] for case in CASES};audit_study.main()
    report={};target=np.array(formal_drawer_camera_quarter_scene().waypoints[-1])
    for case in CASES:
        folder=HERE/'results'/case;done=json.loads((folder/'completion.json').read_text(encoding='utf-8'))
        if done['complete']:continue
        z=np.load(folder/'failed_qp.npz');A=z['A'];lo=z['lo'];hi=z['hi']
        history=np.load(folder/'neo_pre_step_q.npy');scene_folder=next(folder.glob('A-CV-*'))
        model=mujoco.MjModel.from_xml_path(str(scene_folder/'scene.xml'));data=mujoco.MjData(model)
        robot=build_robot_certificate(model);set_configuration(model,data,z['q'])
        positions,jac,radii=certificate_world_state(model,data,robot)
        first_collision=6+3+len(robot)*6
        def feasible(rows):
            mat=A[rows];lower=lo[rows];upper=hi[rows];u=np.isfinite(upper);l=np.isfinite(lower)
            x=linprog(np.zeros(6),A_ub=np.vstack([mat[u],-mat[l]]),b_ub=np.r_[upper[u],-lower[l]],bounds=[(None,None)]*6,method='highs')
            return dict(feasible=bool(x.success),status=int(x.status),message=x.message)
        result=dict(completed_commands=len(history)-1,abort_time_s=(len(history)-1)*.02,
                    last_error_mm=float(np.linalg.norm(attachment_position(model,data)-target)*1000),
                    all_hard_constraints=feasible(slice(None)),
                    without_obstacle_constraints=feasible(slice(0,first_collision)),
                    obstacle_constraints_only=feasible(slice(first_collision,None)),
                    total_rows=len(A),obstacle_rows=len(A)-first_collision)
        if len(z['radii']):
            d=np.linalg.norm(positions[:,None,:]-z['centers'][None,:,:],axis=2)-radii[:,None]-z['radii'][None,:]-z['offsets'][None,:]
            result['minimum_proxy_gap_mm']=float(d.min()*1000)
            result['nearest_robot_body']=robot[np.unravel_index(np.argmin(d),d.shape)[0]].body_name
        penetrating=0;errors=[]
        for q in history:
            set_configuration(model,data,q);errors.append(float(np.linalg.norm(attachment_position(model,data)-target)*1000))
            penetrating+=int(any(data.contact[t].dist < -1e-8 for t in range(data.ncon)))
        result.update(partial_trace_states=len(history),minimum_partial_error_mm=min(errors),partial_trace_penetrating_states=penetrating)
        report[case]=result
    dump(HERE/'results/failure_diagnosis.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
