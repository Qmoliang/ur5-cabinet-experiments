"""Explain final NEO constraints without changing any saved control trajectory."""
import json
import numpy as np
import mujoco
from scipy.optimize import linprog
from neo_online import HERE,BASE,OnlineNEO,quadprog
from model import build_robot_certificate,set_configuration,certificate_world_state
from protocol_drawer_scene import formal_drawer_camera_quarter_scene
from run_study import CASES,dump

def main():
    scene=formal_drawer_camera_quarter_scene();target=np.array(scene.waypoints[-1]);reports={}
    for case,(method,rep,influence) in CASES.items():
        if method!='neo':continue
        outer=HERE/'results'/case
        completion=json.loads((outer/'completion.json').read_text(encoding='utf-8'))
        if not completion['complete']:
            q=np.load(outer/'neo_pre_step_q.npy')
            reports[case]=dict(completed_commands=len(q)-1,failed_solve_index=len(q)-1,
                               abort_simulation_time_s=(len(q)-1)*.02,
                               partial_trace_saved=True,
                               limitation='Failed QP matrices and active proxy snapshot were not saved; infeasibility cause cannot be isolated retrospectively.')
            continue
        summary=json.loads((outer/'summary.json').read_text(encoding='utf-8'));folder=outer/summary['run_name']
        model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));data=mujoco.MjData(model)
        set_configuration(model,data,np.load(folder/'q_history.npy')[-1]);robot=build_robot_certificate(model)
        z=np.load(folder/'causal_proxy_snapshots.npz');sl=slice(z['offsets'][-2],z['offsets'][-1])
        OnlineNEO.influence=influence
        ctrl=OnlineNEO(model,data,scene,robot,z['centers'][sl],representation=rep,
                       obstacle_radii=z['sphere_radii'][sl] if rep=='sphere' else None,
                       obstacle_shapes=z['ellipsoid_shapes'][sl] if rep=='ellipsoid' else None,
                       obstacle_uncertainty_shapes=z['proxy_uncertainty_shapes'][sl] if rep=='ellipsoid' else None,
                       obstacle_offsets=z['uncertainty_offsets'][sl],proxy_ids=z['proxy_ids'][sl])
        H,g,A,lo,hi,info=ctrl.assemble(target)
        # Maximal instantaneous velocity component towards goal under same hard constraints.
        direction=(target-info['ee'])/np.linalg.norm(target-info['ee']);objective=-direction@info['J']
        up=np.isfinite(hi);low=np.isfinite(lo)
        lp=linprog(objective,A_ub=np.vstack([A[up],-A[low]]),b_ub=np.r_[hi[up],-lo[low]],
                   bounds=[(None,None)]*6,method='highs')
        nz=np.linalg.norm(A,axis=1)>1e-14
        aa=A[nz];ll=lo[nz];hh=hi[nz];up2=np.isfinite(hh);low2=np.isfinite(ll)
        C=np.vstack([-aa[up2],aa[low2]]).T; b=np.r_[-hh[up2],ll[low2]]
        original_u=quadprog.solve_qp(H,-g,C,b)[0]
        task_only_g=-info['J'].T@info['v']
        without_manip=quadprog.solve_qp(H,-task_only_g,C,b)[0]
        one_step=dict(original_neo_qdot_norm=float(np.linalg.norm(original_u)),
                      original_neo_goal_progress_m_s=float(direction@info['J']@original_u),
                      without_manipulability_qdot_norm=float(np.linalg.norm(without_manip)),
                      without_manipulability_goal_progress_m_s=float(direction@info['J']@without_manip),
                      without_manipulability_constraint_violation=float(max(0,np.max(lo-A@without_manip),np.max(A@without_manip-hi))),
                      scope='One QP at the saved final state, with only manipulability linear objective term removed; not an executed trajectory or tuned success trial.')
        pos,jac,r=certificate_world_state(model,data,robot);ri,oi,d,n,_,_=ctrl.geometry(pos,r)
        nearest=[]
        for k in np.argsort(d)[:12]:
            j=oi[k]
            item=dict(robot_sphere=int(ri[k]),robot_body=robot[ri[k]].body_name,
                      proxy_id=int(ctrl.proxy_ids[j]),surface_gap_mm=float(d[k]*1000),
                      center_m=ctrl.obstacle_centers[j].tolist(),normal=n[k].tolist())
            if rep=='sphere':item['effective_radius_mm']=float((ctrl.obstacle_radii[j]+ctrl.obstacle_offsets[j])*1000)
            else:
                item.update(core_axes_mm=(np.sqrt(np.maximum(np.linalg.eigvalsh(ctrl.obstacle_shapes[j]),0))*1000).tolist(),
                            uncertainty_axes_mm=(np.sqrt(np.maximum(np.linalg.eigvalsh(ctrl.obstacle_uncertainty_shapes[j]),0))*1000).tolist(),
                            scalar_residual_mm=float(ctrl.obstacle_offsets[j]*1000))
            nearest.append(item)
        diag=json.loads((outer/'neo_diagnostics.json').read_text(encoding='utf-8'))[-1]
        result=dict(final_ee_m=info['ee'].tolist(),final_error_mm=float(np.linalg.norm(info['ee']-target)*1000),
                    final_task_slack_m_s=diag['slack_norm'],closest_pairs=nearest,one_step_objective_diagnostic=one_step,
                    instantaneous_progress_lp_success=bool(lp.success),
                    maximum_instantaneous_goal_progress_m_s=float(-lp.fun) if lp.success else None,
                    scope='Local velocity feasibility at one posture, not a global path-existence test.')
        reports[case]=result;ctrl.close()
    dump(HERE/'results/final_diagnosis.json',reports)
    print(json.dumps({k:{x:y for x,y in v.items() if x!='closest_pairs'} for k,v in reports.items()}),flush=True)

if __name__=='__main__':main()
