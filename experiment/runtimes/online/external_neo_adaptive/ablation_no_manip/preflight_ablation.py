import json
import numpy as np
import mujoco
from run_ablation import HERE,PARENT,CASES,NoManipNEO
from neo_online import quadprog
from model import build_robot_certificate,set_configuration
from protocol_drawer_scene import formal_drawer_camera_quarter_scene
from run_study import dump

def main():
    old=json.loads((PARENT/'results/final_diagnosis.json').read_text(encoding='utf-8'));checks=[]
    scene=formal_drawer_camera_quarter_scene();target=np.array(scene.waypoints[-1])
    for case in CASES:
        outer=PARENT/'results'/case;s=json.loads((outer/'summary.json').read_text(encoding='utf-8'));folder=outer/s['run_name']
        model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));data=mujoco.MjData(model)
        set_configuration(model,data,np.load(folder/'q_history.npy')[-1]);robot=build_robot_certificate(model)
        z=np.load(folder/'causal_proxy_snapshots.npz');sl=slice(z['offsets'][-2],z['offsets'][-1]);rep=s['representation']
        NoManipNEO.influence=s['neo_influence_m']
        ctrl=NoManipNEO(model,data,scene,robot,z['centers'][sl],representation=rep,
                        obstacle_radii=z['sphere_radii'][sl] if rep=='sphere' else None,
                        obstacle_shapes=z['ellipsoid_shapes'][sl] if rep=='ellipsoid' else None,
                        obstacle_uncertainty_shapes=z['proxy_uncertainty_shapes'][sl] if rep=='ellipsoid' else None,
                        obstacle_offsets=z['uncertainty_offsets'][sl],proxy_ids=z['proxy_ids'][sl])
        u,m=ctrl.solve(target);H,g,A,lo,hi,info=ctrl.ablation_last_assembly
        direction=(target-info['ee'])/np.linalg.norm(target-info['ee']);speed=float(direction@info['J']@u)
        expected=old[case]['one_step_objective_diagnostic']['without_manipulability_goal_progress_m_s']
        assert abs(speed-expected)<1e-7
        assert np.max(A@u-hi)<2e-6 and np.max(lo-A@u)<2e-6
        checks.append(dict(case=case,progress_m_s=speed,prior_diagnostic_m_s=expected,
                           objective_only_change_verified=True,constraint_violation=m.qp_primal_residual));ctrl.close()
    dump(HERE/'preflight.json',dict(passed=True,checks=checks));print(json.dumps(checks),flush=True)

if __name__=='__main__':main()
