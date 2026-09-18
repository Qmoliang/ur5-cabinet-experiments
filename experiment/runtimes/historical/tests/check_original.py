"""Offline regression against each accepted original, not an online result."""
from pathlib import Path
import argparse
import inspect
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0,str(ROOT))
from run import CASES, arguments, load_case


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',choices=tuple(CASES))
    args=parser.parse_args()
    module=load_case(args.case)
    from robot import build_model,build_robot_certificate,set_configuration,DT
    from scene import formal_drawer_camera_quarter_scene
    from online_helpers import _new_controller
    import mujoco
    baseline=ROOT/'baselines'/CASES[args.case]
    kwargs=arguments(args.case)
    parameters=inspect.signature(module.PerceptionWorker).parameters
    worker=module.PerceptionWorker(args.case,'mvt_simd',**{k:v for k,v in kwargs.items() if k in parameters})
    snapshot_file=np.load(baseline/'causal_proxy_snapshots.npz')
    configurations=np.load(baseline/'causal_source_configurations.npz')
    q_expected=np.load(baseline/'q_history.npy')
    results=[]
    packet=None
    try:
        for generation in range(2):
            packet=worker.process(configurations['q'][generation],generation,
                int(configurations['source_cycles'][generation]),0.0,include_dense_snapshot=False)
            proxies=packet.proxies
            start,stop=map(int,snapshot_file['offsets'][generation:generation+2])
            fields={'proxy_ids':proxies.proxy_ids,'centers':proxies.centers,
                    'sphere_radii':proxies.sphere_radii,'base_sphere_radii':proxies.base_sphere_radii,
                    'ellipsoid_shapes':proxies.base_ellipsoid_shapes,
                    'ellipsoid_outer_shapes':proxies.ellipsoid_outer_shapes,
                    'proxy_uncertainty_shapes':proxies.proxy_uncertainty_shapes,
                    'uncertainty_offsets':proxies.proxy_offset_radii}
            equal={key:bool(np.array_equal(snapshot_file[key][start:stop],value)) for key,value in fields.items()}
            assert all(equal.values()),(generation,equal)
            results.append({'generation':generation,'proxy_count':len(proxies.centers),'exact_fields':equal})
            if generation==0:
                scene=formal_drawer_camera_quarter_scene();model=build_model(scene);data=mujoco.MjData(model)
                set_configuration(model,data,np.asarray(scene.q0))
                spheres=build_robot_certificate(model)
                controller=_new_controller(args.case,model,data,scene,spheres,proxies,packet.mvt)
                controller.task_gain=kwargs['task_gain'];controller.max_task_speed=kwargs['max_task_speed']
                controller.osqp_absolute_tolerance=kwargs['osqp_tolerance']
                controller.osqp_relative_tolerance=kwargs['osqp_tolerance']
                controller.osqp_max_iterations=kwargs['osqp_max_iterations']
                q_differences=[]
                for cycle in range(20):
                    velocity,metrics=controller.solve(np.asarray(scene.waypoints[-1]))
                    set_configuration(model,data,data.qpos[:6]+DT*velocity)
                    q_differences.append(float(np.max(np.abs(data.qpos[:6]-q_expected[cycle]))))
                assert max(q_differences)<1e-9,q_differences
                results[-1]['first_20_q_max_abs_difference_rad']=max(q_differences)
            packet.close();packet=None
    finally:
        if packet is not None:packet.close()
        worker.close();snapshot_file.close();configurations.close()
    report={'case':args.case,'offline_only':True,'all_passed':True,'snapshots':results}
    out=ROOT/'tests/results'/f'{CASES[args.case]}.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        assert json.loads(out.read_text(encoding='utf-8')) == report
    else:
        with out.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
