"""Run fixed known-volume task with LiuQP and two NEO core configurations."""
from pathlib import Path
import sys,json,time,hashlib,shutil,platform,argparse
import numpy as np
import mujoco
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'src'/'core'))
import run as original
from neo_controller import NEOController
from robot import build_model,build_robot_certificate,set_configuration,attachment_position,DT
from known_volume import build_volume_cover,coverage_audit
from controller import ProtocolLiuQPController
from voxel_index import NativeMultilevelMVT

CASES={'liuqp':None,'neo_matched':.046,'neo_paper_influence':.3}
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def frozen_sources(output):
    paths=[HERE/'neo_controller.py',HERE/'run_comparison.py',HERE/'plan'/'experiment-protocol.md']
    paths += [ROOT/p for p in original.source_hashes()]
    manifest={}
    for p in paths:
        key=p.relative_to(ROOT).as_posix(); dst=output/'source'/key
        dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,dst); manifest[key]=digest(p)
    dump(output/'source_hashes.json',manifest)
    return manifest

def make_baseline(model,data,scene,robot,cover):
    radii=np.array([r.radius for r in robot]); padding=.04+.006+5e-6
    index=NativeMultilevelMVT(cover.centers,cover.aabb_half_extents+5e-6,
        base_voxel_size=.015,maximum_query_half_extent=float(max(radii)+padding),query_padding=padding,simd=True)
    ctrl=ProtocolLiuQPController(model=model,data=data,scene=scene,robot_spheres=robot,
         obstacle_centers=cover.centers,representation='ellipsoid',obstacle_shapes=cover.ellipsoid_shapes,
         obstacle_offsets=np.zeros(len(cover.centers)),proxy_ids=cover.proxy_ids,
         safety_margin=.006,near_distance=.04,contact_distance=0.,obstacle_index=index,
         redundant_plane_pruning=True,ellipsoid_pair_threads=8,ellipsoid_pair_affinity_mask=0)
    ctrl.task_gain=2.;ctrl.max_task_speed=.18;ctrl.osqp_adaptive_row_threshold=512
    ctrl.osqp_absolute_tolerance=1e-4;ctrl.osqp_relative_tolerance=1e-4;ctrl.osqp_max_iterations=6000
    return ctrl,index

def run_case(case,output,cycles):
    folder=output/case;folder.mkdir(parents=True,exist_ok=False)
    scene=original.load_scene(); cover=build_volume_cover(scene.boxes,'ellipsoid',.075)
    audit=coverage_audit(cover); assert audit['complete_solid_cell_volume_covered']
    old=np.load(ROOT/'results'/'ellipsoid'/'proxies.npz')
    for k in ['centers','ellipsoid_shapes','proxy_ids','cell_half_extents']:
        assert np.array_equal(old[k],getattr(cover,k))
    model=build_model(scene);data=mujoco.MjData(model);set_configuration(model,data,np.array(scene.q0))
    robot=build_robot_certificate(model);target=np.array(scene.waypoints[-1]); index=None
    if case=='liuqp': ctrl,index=make_baseline(model,data,scene,robot,cover)
    else: ctrl=NEOController(model,data,robot,cover,CASES[case])
    qs=[data.qpos[:6].copy()];ees=[attachment_position(model,data)];logs=[]
    hold=0; maxhold=0; first_entry=None;confirmed=None;started=time.perf_counter();failure=None
    try:
        for i in range(cycles):
            t0=time.perf_counter();u,metrics=ctrl.solve(target)
            metrics=metrics.as_dict() if hasattr(metrics,'as_dict') else metrics
            set_configuration(model,data,data.qpos[:6].copy()+DT*u)
            ee=attachment_position(model,data);err=float(np.linalg.norm(ee-target))
            distances=[float(data.contact[k].dist) for k in range(data.ncon)]
            pen=sum(d < -1e-8 for d in distances)
            hold=hold+1 if err<.001 and pen==0 else 0;maxhold=max(maxhold,hold)
            if hold==50 and confirmed is None: confirmed=(i+1)*DT;first_entry=(i-48)*DT
            logs.append(dict(metrics,cycle=i,time_s=(i+1)*DT,error_m=err,
                exact_contacts=int(data.ncon),penetrating_contacts=pen,
                min_contact_m=min(distances) if distances else None,
                control_wall_ms=1000*(time.perf_counter()-t0)))
            qs.append(data.qpos[:6].copy());ees.append(ee)
            if (i+1)%250==0: print(f'{case} {i+1}/{cycles} error={err*1000:.4f}mm hold={hold} status={metrics["status"]}',flush=True)
    except Exception as ex: failure=repr(ex)
    finally:
        if index is not None:index.close()
    np.save(folder/'q_history.npy',qs);np.save(folder/'ee_history.npy',ees)
    shutil.copyfile(ROOT/'assets'/'drawer.xml',folder/'scene.xml')
    shutil.copyfile(ROOT/'results'/'ellipsoid'/'proxies.npz',folder/'proxies.npz')
    dump(folder/'cycles.json',logs);dump(folder/'coverage.json',audit)
    wall=[r['control_wall_ms'] for r in logs]
    summary=dict(case=case,method='original LiuQP' if case=='liuqp' else 'NEO core, position-task reimplementation',
        cycles=len(logs),requested_cycles=cycles,simulation_duration_s=len(logs)*DT,wall_duration_s=time.perf_counter()-started,
        final_error_mm=float(np.linalg.norm(ees[-1]-target)*1000),minimum_error_mm=min([r['error_m']*1000 for r in logs],default=None),
        success=maxhold>=50,maximum_success_hold_cycles=maxhold,first_sustained_entry_time_s=first_entry,
        confirmed_success_time_s=confirmed,penetrating_cycles=sum(r['penetrating_contacts']>0 for r in logs),
        maximum_recorded_penetration_m=min([r['min_contact_m'] for r in logs if r['min_contact_m'] is not None]+[0.]),
        fallback_cycles=sum(bool(r.get('fallback',False)) for r in logs),
        nonsolved_status_cycles=sum('solved' not in r['status'].lower() and not r.get('fallback',False) for r in logs),
        maximum_qp_violation=max([r.get('qp_violation',0.) for r in logs],default=0.),
        wall_ms_p50=float(np.percentile(wall,50)) if wall else None,wall_ms_p99=float(np.percentile(wall,99)) if wall else None,
        robot_certificate_count=len(robot),proxy_count=len(cover.centers),safety_margin_m=.006,
        influence_surface_m=CASES[case],target_m=target.tolist(),q0=list(scene.q0),failure=failure,
        original_source_hashes=original.source_hashes(),python=platform.python_version(),mujoco=mujoco.__version__)
    dump(folder/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--cases',nargs='+',choices=list(CASES),default=list(CASES));args=ap.parse_args()
    output=HERE/('smoke' if args.smoke else 'results');output.mkdir(exist_ok=True)
    hashes=frozen_sources(output); results=[]
    for case in args.cases: results.append(run_case(case,output,100 if args.smoke else 3000))
    assert all(digest(ROOT/k)==v for k,v in hashes.items())
    dump(output/'comparison.json',results)
    if any(r['failure'] for r in results):raise SystemExit(1)
if __name__=='__main__': main()
