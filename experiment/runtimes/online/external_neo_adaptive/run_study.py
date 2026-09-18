"""Run NEO / LiuQP on the original v4.3 / v4.4c online experiment pipeline."""
from pathlib import Path
import sys,json,hashlib,shutil,argparse,traceback
import numpy as np
import psutil
HERE=Path(__file__).resolve().parent; BASE=HERE.parent
sys.path.insert(0,str(BASE))
import run_protocol_v3_async_online as online
import run_protocol_v3_online_ablation as common
from run_strict_goal_v4_3_surface_core import _affinities
from neo_online import OnlineNEO

CASES={
    'neo_sphere_300':('neo','sphere',.30),
    'neo_ellipsoid_300':('neo','ellipsoid',.30),
    'neo_sphere_46':('neo','sphere',.046),
    'neo_ellipsoid_46':('neo','ellipsoid',.046),
    'liu_sphere':('liu','sphere',None),
    'liu_ellipsoid':('liu','ellipsoid',None),
}
def dump(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('case',choices=CASES);ap.add_argument('--smoke',type=int);ap.add_argument('--label',default='')
    args=ap.parse_args();method,representation,influence=CASES[args.case]
    output=HERE/('smoke' if args.smoke else 'results')/(args.case+args.label)
    output.mkdir(parents=True,exist_ok=False)
    hashes=online._source_hashes()
    dump(output/'original_source_hashes.json',hashes)
    for path in [HERE/'neo_online.py',HERE/'run_study.py',HERE/'plan/experiment-protocol.md']:
        destination=output/'source'/path.relative_to(HERE);destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,destination)
    dump(output/'adapter_source_hashes.json',{str(p.relative_to(HERE)):sha(p) for p in [HERE/'neo_online.py',HERE/'run_study.py',HERE/'plan/experiment-protocol.md']})
    controllers=[]; oldfactory=online._new_controller; oldwriter=online._write_csv
    def writer(path,rows):
        if method=='neo' and path.name=='pair_states.csv' and not rows:
            from protocol_liuqp_controller import PairRecord
            path.write_text(','.join(('cycle','time_s','active_generation',*PairRecord._fields))+'\n',encoding='utf-8')
        else: oldwriter(path,rows)
    def factory(rep,model,data,scene,robot,proxies,index,**kw):
        ctrl=OnlineNEO(model,data,scene,robot,proxies.centers,representation=rep,
                       obstacle_radii=proxies.sphere_radii if rep=='sphere' else None,
                       obstacle_shapes=proxies.base_ellipsoid_shapes if rep=='ellipsoid' else None,
                       obstacle_uncertainty_shapes=common._controller_uncertainty_shapes(proxies) if rep=='ellipsoid' else None,
                       obstacle_offsets=proxies.proxy_offset_radii,proxy_ids=proxies.proxy_ids,
                       safety_margin=.006,obstacle_index=index,**kw)
        controllers.append(ctrl); return ctrl
    if method=='neo':
        OnlineNEO.influence=influence; online._new_controller=factory;online._write_csv=writer
    control,control_compute,publication,perception=_affinities('perception_heavy')
    process=psutil.Process(); affinity=process.cpu_affinity()
    if control: process.cpu_affinity(list(control))
    summary=None; failure=None
    try:
        summary=online.run_one(representation,'mvt_simd',output,maximum_cycles=args.smoke or 3000,
            unknown_policy='observed_only',realtime_pacing=True,perception_executor='process',
            centervox_size=.0075,maximum_uncertainty_union_inflation=1.05,certificate_radius_limit=.070,
            uncertainty_fusion_mode='separate_uncertainty',direct_thin_axis_inflation=1.,
            direct_tangent_subdivisions=1,direct_partition_mode='grid',
            sphere_cover_mode='adaptive_irredundant' if representation=='sphere' else 'matched',
            ellipsoid_cover_mode='matched',radius_limit_representation='both',
            camera_width=320,camera_height=180,camera_pixel_stride=1,
            camera_names=('ur5_depth_wrist','ur5_depth_forearm'),scene_version='camera_quarter',
            record_dense_map_snapshots=True,sweep_guard_mode='audit_only',
            success_tolerance=.001,success_hold_cycles=50,stop_on_success=False,
            perception_cpu_affinity=perception,control_cpu_affinity=control,
            ellipsoid_pair_threads=8,control_compute_cpu_affinity=control_compute,publication_cpu_affinity=publication,
            task_gain=2.,max_task_speed=.18,osqp_adaptive_row_threshold=512,osqp_tolerance=1e-4,osqp_max_iterations=1000)
        summary.update(experiment='adaptive_online_external_controller_comparison',controller_method=method,
                       neo_influence_m=influence,comparison_scope='independent causal online trajectories, not identical pointcloud',
                       historical_baselines_unmodified=True)
        if method=='neo':
            # Replace legacy bookkeeping claims that describe Liu's implementation.
            for key in list(summary):
                if 'osqp' in key: del summary[key]
            summary['qp_solver']='quadprog 0.1.13 with logged same-QP SLSQP fallback'
            summary['neo_pair_decisions_saved_as_liu_records']=False
            summary['neo_controller_mvt_independently_built_and_audited']=True
            summary['neo_lui_near_penalty_used']=False
            for key in ['liu_qp_command_executed_without_guard_modification']:
                if key in summary: summary['controller_command_executed_without_guard_modification']=summary.pop(key)
            summary['proxy_and_eigensystem_update_off_control_thread']=False
            summary['neo_controller_index_update_on_control_thread']=True
            summary['neo_fallback_cycles']=sum(x['fallback'] for x in controllers[0].diagnostics)
            summary['neo_maximum_constraint_violation']=max(x['violation'] for x in controllers[0].diagnostics)
        run_dir=output/summary['run_name']
        dump(run_dir/'summary.json',summary);dump(output/'summary.json',summary)
    except Exception:
        failure=traceback.format_exc();dump(output/'failure.json',dict(case=args.case,error=failure))
        print(failure,flush=True)
    finally:
        process.cpu_affinity(affinity);online._new_controller=oldfactory;online._write_csv=oldwriter
        for ctrl in controllers:
            if summary is None and hasattr(ctrl,'failed_qp_candidate'):
                H,g,A,lo,hi,info=ctrl.failed_qp_candidate
                np.savez_compressed(output/'failed_qp.npz',H=H,g=g,A=A,lo=lo,hi=hi,
                    q=ctrl.data.qpos[:ctrl.nv].copy(),J=info['J'],v=info['v'],
                    centers=ctrl.obstacle_centers,offsets=ctrl.obstacle_offsets,proxy_ids=ctrl.proxy_ids,
                    radii=ctrl.obstacle_radii if ctrl.representation=='sphere' else np.empty(0),
                    shapes=ctrl.obstacle_shapes if ctrl.representation=='ellipsoid' else np.empty((0,3,3)),
                    uncertainty=ctrl.obstacle_uncertainty_shapes if ctrl.representation=='ellipsoid' else np.empty((0,3,3)))
            dump(output/'neo_diagnostics.json',ctrl.diagnostics);dump(output/'neo_mvt_updates.json',ctrl.index_audits)
            np.save(output/'neo_pre_step_q.npy',np.array(ctrl.trace_q));ctrl.close()
        assert online._source_hashes()==hashes,'Original sources changed'
        dump(output/'completion.json',dict(case=args.case,complete=summary is not None,smoke=bool(args.smoke),failure=failure,
                                         original_sources_unchanged=True))
    if summary:
        keys=['controller_method','representation','success','final_error_m','cycles','final_proxy_count',
              'controller_ms_p99','exact_penetrating_cycles','sweep_audit_failures','observability_late_samples',
              'observability_never_observed_samples','evidence_eligible','neo_fallback_cycles']
        print(json.dumps({k:summary.get(k) for k in keys}),flush=True)
    if failure: raise SystemExit(1)

if __name__=='__main__':main()
