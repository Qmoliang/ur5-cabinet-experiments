"""07.3 development runs; no changes to frozen 07.2 artifacts."""
from pathlib import Path
from datetime import datetime
import argparse,json,hashlib,sys,platform
import numpy as np
from run_protocol_v3_async_online import run_one
from official_drawer_07_3 import scene
from run_experiment_07 import FORMAL_INITIAL_OFFSETS_RAD
ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--groups',nargs='+',default=['ET30','J30'],choices=['ET30','J30','I30']);ap.add_argument('--seeds',nargs='+',type=int,default=list(range(5)));args=ap.parse_args()
    batch=ROOT/'formal_results/experiment_07/development_07_3'/('smoke_' if args.smoke else 'online_')/datetime.now().strftime('%Y%m%d_%H%M%S');batch.mkdir(parents=True,exist_ok=False)
    sources=['model.py','official_drawer_07_3.py','raw_support_intervals.py','incremental_proxy_manager.py','pointcloud_proxy.py','protocol_liuqp_controller.py','run_protocol_v3_async_online.py','run_experiment_07_3.py','plan/experiment-07-3-protocol.md']
    frozen=batch/'source';frozen.mkdir()
    hashes={}
    for name in sources:
        content=(ROOT/name).read_bytes();hashes[name]=hashlib.sha256(content).hexdigest();(frozen/Path(name).name).write_bytes(content)
    manifest=dict(experiment='07.3',development_only=True,mock_data=False,command=sys.argv,python=sys.version,platform=platform.platform(),protocol='plan/experiment-07-3-protocol.md',sources_sha256=hashes,results=[])
    path=batch/'manifest.json';path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for seed in args.seeds:
        # Alternate order across seeds; timing remains descriptive for this development run.
        groups=args.groups if seed%2==0 else args.groups[::-1]
        for group in groups:
            mode={'ET30':'separate_uncertainty','J30':'raw_join_uncertainty','I30':'support_interval_uncertainty'}[group]
            out=batch/group/f'seed{seed}';out.mkdir(parents=True)
            print(json.dumps(dict(start_group=group,seed=seed,output=str(out))),flush=True)
            summary=run_one('ellipsoid','mvt_simd',out,maximum_cycles=50 if args.smoke else None,
                unknown_policy='observed_only',realtime_pacing=not args.smoke,perception_executor='thread' if args.smoke else 'process',
                centervox_size=.0075,maximum_uncertainty_union_inflation=1.05,certificate_radius_limit=.070,
                uncertainty_fusion_mode=mode,direct_thin_axis_inflation=1.,minimum_core_semi_axis=.003,
                direct_tangent_subdivisions=1,direct_partition_mode='grid',sphere_cover_mode='matched',ellipsoid_cover_mode='adaptive_irredundant',
                radius_limit_representation='ellipsoid',camera_width=160,camera_height=90,camera_pixel_stride=1,camera_names=('ur5_depth_wrist','ur5_depth_forearm'),
                scene_version='exp07_drawer_official',record_dense_map_snapshots=True,sweep_guard_mode='audit_only',
                success_tolerance=.001,success_hold_cycles=50,stop_on_success=False,ellipsoid_pair_threads=6,
                control_process_priority='normal',perception_process_priority='normal',task_gain=2.,max_task_speed=.18,
                osqp_adaptive_row_threshold=512,osqp_tolerance=1e-4,osqp_max_iterations=1000,
                initial_configuration=np.array(scene().q0)+FORMAL_INITIAL_OFFSETS_RAD[seed])
            record=dict(group=group,seed=seed,summary=str(out/summary['run_name']/'summary.json'))
            for key in ['success','ever_sustained_success','final_error_m','evidence_eligible','final_proxy_count','deadline_misses','all_centervox_coverage_checks_passed','all_mvt_oracle_checks_passed']:
                record[key]=summary.get(key)
            actual=list(out.rglob('summary.json'));assert len(actual)==1;record['summary']=str(actual[0]);manifest['results'].append(record);path.write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(record),flush=True)
    print('BATCH_MANIFEST='+str(path),flush=True)
if __name__=='__main__':main()
