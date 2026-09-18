"""Run one of the two accepted original variants with the original settings."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib
import json
import os
import sys
import psutil

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
CASES = {'ellipsoid':'v43_ellipsoid', 'sphere':'v44_sphere'}


def load_case(case):
    chosen = ROOT / 'src' / CASES[case]
    loaded = sys.modules.get('pipeline')
    if loaded is not None and Path(loaded.__file__).resolve().parent != chosen:
        raise RuntimeError('Run the two original versions in separate processes')
    sys.path[:0] = [str(chosen), str(ROOT / 'src/common')]
    return importlib.import_module('pipeline')


def arguments(case):
    kwargs = json.loads((ROOT / 'configs/original.json').read_text(encoding='utf-8'))
    for key in ('camera_names','control_cpu_affinity','perception_cpu_affinity'):
        kwargs[key] = tuple(kwargs[key])
    if case == 'sphere':
        kwargs['sphere_cover_mode'] = 'adaptive_irredundant'
    return kwargs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',choices=tuple(CASES))
    parser.add_argument('--check',action='store_true',help='Check imports and frozen scene without starting a run')
    args = parser.parse_args()
    module = load_case(args.case)
    from robot import build_model, build_robot_certificate, build_xml
    from scene import formal_drawer_camera_quarter_scene
    from package_integrity import source_hashes, forbid_external_experiment_reads
    scene = formal_drawer_camera_quarter_scene()
    assert hashlib.sha256(build_xml(scene).encode()).hexdigest() == 'd0918f3492e2f502221cf25c9d94606f1d4e5f30d535ce97654c58136e71ac24'
    assert len(build_robot_certificate(build_model(scene))) == 65
    kwargs = arguments(args.case)
    before = source_hashes()
    if args.check:
        print(json.dumps({'case':args.case,'scene_matches_original':True,'robot_spheres':65,
                          'pipeline':module.__file__,'source_files':len(before)},indent=2))
        return
    if (os.cpu_count() or 0) < 32:
        raise RuntimeError('Original CPU layout needs the 32-logical-CPU experimental machine')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'runs' / CASES[args.case] / stamp
    output.mkdir(parents=True,exist_ok=False)
    (output/'arguments.json').write_text(json.dumps(kwargs,indent=2),encoding='utf-8')
    (output/'source_hashes.json').write_text(json.dumps(before,indent=2),encoding='utf-8')
    forbid_external_experiment_reads()
    process = psutil.Process()
    affinity,priority = process.cpu_affinity(),process.nice()
    print('NEW ONLINE RUN: '+str(output),flush=True)
    try:
        process.cpu_affinity(list(kwargs['control_cpu_affinity']))
        process.nice(psutil.NORMAL_PRIORITY_CLASS)
        summary = module.run_one(args.case,'mvt_simd',output,**kwargs)
    finally:
        process.cpu_affinity(affinity)
        process.nice(priority)
    assert source_hashes() == before
    print(json.dumps({'case':args.case,'final_error_mm':1000*summary['final_error_m'],
        'below_0_18mm':summary['final_error_m']<0.00018,'original_success':summary['success'],
        'cycles':summary['cycles'],'evidence_eligible':summary['evidence_eligible'],
        'output':str(output)},indent=2),flush=True)


if __name__ == '__main__':
    main()
