"""Dispatch one fixed configuration; retain original algorithms in isolated runtimes."""
from pathlib import Path
import argparse,json,sys,shutil,traceback,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parent
ONLINE_CASES={'O01':'liu_sphere','O02':'liu_ellipsoid','O03':'neo_sphere_300','O04':'neo_ellipsoid_300','O05':'neo_sphere_46','O06':'neo_ellipsoid_46','A01':'neo_sphere_300','A02':'neo_ellipsoid_300','A03':'neo_ellipsoid_46'}

def dump(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('case');ap.add_argument('--smoke',type=int);args=ap.parse_args();case=args.case
    output=ROOT/('smoke' if args.smoke else 'results')/case
    output.mkdir(parents=True,exist_ok=False)
    start=time.time();source=None;summary=None;error=None
    try:
        if case.startswith(('K','N')):
            rt=ROOT/'runtimes/known';sys.path[:0]=[str(rt),str(rt/'src'),str(rt/'src/core'),str(rt/'external_neo'),str(rt/'external_neo/sphere_study')]
            import run as original
            if case in ('K01','K02'):
                rep='sphere' if case=='K01' else 'ellipsoid';summary=original.run(rep,maximum_cycles=args.smoke or 3000);source=rt/'results'/rep
            elif case=='K03':
                sys.path.insert(0,str(rt/'tests'));import run_sphere_same_qp_slsqp as validation
                validation.main();source=rt/'results/sphere_same_qp_slsqp_validation'
                shutil.copy2(rt/'assets/drawer.xml',source/'scene.xml')
                summary=json.loads((source/'summary.json').read_text(encoding='utf-8'))
            elif case in ('N01','N02','N03'):
                import run_comparison as comparison
                key={'N01':'liuqp','N02':'neo_matched','N03':'neo_paper_influence'}[case]
                summary=comparison.run_case(key,rt/'external_neo/results',args.smoke or 3000);source=rt/'external_neo/results'/key
            else:
                import experiment as sphere_neo
                key={'N04':'neo_sphere_matched','N05':'neo_sphere_paper'}[case]
                summary=sphere_neo.run_case(key);source=rt/'external_neo/sphere_study/results'/key
        elif case.startswith('H'):
            rt=ROOT/'runtimes/historical';sys.path.insert(0,str(rt));import run as historical
            import psutil
            rep='ellipsoid' if case=='H01' else 'sphere';pipeline=historical.load_case(rep);kwargs=historical.arguments(rep)
            if args.smoke:kwargs['maximum_cycles']=args.smoke
            dump(output/'arguments.json',kwargs)
            p=psutil.Process();old=p.cpu_affinity();p.cpu_affinity(list(kwargs['control_cpu_affinity']))
            try:summary=pipeline.run_one(rep,'mvt_simd',output,**kwargs)
            finally:p.cpu_affinity(old)
        else:
            rt=ROOT/'runtimes/online';adapter=rt/'external_neo_adaptive';sys.path[:0]=[str(adapter),str(rt)]
            key=ONLINE_CASES[case]
            if case.startswith('A'):
                sys.path.insert(0,str(adapter/'ablation_no_manip'));import run_ablation as runner
                source=adapter/'ablation_no_manip/results'/key;sys.argv=[str(runner.__file__),key]
            else:
                import run_study as runner
                source=adapter/('smoke' if args.smoke else 'results')/key
                sys.argv=[str(runner.__file__),key]+(['--smoke',str(args.smoke)] if args.smoke else [])
            try:runner.main()
            except SystemExit as e:
                if e.code:raise RuntimeError(f'Original runner exit {e.code}')
            if (source/'summary.json').exists():summary=json.loads((source/'summary.json').read_text(encoding='utf-8'))
    except Exception:
        error=traceback.format_exc();print(error,flush=True)
    finally:
        if source is not None and source.exists():shutil.copytree(source,output,dirs_exist_ok=True)
        if summary is not None:
            dump(output/'summary.json',summary)
            if summary.get('failure'):error=summary['failure']
        dump(output/'run_identity.json',dict(case=case,new_physical_scene=True,planned_cycles=args.smoke or 3000,source_runtime=str(rt),original_case=case,elapsed_wall_s=time.time()-start,scene_sha256=hashlib.sha256((ROOT/'assets/scene.xml').read_bytes()).hexdigest(),completed=summary is not None and error is None,error=error))
    if error:raise SystemExit(1)
    print(case,'FINISHED',flush=True)

if __name__=='__main__':main()
