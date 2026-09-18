from pathlib import Path
import sys,subprocess,json,time
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    rows=[]
    for case in ['neo_sphere_300','neo_ellipsoid_300','neo_ellipsoid_46']:
        print('START',case,flush=True);start=time.time()
        result=subprocess.run([sys.executable,'-B',str(HERE/'run_ablation.py'),case],cwd=str(HERE.parents[1]))
        rows.append(dict(case=case,exit_code=result.returncode,wall_s=time.time()-start))
        (HERE/'batch_progress.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print('FINISHED',case,result.returncode,flush=True)
