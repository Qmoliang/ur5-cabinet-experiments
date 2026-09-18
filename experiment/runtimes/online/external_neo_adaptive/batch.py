"""Sequential fixed cases: avoids perception/CPU contention between trials."""
from pathlib import Path
import subprocess,sys,json,time
HERE=Path(__file__).resolve().parent
CASES=['neo_sphere_300','neo_ellipsoid_300','neo_sphere_46','neo_ellipsoid_46','liu_sphere','liu_ellipsoid']
if __name__=='__main__':
    records=[]
    for case in CASES:
        print('START',case,flush=True); started=time.time()
        result=subprocess.run([sys.executable,'-B',str(HERE/'run_study.py'),case],cwd=str(HERE.parent))
        records.append(dict(case=case,exit_code=result.returncode,wall_s=time.time()-started))
        (HERE/'batch_progress.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        print('FINISHED',case,result.returncode,flush=True)
    print('BATCH COMPLETE',json.dumps(records),flush=True)
