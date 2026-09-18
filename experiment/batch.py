from pathlib import Path
import json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
CASES=['K01','K02','K03','H01','H02','N01','N02','N03','N04','N05','O01','O02','O03','O04','O05','O06','A01','A02','A03']
def main():
    (ROOT/'logs').mkdir(exist_ok=True);progress=[]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
    for case in CASES:
        stamp=ROOT/'results'/case/'run_identity.json'
        if stamp.exists():
            print('Existing attempt retained',case,flush=True);progress.append(json.loads(stamp.read_text(encoding='utf-8')));continue
        print('START',case,flush=True)
        (ROOT/'current_case.json').write_text(json.dumps(dict(case=case,start_unix=time.time())),encoding='utf-8')
        with (ROOT/'logs'/f'{case}.log').open('w',encoding='utf-8') as log:
            done=subprocess.run([sys.executable,'-B',str(ROOT/'run_case.py'),case],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        entry=json.loads(stamp.read_text(encoding='utf-8')) if stamp.exists() else dict(case=case,completed=False,error='Process terminated before identity was saved')
        entry['exit_code']=done.returncode;progress.append(entry)
        (ROOT/'batch_progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2),encoding='utf-8')
        print('END',case,'exit',done.returncode,flush=True)
    (ROOT/'current_case.json').write_text(json.dumps(dict(finished=True,attempts=len(progress))),encoding='utf-8')
    print('ALL 19 ATTEMPTS FINISHED',flush=True)
if __name__=='__main__':main()
