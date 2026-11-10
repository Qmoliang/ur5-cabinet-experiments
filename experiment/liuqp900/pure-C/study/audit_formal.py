from pathlib import Path
import json,time,subprocess,os,queue,concurrent.futures,traceback,datetime
R=Path(__file__).resolve().parent;jobs=json.loads((R/'FORMAL_JOBS.json').read_text());slots=queue.Queue()
for base in [48,52,56,60]:slots.put(','.join(map(str,range(base,base+4))))
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6')
def audit(j):
 f=Path(j['folder']);cpu=slots.get();started=time.monotonic()
 try:
  with (f/'audit_driver.log').open('w') as log:p=subprocess.run(['/home/hsjed/envs/ur5cab/bin/python','-u',str(R/'audit_run.py'),j['runtime'],str(f),cpu],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=7500)
  if not (f/'VALIDATION.json').exists():raise RuntimeError('Audit did not write validation, exit='+str(p.returncode))
  v=json.loads((f/'VALIDATION.json').read_text());print(json.dumps(v),flush=True);return v
 except Exception as e:
  out=dict(id=j['id'],arm='C',strict_arrival=False,physical_safe_arrival=False,audit_error=repr(e),traceback=traceback.format_exc(),audit_wall_s=time.monotonic()-started);(f/'VALIDATION.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True);return out
 finally:slots.put(cpu)
pending=list(jobs);futures=[];results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 while pending or futures:
  for j in pending[:]:
   if (Path(j['folder'])/'launcher.json').exists():futures.append(pool.submit(audit,j));pending.remove(j)
  for f in futures[:]:
   if f.done():results.append(f.result());futures.remove(f);p=R/'FORMAL_AUDITS.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(results,indent=2));os.replace(tmp,p)
  if pending or futures:time.sleep(5)
(R/'AUDIT_COMPLETE.json').write_text(json.dumps(dict(runs=len(results),strict_arrivals=sum(x.get('strict_arrival',False) for x in results),completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
