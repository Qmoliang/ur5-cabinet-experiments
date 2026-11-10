from pathlib import Path
import os,json,time,subprocess,concurrent.futures,queue,threading,hashlib,signal,traceback,datetime
R=Path(__file__).resolve().parent;PY='/home/hsjed/envs/ur5cab/bin/python';jobs=json.loads((R/'FORMAL_JOBS.json').read_text());freeze=json.loads((R/'FORMAL_FREEZE.json').read_text());slots=queue.Queue();lock=threading.Lock();records=[]
for k in range(12):slots.put(list(range(k*4,k*4+4)))
envbase=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2));os.replace(t,p)
for name,h in freeze['drivers'].items():assert digest(R/name)==h,name
for name,h in freeze['files'].items():assert digest(R/'frozen'/name)==h,name
assert digest(R/'FORMAL_JOBS.json')==freeze['jobs_sha256'] and digest(R/'PROTOCOL.json')==freeze['protocol_sha256']
def run(spec):
 cpus=slots.get();f=Path(spec['folder']);rt=Path(spec['runtime']);job=dict(spec['job']);worker=None
 try:
  tick=time.monotonic()
  for name,h in freeze['runtime_files'].items():assert digest(rt/name)==h,name
  assert json.loads((rt/'INPUTS.json').read_text())==[spec['input']]
  for name,key in [('input_file','input_sha256'),('scene','scene_sha256')]:assert digest(Path(spec['input'][name]))==spec['input'][key],name
  assert not (rt/'GOAL_PORTS.json').exists() and not (rt/'GOAL_BRANCH_PORTS.json').exists()
  integrity_s=time.monotonic()-tick;start=time.monotonic();deadline=start+1200;job.update(cpus=cpus,arm_start_monotonic=start,run_deadline_monotonic=deadline);assert not any(k.startswith('_preflight') for k in job);save(f/'job.json',job);env=dict(envbase,STUDY_CPUS=','.join(map(str,cpus)))
  worker_start=time.monotonic();timed=False;killed=False
  with (f/'worker.log').open('x') as log:
   p=subprocess.Popen([PY,'-u',str(rt/'pure_worker.py'),str(f)],stdout=log,stderr=subprocess.STDOUT,env=env,stdin=subprocess.DEVNULL)
   try:exit_code=p.wait(timeout=max(1,deadline-time.monotonic()+20))
   except subprocess.TimeoutExpired:
    timed=True;p.send_signal(signal.SIGINT)
    try:exit_code=p.wait(timeout=60)
    except subprocess.TimeoutExpired:p.kill();killed=True;exit_code=p.wait()
  worker=dict(exit_code=exit_code,wall_s=time.monotonic()-worker_start,supervisor_timeout=timed,forced_kill=killed)
  if not (f/'result.json').exists():
   save(f/'result.json',dict(task=job['task'],arm='C',method='baseline',status='worker_timeout' if timed else 'worker_process_failure',goal_hold_reached=False,cycles=0,sim_s=0.,wall_s=time.monotonic()-start,final_error_mm=None,within_total_compute_budget=False,goal_initialization_s=0,forbidden_calls={},baseline_forbidden_endpoint_reads=[],pure_original_objective=True,new_certified_executor_used=False));save(f/'error.json',dict(error='Worker did not produce a complete result',worker=worker))
  a=json.loads((f/'result.json').read_text());record=dict(id=spec['id'],case=spec['case'],task=spec['task'],seed=spec['seed'],arm='C',cpus=cpus,status=a['status'],goal_reached=a['goal_hold_reached'],sim_s=a['sim_s'],error_mm=a.get('final_error_mm'),worker=worker,goal_initialization_s=0,arm_total_wall_s=time.monotonic()-start,shared_integrity_s=integrity_s,shared_input_preparation_s=spec['input_preparation_s'])
  save(f/'launcher.json',record)
  with lock:records.append(record);save(R/'FORMAL_STATUS.json',records)
  print(json.dumps(record),flush=True);return record
 except Exception as e:
  save(R/('INTEGRITY_OR_PIPELINE_STOP_'+spec['id']+'.json'),dict(error=repr(e),traceback=traceback.format_exc()));raise
 finally:slots.put(cpus)
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
 for x in pool.map(run,jobs):pass
save(R/'BATCH_COMPLETE.json',dict(runs=len(records),completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));print('300 pure outcomes retained',flush=True)
