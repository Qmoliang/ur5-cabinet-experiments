from pathlib import Path
import sys,os,json,time,subprocess,numpy as np
R=Path(__file__).resolve().parent;rt=Path(sys.argv[1]);f=Path(sys.argv[2]);cpus=sys.argv[3];os.sched_setaffinity(0,set(map(int,cpus.split(','))));sys.path.insert(0,str(rt));started=time.monotonic()
from pilot_adapter import build,POLICY
j=json.loads((f/'job.json').read_text());r=json.loads((f/'result.json').read_text());out=dict(id=f.name,arm='C',status=r['status']);env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6');PY='/home/hsjed/envs/ur5cab/bin/python'
if not r.get('cycles',0) or not (f/'trajectory.npz').exists():
 out.update(executed=False,strict_arrival=False,physical_safe_arrival=False,physical=None,feasibility=None,memory=None,strict_memory=None,reason='no executed trajectory, retained failure')
else:
 with np.load(f/'trajectory.npz') as z:
  q=z['q'][-1].copy();prev=z['velocity'][-1].copy();assert len(z['q'])==r['cycles']+1;assert len(z['velocity'])==r['cycles'];assert r['sim_s']==r['cycles']*.02;assert np.max(abs(z['gradient_delta']))==0
 assert r.get('pure_original_objective') and not r.get('new_certified_executor_used') and not r.get('baseline_forbidden_endpoint_reads') and not r.get('forbidden_calls')
 # This witness is reconstructed strictly after the run. It is never fed to C.
 ctx=build(j['task'],start=q,previous=prev);speed=prev-np.clip(prev,-POLICY['acceleration_limit']*.02,POLICY['acceleration_limit']*.02);physical_tail=bool(ctx.guard.braking_safe(ctx.proxy_guard,q,speed));steps=[];current=q.copy();last=prev.copy()
 for k in range(11):
  terminal=bool(np.all(speed==0.) and np.all(last==0.));steps.append(dict(q=current.tolist(),previous=last.tolist(),velocity=speed.tolist(),terminal=terminal))
  if terminal:break
  current=current+.02*speed;last=speed.copy();speed=speed-np.clip(speed,-POLICY['acceleration_limit']*.02,POLICY['acceleration_limit']*.02)
 tail=dict(created_after_execution=True,used_by_controller=False,physical_braking_passed=physical_tail,terminal_rest_reached=bool(steps[-1]['terminal']));(f/'POSTHOC_BRAKING.json').write_text(json.dumps(tail,indent=2));(f/'final_backup.json').write_text(json.dumps(dict(steps=steps,provenance=tail),indent=2));del ctx
 for kind,script,extra,name in [('physical','audit_wrapper_mixed.py',['physical'],'audit'),('feasibility','audit_original_feasibility.py',[],'original_feasibility_audit')]:
  with (f/(name+'.log')).open('w') as log:p=subprocess.run([PY,str(R/script),str(rt),str(f),cpus,*extra],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
  data=json.loads((f/(name+'.json')).read_text()) if (f/(name+'.json')).exists() else {};out[kind]=bool(data.get('safety_passed' if kind=='physical' else 'passed',False));out[kind+'_exit']=p.returncode
 out['posthoc_braking']=bool(physical_tail and steps[-1]['terminal'])
 assert json.loads((f/'goalward_memories.json').read_text())==[]
 for kind,name in [('memory','memory_audit'),('strict_memory','strict_memory_audit')]:
  out[kind]=True;(f/(name+'.json')).write_text(json.dumps(dict(passed=True,memory_count=0,scope='No memories in C; not exercised')))
 launch=json.loads((f/'launcher.json').read_text()) if (f/'launcher.json').exists() else {};budget=bool(r.get('within_total_compute_budget',False) and r['sim_s']<=300 and launch.get('arm_total_wall_s',0)<=1200)
 out.update(executed=True,goal_reached=bool(r['goal_hold_reached']),budget_passed=budget,physical_safe_arrival=bool(r['goal_hold_reached'] and budget and out['physical']),strict_arrival=bool(r['goal_hold_reached'] and budget and out['physical'] and out['feasibility'] and out['posthoc_braking']))
 (f/'PURE_SEMANTICS.json').write_text(json.dumps(dict(passed=True,objective_hook=False,added_gradient_max=0,terminal_file_reads=0,new_executor=False,posthoc_witness_not_used_online=True),indent=2))
out['audit_wall_s']=time.monotonic()-started;(f/'VALIDATION.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
