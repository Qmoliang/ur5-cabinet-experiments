from pathlib import Path
import os,sys,time,json,traceback,importlib
import numpy as np
ROOT=Path(__file__).resolve().parent;folder=Path(sys.argv[1]);job=json.loads((folder/'job.json').read_text());os.sched_setaffinity(0,set(job['cpus']));os.environ['UNIFIED_CASE_DIR']=str(folder);sys.modules['verified_qp']=importlib.import_module('verified_qp_recovery')
from pilot_adapter import build,online,POLICY,FORBIDDEN_COUNTS
baseline_forbidden_reads=[]
if job.get('arm')=='A':
 def input_audit(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)) and str(args[0]).replace('\\','/').rsplit('/',1)[-1] in ['GOAL_PORTS.json','GOAL_BRANCH_PORTS.json']:
   baseline_forbidden_reads.append(str(args[0]));raise RuntimeError('Baseline must not read endpoint/posture data')
 sys.addaudithook(input_audit)
from goal_plane_memory_policy import LowDimPolicy

def save(name,data):
 p=folder/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)));os.replace(tmp,p)
physical_cycle=0;logical_cycle=0;observed_logical_cycle=-1;logical_to_physical={};events=[]
def event(**data):
 logical=data.get('cycle',logical_cycle);data=dict(data,policy_cycle=logical,cycle=physical_cycle)
 with (folder/'events.jsonl').open('a') as f:f.write(json.dumps(data,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')
boot=time.monotonic();ctx=policy=None;qs=[];vel=[];proposed=[];ees=[];errors=[];timings=[];deltas=[];modes=[];raw_speeds=[];decision_cycles=[];decision_ticks=[];decision_q=[];decision_v=[];decision_modes=[];applied_ids=[];hold=best=0;status='not_started';command_blocks=0;held_compute_cycles=0;certificate_cycles=0;control_calls=0;trace=[];forward_errors=[]
try:
 start=previous=None
 if job.get('start_snapshot'):
  z=np.load(job['start_snapshot']);start=z['q'];previous=z['previous']
 ctx=build(job['task'],job.get('representation','ellipsoid'),start,previous);c=ctx.controller;c.osqp_adaptive_row_threshold=10**9;c.osqp_absolute_tolerance=1e-7;c.osqp_relative_tolerance=1e-7;c.osqp_max_iterations=3000;c.pilot_verified_fallback=True;c.pilot_use_certified_auxiliary=True
 policy=LowDimPolicy(ctx,job,event);c.pilot_objective_hook=policy;policy.last_delay_ticks=1
 q=ctx.q0.copy();previous=np.zeros(6) if previous is None else previous.copy();qs=[q.copy()];dt=POLICY['dt_s'];limit=job.get('max_cycles',15000);started=time.monotonic();status='motion_budget_exhausted';queue=[];wait_ticks=0;block_epoch=policy.command_epoch;current_delta=np.zeros(6);decision_id=-1;inherited_count=len(policy.memories)
 save('ready.json',dict(setup_s=time.monotonic()-boot,proxies=len(ctx.arrays['centers']),observed=ctx.observed,frontier_boxes=ctx.boxes,self_pairs=len(ctx.selfrows.pairs),target=ctx.target.tolist(),q0=q.tolist(),planning_disabled=True,path_loaded=False,perception_scope='same frozen historical observation map',execution='certified constant-command blocks; compute waits at a certified original-row-feasible rest state'))
 def install(cert):
  state=policy._key(cert['steps'][0]['q'],cert['steps'][0]['previous']);policy.hold_registry.setdefault(state,{})[cert['id']]=(policy.command_epoch,cert)
  policy.published_certificates[cert['id']]=(policy.command_epoch,cert)
 def apply(v,cert,compute_wait=False,mode=None):
  global q,previous,physical_cycle,logical_cycle,observed_logical_cycle,hold,best,status,held_compute_cycles,certificate_cycles
  c.applied_velocity=previous.copy()
  if not compute_wait:
   logical_to_physical.setdefault(logical_cycle,physical_cycle)
   if observed_logical_cycle!=logical_cycle:policy.observe(logical_cycle,q,previous);observed_logical_cycle=logical_cycle
  else:held_compute_cycles+=1
  if cert is not None:install(cert)
  lo,hi=c.joint_velocity_bounds();command=np.clip(v,lo,hi);checked=ctx.guard.filter(ctx.proxy_guard,q,previous,command,lo,hi,physical_cycle,ctx.metadata['generation']);actual=policy.filter_viability(q,previous,checked)
  if compute_wait and not np.array_equal(actual,np.zeros(6)):raise RuntimeError('Computation may wait only at an exactly certified rest state')
  modified=np.max(abs(actual-command))>1e-10;q=q+dt*actual;online.set_configuration(ctx.model,ctx.data,q);ee=online.attachment_position(ctx.model,ctx.data).copy();error=float(np.linalg.norm(ee-ctx.target));qs.append(q.copy());vel.append(actual.copy());proposed.append(command.copy());ees.append(ee);errors.append(error);deltas.append(current_delta.copy() if mode not in ('BRAKE','COMPUTE_HOLD') else np.zeros(6));modes.append(mode or policy.mode);raw_speeds.append(float(np.max(abs(c.pilot_nominal['x']))) if getattr(c,'pilot_nominal',None) is not None and c.pilot_nominal.get('x') is not None else 0.);applied_ids.append(decision_id if mode not in ('BRAKE','COMPUTE_HOLD') else -1);previous=actual.copy();c.previous_velocity=actual.copy()
  if not compute_wait:policy.feedback(logical_cycle,q,actual,modified);logical_cycle+=1
  if any(ctx.data.contact[i].dist<-1e-8 for i in range(ctx.data.ncon)):raise RuntimeError('Physical penetration detected')
  hold=hold+1 if error<POLICY['goal_tolerance_m'] else 0;best=max(best,hold);physical_cycle+=1
  if physical_cycle%100==1:save('progress.json',dict(cycles=physical_cycle,sim_s=physical_cycle*dt,error_mm=error*1000,wall_s=time.monotonic()-started,mode=policy.mode,logical_motion_cycles=logical_cycle,compute_hold_cycles=held_compute_cycles))
  if best>=50:status='safe_arrival_candidate'
  return modified
 while physical_cycle<limit and status!='safe_arrival_candidate':
  if time.monotonic()-started>job.get('wall_limit_s',1200) or time.monotonic()>=job.get('run_deadline_monotonic',float('inf')):status='wall_budget_exhausted';break
  if job.get('preflight_feedback_cycles') is not None and logical_cycle>=job['preflight_feedback_cycles']:status='interface_preflight_complete';break
  if queue and block_epoch!=policy.command_epoch:queue=[]
  if wait_ticks:
   first=policy.backup['steps'][0]
   if not first['terminal']:raise RuntimeError('Pending computation has no reusable rest certificate')
   apply(np.zeros(6),None,True,'COMPUTE_HOLD');wait_ticks-=1;continue
  if queue:
   cert=queue.pop(0);first=cert['steps'][0]
   if not np.array_equal(q,first['q']) or not np.array_equal(previous,first['previous']):raise RuntimeError('Certified block state mismatch')
   if apply(first['velocity'],cert,False,policy.mode):queue=[]
   continue
  # Finish the existing certified brake before computing another task command.
  first=policy.backup['steps'][0]
  if not first['terminal'] or not np.array_equal(previous,np.zeros(6)):
   apply(first['velocity'],None,False,'BRAKE');continue
  c.applied_velocity=previous.copy();c.pilot_cycle=logical_cycle;logical_to_physical.setdefault(logical_cycle,physical_cycle)
  if observed_logical_cycle!=logical_cycle:policy.observe(logical_cycle,q,previous);observed_logical_cycle=logical_cycle
  policy.expected=None;c.pilot_supplied_solution=None;policy.last_delay_ticks=1
  t=time.perf_counter();command,metrics=c.solve(ctx.target);control_calls+=1;decision_id=control_calls-1
  if not str(metrics.status).lower().startswith('solved'):raise RuntimeError('Original QP failed at certified rest: '+str(metrics.status))
  if policy.expected is not None:
   difference=float(np.max(abs(c.pilot_executed_raw-policy.expected)));forward_errors.append(difference)
   if difference>1e-5:raise RuntimeError('Auxiliary QP realization mismatch')
  command=np.zeros(6) if policy.brake_pending or policy.terminal else np.clip(command,*c.joint_velocity_bounds());current_delta=c.pilot_delta.copy();block_epoch=policy.command_epoch;certificates=[];future_q=q.copy();future_previous=previous.copy()
  # This is validation of a constant held velocity, never optimization of future states.
  for k in range(10 if np.max(abs(command))>1e-12 else 1):
   cert=policy._certify(future_q,future_previous,command)
   if cert is None:break
   certificates.append(cert);future_q=future_q+dt*command;future_previous=command.copy()
  if not certificates:
   command=np.zeros(6);cert=policy._certify(q,previous,command)
   if cert is None:raise RuntimeError('Original certified rest witness lost')
   certificates=[cert]
  # The policy may have committed a memory during this solve. Waiting must
  # use a new rest certificate under that same memory set.
  rest=policy._certify(q,previous,np.zeros(6))
  if rest is None:raise RuntimeError('New control state has no certified computation-wait rest')
  policy.backup=rest
  elapsed=time.perf_counter()-t;timings.append(elapsed*1000);ticks=max(1,int(np.ceil(elapsed/dt-1e-9)));decision_cycles.append(physical_cycle);decision_ticks.append(ticks);decision_q.append(q.copy());decision_v.append(command.copy());decision_modes.append(policy.mode);queue=certificates;wait_ticks=ticks-1;command_blocks+=1;certificate_cycles+=len(certificates)
  event(event='certified_constant_command_block',cycle=logical_cycle,steps=len(certificates),charged_compute_ticks=ticks,epoch=block_epoch)
except Exception as exc:
 status='execution_error';save('error.json',dict(error=repr(exc),traceback=traceback.format_exc()))
finally:
 if qs:
  np.savez_compressed(folder/'trajectory.npz',q=np.asarray(qs),velocity=np.asarray(vel),proposed_velocity=np.asarray(proposed),ee=np.asarray(ees),goal_error=np.asarray(errors),controller_ms=np.asarray(timings),nominal_raw_speed=np.asarray(raw_speeds),gradient_delta=np.asarray(deltas),control_mode=np.asarray(modes),realization_errors=np.asarray(forward_errors))
  np.savez_compressed(folder/'decision_trace.npz',cycles=np.asarray(decision_cycles),ticks=np.asarray(decision_ticks),q=np.asarray(decision_q),proposed=np.asarray(decision_v),modes=np.asarray(decision_modes),offsets=np.zeros(len(decision_cycles)+1,dtype=int),b=np.empty((0,6)),h=np.empty(0),fraction=np.empty(0),kind=np.empty(0),ids=np.empty((0,2),dtype=int),refresh=np.zeros(len(decision_cycles),dtype=int),applied_ids=np.asarray(applied_ids))
  np.savez_compressed(folder/'snapshot_actual_final.npz',q=qs[-1],previous=previous,target=ctx.target)
 summary=None if policy is None else policy.summary()
 if policy is not None:
  raw=json.loads(json.dumps(policy.memories));save('policy_memories_motion_clock.json',raw);mapped=[]
  for i,m in enumerate(raw):
   if i>=inherited_count:
    for field in ['hit_cycle','activation_cycle']:
     if field in m:m[field]=logical_to_physical[int(m[field])]
   mapped.append(m)
  save('goalward_memories.json',mapped);save('motion_clock_mapping.json',logical_to_physical)
 result=dict(task=job['task'],method=job.get('method','release'),representation=job.get('representation','ellipsoid'),status=status,goal_hold_reached=best>=50,maximum_hold=best,cycles=len(vel),sim_s=len(vel)*.02,wall_s=time.monotonic()-boot,final_error_mm=None if not errors else errors[-1]*1000,controller_calls=control_calls,controller_ms_p99=None if not timings else float(np.percentile(timings,99)),compute_delay_held_cycles=held_compute_cycles,guard={} if ctx is None or not ctx.guard.ms else ctx.guard.summary(),truth_penetrating_cycles=0,forbidden_calls=FORBIDDEN_COUNTS,snapshots=['snapshot_actual_final'] if qs else [],policy_summary=summary,solver_profile='verified_common',latency_model='measured solve plus block-certificate computation charged as certified zero-velocity waiting; outer 300s includes every hold',maximum_auxiliary_forward_difference=max(forward_errors,default=0.),scope='same kinematic feedback, observed map, safety and total budgets; deterministic constant-command blocks and a motion/feedback clock excluding only calculation waits; no route search',command_blocks=command_blocks,certified_hold_prefixes=certificate_cycles,policy_motion_cycles=logical_cycle)
 result.update(arm=job.get('arm'),baseline_forbidden_endpoint_reads=baseline_forbidden_reads,goal_initialization_s=job.get('goal_initialization_s',0.),arm_elapsed_s=time.monotonic()-job.get('arm_start_monotonic',boot),within_total_compute_budget=time.monotonic()<=job.get('run_deadline_monotonic',float('inf')))
 save('result.json',result);print(json.dumps({k:result[k] for k in ['status','sim_s','wall_s','final_error_mm','command_blocks']}),flush=True)
