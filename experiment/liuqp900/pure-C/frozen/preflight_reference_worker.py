from pathlib import Path
import os,sys,time,json,traceback
import numpy as np

ROOT=Path(__file__).resolve().parent; folder=Path(sys.argv[1]);job=json.loads((folder/'job.json').read_text())
os.sched_setaffinity(0,set(job.get('cpus',[48,49,50,51])));os.environ['UNIFIED_CASE_DIR']=str(folder)
import importlib
sys.modules['verified_qp']=importlib.import_module('verified_qp_recovery')
from pilot_adapter import build,online,POLICY,FORBIDDEN_COUNTS

def save(name,data):
    path=folder/name; temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)));os.replace(temp,path)

def event(**data):
    with (folder/'events.jsonl').open('a') as f:f.write(json.dumps(data,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')

def new_delay():
    return dict(remaining=0,current=None,pending=None,metrics=None,current_delta=np.zeros(6),pending_delta=np.zeros(6),current_mode='TRACK',pending_mode='TRACK')

def snapshot(ctx,k,label):
    ctl=ctx.controller;s=ctl.pilot_nominal;H,g,A,l,u=ctl.last_qp_arrays
    name=f'snapshot_{label}_{k:05d}'
    np.savez_compressed(folder/(name+'.npz'),q=s['q'],previous=s['applied_velocity'],target=s['target'],J=s['J'],desired=s['desired'],H=H,g=g,A=A,l=l,u=u,x=s['x'],y=s['y'],delta=ctl.pilot_delta)
    save(name+'.json',dict(cycle=k,label=label,row_ids=s['row_ids'],status=s['status'],prim_res=s['prim_res'],dual_res=s['dual_res'],controller_margin=ctl.safety_margin,task_weight=ctl.task_weight,velocity_regularization=ctl.velocity_regularization))
    return name

ctx=None; rows=[]; velocities=[]; proposed=[]; endpoints=[];errors=[];timings=[]; raw_speeds=[];snapshots=[]; delta_history=[];mode_history=[];realization_errors=[];status='not_started';hold=best=0;truth_bad=0;fresh_calls=0;delayed=0
boot=time.monotonic();policy=None
try:
    start=previous=None
    if job.get('initial_q_override') is not None:start=np.asarray(job['initial_q_override'],dtype=float)
    if job.get('start_snapshot'):
        z=np.load(job['start_snapshot']);start=z['q'];previous=z['previous']
    ctx=build(job['task'],job.get('representation','ellipsoid'),start,previous)
    if job.get('target_override') is not None:ctx.target=np.asarray(job['target_override'],dtype=float)
    if job.get('solver_profile') in ('accurate_common','verified_common'):
        # Shared numerical policy only: objective and constraints unchanged.
        ctx.controller.osqp_adaptive_row_threshold=10**9
        ctx.controller.osqp_absolute_tolerance=1e-7
        ctx.controller.osqp_relative_tolerance=1e-7
        ctx.controller.osqp_max_iterations=50000
        if job.get('solver_profile')=='verified_common':
            ctx.controller.osqp_max_iterations=3000
            ctx.controller.pilot_verified_fallback=True
    if job.get('verified_forward',False):ctx.controller.pilot_force_verified_forward=True
    if job.get('certified_auxiliary',False):ctx.controller.pilot_use_certified_auxiliary=True
    if job.get('method','baseline')!='baseline':
        if job.get('policy_version')=='goal_entry_v7':
            from goal_entry_policy_v7 import LowDimPolicy
        elif job.get('policy_version')=='goal_entry_v6':
            from goal_entry_policy_v6 import LowDimPolicy
        elif job.get('policy_version')=='goal_entry_v5':
            from goal_entry_policy_v5 import LowDimPolicy
        elif job.get('policy_version')=='goal_entry_v4':
            from goal_entry_policy_v4 import LowDimPolicy
        elif job.get('policy_version')=='goal_entry_v3':
            from goal_entry_policy_v3 import LowDimPolicy
        elif job.get('policy_version')=='goal_entry_v2':
            from goal_entry_policy_v2 import LowDimPolicy
        elif job.get('policy_version')=='goal_entry':
            from goal_entry_policy import LowDimPolicy
        elif job.get('policy_version')=='retained_blocker':
            from retained_blocker_policy import LowDimPolicy
        elif job.get('policy_version')=='blocker_v5':
            from blocker_policy_v5 import LowDimPolicy
        elif job.get('policy_version')=='blocker_v4':
            from blocker_policy_v4 import LowDimPolicy
        elif job.get('policy_version')=='blocker_v3':
            from blocker_policy_v3 import LowDimPolicy
        elif job.get('policy_version')=='blocker_v2':
            from blocker_policy_v2 import LowDimPolicy
        elif job.get('policy_version')=='blocker':
            from blocker_policy import LowDimPolicy
        elif job.get('policy_version')=='polished':
            from polished_policy import LowDimPolicy
        elif job.get('policy_version')=='viability':
            from viability_policy import LowDimPolicy
        elif job.get('policy_version')=='reserve_v3':
            from release_policy_v3 import LowDimPolicy
        elif job.get('policy_version')=='reserve_v2':
            from release_policy_v2 import LowDimPolicy
        elif job.get('policy_version')=='admission':
            from admission_policy import LowDimPolicy
        elif job.get('policy_version')=='reserve':
            from release_policy import LowDimPolicy
        elif job.get('policy_version')=='trace_v4':
            from trace_policy import LowDimPolicy
        elif job.get('policy_version')=='v4':
            from lowdim_policy_v4 import LowDimPolicy
        elif job.get('policy_version')=='v3':
            from lowdim_policy_v3 import LowDimPolicy
        else:
            from lowdim_policy import LowDimPolicy
        policy=LowDimPolicy(ctx,job,event);ctx.controller.pilot_objective_hook=policy
    q=ctx.q0.copy();previous=np.zeros(6) if previous is None else previous.copy();rows=[q.copy()]
    dt=POLICY['dt_s'];max_cycles=job.get('max_cycles',3000)
    save('ready.json',dict(setup_s=time.monotonic()-boot,proxies=len(ctx.arrays['centers']),observed=ctx.observed,frontier_boxes=ctx.boxes,self_pairs=len(ctx.selfrows.pairs),target=ctx.target.tolist(),q0=q.tolist(),planning_disabled=True,path_loaded=False,perception_scope='frozen causal published map from prior online run; no new sensing'))
    delay=new_delay();started=time.monotonic();status='motion_budget_exhausted';stall_saved=False;escape_saved=False;posture_saved=False;last_epoch=0;mode_flushes=0
    for k in range(max_cycles):
        if time.monotonic()-started>job.get('wall_limit_s',600):status='wall_budget_exhausted';break
        ctl=ctx.controller;ctl.applied_velocity=previous.copy();ctl.pilot_cycle=k
        if policy is not None:policy.observe(k,q,previous)
        if policy is not None and getattr(policy,'command_epoch',0)!=last_epoch:
            delay=new_delay();last_epoch=policy.command_epoch;mode_flushes+=1
            if np.max(np.abs(previous))>1e-8:policy.brake_pending=True
            event(event='stale_command_queue_flushed',cycle=k,epoch=last_epoch,mode=policy.mode)
        fresh=False
        if delay['remaining']>0:delay['remaining']-=1
        if delay['remaining']==0:
            if delay['pending'] is not None:
                delay['current'],delay['pending']=delay['pending'],None
                delay['current_delta']=delay['pending_delta'].copy();delay['current_mode']=delay['pending_mode']
            t=time.perf_counter();command,metrics=ctl.solve(ctx.target);elapsed=time.perf_counter()-t
            if policy is not None and getattr(policy,'command_epoch',0)!=last_epoch:
                delay=new_delay();last_epoch=policy.command_epoch;mode_flushes+=1
                if np.max(np.abs(previous))>1e-8:policy.brake_pending=True
                event(event='stale_command_queue_flushed',cycle=k,epoch=last_epoch,mode=policy.mode)
            timings.append(elapsed*1000);fresh_calls+=1;fresh=True;delay['metrics']=metrics
            ticks=max(1,int(np.ceil(elapsed/dt-1e-9))) if job.get('compute_delay',True) else 1
            if job.get('_preflight_delay_ticks') is not None:
                assert job.get('study_phase')=='preflight';ticks=job['_preflight_delay_ticks'][(fresh_calls-1)%len(job['_preflight_delay_ticks'])]
            if policy is not None and policy.expected is not None:
                difference=float(np.max(np.abs(ctl.pilot_executed_raw-policy.expected)));realization_errors.append(difference)
                if difference>1e-5:policy.terminal='qp_realization_mismatch';event(event='qp_realization_mismatch',cycle=k,difference=difference)
            if ticks<=1:
                delay['current']=command;delay['current_delta']=ctl.pilot_delta.copy();delay['current_mode']='TRACK' if policy is None else policy.mode
            else:
                delay['pending']=command;delay['remaining']=ticks;delay['pending_delta']=ctl.pilot_delta.copy();delay['pending_mode']='TRACK' if policy is None else policy.mode
            if k==0:snapshots.append(snapshot(ctx,k,'initial'))
            if policy is not None and np.any(ctl.pilot_delta) and not escape_saved:snapshots.append(snapshot(ctx,k,'escape'));escape_saved=True
            if policy is not None and policy.mode in ('POSTURE','MIXED_ESCAPE') and not posture_saved:snapshots.append(snapshot(ctx,k,'posture'));posture_saved=True
            if not str(metrics.status).lower().startswith('solved'):
                status='qp_not_converged';event(event=status,cycle=k,solver_status=str(metrics.status));break
        else:delayed+=1
        command=np.zeros(6) if delay['current'] is None else np.asarray(delay['current']).copy()
        applied_delta=delay['current_delta'].copy();applied_mode=delay['current_mode']
        if policy is not None and (policy.brake_pending or policy.terminal):
            command=previous-np.clip(previous,-POLICY['acceleration_limit']*dt,POLICY['acceleration_limit']*dt)
            delay=new_delay();applied_delta=np.zeros(6);applied_mode='BRAKE'
        lo,hi=ctl.joint_velocity_bounds()
        if policy is not None and hasattr(policy,'filter_memory'):
            memory_command=policy.filter_memory(q,previous,command,lo,hi)
            if np.max(np.abs(memory_command-command))>1e-10:
                delay=new_delay();ctl.previous_velocity=memory_command.copy()
            command=memory_command
        actual=ctx.guard.filter(ctx.proxy_guard,q,previous,command,lo,hi,k,ctx.metadata['generation'])
        modified=float(np.max(np.abs(actual-command)))>1e-10
        if modified:delay=new_delay();ctl.previous_velocity=actual.copy()
        q=q+dt*actual;online.set_configuration(ctx.model,ctx.data,q)
        ee=online.attachment_position(ctx.model,ctx.data).copy();error=float(np.linalg.norm(ee-ctx.target))
        rows.append(q.copy());velocities.append(actual.copy());proposed.append(command);endpoints.append(ee);errors.append(error);previous=actual.copy()
        delta_history.append(applied_delta);mode_history.append(applied_mode)
        raw_speeds.append(float(np.max(np.abs(ctl.pilot_nominal['x']))) if ctl.pilot_nominal['x'] is not None else float('inf'))
        if policy is not None:policy.feedback(k,q,actual,modified)
        if any(float(ctx.data.contact[i].dist)<-1e-8 for i in range(ctx.data.ncon)):
            truth_bad+=1;status='physical_contact_violation';event(event=status,cycle=k);break
        hold=hold+1 if error<POLICY['goal_tolerance_m'] else 0;best=max(best,hold)
        if fresh and not stall_saved and k>=100 and error>.002 and abs(errors[-100]-error)<.0002 and max(raw_speeds[-50:])<.002:
            snapshots.append(snapshot(ctx,k,'stall'));stall_saved=True
            event(event='real_stall_snapshot',cycle=k,error_m=error,max_raw_speed=max(raw_speeds[-50:]))
            if job.get('stop_after_stall',False):status='baseline_stall_captured';break
        if k%100==0:save('progress.json',dict(cycles=k+1,sim_s=(k+1)*dt,error_mm=1000*error,raw_speed=raw_speeds[-1],applied_speed=float(np.max(np.abs(actual))),guard_modified=ctx.guard.modified,wall_s=time.monotonic()-started,mode=None if policy is None else policy.mode))
        if hold>=POLICY['goal_hold_cycles']:status='safe_arrival_candidate';break
        if policy is not None and policy.terminal and np.max(np.abs(actual))<1e-8:status=policy.terminal;break
    if ctx is not None and fresh_calls and not stall_saved:snapshots.append(snapshot(ctx,k,'final'))
except Exception as exc:
    status='execution_error';save('error.json',dict(error=repr(exc),traceback=traceback.format_exc()))
finally:
    np.savez_compressed(folder/'trajectory.npz',q=np.asarray(rows),velocity=np.asarray(velocities),proposed_velocity=np.asarray(proposed),ee=np.asarray(endpoints),goal_error=np.asarray(errors),controller_ms=np.asarray(timings),nominal_raw_speed=np.asarray(raw_speeds),gradient_delta=np.asarray(delta_history),control_mode=np.asarray(mode_history),realization_errors=np.asarray(realization_errors))
    result=dict(task=job['task'],method=job.get('method','baseline'),representation=job.get('representation','ellipsoid'),status=status,goal_hold_reached=best>=50,maximum_hold=best,cycles=len(velocities),sim_s=len(velocities)*.02,wall_s=time.monotonic()-boot,final_error_mm=None if not errors else errors[-1]*1000,controller_calls=fresh_calls,controller_ms_p99=None if not timings else float(np.percentile(timings,99)),compute_delay_held_cycles=delayed,guard={} if ctx is None else ctx.guard.summary(),truth_penetrating_cycles=truth_bad,forbidden_calls=FORBIDDEN_COUNTS,snapshots=snapshots,policy_summary=None if policy is None else policy.summary(),scope='actual UR5e kinematic feedback, final Cartesian goal only, frozen original observed map, no path/planner/waypoint; independent audit pending')
    result['solver_profile']=job.get('solver_profile','original')
    result['verified_numerical_fallback_calls']=0 if ctx is None else int(getattr(ctx.controller,'pilot_fallback_calls',0))
    result['maximum_auxiliary_forward_difference']=None if not realization_errors else max(realization_errors)
    result['mode_command_queue_flushes']=locals().get('mode_flushes',0)
    result['direct_auxiliary_certificates']=None if ctx is None else getattr(ctx.controller,'pilot_certificate_maxima',None)
    save('result.json',result);print(json.dumps(result,default=str),flush=True)
