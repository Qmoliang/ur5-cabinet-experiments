from pathlib import Path
import sys,os,json,time,numpy as np
R=Path(__file__).resolve().parent;runtime=Path(sys.argv[1]);f=Path(sys.argv[2]);os.sched_setaffinity(0,set(map(int,sys.argv[3].split(','))));sys.path.insert(0,str(runtime))
import pilot_adapter
from original_row_certificate import build_assembler,check_rows,feasible
j=json.loads((f/'job.json').read_text());r=json.loads((f/'result.json').read_text());z=dict(np.load(f/'trajectory.npz'));q=z['q'];vel=z['velocity'];target=np.array(json.loads((f/'ready.json').read_text())['target']);asm=build_assembler(j['task'],j.get('representation','ellipsoid'));previous=np.load(j['start_snapshot'])['previous'] if j.get('start_snapshot') else np.zeros(6);maximum=0.;bad=[];infeasible=[];first=[];started=time.perf_counter();last_state=None;cache_hits=0
for k,v in enumerate(vel):
 key=q[k].tobytes()+previous.tobytes()
 if key!=last_state:arrays,ids,J=asm.rows(q[k],previous,target);last_state=key
 else:cache_hits+=1
 violation=check_rows(arrays,v);maximum=max(maximum,violation)
 if violation>1e-7:
  witness=feasible(arrays);bad.append(k)
  if not witness['feasible']:infeasible.append(k)
  if len(first)<20:first.append(dict(cycle=k,violation=violation,feasibility=witness))
 previous=v
backup=None
if (f/'final_backup.json').exists():
 data=json.loads((f/'final_backup.json').read_text());steps=data['steps'];current=q[-1].copy();previous=vel[-1].copy();violations=[];state_match=True
 for step in steps:
  sq=np.asarray(step['q']);sp=np.asarray(step['previous']);v=np.asarray(step['velocity']);state_match=state_match and np.array_equal(current,sq) and np.array_equal(previous,sp);arrays,ids,J=asm.rows(current,previous,target);violations.append(check_rows(arrays,v));current=current+.02*v;previous=v
 backup=dict(state_match=bool(state_match),maximum_original_row_violation=max(violations),ends_at_rest=bool(np.max(abs(previous))<1e-12),terminal_zero_witness=bool(steps[-1]['terminal']))
 arrays,ids,J=asm.rows(current,previous,target);backup['rest_zero_violation']=check_rows(arrays,np.zeros(6));backup['passed']=state_match and max(violations)<=1e-7 and backup['ends_at_rest'] and backup['rest_zero_violation']<=1e-7
else:
 arrays,ids,J=asm.rows(q[-1],previous,target);backup=dict(available=False,final_state_feasibility=feasible(arrays))
out=dict(passed=not bad and backup.get('passed',backup.get('final_state_feasibility',{}).get('feasible',False)),executed_states=len(vel),maximum_original_row_violation=maximum,original_row_violating_cycles=bad,original_qp_infeasible_cycles=infeasible,examples=first,final_backup=backup,row_tolerance=1e-7,source_hash=asm.source_hash,audit_wall_s=time.perf_counter()-started,identical_state_row_reuses=cache_hits,scope='fresh reconstruction of original rows at every executed state; actual next velocity is a feasibility witness; independent LP when that witness fails; final stored backup checked through rest')
(f/'original_feasibility_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(id=f.name,passed=out['passed'],states=len(vel),bad_rows=len(bad),infeasible_states=len(infeasible),maximum=maximum,backup=backup)),flush=True)
