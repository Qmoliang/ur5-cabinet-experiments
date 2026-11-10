from pathlib import Path
import os,sys,json,time
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
import numpy as np,mujoco
from pilot_adapter import ROOT,U,BASE,online,POLICY
sys.path.insert(0,str(BASE/'audit_tools'))
from continuous_robot_audit import audit_motion
from independent_geometry import IndependentGuard
from audit_shared_inputs import audit_workspace

folder=Path(sys.argv[1]);job=json.loads((folder/'job.json').read_text());entry=next(x for x in json.loads((ROOT/'INPUTS.json').read_text()) if x['task']==job['task'])
audit_cpus=list(map(int,sys.argv[2].split(','))) if len(sys.argv)>2 else job.get('cpus',[48,49,50,51])
os.sched_setaffinity(0,set(audit_cpus));started=time.monotonic()
z=np.load(entry['input_file']);tr=np.load(folder/'trajectory.npz');result=json.loads((folder/'result.json').read_text());q=tr['q']
model=mujoco.MjModel.from_xml_path(entry['scene']);data=mujoco.MjData(model);robot=online.build_robot_certificate(model)
body=np.array([x.body_id for x in robot]);local=np.array([x.local_center for x in robot]);radii=np.array([x.radius for x in robot]);site=model.site('attachment_site').id
def pos(qi):
    data.qpos[:6]=qi;mujoco.mj_kinematics(model,data);return data.xpos[body]+np.einsum('nij,nj->ni',data.xmat[body].reshape(-1,3,3),local)
p0=pos(z['q0']);parent=model.body_parentid
pairs=np.array([(i,j) for i in range(len(radii)) for j in range(i+1,len(radii)) if body[i]!=body[j] and parent[body[i]]!=body[j] and parent[body[j]]!=body[i] and np.linalg.norm(p0[j]-p0[i])-radii[i]-radii[j]>=.005])
meta=json.loads(Path(entry['metadata']).read_text());assert len(pairs)==meta['original_self_pairs']
lever=[]
for bi,point in zip(body,local):
    length=np.linalg.norm(point);bi=int(bi)
    while bi:
        length+=np.linalg.norm(model.body_pos[bi]);j0=model.body_jntadr[bi]
        for j in range(j0,j0+model.body_jntnum[bi]):length+=2*np.linalg.norm(model.jnt_pos[j])
        bi=int(parent[bi])
    lever.append(length)
K=2*max(lever);positions=[];ee=[]
for qi in q:positions.append(pos(qi));ee.append(data.site_xpos[site].copy())
positions=np.asarray(positions);ee=np.asarray(ee)
assert len(q)>1 and np.max(np.abs(ee[1:]-tr['ee']))<1e-8
workspace=audit_workspace(q,positions,radii,K,pos)
if not entry.get('synthetic_fixture'):
    scene=json.loads((U/'scenes'/(entry.get('scene_task',job['task'])+'.json')).read_text());centers=np.asarray([b['center'] for b in scene['boxes']]);half=np.asarray([b['half_size'] for b in scene['boxes']])
    physical=audit_motion(q,positions,radii,centers,half,pairs,K,pos)
else:
    # Independent self-pair interval check; physical ellipsoid checked below.
    aa,bb=pairs.T;self_min=float('inf');self_bad=[];self_sub=0
    def self_gap(p):
        global self_min
        value=float(np.min(np.linalg.norm(p[bb]-p[aa],axis=1)-radii[aa]-radii[bb]));self_min=min(self_min,value);return value
    def self_interval(a,b,pa,pb,depth=0):
        global self_sub
        if min(self_gap(pa),self_gap(pb))<.005-1e-7:return False
        x=pa[bb]-pa[aa];y=pb[bb]-pb[aa];d=y-x;t=np.clip(-np.sum(x*d,axis=1)/np.maximum(np.sum(d*d,axis=1),1e-30),0,1)
        lower=np.min(np.linalg.norm(x+t[:,None]*d,axis=1)-radii[aa]-radii[bb])-2*(K*np.sum(np.abs(b-a))**2/8+1e-12)
        if lower>=.005-1e-7:return True
        if depth>=12:return False
        self_sub+=1;mid=(a+b)/2;pm=pos(mid);return self_interval(a,mid,pa,pm,depth+1) and self_interval(mid,b,pm,pb,depth+1)
    for i in range(len(q)-1):
        if not self_interval(q[i],q[i+1],positions[i],positions[i+1]):self_bad.append(i)
    physical=dict(passed=not self_bad,self_passed=not self_bad,self_bad_intervals=self_bad,minimum_self_m=self_min,subdivisions=self_sub,environment='exact ellipsoid physical shape audited by independent proxy geometry below')

guard=IndependentGuard(z['centers'],z['shapes'],z['offsets'],z['radii'],z['boxes'],job.get('representation','ellipsoid'))
minimum=.020;worst=q[0].copy();bad=[];subdivisions=0
def state(qi,p):
    global minimum,worst
    gap=min(.020,float(guard.certify(p,p,radii+.020)+.020))
    if gap<minimum:minimum=gap;worst=qi.copy()
    return gap
def interval(a,b,pa,pb,depth=0):
    global subdivisions
    pad=K*np.sum(np.abs(b-a))**2/8+1e-10
    if guard.certify(pa,pb,radii+.006+pad)>=-1e-7:return True
    if depth>=12:return False
    subdivisions+=1;mid=(a+b)/2;pm=pos(mid)
    if state(mid,pm)<.006-1e-7:return False
    return interval(a,mid,pa,pm,depth+1) and interval(mid,b,pm,pb,depth+1)
for i in range(len(q)-1):
    if state(q[i],positions[i])<.006-1e-7 or state(q[i+1],positions[i+1])<.006-1e-7 or not interval(q[i],q[i+1],positions[i],positions[i+1]):bad.append(i)
    if i%1000==0:print('audit',folder.name,i,len(q)-1,flush=True)
reference=float(guard.reference_gap(pos(worst),radii))
assert abs(reference-minimum)<1e-6 or minimum==.020
dt=.02;velocity=np.diff(q,axis=0)/dt;previous0=np.zeros(6)
if job.get('start_snapshot'):previous0=np.load(job['start_snapshot'])['previous']
acc=np.diff(np.vstack([previous0,velocity]),axis=0)/dt
assert np.max(np.abs(velocity-tr['velocity']))<1e-9
lo=model.jnt_range[:6,0]+.015;hi=model.jnt_range[:6,1]-.015;speed=np.abs(velocity).copy();travel=np.zeros_like(speed)
while np.max(speed)>1e-15:travel+=dt*speed;speed=np.maximum(speed-.2,0)
stop=q[:-1]+np.sign(velocity)*travel
joint_slack=float(min((q-lo).min(),(hi-q).min()));brake_slack=float(min((stop-lo).min(),(hi-stop).min()));v_ratio=float(np.max(np.abs(velocity)/np.asarray(POLICY['velocity_limits'])));acc_peak=float(np.max(np.abs(acc)))
target=np.asarray(job.get('target_override',z['target']));err=np.linalg.norm(ee[1:]-target,axis=1);hold=best=0
for value in err:hold=hold+1 if value<.001 else 0;best=max(best,hold)
safe=physical['passed'] and workspace['passed'] and not bad and minimum>=.006-1e-7 and joint_slack>=-1e-10 and brake_slack>=-1e-10 and v_ratio<=1+1e-8 and acc_peak<=10.0001 and result['truth_penetrating_cycles']==0 and result['guard']['failed']==0
out=dict(safety_passed=bool(safe),goal_passed=bool(best>=50),safe_arrival=bool(safe and best>=50),final_error_mm=float(err[-1])*1000,maximum_goal_hold=best,minimum_proxy_gap_mm=minimum*1000,proxy_failed_intervals=bad,proxy_subdivisions=subdivisions,independent_reference_gap_mm=reference*1000,physical_motion=physical,workspace=workspace,joint_slack_rad=joint_slack,braking_slack_rad=brake_slack,peak_velocity_ratio=v_ratio,peak_acceleration=acc_peak,self_pairs=len(pairs),forbidden_calls=result['forbidden_calls'],audit_wall_s=time.monotonic()-started,scope='full actual trajectory audited independently; no controller use of truth or witness route')
(folder/'audit.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['physical_motion','workspace','proxy_failed_intervals']},indent=2))
