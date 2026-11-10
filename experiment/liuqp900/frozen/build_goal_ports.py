"""Static target-end IK and one local entry-neighborhood certificate.

No start-to-goal connection, graph, tree, route, waypoint list or witness read.
The IK sampling parameters have dimension two; q6 stays at the public start.
"""
from pathlib import Path
import os,time,json,hashlib
os.sched_setaffinity(0,set(map(int,os.environ['STUDY_CPUS'].split(','))))
import numpy as np,mujoco
from scipy.stats import qmc
from scipy.optimize import minimize
from pilot_adapter import ROOT,build,POLICY
from position_ik import candidates
from collision_oracle import CollisionOracle

started=time.monotonic();ctx=build(json.loads((ROOT/'INPUTS.json').read_text())[0]['task']);qref=ctx.q0.copy();target=ctx.target.copy()
oracle=CollisionOracle(ctx.model,ctx.robot,ctx.arrays,qref,margin=POLICY['goal_margin_m'])
zero=mujoco.MjData(ctx.model);mujoco.mj_kinematics(ctx.model,zero);shoulder=ctx.model.body('shoulder_link').id
t=zero.xmat[shoulder].reshape(3,3).T@(target-zero.xpos[shoulder]);radius=np.hypot(t[0],t[1]);azimuth=np.arctan2(t[1],t[0])
# The exact position IK requires ty in [0.134-0.07, 0.134+0.07].
# Restrict its two free coordinates analytically instead of sampling impossible base angles.
ty=np.linspace(max(-radius,.064),min(radius,.204),33)
arcs=np.arcsin(np.clip(ty/radius,-1,1));bases=np.r_[azimuth-arcs,azimuth-(np.pi-arcs)]
params=np.asarray([(u,v,qref[5]) for u in bases for v in np.linspace(-np.pi,np.pi,65)])
pool=candidates(ctx.model,target,params,qref)
def branch(q):return (int(np.cos(q[0]-np.arctan2(target[1],target[0]))>=0),int(np.sin(q[2])>=0),int(np.sin(q[4])>=0))
groups={}
for q in pool:groups.setdefault(branch(q),[]).append(q)
for key in groups:groups[key].sort(key=lambda q:np.linalg.norm(q-qref))
ordered=[]
for index in range(max((len(x) for x in groups.values()),default=0)):
    for key in sorted(groups):
        if index<len(groups[key]):ordered.append((key,groups[key][index]))
records=[];best={}
for index,(key,q) in enumerate(ordered[:384]):
    error=float(np.linalg.norm(oracle.gripper_position(q)-target));valid=error<.0007 and oracle.free(q)
    rec=dict(index=index,branch=key,q=q.tolist(),error_m=error,feasible=bool(valid),static_slack_m=float(oracle.exact_clearance(q)))
    if valid:
        ee,J,gaps,G,work,W=oracle.evaluate(q);env=float(gaps.min());j,k=oracle.pairs.T;p=oracle.sphere_positions(q)
        selfgap=float(np.min(np.linalg.norm(p[k]-p[j],axis=1)-oracle.radii[j]-oracle.radii[k]));clear=min(env,selfgap)
        rec.update(environment_gap_m=env,self_gap_m=selfgap,clearance_score=clear)
        if key not in best or clear>best[key]['clearance_score']:best[key]=rec
    records.append(rec)
    if index%40==0:print(json.dumps(dict(checked=index+1,pool=len(pool),feasible=sum(r['feasible'] for r in records),elapsed_s=time.monotonic()-started)),flush=True)
refinements=[]
for seed in sorted(records,key=lambda r:-r['static_slack_m'])[:12]:
    if len(best)>=4:break
    initial=np.array(seed['q']);fixed=initial[5]
    def unpack(x):return np.r_[x[:5],fixed]
    def fun(x):return -x[5]+1e-6*np.sum((x[:5]-initial[:5])**2)
    def jac(x):return np.r_[2e-6*(x[:5]-initial[:5]),-1.]
    def eq(x):return oracle.evaluate(unpack(x))[0]-target
    def eqj(x):return np.c_[oracle.evaluate(unpack(x))[1][:,:5],np.zeros(3)]
    def con(x):
        ee,J,gaps,G,work,W=oracle.evaluate(unpack(x));return np.r_[gaps-x[5],work]
    def conj(x):
        ee,J,gaps,G,work,W=oracle.evaluate(unpack(x));return np.vstack([np.c_[G[:,:5],-np.ones(len(G))],np.c_[W[:,:5],np.zeros(len(W))]])
    result=minimize(fun,np.r_[initial[:5],.0066],jac=jac,bounds=list(zip(oracle.lower[:5],oracle.upper[:5]))+[(-.03,.02)],constraints=[dict(type='eq',fun=eq,jac=eqj),dict(type='ineq',fun=con,jac=conj)],method='SLSQP',options=dict(maxiter=100,ftol=1e-10))
    q=unpack(result.x);error=float(np.linalg.norm(oracle.gripper_position(q)-target));valid=error<.0007 and oracle.free(q)
    rec=dict(q=q.tolist(),error_m=error,feasible=bool(valid),static_slack_m=float(oracle.exact_clearance(q)),iterations=int(result.nit),message=str(result.message),seed_index=seed['index'])
    if valid:
        ee,J,gaps,G,work,W=oracle.evaluate(q);p=oracle.sphere_positions(q);j,k=oracle.pairs.T;env=float(gaps.min());selfgap=float(np.min(np.linalg.norm(p[k]-p[j],axis=1)-oracle.radii[j]-oracle.radii[k]));key=branch(q)
        rec.update(branch=key,environment_gap_m=env,self_gap_m=selfgap,clearance_score=min(env,selfgap))
        if key not in best or rec['clearance_score']>best[key]['clearance_score']:best[key]=rec
    refinements.append(rec);print(json.dumps(dict(static_refinement=rec)),flush=True)
ports=[];pstart=oracle.gripper_position(qref);reference_direction=(pstart-target)/np.linalg.norm(pstart-target)
for key,rec in sorted(best.items()):
    qg=np.array(rec['q']);ee,J,gaps,G,work,W=oracle.evaluate(qg);N=np.eye(6)-np.linalg.pinv(J,rcond=1e-8)@J
    _,sv,vt=np.linalg.svd((G/np.maximum(gaps,.0065)[:,None])@N,full_matrices=False)
    basis=vt[:2].copy()
    for row in basis:
        if row[np.argmax(np.abs(row))]<0:row*=-1
    lever=float(np.median(np.linalg.norm(J,axis=0)[np.linalg.norm(J,axis=0)>1e-6]));P=lever*basis
    lift=np.linalg.pinv(np.vstack([J,P]),rcond=1e-8);directions=[reference_direction]+[sign*np.eye(3)[axis] for axis in range(3) for sign in [1.,-1.]]
    rays=[]
    for d in directions:
        oracle.margin=POLICY['operating_margin_m']
        radius=ctx.controller.near_distance;ok=False
        for _ in range(7):
            qb=qg+lift@np.r_[radius*d,np.zeros(2)]
            if np.all(qb>=oracle.lower) and np.all(qb<=oracle.upper) and oracle.segment_certified(qg,qb):ok=True;break
            radius*=.5
        if not ok:continue
        actual=oracle.gripper_position(qb)-target;length=float(np.linalg.norm(actual));outward=actual/max(length,1e-30)
        rays.append(dict(outward=outward.tolist(),radius_m=length,linear_radius_m=radius,alignment=float(outward@reference_direction),backward_configuration_for_certificate_only=qb.tolist()))
    if not rays:continue
    ray=max(rays,key=lambda x:(x['linear_radius_m'],x['alignment']))
    port=dict(id=len(ports),branch=key,goal_q=qg.tolist(),target=target.tolist(),posture_matrix=P.tolist(),posture_basis=basis.tolist(),lever_m=lever,goal_environment_gap_m=rec['environment_gap_m'],goal_self_gap_m=rec['self_gap_m'],posture_singular_values=sv.tolist(),**ray,all_entry_rays=rays)
    ports.append(port);print(json.dumps(dict(port=port['id'],branch=key,goal_q=qg.tolist(),clearance=rec['clearance_score'],radius=ray['radius_m'],outward=ray['outward'])),flush=True)
if not ports:
    (ROOT/'GOAL_PORT_FAILURE.json').write_text(json.dumps(dict(records=records,refinements=refinements),indent=2))
    raise RuntimeError('No feasible static target-entry hypothesis; do not substitute witness data')
ports.sort(key=lambda p:(-p['linear_radius_m'],-min(p['goal_environment_gap_m'],p['goal_self_gap_m']),-p['alignment'],p['id']))
for rank,p in enumerate(ports):p['rank']=rank
out=dict(task=ctx.entry['task'],input_manifest=json.loads((ROOT/'INPUTS.json').read_text()),scope='Static endpoint hypotheses and local backward entry certificates only; never connect to current robot or create an execution route',path_read=False,planner_calls=0,parameter_dimension=2,posture_feature_dimension=2,pose_candidates=len(pool),checked=len(records),feasible=sum(r['feasible'] for r in records),ports=ports,records=records,refinements=refinements,wall_s=time.monotonic()-started)
(ROOT/'GOAL_PORTS.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(ports=len(ports),wall_s=out['wall_s'])),flush=True)
