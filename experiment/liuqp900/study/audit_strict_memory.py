from pathlib import Path
import os,sys,json
os.sched_setaffinity(0,set(map(int,sys.argv[2].split(','))) if len(sys.argv)>2 else {56,57,58,59})
import numpy as np,mujoco
from pilot_adapter import ROOT
folder=Path(sys.argv[1]);job=json.loads((folder/'job.json').read_text());result=json.loads((folder/'result.json').read_text());summary=result.get('policy_summary') or {};memories=json.loads((folder/'goalward_memories.json').read_text())
entry=json.loads((ROOT/'INPUTS.json').read_text())[0];model=mujoco.MjModel.from_xml_path(entry['scene']);data=mujoco.MjData(model);site=model.site('attachment_site').id
port=summary['goal_port'];P=np.asarray(port['posture_matrix']);terminal_q=np.asarray(port['goal_q']);qg=np.asarray(port.get('memory_feature_origin_q',port['goal_q']));target=np.asarray(port['target']);q=np.load(folder/'trajectory.npz')['q']
body=int(model.site_bodyid[site]);length=float(np.linalg.norm(model.site_pos[site]))
while body:
    length+=float(np.linalg.norm(model.body_pos[body]));j0=model.body_jntadr[body]
    for j in range(j0,j0+model.body_jntnum[body]):length+=2*float(np.linalg.norm(model.jnt_pos[j]))
    body=int(model.body_parentid[body])
curvature=2*length
def feature(qi):
    data.qpos[:6]=qi;mujoco.mj_kinematics(model,data);return np.r_[data.site_xpos[site].copy(),P@(qi-qg)]
features=np.asarray([feature(qi) for qi in q]);records=[];bad=[];subdivisions=0
inherited=[];source_q=None
if job.get('initial_memories'):
    source=Path(job['initial_memories']);inherited=json.loads(source.read_text());source_q=np.load(source.parent/'trajectory.npz')['q']
    assert memories[:len(inherited)]==inherited,'Inherited no-return constraints changed'

for index,m in enumerate(memories):
    ball=m.get('memory_kind')=='local_feature_exclusion';activation=0 if index<len(inherited) else m['activation_cycle'];start=m['hit_cycle'];hit_feature=feature(source_q[start]) if index<len(inherited) else features[start];hit_error=float(np.max(np.abs(hit_feature-m['hit_feature'])))
    if ball:
        center=np.asarray(m['hit_feature']);radius=m['radius_m'];values=np.linalg.norm(features-center,axis=1)-radius;goal=float(np.linalg.norm(target-center[:3])-radius-.001);hit_value=float(np.linalg.norm(hit_feature-center)-radius)
    else:
        normal=np.asarray(m['normal']);values=features@normal-m['offset'];goal=float(np.r_[target,P@(terminal_q-qg)]@normal-m['offset']);hit_value=float(hit_feature@normal-m['offset'])
    minimum=float(values[activation:].min())
    failures=[]
    def interval(a,b,fa,fb,depth=0):
        global subdivisions
        if ball:
            x=fa-center;y=fb-center;d=y-x;curve=curvature*np.sum(np.abs(b-a))**2/8;t=np.clip(-x@d/max(d@d,1e-30),0,1);lower=float(np.linalg.norm(x+t*d)-curve)
            for endpoint in [x,y]:
                direction=endpoint/max(np.linalg.norm(endpoint),1e-30);initial=float(direction@x);slope=float(direction@d);B=4*curve*np.linalg.norm(direction[:3]);tau=np.clip((B-slope)/max(2*B,1e-30),0,1);lower=max(lower,float(initial+tau*(slope-B)+B*tau*tau))
            bound=lower-radius;va=float(np.linalg.norm(x)-radius);vb=float(np.linalg.norm(y)-radius)
        else:
            va=float(normal@fa-m['offset']);vb=float(normal@fb-m['offset']);B=np.linalg.norm(normal[:3])*curvature*np.sum(np.abs(b-a))**2/2
            slope=vb-va;tau=np.clip((B-slope)/max(2*B,1e-30),0,1)
            bound=float(va+tau*(slope-B)+B*tau*tau)
        if bound>=-1e-10:return True
        if min(va,vb)<-1e-10 or depth>=12:return False
        subdivisions+=1;mid=(a+b)/2;fm=feature(mid)
        return interval(a,mid,fa,fm,depth+1) and interval(mid,b,fm,fb,depth+1)
    for k in range(activation,len(q)-1):
        if not interval(q[k],q[k+1],features[k],features[k+1]):failures.append(k)
    record=dict(memory_kind='local_feature_exclusion' if ball else 'plane',memory_index=index,activation_cycle=activation,hit_cycle=start,minimum_value_m=minimum,goal_value_m=goal,hit_value_m=hit_value,inherited_at_initial_state=index<len(inherited),source_activation_cycle=m['activation_cycle'],hit_feature_reconstruction_error=hit_error,failed_intervals=failures)
    records.append(record)
    if failures or goal<=0 or hit_error>1e-8 or hit_value>=0:bad.append(index)
out=dict(passed=not bad,memory_count=len(memories),bad_memories=bad,records=records,subdivisions=subdivisions,memory_tolerance_m=1e-10,scope='independent endpoint-consistent continuous plane/compact-exclusion memory audit; compact goals checked over the full 1mm Cartesian goal tolerance region; zero memories means the mechanism was not exercised',runtime_memory_failures=summary.get('memory_certificate_failures'),inherited_count=len(inherited),inherited_audit_starts_at_cycle_zero=True)
(folder/'strict_memory_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(name=folder.name,passed=out['passed'],memories=len(memories),bad=bad)))
