"""Independent audits of saved task, endpoints and continuous cabinet separation."""
from pathlib import Path
import json,hashlib
import numpy as np
import mujoco
from run_comparison import ROOT,HERE,original,CASES,dump
from robot import build_model,build_robot_certificate,set_configuration,certificate_world_positions,attachment_position,DT
from known_volume import build_volume_cover
from neo_controller import EllipsoidDistances,VELOCITY_LIMITS,WA,WB

def lever_bounds(model,robot):
    L=np.zeros((len(robot),model.nv))
    for i,s in enumerate(robot):
        body=s.body_id;reach=float(np.linalg.norm(s.local_center))
        while body:
            for j in range(model.body_jntadr[body],model.body_jntadr[body]+model.body_jntnum[body]):
                assert model.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE
                L[i,model.jnt_dofadr[j]]=reach+np.linalg.norm(model.jnt_pos[j])
            reach+=np.linalg.norm(model.body_pos[body]);body=int(model.body_parentid[body])
    return L

def main():
 scene=original.load_scene();model=build_model(scene);data=mujoco.MjData(model)
 robot=build_robot_certificate(model);radii=np.array([s.radius for s in robot]);L=lever_bounds(model,robot)
 centers=np.array([b.center for b in scene.boxes]);halves=np.array([b.half_size for b in scene.boxes])
 geometry=EllipsoidDistances(build_volume_cover(scene.boxes,'ellipsoid'));target=np.array(scene.waypoints[-1])
 def gaps(q):
    set_configuration(model,data,q);p=certificate_world_positions(data,robot)
    v=np.abs(p[:,None,:]-centers[None,:,:])-halves[None,:,:]
    signed=np.linalg.norm(np.maximum(v,0),axis=2)+np.minimum(np.max(v,axis=2),0)
    return signed-radii[:,None]
 hashes=json.loads((HERE/'results'/'source_hashes.json').read_text())
 assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in hashes.items())
 results={}
 for case in CASES:
    f=HERE/'results'/case;s=json.loads((f/'summary.json').read_text());logs=json.loads((f/'cycles.json').read_text())
    q=np.load(f/'q_history.npy');ee=np.load(f/'ee_history.npy')
    assert q.shape==(3001,6) and ee.shape==(3001,3) and np.isfinite(q).all()
    assert np.array_equal(q[0],scene.q0);assert len(logs)==3000
    if case=='liuqp': assert np.array_equal(q,np.load(ROOT/'results'/'ellipsoid'/'q_history.npy'))
    velocity=np.max(np.abs(np.diff(q,axis=0)/DT)-VELOCITY_LIMITS)
    qlo=np.array([model.jnt_range[model.joint(n).id,0] for n in original.JOINT_NAMES])+.015
    qhi=np.array([model.jnt_range[model.joint(n).id,1] for n in original.JOINT_NAMES])-.015
    joint_margin=min(float(np.min(q-qlo)),float(np.min(qhi-q)))
    assert velocity<1e-6 and joint_margin>=-1e-7
    min_proxy=np.inf;minimum_workspace=np.inf;pen=0;max_ee_diff=0;maxhold=0;hold=0
    for k,x in enumerate(q):
        set_configuration(model,data,x);pos=certificate_world_positions(data,robot)
        ws=WB[None,:]-radii[:,None]-pos@WA.T;minimum_workspace=min(minimum_workspace,float(ws.min()))
        max_ee_diff=max(max_ee_diff,float(np.max(np.abs(attachment_position(model,data)-ee[k]))))
        contacts=sum(data.contact[t].dist < -1e-8 for t in range(data.ncon));pen+=int(contacts>0)
        if k:
            err=float(np.linalg.norm(ee[k]-target));hold=hold+1 if err<.001 and contacts==0 else 0;maxhold=max(maxhold,hold)
            assert abs(err-logs[k-1]['error_m'])<1e-12
        # Full pair check at every saved endpoint, independent of broadphase/pruning.
        _,_,d,_,res=geometry.query(pos,radii,np.inf)
        assert res<1e-8
        min_proxy=min(min_proxy,float(d.min()))
    bound_min=np.inf;unresolved=0;checks=0
    def interval(a,b,depth=0):
        nonlocal bound_min,unresolved,checks
        value=float((gaps((a+b)/2)-(.5*L@np.abs(b-a))[:,None]).min());checks+=1
        if value>=.006:
            bound_min=min(bound_min,value);return
        if depth>=8:
            unresolved+=1;bound_min=min(bound_min,value);return
        mid=(a+b)/2;interval(a,mid,depth+1);interval(mid,b,depth+1)
    for a,b in zip(q[:-1],q[1:]):interval(a,b)
    assert maxhold==s['maximum_success_hold_cycles'];assert pen==0;assert max_ee_diff<1e-12
    result=dict(saved_states=3001,all_pair_endpoint_checks=3001*65*368,
       minimum_proxy_surface_gap_mm=min_proxy*1000,minimum_endpoint_proxy_safety_slack_mm=(min_proxy-.006)*1000,
       maximum_velocity_limit_excess_rad_s=float(velocity),minimum_joint_padding_slack_rad=joint_margin,
       minimum_workspace_slack_mm=minimum_workspace*1000,penetrating_states=pen,
       max_reconstructed_ee_error_m=max_ee_diff,success_hold_recomputed=maxhold,
       continuous_box_segments=3000,interval_checks=checks,unresolved_for_6mm=unresolved,
       continuous_cabinet_6mm_certified=unresolved==0,continuous_box_gap_lower_bound_mm=bound_min*1000,
       continuous_self_collision_certified=False,continuous_floor_certified=False,
       scope='Linear joint interpolation between saved kinematic states; cabinet boxes use full sphere certificates. No dynamics tracking claim.')
    results[case]=result;dump(f/'audit.json',result);print(case,json.dumps(result),flush=True)
 vendor={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'vendor').rglob('*') if p.is_file() and p.suffix in ['.pyd','.dll']}
 report=dict(passed=True,original_sources_unchanged=True,baseline_trajectory_bit_identical=True,quadprog_version='0.1.13',vendor_binary_hashes=vendor,cases=results)
 dump(HERE/'results'/'verification.json',report)
if __name__=='__main__':main()
