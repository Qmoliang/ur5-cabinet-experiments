"""Continuous certificate-to-box audit of linearly interpolated joint motion.

The sphere-center distance to a fixed box is 1-Lipschitz in center position.
A serial hinge-chain lever-arm upper bound L gives displacement <= L @ abs(dq).
Midpoint distance minus half that bound certifies the whole segment. Recursive
bisection only sharpens the certificate; no samples are reported as proofs.
This checks environment boxes, not self-collision or unknown-space observability.
"""
from pathlib import Path
import json,argparse
import numpy as np
import mujoco
from model import build_robot_certificate,set_configuration,certificate_world_positions
from official_drawer_07_3 import scene
from experiment_07_3_geometry import distances,OUT

def lever_bounds(model,robot):
    L=np.zeros((len(robot),model.nv))
    for i,s in enumerate(robot):
        b=s.body_id;reach=np.linalg.norm(s.local_center)
        while b:
            for j in range(model.body_jntadr[b],model.body_jntadr[b]+model.body_jntnum[b]):
                assert model.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE
                L[i,model.jnt_dofadr[j]]=reach+np.linalg.norm(model.jnt_pos[j])
            reach+=np.linalg.norm(model.body_pos[b]);b=int(model.body_parentid[b])
    return L

def audit(folder):
    m=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));d=mujoco.MjData(m);robot=build_robot_certificate(m);r=np.array([s.radius for s in robot]);L=lever_bounds(m,robot);q=np.load(folder/'q_history.npy');s=scene()
    # Online history contains post-step states; restore the actual configured initial q.
    meta=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    if not meta.get('diagnostic_true_geometry',False):q=np.vstack([meta['initial_configuration_rad'],q])
    min_bound=float('inf');unresolved=0;proven=0;checks=0
    def interval(a,b,depth):
        nonlocal min_bound,unresolved,proven,checks
        set_configuration(m,d,(a+b)/2);g=distances(certificate_world_positions(d,robot),s.boxes,r)
        bounds=g-(.5*L@np.abs(b-a))[:,None];value=float(bounds.min());checks+=1
        if value>0:
            min_bound=min(min_bound,value);proven+=1;return
        if depth>=8:unresolved+=1;min_bound=min(min_bound,value);return
        mid=(a+b)/2;interval(a,mid,depth+1);interval(mid,b,depth+1)
    for a,b in zip(q[:-1],q[1:]):interval(a,b,0)
    result=dict(method='analytic_hinge_lever_bound_and_box_distance_lipschitz',environment_boxes_only=True,self_collision_certified=False,ground_certified=False,segments=len(q)-1,subinterval_checks=checks,unresolved=unresolved,all_environment_box_segments_certified=unresolved==0,minimum_continuous_gap_lower_bound_mm=1000*min_bound,safety_margin_6mm_certified=min_bound>=.006)
    (folder/'continuous_box_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path);a=p.parse_args();folders=[a.folder] if a.folder else sorted(OUT.glob('exact_box_seed*'))
    rows=[]
    for f in folders:
        row=dict(folder=str(f),**audit(f));rows.append(row);print(json.dumps(row),flush=True)
    if a.folder is None:(OUT/'exact_continuous_audit.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
