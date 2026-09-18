"""Reconstruct saved states independently; report failed gates without erasing them."""
from pathlib import Path
import json,csv,sys,hashlib
import numpy as np
import mujoco
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;sys.path.insert(0,str(BASE))
from model import build_robot_certificate,certificate_world_positions,attachment_position,set_configuration,DT,JOINT_NAMES
from protocol_drawer_scene import formal_drawer_camera_quarter_scene
from native_ellipsoid_support import NativeEllipsoidSupport
from run_study import CASES,dump

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
    results={};scene=formal_drawer_camera_quarter_scene();target=np.array(scene.waypoints[-1])
    kernel=NativeEllipsoidSupport();kernel.set_pair_threads(8)
    for case in CASES:
        outer=HERE/'results'/case
        if not (outer/'completion.json').exists(): continue
        complete=json.loads((outer/'completion.json').read_text(encoding='utf-8'))
        paths=list(outer.glob('A-CV-*'));assert len(paths)==1;folder=paths[0]
        if not complete['complete']:
            results[case]=dict(complete=False,failure=complete['failure'])
            continue
        summary=json.loads((outer/'summary.json').read_text(encoding='utf-8'))
        with (folder/'cycles.csv').open(encoding='utf-8',newline='') as f:logs=list(csv.DictReader(f))
        history=np.load(folder/'q_history.npy');ee=np.load(folder/'ee_history.npy')
        assert history.shape==(3000,6) and ee.shape==(3000,3) and len(logs)==3000
        assert np.isfinite(history).all()
        q=np.vstack([scene.q0,history]);model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));data=mujoco.MjData(model)
        robot=build_robot_certificate(model);radii=np.array([s.radius for s in robot]);L=lever_bounds(model,robot)
        centers=np.array([b.center for b in scene.boxes]);halves=np.array([b.half_size for b in scene.boxes])
        def boxgap(pos):
            v=np.abs(pos[:,None,:]-centers[None,:,:])-halves[None,:,:]
            return np.linalg.norm(np.maximum(v,0),axis=2)+np.minimum(np.max(v,axis=2),0)-radii[:,None]
        min_box=np.inf;minws=np.inf;penetrating=0;maxeediff=0;hold=0;maxhold=0;entry=None;confirmed=None
        wa=np.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]])
        wb=np.array([.95,.95,.95,.95,0,1.35]);positions=[]
        for k,x in enumerate(q):
            set_configuration(model,data,x);pos=certificate_world_positions(data,robot);positions.append(pos)
            min_box=min(min_box,float(boxgap(pos).min()));minws=min(minws,float((wb[None,:]-radii[:,None]-pos@wa.T).min()))
            pen=sum(data.contact[t].dist < -1e-8 for t in range(data.ncon))
            if k:
                penetrating+=int(pen>0)
                maxeediff=max(maxeediff,float(np.max(np.abs(attachment_position(model,data)-ee[k-1]))))
                err=float(np.linalg.norm(ee[k-1]-target));assert abs(err-float(logs[k-1]['error_m']))<1e-10
                assert pen==int(logs[k-1]['exact_penetrating_contact_count'])
                hold=hold+1 if err<.001 and pen==0 else 0;maxhold=max(maxhold,hold)
                if hold==50 and confirmed is None:confirmed=k*DT;entry=(k-49)*DT
        assert maxeediff<1e-12 and maxhold==summary['maximum_success_hold_cycles']
        assert penetrating==summary['exact_penetrating_cycles']
        velocity=float(np.max(np.abs(np.diff(q,axis=0)/DT)-np.array([1,1,1,1.4,1.4,1.4])))
        qlo=np.array([model.jnt_range[model.joint(n).id,0] for n in JOINT_NAMES])+.015
        qhi=np.array([model.jnt_range[model.joint(n).id,1] for n in JOINT_NAMES])-.015
        joint=min(float((q-qlo).min()),float((qhi-q).min()))
        bound=np.inf;unresolved=0;checks=0
        def interval(a,b,depth=0):
            nonlocal bound,unresolved,checks
            set_configuration(model,data,(a+b)/2);gap=boxgap(certificate_world_positions(data,robot));checks+=1
            lower=float((gap-(.5*L@np.abs(b-a))[:,None]).min())
            if lower>=.006: bound=min(bound,lower);return
            if gap.min()<.006 or depth>=6:
                unresolved+=1;bound=min(bound,lower);return
            mid=(a+b)/2;interval(a,mid,depth+1);interval(mid,b,depth+1)
        for a,b in zip(q[:-1],q[1:]):interval(a,b)
        archive=np.load(folder/'causal_proxy_snapshots.npz');pub=archive['publish_cycles'];src=archive['source_cycles'];off=archive['offsets']
        assert np.all(src<=pub) and np.all(np.diff(pub)>=0)
        probe_cycles=sorted(set(range(0,3000,25))|set(int(i) for i in pub)|{2999})
        min_proxy=np.inf;support_residual=0.;proxy_pairs=0;proxy_inside_samples=0
        for cycle in probe_cycles:
            snapshot=int(np.searchsorted(pub,cycle,side='right')-1);sl=slice(off[snapshot],off[snapshot+1])
            c=archive['centers'][sl];offset=archive['uncertainty_offsets'][sl];pos=positions[cycle+1]
            if summary['representation']=='sphere':
                d=np.linalg.norm(pos[:,None,:]-c[None,:,:],axis=2)-radii[:,None]-archive['sphere_radii'][sl][None,:]-offset[None,:]
                proxy_pairs+=d.size;gap=float(d.min())
            else:
                Q=archive['ellipsoid_shapes'][sl];U=archive['proxy_uncertainty_shapes'][sl]
                half=np.sqrt(np.maximum(np.diagonal(Q,axis1=1,axis2=2),0))+np.sqrt(np.maximum(np.diagonal(U,axis1=1,axis2=2),0))+offset[:,None]
                lower=np.linalg.norm(np.maximum(np.abs(pos[:,None,:]-c[None,:,:])-half[None,:,:],0),axis=2)-radii[:,None]
                ri,oi=np.nonzero(lower<.050);proxy_pairs+=len(ri)
                if not len(ri):continue
                R=np.eye(3)[None,:,:]*radii[ri,None,None]**2
                n,_,res=kernel.normals_sum_pairs_newton_warm(pos[ri],R,c[oi],Q[oi],U[oi],np.zeros((len(ri),3)),max_iterations=64)
                support_residual=max(support_residual,float(res.max()))
                d=np.einsum('ij,ij->i',c[oi]-pos[ri],n)-radii[ri]-offset[oi]
                for shape in [Q[oi],U[oi]]:d-=np.sqrt(np.maximum(np.einsum('ni,nij,nj->n',n,shape,n),0))
                gap=float(d.min())
            min_proxy=min(min_proxy,gap);proxy_inside_samples+=int(gap<0)
        for key,expected in json.loads((outer/'original_source_hashes.json').read_text(encoding='utf-8')).items():
            assert hashlib.sha256((BASE/key).read_bytes()).hexdigest()==expected
        result=dict(complete=True,data_integrity_passed=True,states=len(q),reconstructed_contacts_match=True,
                    penetrating_cycles=penetrating,maximum_reconstructed_ee_error_m=maxeediff,
                    max_success_hold=maxhold,final_success_hold=hold,first_entry_post_step_s=entry,confirmed_post_step_s=confirmed,
                    velocity_limit_excess_rad_s=velocity,joint_padding_slack_rad=joint,workspace_slack_mm=minws*1000,
                    minimum_true_box_endpoint_gap_mm=min_box*1000,continuous_true_box_gap_lower_bound_mm=bound*1000,
                    continuous_true_box_6mm_certified=unresolved==0,interval_checks=checks,unresolved_intervals=unresolved,
                    proxy_probe_cycles=len(probe_cycles),proxy_probe_pair_checks=proxy_pairs,minimum_probed_proxy_gap_mm=min_proxy*1000,
                    proxy_probe_inside_samples=proxy_inside_samples,proxy_probe_max_support_residual=support_residual,
                    proxy_probe_scope='Post-step states every 25 cycles and each map publication; current causal snapshot; not full continuous proxy proof.',
                    continuous_self_collision_certified=False,continuous_floor_certified=False,source_hashes_unchanged=True)
        dump(outer/'independent_audit.json',result);results[case]=result
        print(case,json.dumps(result),flush=True)
    dump(HERE/'results/independent_audits.json',results)

if __name__=='__main__':main()
