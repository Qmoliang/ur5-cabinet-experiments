"""Independent new-scene trajectory, geometry, observation and failure audits."""
from pathlib import Path
import json,sys,hashlib
import numpy as np,mujoco
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'runtimes/online'),str(ROOT/'runtimes/online/external_neo_adaptive')]
from model import build_robot_certificate,certificate_world_positions,set_configuration,attachment_position,JOINT_NAMES
from depth_camera_perception import UR5MountedDepthCamera
from native_ellipsoid_support import NativeEllipsoidSupport
from audit_study import lever_bounds
DT=.02

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
def feasible(A,lo,hi):
    upper=np.isfinite(hi);lower=np.isfinite(lo)
    ans=linprog(np.zeros(A.shape[1]),A_ub=np.vstack((A[upper],-A[lower])),b_ub=np.r_[hi[upper],-lo[lower]],bounds=[(None,None)]*A.shape[1],method='highs')
    return dict(feasible=bool(ans.success),status=ans.status,message=ans.message)

def audit(case,kernel):
    out=ROOT/'results'/case;identity=read(out/'run_identity.json');summary=read(out/'summary.json') if (out/'summary.json').exists() else {}
    recs=list(out.glob('A-CV-*'));rec=recs[0] if recs else out
    path=rec/'q_history.npy'
    if not path.exists():path=out/'neo_pre_step_q.npy'
    if not path.exists():return dict(case=case,complete=False,error=identity['error'],saved_trajectory_available=False)
    scene=read(ROOT/'assets/scene.json');q0=np.array(scene['q0']);target=np.array(scene['waypoints'][-1])
    q=np.load(path);post_only=len(q)==summary.get('cycles')
    if post_only:q=np.vstack((q0,q))
    assert q.ndim==2 and q.shape[1]==6 and np.isfinite(q).all() and np.allclose(q[0],q0,rtol=0,atol=1e-12)
    xml=rec/'scene.xml' if (rec/'scene.xml').exists() else ROOT/'assets/scene.xml'
    m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m);robot=build_robot_certificate(m);radii=np.array([x.radius for x in robot]);assert len(robot)==65
    centers=np.array([b['center'] for b in scene['boxes']]);halves=np.array([b['half_size'] for b in scene['boxes']]);assert len(centers)==12
    feet=[i for i in range(m.ngeom) if m.geom(i).name.startswith('cabinet_ground_')];assert len(feet)==4 and np.all(m.geom_group[feet]==0)
    def boxgap(p):
        v=np.abs(p[:,None]-centers)-halves
        return np.linalg.norm(np.maximum(v,0),axis=2)+np.minimum(np.max(v,axis=2),0)-radii[:,None]
    positions=[];ees=[];penetrations=[];gap_min=np.inf;foot_min=np.inf;workspace=np.inf
    wa=np.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]]);wb=np.array([.95,.95,.95,.95,0,1.35])
    for x in q:
        set_configuration(m,d,x);pos=certificate_world_positions(d,robot);positions.append(pos);ees.append(attachment_position(m,d))
        gaps=boxgap(pos);gap_min=min(gap_min,float(gaps.min()));foot_min=min(foot_min,float(gaps[:,8:].min()))
        workspace=min(workspace,float((wb-radii[:,None]-pos@wa.T).min()))
        penetrations.append(sum(d.contact[i].dist < -1e-8 for i in range(d.ncon)))
    ees=np.array(ees);error=np.linalg.norm(ees-target,axis=1);hold=0;maxhold=0;confirmed=None
    for k,(e,pen) in enumerate(zip(error[1:],penetrations[1:]),1):
        hold=hold+1 if e<.001 and pen==0 else 0;maxhold=max(maxhold,hold)
        if hold==50 and confirmed is None:confirmed=k*DT
    if (rec/'ee_history.npy').exists():
        recorded=np.load(rec/'ee_history.npy');compare=ees[1:] if post_only else ees
        assert recorded.shape==compare.shape and np.allclose(recorded,compare,rtol=0,atol=1e-12)
    logged=summary.get('final_error_m',summary.get('final_error_mm',error[-1]*1000)/1000)
    assert abs(logged-error[-1])<1e-10
    L=lever_bounds(m,robot);lower_min=np.inf;unresolved=0;checks=0
    def interval(a,b,depth=0):
        nonlocal lower_min,unresolved,checks
        set_configuration(m,d,(a+b)/2);g=boxgap(certificate_world_positions(d,robot));checks+=1
        low=float((g-(.5*L@np.abs(b-a))[:,None]).min())
        if low>=.006:lower_min=min(lower_min,low);return
        if g.min()<.006 or depth>=6:unresolved+=1;lower_min=min(lower_min,low);return
        mid=(a+b)/2;interval(a,mid,depth+1);interval(mid,b,depth+1)
    for a,b in zip(q[:-1],q[1:]):interval(a,b)
    rep=summary.get('representation') or ('sphere' if case in ('K01','K03','H02','N04','N05','O01','O03','O05','A01') else 'ellipsoid')
    proxyfile=rec/'causal_proxy_snapshots.npz';dynamic=proxyfile.exists()
    if not dynamic:proxyfile=rec/'proxies.npz'
    proxy_min=np.inf;probe_n=0;residual=0.;probe_pairs=0
    if proxyfile.exists():
        with np.load(proxyfile) as f:z={k:f[k] for k in f.files}
        cycles=sorted(set(range(0,len(q)-1,25))|{len(q)-2})
        if dynamic:
            assert np.all(z['source_cycles']<=z['publish_cycles']);cycles=sorted(set(cycles)|set(int(p) for p in z['publish_cycles'] if p<len(q)-1))
        for cycle in cycles:
            if dynamic:
                idx=int(np.searchsorted(z['publish_cycles'],cycle,side='right'))-1
                assert idx>=0;sl=slice(z['offsets'][idx],z['offsets'][idx+1])
            else:sl=slice(None)
            c=z['centers'][sl];pos=positions[cycle+1];offset=z['uncertainty_offsets'][sl] if dynamic else np.zeros(len(c))
            if rep=='sphere':
                distances=np.linalg.norm(pos[:,None]-c,axis=2)-radii[:,None]-z['sphere_radii'][sl]-offset;gap=float(distances.min());probe_pairs+=distances.size
            else:
                Q=z['ellipsoid_shapes'][sl];U=z['proxy_uncertainty_shapes'][sl] if dynamic else np.zeros_like(Q)
                half=np.sqrt(np.maximum(np.diagonal(Q,axis1=1,axis2=2),0))+np.sqrt(np.maximum(np.diagonal(U,axis1=1,axis2=2),0))+offset[:,None]
                low=np.linalg.norm(np.maximum(np.abs(pos[:,None]-c)-half,0),axis=2)-radii[:,None]
                ri,oi=np.nonzero(low<.050)
                if not len(ri):continue
                R=np.eye(3)[None]*radii[ri,None,None]**2
                n,it,res=kernel.normals_sum_pairs_newton_warm(pos[ri],R,c[oi],Q[oi],U[oi],np.zeros((len(ri),3)),max_iterations=64)
                residual=max(residual,float(res.max()));probe_pairs+=len(ri)
                dd=np.einsum('ij,ij->i',c[oi]-pos[ri],n)-radii[ri]-offset[oi]
                for shape in (Q[oi],U[oi]):dd-=np.sqrt(np.maximum(np.einsum('ni,nij,nj->n',n,shape,n),0))
                gap=float(dd.min())
            proxy_min=min(proxy_min,gap);probe_n+=1
    cameras={}
    if (rec/'causal_source_configurations.npz').exists():
        configs=np.load(rec/'causal_source_configurations.npz');qs=configs['q'];total=0;seen=set();frames=0
        with UR5MountedDepthCamera(m,camera_names=('ur5_depth_wrist','ur5_depth_forearm'),width=320,height=180,pixel_stride=1,minimum_range=.07,optical_depth_error_bound=.003,occluding_self_filter=True) as cam:
            for x in qs:
                set_configuration(m,d,x)
                for obs in cam.capture(d):
                    mask=np.isin(obs.geom_ids,feet);total+=int(mask.sum());seen.update(int(i) for i in obs.geom_ids[mask]);frames+=1
        cameras=dict(published_source_poses=len(qs),reconstructed_camera_frames=frames,foot_depth_returns=total,observed_feet=[m.geom(i).name for i in sorted(seen)],observations_recomputed_from_actual_run_source_poses=True)
    result=dict(case=case,complete=identity['completed'],saved_trajectory_available=True,states=len(q),executed_cycles=len(q)-1,duration_s=(len(q)-1)*DT,representation=rep,final_error_mm=float(error[-1]*1000),minimum_error_mm=float(error.min()*1000),confirmed_time_s=confirmed,final_hold_cycles=hold,max_hold_cycles=maxhold,ever_sustained_success=maxhold>=50,final_task_success=hold>=50,penetrating_cycles=sum(p>0 for p in penetrations[1:]),new_scene_foot_geoms=4,minimum_true_box_endpoint_gap_mm=gap_min*1000,minimum_foot_endpoint_gap_mm=foot_min*1000,continuous_12_box_gap_lower_bound_mm=lower_min*1000,continuous_12_box_6mm_certified=unresolved==0,interval_checks=checks,unresolved_intervals=unresolved,minimum_workspace_slack_mm=workspace*1000,velocity_limit_excess_rad_s=float(np.max(np.abs(np.diff(q,axis=0)/DT)-np.array([1,1,1,1.4,1.4,1.4]))) if len(q)>1 else None,proxy_probe_cycles=probe_n,proxy_probe_pairs=probe_pairs,minimum_probed_proxy_gap_mm=proxy_min*1000 if probe_n else None,maximum_support_residual=residual,proxy_probe_scope='Every 25 cycles, each published map, and terminal; no continuous proxy proof.',camera_audit=cameras,source_summary_evidence_eligible=summary.get('evidence_eligible'),full_self_floor_dynamics_certification=False)
    failure_record=read(out/'failure.json') if (out/'failure.json').exists() else {}
    failure_text=failure_record.get('error',identity.get('error') or '')
    geometry_failure='support KKT residual' in failure_text
    result['failure_stage']='distance_kernel_precision' if geometry_failure else 'controller_or_runner' if not identity['completed'] else None
    result['failure_detail']=failure_text if not identity['completed'] else None
    fail=out/'failed_qp.npz'
    result['saved_qp_valid_for_failure']=bool(fail.exists() and not geometry_failure)
    if fail.exists() and geometry_failure:
        dump(out/'failed_qp_validity.json',dict(valid_for_failure=False,reason='Geometry failed before assembling a new QP; saved candidate is from the previous successful assembly and cannot diagnose current infeasibility.'))
    if fail.exists() and not geometry_failure:
        z=np.load(fail);A,lo,hi=z['A'],z['lo'],z['hi'];base=6+3+65*6
        result['failure_feasibility']=dict(all=feasible(A,lo,hi),no_obstacles=feasible(A[:base],lo[:base],hi[:base]),obstacles_only=feasible(A[base:],lo[base:],hi[base:]))
    np.save(out/'audited_q.npy',q);np.save(out/'audited_error_mm.npy',error*1000)
    dump(out/'independent_new_scene_audit.json',result)
    return result

def main():
    CASES=[c["id"] for c in read(ROOT.parent/"catalog.json")["cases"]]
    result={};kernel=NativeEllipsoidSupport();kernel.set_pair_threads(8)
    for case in CASES:
        result[case]=audit(case,kernel);dump(ROOT/'audits/results.json',result)
        r=result[case];print(case,'complete',r['complete'],'error',r.get('final_error_mm'),'12-box 6mm',r.get('continuous_12_box_6mm_certified'),flush=True)
    manifest=read(ROOT/'formal_runtime_manifest.json')
    for k,v in manifest.items():assert hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v,k
    dump(ROOT/'audits/source_integrity.json',dict(files=len(manifest),all_runtime_sources_unchanged=True))
    print('All result audits and source checks finished.',flush=True)
if __name__=='__main__':main()
