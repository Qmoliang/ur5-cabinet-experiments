"""Scene validation, exact-box diagnostic, and simple MuJoCo replay for 07.3."""
from pathlib import Path
import argparse, json, hashlib, time
from collections import Counter
import numpy as np
import mujoco
from model import build_model, build_xml, build_robot_certificate, set_configuration, attachment_position, certificate_world_positions
from official_drawer_07_3 import scene
from experiment_07_scenes import experiment_07_drawer_scene
from run_experiment_07 import FORMAL_INITIAL_OFFSETS_RAD
from protocol_liuqp_controller import ProtocolLiuQPController, ProtocolSeparatingPlane
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'formal_results/experiment_07/development_07_3'
ASSET=ROOT/'assets/experiment_07_3'

def distances(pos, boxes, radii):
    centers=np.array([b.center for b in boxes]); half=np.array([b.half_size for b in boxes])
    d=np.abs(pos[:,None,:]-centers)-half
    signed=np.linalg.norm(np.maximum(d,0),axis=2)+np.minimum(d.max(axis=2),0)
    return signed-radii[:,None]

class ExactBoxController(ProtocolLiuQPController):
    """Diagnostic only: true boxes supply planes; all QP assembly is inherited."""
    def __init__(self,m,d,s,r):
        super().__init__(m,d,s,r,np.array([b.center for b in s.boxes]),representation='sphere',obstacle_radii=np.zeros(len(s.boxes)),native_exact_batch=False,native_exact_pair_batch=False,redundant_plane_pruning=False,ellipsoid_pair_threads=2)
        self.boxes=s.boxes
        self.osqp_absolute_tolerance=self.osqp_relative_tolerance=1e-4
        self.osqp_max_iterations=1000
    def active_planes_for_robot_sphere(self,ri,center,radius,*unused):
        planes=[]
        for i,b in enumerate(self.boxes):
            delta=center-np.array(b.center);half=np.array(b.half_size)
            nearest=np.clip(delta,-half,half)+b.center
            v=nearest-center; dist=np.linalg.norm(v)
            if dist>1e-12:normal=v/dist; signed=dist
            else:
                slack=half-np.abs(delta); axis=int(np.argmin(slack))
                outward=np.zeros(3);outward[axis]=1 if delta[axis]>=0 else -1
                normal=-outward;signed=-slack[axis];nearest=center+outward*slack[axis]
            h=signed-radius-self.safety_margin
            if h>self.near_distance:continue
            planes.append(ProtocolSeparatingPlane(ri,i,i,normal,nearest,float(normal@nearest),float(h)))
        self._last_support_batch_stats=(0,0.,0);self._last_closest_batch_stats=(0,0,0.,0,False)
        return planes,len(self.boxes),0,0.,0.

def camera():
    c=mujoco.MjvCamera();mujoco.mjv_defaultCamera(c)
    c.lookat[:]=[.34,.22,.42];c.distance=2.15;c.azimuth=35;c.elevation=-20
    return c

def render(m,d,path):
    from PIL import Image
    opt=mujoco.MjvOption();opt.geomgroup[1]=0;opt.geomgroup[5]=1
    with mujoco.Renderer(m,height=900,width=1440) as r:
        r.update_scene(d,camera=camera(),scene_option=opt);Image.fromarray(r.render()).save(path)

def validate():
    ASSET.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    s=scene();old=experiment_07_drawer_scene();m=build_model(s);ref=build_model(old)
    a=build_robot_certificate(m);b=build_robot_certificate(ref)
    assert len(a)==len(b)==65
    for x,y in zip(a,b):
        assert x.body_name==y.body_name and x.radius==y.radius and np.array_equal(x.local_center,y.local_center)
    for box in old.boxes:
        g=m.geom(box.name);oldg=ref.geom(box.name)
        assert np.array_equal(g.pos,oldg.pos) and np.array_equal(g.size,oldg.size)
    for name in ['ur5_depth_wrist','ur5_depth_forearm']:
        for attr in ['pos','quat','fovy']:
            assert np.array_equal(getattr(m.camera(name),attr),getattr(ref.camera(name),attr))
    d=mujoco.MjData(m);radii=np.array([r.radius for r in a]);initial=[]
    for seed,off in enumerate(FORMAL_INITIAL_OFFSETS_RAD):
        set_configuration(m,d,np.array(s.q0)+off)
        gap=distances(certificate_world_positions(d,a),s.boxes,radii)
        penetrations=sum(d.contact[i].dist < -1e-8 for i in range(d.ncon))
        assert penetrations==0
        initial.append(dict(seed=seed,minimum_box_gap_mm=float(gap.min()*1000),penetrations=int(penetrations)))
    (ASSET/'scene.xml').write_text(build_xml(s),encoding='utf-8')
    sources=['official_drawer_07_3.py','model.py','third_party/iris_benchmarks/iris_environments/assets/shelves1.sdf','assets/experiment_07_3/scene.xml']
    manifest=dict(scene=s.name,adapted_official_asset=True,original_4shelves=False,robot_certificate_equal=True,cameras_equal=True,old_boxes_equal=True,initial_states=initial,visual_mesh_group=5,visual_meshes_excluded_from_depth=True,sources={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources})
    (ASSET/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    set_configuration(m,d,s.q0);render(m,d,ASSET/'scene_preview.png')
    print(json.dumps(manifest),flush=True)

def run():
    OUT.mkdir(parents=True,exist_ok=True);s=scene();m=build_model(s);d=mujoco.MjData(m);robot=build_robot_certificate(m);radii=np.array([r.radius for r in robot]);target=np.array(s.waypoints[-1]);results=[]
    for seed,offset in enumerate(FORMAL_INITIAL_OFFSETS_RAD):
        q=np.array(s.q0)+offset;set_configuration(m,d,q);c=ExactBoxController(m,d,s,robot)
        hist=[q.copy()];trace=[];hold=0;first=None;min_gap=np.inf;penetrations=0;statuses=Counter()
        for k in range(2250):
            v,met=c.solve(target);statuses[met.status]+=1
            q=q+.02*v;set_configuration(m,d,q);hist.append(q.copy())
            gap=distances(certificate_world_positions(d,robot),s.boxes,radii);min_gap=min(min_gap,float(gap.min()))
            pen=any(d.contact[j].dist < -1e-8 for j in range(d.ncon));penetrations+=int(pen)
            err=float(np.linalg.norm(target-attachment_position(m,d)));hold=hold+1 if err<.001 and not pen else 0
            if hold>=50 and first is None:first=(k+1)*.02
            if k%50==0 or k==2249:trace.append(dict(cycle=k,time_s=(k+1)*.02,error_mm=err*1000,gap_mm=float(gap.min()*1000)))
        folder=OUT/f'exact_box_seed{seed}';folder.mkdir(exist_ok=True);np.save(folder/'q_history.npy',np.array(hist));(folder/'scene.xml').write_text(build_xml(s),encoding='utf-8')
        result=dict(seed=seed,diagnostic_true_geometry=True,final_error_mm=1000*err,first_hold50_s=first,final_hold=hold,minimum_discrete_certificate_gap_mm=min_gap*1000,penetrating_cycles=penetrations,statuses=dict(statuses),trace=trace,continuous_audit=False)
        (folder/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');results.append(result)
        (OUT/'exact_geometry.json').write_text(json.dumps(dict(diagnostic_only=True,near_terms_count_boxes_not_proxies=True,runs=results),indent=2),encoding='utf-8')
        render(m,d,folder/'final.png');print(json.dumps({k:v for k,v in result.items() if k!='trace'}),flush=True)

def view(folder):
    from mujoco import viewer
    m=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));d=mujoco.MjData(m);q=np.load(folder/'q_history.npy') if (folder/'q_history.npy').exists() else np.array([scene().q0]);state={'pause':False,'k':0}
    def key(k):
        if k==32:state['pause']=not state['pause']
        if k==48:state['k']=0
    with viewer.launch_passive(m,d,key_callback=key) as v:
        v.opt.geomgroup[1]=0;v.opt.geomgroup[5]=1;c=camera()
        for name in ['lookat','distance','azimuth','elevation']:setattr(v.cam,name,getattr(c,name))
        while v.is_running():
            set_configuration(m,d,q[state['k']]);v.sync()
            if not state['pause']:state['k']=min(state['k']+1,len(q)-1)
            time.sleep(.02)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--validate',action='store_true');ap.add_argument('--run',action='store_true');ap.add_argument('--view',type=Path);args=ap.parse_args()
    if args.validate:validate()
    if args.run:run()
    if args.view:view(args.view)
