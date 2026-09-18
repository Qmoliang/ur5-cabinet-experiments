"""Causal proxy replay for the three no-manipulability online trials, including partial failure."""
from pathlib import Path
import sys,json,queue,time,argparse
import numpy as np
import mujoco
import mujoco.viewer
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];sys.path.insert(0,str(BASE))
from model import build_robot_certificate,certificate_world_positions,attachment_position,set_configuration,DT
from protocol_drawer_scene import formal_drawer_camera_quarter_scene
from formal_protocol_v3_viewer import RaggedSnapshots
CASES={'neo_sphere_300':('neo','sphere',.3),'neo_ellipsoid_300':('neo','ellipsoid',.3),'neo_ellipsoid_46':('neo','ellipsoid',.046)}

def append(scene,kind,size,pos,rotation,color):
    if scene.ngeom>=scene.maxgeom:return
    g=scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g,kind,np.asarray(size,float),np.asarray(pos,float),np.asarray(rotation,float).ravel(),np.asarray(color,np.float32))
    scene.ngeom+=1

def camera(cam):
    cam.lookat[:]=[.42,.32,.52];cam.distance=1.55;cam.azimuth=10;cam.elevation=-8

def load():
    results=[]
    for case,(method,rep,di) in CASES.items():
        outer=HERE/'results'/case;completion=json.loads((outer/'completion.json').read_text(encoding='utf-8'))
        folder=next(outer.glob('A-CV-*'));model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'))
        model.vis.headlight.ambient[:]=[.5,.5,.5];model.vis.headlight.diffuse[:]=[.7,.7,.7]
        q=np.load(folder/'q_history.npy') if completion['complete'] else np.load(outer/'neo_pre_step_q.npy')
        assert len(q)>0 and np.isfinite(q).all()
        summary=json.loads((outer/'summary.json').read_text(encoding='utf-8')) if completion['complete'] else None
        results.append(dict(case=case,model=model,data=mujoco.MjData(model),robot=build_robot_certificate(model),
                            q=q,summary=summary,complete=completion['complete'],kind=rep,
                            proxies=RaggedSnapshots(folder/'causal_proxy_snapshots.npz') if completion['complete'] else None))
    return results

def draw(scene,r,frame,proxies=False,robot=False):
    if proxies and r['proxies'] is not None:
        _,p=r['proxies'].at_cycle(frame)
        for i,center in enumerate(p['centers']):
            if r['kind']=='sphere':
                append(scene,mujoco.mjtGeom.mjGEOM_SPHERE,np.full(3,p['sphere_radii'][i]+p['uncertainty_offsets'][i]),center,np.eye(3),[.2,.65,1,.16])
            else:
                values,rotation=np.linalg.eigh(p['ellipsoid_shapes'][i])
                if np.linalg.det(rotation)<0:rotation[:,0]*=-1
                append(scene,mujoco.mjtGeom.mjGEOM_ELLIPSOID,np.sqrt(np.maximum(values,0)),center,rotation,[.2,.65,1,.25])
    if robot:
        for pos,s in zip(certificate_world_positions(r['data'],r['robot']),r['robot']):
            append(scene,mujoco.mjtGeom.mjGEOM_SPHERE,np.full(3,s.radius),pos,np.eye(3),[.2,1,.4,.2])
    append(scene,mujoco.mjtGeom.mjGEOM_SPHERE,np.full(3,.008),formal_drawer_camera_quarter_scene().waypoints[-1],np.eye(3),[1,.2,.1,.8])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');ap.add_argument('--render',action='store_true');args=ap.parse_args()
    results=load();target=np.array(formal_drawer_camera_quarter_scene().waypoints[-1])
    if args.check:
        for r in results:
            for frame in [0,len(r['q'])//2,len(r['q'])-1]:
                set_configuration(r['model'],r['data'],r['q'][frame])
                if r['proxies'] is not None:
                    i,p=r['proxies'].at_cycle(frame)
                    assert r['proxies'].source_cycles[i]<=r['proxies'].publish_cycles[i]<=frame
        print('PASS: three replays load, including the aborted partial run; proxies selected causally.');return
    if args.render:
        from PIL import Image,ImageDraw,ImageFont
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',23)
        for r in results:
            model=r['model'];model.vis.global_.offwidth=1100;model.vis.global_.offheight=800
            renderer=mujoco.Renderer(model,height=800,width=1100);cam=mujoco.MjvCamera();camera(cam)
            try:
                set_configuration(model,r['data'],r['q'][-1]);renderer.update_scene(r['data'],camera=cam)
                draw(renderer.scene,r,len(r['q'])-1)
                im=Image.fromarray(renderer.render());canvas=ImageDraw.Draw(im)
                canvas.rectangle((0,0,1100,78),fill=(18,22,30))
                error=np.linalg.norm(attachment_position(model,r['data'])-target)*1000
                canvas.text((18,8),r['case']+' | '+('complete 60 s trial' if r['complete'] else 'ABORTED: last recorded state'),font=font,fill='white')
                canvas.text((18,43),f'error={error:.3f} mm',font=font,fill='white')
                im.save(HERE/'results'/(r['case']+'_final.png'))
            finally:renderer.close()
        print('Three actual final/partial-state images rendered.');return
    commands=queue.Queue()
    def key(k):
        if 0<=k<128:commands.put(chr(k).upper())
    # All trials use byte-identical scenes, so one viewer can replay them.
    r=results[2];model=r['model'];data=r['data'];case=2;frame=0;paused=False;show=False;show_robot=False;last=time.perf_counter()
    with mujoco.viewer.launch_passive(model,data,key_callback=key) as viewer:
        camera(viewer.cam)
        while viewer.is_running():
            while not commands.empty():
                c=commands.get_nowait()
                if c in '123':case=int(c)-1;frame=0;paused=False
                elif c==' ':paused=not paused
                elif c=='0':frame=0;paused=False
                elif c=='F':frame=len(results[case]['q'])-1;paused=True
                elif c in 'SBA':
                    case=2;frame=round({'S':6.36,'B':41.96,'A':42.50}[c]/DT)-1;paused=True
                elif c=='V':show=not show
                elif c=='R':show_robot=not show_robot
            r=results[case];now=time.perf_counter()
            if not paused and now-last>=DT:
                frame=min(frame+1,len(r['q'])-1);last=now
                if frame==len(r['q'])-1:paused=True
            set_configuration(model,data,r['q'][frame]);viewer.user_scn.ngeom=0
            # draw uses the viewer's current data, not a stale per-run MjData.
            display=dict(r,data=data);draw(viewer.user_scn,display,frame,show,show_robot)
            error=np.linalg.norm(attachment_position(model,data)-target)*1000
            text=f't={(frame+int(r["complete"]))*DT:.2f}s  error={error:.3f}mm  '+('PAUSED' if paused else 'PLAY')+'\n'
            text+='NO MANIPULABILITY OBJECTIVE\n1 sphere300 abort | 2 ellipsoid300 abort | 3 ellipsoid46\n'
            text+='Space pause | 0 restart | F final | V proxies | R robot balls\nS reached (6.36s) | B before map (41.96s) | A after map (42.50s)\n'
            text+=('Ellipsoid overlay: core only; controller also includes U + residual.' if r['kind']=='ellipsoid' else 'Sphere overlay: effective radius + residual.')
            if not r['complete']:text+='\nABORTED: proxy history unavailable for this partial run.'
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,r['case'],text))
            viewer.sync();time.sleep(.001)

if __name__=='__main__':main()
