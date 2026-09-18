"""Replay actual known-volume LiuQP/NEO trajectories; no live control."""
import argparse,json,queue,time
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
from run_comparison import HERE,ROOT,original,CASES
from robot import build_model,build_robot_certificate,certificate_world_positions,set_configuration,DT
from view import append_geom,shape_to_axes_rotation,IDENTITY,TARGET_RGBA,ROBOT_RGBA,ELLIPSOID_RGBA
LABELS={'liuqp':'1 LiuQP / ellipsoids','neo_matched':'2 NEO core / influence 46 mm','neo_paper_influence':'3 NEO core / influence 300 mm'}
def load():
    results={}
    for case in CASES:
        f=HERE/'results'/case
        results[case]=dict(q=np.load(f/'q_history.npy'),ee=np.load(f/'ee_history.npy'),summary=json.loads((f/'summary.json').read_text()))
        assert results[case]['q'].shape==(3001,6)
    return results

def camera_setup(cam):
    cam.lookat[:]=[.42,.32,.52];cam.distance=1.55;cam.azimuth=10.;cam.elevation=-8.

def display_lighting(model):
    model.vis.headlight.ambient[:]=[.5,.5,.5]
    model.vis.headlight.diffuse[:]=[.7,.7,.7]

def draw(overlay,data,robot,proxies,target,show_proxies=False,show_robot=False):
    if show_proxies:
        for center,shape in zip(proxies['centers'],proxies['ellipsoid_shapes']):
            axes,rotation=shape_to_axes_rotation(shape)
            append_geom(overlay,mujoco.mjtGeom.mjGEOM_ELLIPSOID,axes,center,rotation,ELLIPSOID_RGBA)
    if show_robot:
        for p,s in zip(certificate_world_positions(data,robot),robot):
            append_geom(overlay,mujoco.mjtGeom.mjGEOM_SPHERE,np.full(3,s.radius),p,IDENTITY,ROBOT_RGBA)
    append_geom(overlay,mujoco.mjtGeom.mjGEOM_SPHERE,np.full(3,.008),target,IDENTITY,TARGET_RGBA)

def render(results):
    from PIL import Image,ImageDraw,ImageFont
    model=build_model(original.load_scene());data=mujoco.MjData(model);robot=build_robot_certificate(model)
    model.vis.global_.offwidth=960;model.vis.global_.offheight=720
    display_lighting(model)
    cam=mujoco.MjvCamera();camera_setup(cam)
    proxies=np.load(ROOT/'results'/'ellipsoid'/'proxies.npz');target=np.array(original.load_scene().waypoints[-1])
    renderer=mujoco.Renderer(model,height=720,width=960)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',23)
    try:
        for case,r in results.items():
            for name,frame in [('at_6s',300),('final',3000)]:
                set_configuration(model,data,r['q'][frame]);renderer.update_scene(data,camera=cam)
                draw(renderer.scene,data,robot,proxies,target)
                im=Image.fromarray(renderer.render());d=ImageDraw.Draw(im)
                d.rectangle((0,0,960,72),fill=(18,22,30))
                err=np.linalg.norm(r['ee'][frame]-target)*1000
                d.text((18,8),LABELS[case],font=font,fill='white')
                d.text((18,39),f't={frame*DT:.2f} s   error={err:.6f} mm',font=font,fill='white')
                im.save(HERE/'results'/f'{case}_{name}.png')
    finally: renderer.close()
    print('Rendered actual replay frames.')

def replay(results,case,speed):
    scene=original.load_scene();model=build_model(scene);data=mujoco.MjData(model);robot=build_robot_certificate(model)
    proxies=np.load(ROOT/'results'/'ellipsoid'/'proxies.npz');target=np.array(scene.waypoints[-1]);cmd=queue.Queue()
    def key(k):
        if 0<=k<128:cmd.put(chr(k).upper())
    frame=0;paused=False;show_proxies=False;show_robot=False;last=time.perf_counter()
    display_lighting(model)
    with mujoco.viewer.launch_passive(model,data,key_callback=key) as viewer:
        camera_setup(viewer.cam)
        while viewer.is_running():
            one=False
            while not cmd.empty():
                c=cmd.get_nowait()
                if c in ['1','2','3']:case=list(CASES)[int(c)-1];frame=0;paused=False
                elif c==' ':paused=not paused
                elif c=='0':frame=0;paused=False
                elif c=='F':frame=3000;paused=True
                elif c=='S':frame=round(results[case]['summary']['confirmed_success_time_s']/DT);paused=True
                elif c=='V':show_proxies=not show_proxies
                elif c=='R':show_robot=not show_robot
                elif c=='N':one=True;paused=True
            r=results[case];now=time.perf_counter()
            if one or (not paused and now-last>=DT/speed):
                frame=min(frame+1,3000);last=now
                # Pause once success has been held for 50 steps; full path remains available.
                if frame==round(r['summary']['confirmed_success_time_s']/DT):paused=True
            set_configuration(model,data,r['q'][frame]);viewer.user_scn.ngeom=0
            draw(viewer.user_scn,data,robot,proxies,target,show_proxies,show_robot)
            err=float(np.linalg.norm(r['ee'][frame]-target)*1000)
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                LABELS[case],f't={frame*DT:.2f}s  error={err:.6f}mm  '+('PAUSED' if paused else 'PLAY')+'\n'
                'Same robot, goal, 368 ellipsoids, safety 6mm\n1/2/3 switch | Space pause | 0 restart | S success | F final | V proxies | R robot spheres'))
            viewer.sync();time.sleep(.001)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');ap.add_argument('--render',action='store_true');ap.add_argument('--initial',choices=list(CASES),default='neo_matched');ap.add_argument('--speed',type=float,default=1.)
    args=ap.parse_args();results=load()
    if args.check:
        model=build_model(original.load_scene());data=mujoco.MjData(model)
        for r in results.values():
            for q in r['q'][::250]:set_configuration(model,data,q)
        print('PASS: all three 3001-state MuJoCo replays load correctly.')
    elif args.render:render(results)
    else:replay(results,args.initial,args.speed)
if __name__=='__main__':main()
